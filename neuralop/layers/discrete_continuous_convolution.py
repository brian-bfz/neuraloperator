import abc
import math
import numpy

import torch
import torch.nn as nn
import os

from typing import Union, Optional, Literal

# import the base class from torch-harmonics
try:
    from torch_harmonics.quadrature import _precompute_grid
    from torch_harmonics.filter_basis import (
        PiecewiseLinearFilterBasis,
        PiecewiseLinearFilterBasis3d,
        MorletFilterBasis,
        MorletFilterBasis3d,
        ZernikeFilterBasis,
    )
except ModuleNotFoundError:
    print(
        "Error: trying to import DISCO convolutions without optional dependency torch-harmonics. ",
        "Please install with `pip install torch-harmonics` and retry.",
    )

basis_type_classes = {
    "piecewise_linear": PiecewiseLinearFilterBasis,
    "piecewise_linear3d": PiecewiseLinearFilterBasis3d,
    "morlet": MorletFilterBasis,
    "morlet3d": MorletFilterBasis3d,
    "zernike": ZernikeFilterBasis,
}


def _normalize_convolution_filter_matrix(
    psi_idx,
    psi_vals,
    grid_in,
    grid_out,
    kernel_shape,
    quadrature_weights,
    transpose_normalization=False,
    eps=1e-9,
):
    """
    Discretely normalizes the convolution tensor.
    """

    n_in = grid_in.shape[-1]
    n_out = grid_out.shape[-2]

    if len(kernel_shape) == 1:
        kernel_size = math.ceil(kernel_shape[0] / 2)
    elif len(kernel_shape) == 2:
        kernel_size = (kernel_shape[0] // 2) * kernel_shape[1] + kernel_shape[0] % 2
    elif len(kernel_shape) == 3:
        kernel_size = (
            (kernel_shape[0] // 2) * kernel_shape[1] * kernel_shape[2]
            + kernel_shape[0] % 2
        )
    else:
        raise ValueError(f"Unsupported kernel_shape dimensionality: {kernel_shape}")

    # # reshape the indices implicitly to be ikernel, n_in, n_out
    # idx = torch.stack([psi_idx[0], psi_idx[1], psi_idx[2] // nlon_in, psi_idx[2] % nlon_in], dim=0)
    idx = psi_idx

    if transpose_normalization:
        # pre-compute the quadrature weights
        q = quadrature_weights[idx[1]].reshape(-1)

        # loop through dimensions which require normalization
        for ik in range(kernel_size):
            for iin in range(n_in):
                # get relevant entries
                iidx = torch.argwhere((idx[0] == ik) & (idx[2] == iin))
                # normalize, while summing also over the input longitude dimension here as this is not available for the output
                vnorm = torch.sum(psi_vals[iidx] * q[iidx])
                psi_vals[iidx] = psi_vals[iidx] / (vnorm + eps)
    else:
        # pre-compute the quadrature weights
        q = quadrature_weights[idx[2]].reshape(-1)

        # loop through dimensions which require normalization
        for ik in range(kernel_size):
            for iout in range(n_out):
                # get relevant entries
                iidx = torch.argwhere((idx[0] == ik) & (idx[1] == iout))
                # normalize
                vnorm = torch.sum(psi_vals[iidx] * q[iidx])
                psi_vals[iidx] = psi_vals[iidx] / (vnorm + eps)

    return psi_vals


def _precompute_convolution_filter_matrix(
    grid_in,
    grid_out,
    kernel_shape,
    quadrature_weights,
    normalize=True,
    basis_type="piecewise_linear",
    radius_cutoff=0.01,
    periodic=False,
    transpose_normalization=False,
):
    """
    Precomputes the values stored in Psi, the local convolution filter matrix.
    The values are the results of a set of kernel basis "hat" functions applied to
    pairwise distances between each points on the input and output grids.

    The hat functions are the absolute differences between a squared distance and a
    multiple of the radius scaled by the kernel size.

    Assume the kernel is an array of shape ``(k0, k1)``. Then:

    If the kernel is isotropic (``k0 == k1``), the basis functions are a series of
    ``k0`` distances re-centered around multiples of the discretization size of the
    convolution's radius. If the kernel is anisotropic, the outputs of these hat
    functions are then multiplied by the outputs of another series of ``k1`` hat
    functions evaluated on the arctangents of these pairwise distances.

    Compared to the ``torch_harmonics`` routine for spherical support values, this
    function also returns the translated filters at positions
    $T^{-1}_j \omega_i = T^{-1}_j T_i \nu$, but assumes a non-periodic subset of the
    euclidean plane.

    Parameters
    ----------
    grid_in : ``torch.Tensor``
        coordinate grid on which input function is provided
    grid_out : ``torch.Tensor``
        coordinate grid on which output function is queried
    kernel_shape: ``Union[int, List[int]]``
        Dimensions of the convolution kernel, either one int or a two-int tuple.
        If one int k, the kernel will be a square of shape (k,k), meaning the convolution
        will be 'isotropic': both directions are equally scaled in feature space
        If two ints (k1,k2), the kernel will be a rectangle of shape (k1,k2), meaning the convolution
        will  be 'anisotropic': one direction will be compressed or stretched in feature space.
    quadrature_weights : ``torch.Tensor``
        tensor of weights for each grid point in the input
    normalize : ``bool``, optional
        whether to normalize the precomputed filter tensor, by default True
    basis_type: str literal ``{'piecewise_linear', 'morlet', 'zernike'}``
        choice of basis functions to use for convolution filter tensor.
    radius_cutoff : ``float``, optional
        radius cutoff parameter for computing basis functions, by default 0.01
    periodic : ``bool``, optional
        whether the domain is assumed to be periodic, by default False
    transpose_normalization : ``bool``, optional
        whether to transpose the normalized filter tensor, by default False

    """

    # check that input arrays are valid point clouds in 2D
    assert len(grid_in) == 2, "grid_in must be a 2d tensor."
    assert len(grid_out) == 2, "grid_out must be a 2d tensor."
    assert grid_in.shape[0] == 2, "grid_in must be a 2d tensor."
    assert grid_out.shape[0] == 2, "grid_out must be a 2d tensor."

    n_in = grid_in.shape[-1]
    n_out = grid_out.shape[-1]

    grid_in = grid_in.reshape(2, 1, n_in)
    grid_out = grid_out.reshape(2, n_out, 1)

    diffs = grid_in - grid_out
    if periodic:
        periodic_diffs = torch.where(diffs > 0.0, diffs - 1, diffs + 1)
        diffs = torch.where(diffs.abs() < periodic_diffs.abs(), diffs, periodic_diffs)

    r = torch.sqrt(diffs[0] ** 2 + diffs[1] ** 2)
    phi = torch.arctan2(diffs[1], diffs[0]) + torch.pi

    assert basis_type in basis_type_classes.keys(), f"Error: expected one of "
    f"{basis_type_classes.keys()}, got {basis_type}"

    # if a basis name is provided, instantiate it with the provided kernel shape
    basis_type = basis_type_classes[basis_type](kernel_shape)
    idx, vals = basis_type.compute_support_vals(r, phi, r_cutoff=radius_cutoff)

    idx = idx.permute(1, 0)

    if normalize:
        vals = _normalize_convolution_filter_matrix(
            idx,
            vals,
            grid_in,
            grid_out,
            kernel_shape,
            quadrature_weights,
            transpose_normalization=transpose_normalization,
        )

    return idx, vals

def _precompute_convolution_filter_matrix_3d(
    grid_in,
    grid_out,
    kernel_shape,
    quadrature_weights,
    basis_type="piecewise_linear3d",
    radius_cutoff=0.01,
    periodic=False,
    normalize=True,
    transpose_normalization=False,
):
    # Validate dimensions (3D)
    assert grid_in.shape[0] == 3, "grid_in must be 3D coordinates."
    assert grid_out.shape[0] == 3, "grid_out must be 3D coordinates."

    n_in = grid_in.shape[-1]
    n_out = grid_out.shape[-1]

    # reshape for broadcasting: [3, 1, N_in] and [3, N_out, 1]
    grid_in = grid_in.reshape(3, 1, n_in)    # [3, 1, N]
    grid_out = grid_out.reshape(3, n_out, 1) # [3, M, 1]

    # Pairwise differences -> [3, M, N]
    diffs = grid_in - grid_out

    # Periodic wrap (componentwise)
    if periodic:
        periodic_diffs = torch.where(diffs > 0.0, diffs - 1.0, diffs + 1.0)
        diffs = torch.where(diffs.abs() < periodic_diffs.abs(), diffs, periodic_diffs)

    # radial distance r: [M, N]
    r = torch.sqrt((diffs ** 2).sum(dim=0) + 1e-12)

    # spherical angles (same shapes [M, N])
    # phi: azimuth angle in [-pi, pi]; add +pi if you want [0,2pi)
    # phi = torch.atan2(diffs[1], diffs[0])  # shape [M, N]
    phi = torch.atan2(diffs[1], diffs[2])
    # polar angle theta in [0, pi]; clamp safe division
    # theta = torch.acos(torch.clamp(diffs[2] / (r + 1e-12), -1.0, 1.0))
    theta = torch.acos(torch.clamp(diffs[0] / (r + 1e-12), -1.0, 1.0))

    # Instantiate basis
    assert basis_type in basis_type_classes, f"Unknown basis {basis_type}"
    basis = basis_type_classes[basis_type](kernel_shape)

    # Compute sparse support representation. We pass r, theta, phi.
    # Note: basis.compute_support_vals must accept (r, theta, phi, r_cutoff=...)
    idx, vals = basis.compute_support_vals(r, theta, phi, r_cutoff=radius_cutoff)

    # Some basis implementations may return idx in shape [K, nnz] or [nnz, K].
    # We need to ensure idx has shape [4, nnz] where rows are:
    # [basis_idx, output_idx, input_idx, unused]
    # First, normalize to [K, nnz] format (transpose if needed)
    if idx.dim() == 2 and idx.shape[0] > idx.shape[1]:
        # idx is [nnz, K] with nnz > K, transpose to [K, nnz]
        idx = idx.permute(1, 0).contiguous()
    # If idx.shape[0] < idx.shape[1], it's already [K, nnz] format, no transpose needed
    
    # Now idx should be [K, nnz]. Ensure it has 4 rows
    K, nnz = idx.shape
    if K == 2:
        # Basis returned [basis_idx, input_idx] (2 rows)
        # Since we have a single output point (grid_out has shape [3, 1]), output_idx should be 0
        # The second row is the input index into the input grid
        basis_idx = idx[0, :]  # [nnz]
        input_idx = idx[1, :]  # [nnz] - linear index into input grid
        
        # Construct proper idx format: [4, nnz]
        # Format: [basis_idx, output_idx, input_idx, unused]
        # Since M=1 (single output point), output_idx is always 0
        idx = torch.stack([
            basis_idx,
            torch.zeros(nnz, dtype=idx.dtype, device=idx.device),  # output_idx = 0
            input_idx,
            torch.zeros(nnz, dtype=idx.dtype, device=idx.device)   # unused = 0
        ], dim=0)
    elif K < 4:
        # Pad with zeros if needed
        padding = torch.zeros(4 - K, nnz, dtype=idx.dtype, device=idx.device)
        idx = torch.cat([idx, padding], dim=0)
    elif K > 4:
        # Truncate to 4 rows
        idx = idx[:4, :]

    # If normalization requested
    if normalize:
        vals = _normalize_convolution_filter_matrix(
            idx,
            vals,
            grid_in,
            grid_out,
            kernel_shape,
            quadrature_weights,
            transpose_normalization=transpose_normalization,
        )

    return idx, vals


class DiscreteContinuousConv(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for DISCO convolutions, reproduced with permission
    from ``torch_harmonics.convolution``. If you use DISCO convs, please cite
    [1]_ and [2]_.

    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    kernel_shape : ``Union[int, List[int]]``
        Dimensions of the convolution kernel, either one int or a two-int tuple.
        * If one int k, the kernel will be a square of shape (k,k), meaning the convolution
        will be 'isotropic': both directions are equally scaled in feature space.

        * If two ints (k1,k2), the kernel will be a rectangle of shape (k1,k2), meaning the convolution
        will  be 'anisotropic': one direction will be compressed or stretched in feature space.
    groups : int, optional
        number of groups in the convolution, default 1
    bias : bool, optional
        whether to create a separate bias parameter, default True
    transpose : bool, optional
        whether conv is a transpose conv, default False
    References
    ----------
    .. [1] : Bonev B., Kurth T., Hundt C., Pathak J., Baust M., Kashinath K., Anandkumar A.
        Spherical Neural Operators: Learning Stable Dynamics on the Sphere; arxiv:2306.03838

    .. [2] : Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, list[int]],
        basis_type: Literal[
            "piecewise_linear", "piecewise_linear3d", "morlet", "zernike", "morlet3d"
        ] = "piecewise_linear",
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
    ):
        super().__init__()

        if isinstance(kernel_shape, int):
            self.kernel_shape = [kernel_shape, kernel_shape]
        else:
            self.kernel_shape = kernel_shape

        if basis_type in ["morlet", "morlet3d"]:
            self.kernel_size = math.prod(self.kernel_shape)
        elif basis_type == 'piecewise_linear3d':
            self.kernel_size = (self.kernel_shape[0] - 1) * self.kernel_shape[1] * self.kernel_shape[2] + 1
        else:
            self.kernel_size = (self.kernel_shape[0] - 1) * self.kernel_shape[1] + 1

        # groups
        self.groups = groups

        # weight tensor
        if in_channels % self.groups != 0:
            raise ValueError(
                "Error, the number of input channels has to be an integer multiple of the group size"
            )
        if out_channels % self.groups != 0:
            raise ValueError(
                "Error, the number of output channels has to be an integer multiple of the group size"
            )

        self.groupsize = in_channels // self.groups

        scale = math.sqrt(1.0 / self.groupsize)

        self.weight = nn.Parameter(
            scale * torch.randn(out_channels, self.groupsize, self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class DiscreteContinuousConv2d(DiscreteContinuousConv):
    """
    Discrete-continuous convolutions (DISCO) on arbitrary 2d grids
    as implemented in [1]_. To evaluate continuous convolutions on a
    computer, they can be evaluated semi-discretely, where the translation
    operation is performed continuously, and the quadrature/projection is
    performed discretely on a grid [2]_. They are the main building blocks
    for local Neural Operators [1]_. Forward call expects an input of shape
    (batch_size, in_channels, n_in).

    Parameters
    ----------
    in_channels: int
        input channels to DISCO convolution
    out_channels: int
        output channels of DISCO convolution
    grid_in: torch.Tensor or literal ``{'equidistant', 'legendre-gauss', 'equiangular', 'lobatto'}``
        input grid in the form of a point cloud of shape (n_in, 2).
        Can also pass a string to generate a regular (tensor) grid.
        For exact options see ``torch_harmonics.quadrature``.
    grid_out: torch.Tensor or literal ``{'equidistant', 'legendre-gauss', 'equiangular', 'lobatto'}``
        output grid in the form of a point cloud (n_out, 2).
        Can also pass a string to generate a regular (tensor) grid.
        For exact options see ``torch_harmonics.quadrature``.
    kernel_shape : ``Union[int, List[int]]``
        Dimensions of the convolution kernel, either one int or a two-int tuple.
        * If one int k, the kernel will be a square of shape (k,k), meaning the convolution
        will be 'isotropic': both directions are equally scaled in feature space.

        * If two ints (k1,k2), the kernel will be a rectangle of shape (k1,k2), meaning the convolution
        will  be 'anisotropic': one direction will be compressed or stretched in feature space.
    basis_type: str literal ``{'piecewise_linear', 'morlet', 'zernike'}``
        choice of basis functions to use for convolution filter tensor.
    n_in: Tuple[int], optional
        number of input points along each dimension. Only used
        if grid_in is passed as a str. See ``torch_harmonics.quadrature``.
    n_out: Tuple[int], optional
        number of output points along each dimension. Only used
        if grid_out is passed as a str. See ``torch_harmonics.quadrature``.
    quadrature_weights: torch.Tensor, optional
        quadrature weights on the input grid
        expects a tensor of shape (n_in,)
    periodic: bool, optional
        whether the domain is periodic, by default False
    groups: int, optional
        number of groups in the convolution, by default 1
    bias: bool, optional
        whether to use a bias, by default True
    radius_cutoff: float, optional
        cutoff radius for the kernel. For a point ``x`` on the input grid,
        every point ``y`` on the output grid with ``||x - y|| <= radius_cutoff``
        will be affected by the value at ``x``.
        By default, set to 2 / sqrt(# of output points)

    References
    ----------
    .. [1] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845

    .. [2] Ocampo J., Price M.A. , McEwen J.D.; Scalable and equivariant spherical CNNs by
        discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_in: torch.Tensor,
        grid_out: torch.Tensor,
        kernel_shape: Union[int, list[int]],
        basis_type: str = "piecewise_linear",
        n_in: Optional[tuple[int]] = None,
        n_out: Optional[tuple[int]] = None,
        quadrature_weights: Optional[torch.Tensor] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            groups=groups,
            bias=bias,
        )

        # the instantiator supports convenience constructors for the input and output grids
        if isinstance(grid_in, torch.Tensor):
            assert isinstance(quadrature_weights, torch.Tensor)
            assert not periodic
        elif isinstance(grid_in, str):
            assert n_in is not None
            assert len(n_in) == 2
            x, wx = _precompute_grid(n_in[0], grid=grid_in, periodic=periodic)
            y, wy = _precompute_grid(n_in[1], grid=grid_in, periodic=periodic)
            x, y = torch.meshgrid(
                torch.from_numpy(x), torch.from_numpy(y), indexing="ij"
            )
            wx, wy = torch.meshgrid(
                torch.from_numpy(wx), torch.from_numpy(wy), indexing="ij"
            )
            grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
            quadrature_weights = (wx * wy).reshape(-1)
        else:
            raise ValueError(f"Unknown grid input type of type {type(grid_in)}")

        if isinstance(grid_out, torch.Tensor):
            pass
        elif isinstance(grid_out, str):
            assert n_out is not None
            assert len(n_out) == 2
            x, wx = _precompute_grid(n_out[0], grid=grid_out, periodic=periodic)
            y, wy = _precompute_grid(n_out[1], grid=grid_out, periodic=periodic)
            x, y = torch.meshgrid(
                torch.from_numpy(x), torch.from_numpy(y), indexing="ij"
            )
            grid_out = torch.stack([x.reshape(-1), y.reshape(-1)])
        else:
            raise ValueError(f"Unknown grid output type of type {type(grid_out)}")

        # check that input arrays are valid point clouds in 2D
        assert len(grid_in.shape) == 2
        assert len(grid_out.shape) == 2
        assert len(quadrature_weights.shape) == 1
        assert grid_in.shape[0] == 2
        assert grid_out.shape[0] == 2

        self.n_in = grid_in.shape[-1]
        self.n_out = grid_out.shape[-1]

        # compute the cutoff radius based on the bandlimit of the input field
        # TODO: Attention - this heuristic is ad-hoc! Make sure to set it yourself!
        if radius_cutoff is None:
            radius_cutoff = radius_cutoff = 2 / float(math.sqrt(self.n_out) - 1)

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # integration weights
        self.register_buffer("quadrature_weights", quadrature_weights, persistent=False)

        idx, vals = _precompute_convolution_filter_matrix(
            grid_in,
            grid_out,
            self.kernel_shape,
            quadrature_weights,
            basis_type=basis_type,
            radius_cutoff=radius_cutoff,
            periodic=periodic,
        )

        # to improve performance, we make psi a matrix by merging the first two dimensions
        # This has to be accounted for in the forward pass
        idx = torch.stack([idx[0] * self.n_out + idx[1], idx[2]], dim=0)

        self.register_buffer("psi_idx", idx.contiguous(), persistent=False)
        self.register_buffer("psi_vals", vals.contiguous(), persistent=False)

    def get_local_filter_matrix(self):
        """
        Returns the precomputed local convolution filter matrix Psi.
        Psi parameterizes the kernel function as triangular basis functions
        evaluated on pairs of points on the convolution's input and output grids,
        such that Psi[l, i, j] is the l-th basis function evaluated on point i in
        the output grid and point j in the input grid.
        """

        psi = torch.sparse_coo_tensor(
            self.psi_idx, self.psi_vals, size=(self.kernel_size * self.n_out, self.n_in)
        )
        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call. Expects an input of shape batch_size x in_channels x n_in.
        """

        # pre-multiply x with the quadrature weights
        x = self.quadrature_weights * x

        psi = self.get_local_filter_matrix()

        # extract shape
        B, C, _ = x.shape

        # bring x into the right shape for the bmm (batch_size x channels, n_in) and pre-apply psi to x
        x = x.reshape(B * C, self.n_in).permute(1, 0).contiguous()
        x = torch.mm(psi, x)
        x = x.permute(1, 0).reshape(B, C, self.kernel_size, self.n_out)
        x = x.reshape(B, self.groups, self.groupsize, self.kernel_size, self.n_out)

        # do weight multiplication
        out = torch.einsum(
            "bgckx,gock->bgox",
            x,
            self.weight.reshape(
                self.groups, -1, self.weight.shape[1], self.weight.shape[2]
            ),
        )
        out = out.reshape(out.shape[0], -1, out.shape[-1])

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1)

        return out


class DiscreteContinuousConvTranspose2d(DiscreteContinuousConv):
    """
    Transpose variant of discrete-continuous convolutions on arbitrary
    2d grids as implemented for [1]_. Forward call expects an input of shape
    (batch_size, in_channels, n_in).

    Parameters
    ----------
    in_channels: int
        input channels to DISCO convolution
    out_channels: int
        output channels of DISCO convolution
    grid_in: torch.Tensor or literal ``{'equidistant', 'legendre-gauss', 'equiangular', 'lobatto'}``
        input grid in the form of a point cloud of shape (n_in, 2).
        Can also pass a string to generate a regular (tensor) grid.
        For exact options see ``torch_harmonics.quadrature``.
    grid_out: torch.Tensor or literal ``{'equidistant', 'legendre-gauss', 'equiangular', 'lobatto'}``
        output grid in the form of a point cloud (n_out, 2).
        Can also pass a string to generate a regular (tensor) grid.
        For exact options see ``torch_harmonics.quadrature``.
    kernel_shape : ``Union[int, List[int]]``
        Dimensions of the convolution kernel, either one int or a two-int tuple.
        * If one int k, the kernel will be a square of shape (k,k), meaning the convolution
        will be 'isotropic': both directions are equally scaled in feature space.

        * If two ints (k1,k2), the kernel will be a rectangle of shape (k1,k2), meaning the convolution
        will  be 'anisotropic': one direction will be compressed or stretched in feature space.
    basis_type: str literal ``{'piecewise_linear', 'morlet', 'zernike'}``
        choice of basis functions to use for convolution filter tensor.
    n_in: Tuple[int], optional
        number of input points along each dimension. Only used
        if grid_in is passed as a str. See ``torch_harmonics.quadrature``.
    n_out: Tuple[int], optional
        number of output points along each dimension. Only used
        if grid_out is passed as a str. See ``torch_harmonics.quadrature``.
    quadrature_weights: torch.Tensor, optional
        quadrature weights on the input grid
        expects a tensor of shape (n_in,)
    periodic: bool, optional
        whether the domain is periodic, by default False
    groups: int, optional
        number of groups in the convolution, by default 1
    bias: bool, optional
        whether to use a bias, by default True
    radius_cutoff: float, optional
        cutoff radius for the kernel. For a point ``x`` on the input grid,
        every point ``y`` on the output grid with ``||x - y|| <= radius_cutoff``
        will be affected by the value at ``x``.
        By default, set to 2 / sqrt(# of output points)

    References
    ----------
    .. [1] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.;
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_in: torch.Tensor,
        grid_out: torch.Tensor,
        kernel_shape: Union[int, list[int]],
        basis_type: str = "piecewise_linear",
        n_in: Optional[tuple[int]] = None,
        n_out: Optional[tuple[int]] = None,
        quadrature_weights: Optional[torch.Tensor] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            groups=groups,
            bias=bias,
        )
        # the instantiator supports convenience constructors for the input and output grids
        if isinstance(grid_in, torch.Tensor):
            assert isinstance(quadrature_weights, torch.Tensor)
            assert not periodic
        elif isinstance(grid_in, str):
            assert n_in is not None
            assert len(n_in) == 2
            x, wx = _precompute_grid(n_in[0], grid=grid_in, periodic=periodic)
            y, wy = _precompute_grid(n_in[1], grid=grid_in, periodic=periodic)
            x, y = torch.meshgrid(
                torch.from_numpy(x), torch.from_numpy(y), indexing="ij"
            )
            wx, wy = torch.meshgrid(
                torch.from_numpy(wx), torch.from_numpy(wy), indexing="ij"
            )
            grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
            quadrature_weights = (wx * wy).reshape(-1)
        else:
            raise ValueError(f"Unknown grid input type of type {type(grid_in)}")

        if isinstance(grid_out, torch.Tensor):
            pass
        elif isinstance(grid_out, str):
            assert n_out is not None
            assert len(n_out) == 2
            x, wx = _precompute_grid(n_out[0], grid=grid_out, periodic=periodic)
            y, wy = _precompute_grid(n_out[1], grid=grid_out, periodic=periodic)
            x, y = torch.meshgrid(
                torch.from_numpy(x), torch.from_numpy(y), indexing="ij"
            )
            grid_out = torch.stack([x.reshape(-1), y.reshape(-1)])
        else:
            raise ValueError(f"Unknown grid output type of type {type(grid_out)}")

        # check that input arrays are valid point clouds in 2D
        assert len(grid_in.shape) == 2
        assert len(grid_out.shape) == 2
        assert len(quadrature_weights.shape) == 1
        assert grid_in.shape[0] == 2
        assert grid_out.shape[0] == 2

        self.n_in = grid_in.shape[-1]
        self.n_out = grid_out.shape[-1]

        # compute the cutoff radius based on the bandlimit of the input field
        # TODO: Attention - this heuristic is ad-hoc! Make sure to set it yourself!
        if radius_cutoff is None:
            radius_cutoff = 2 / float(math.sqrt(self.n_in) - 1)

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # integration weights
        self.register_buffer("quadrature_weights", quadrature_weights, persistent=False)

        # precompute the transposed tensor
        idx, vals = _precompute_convolution_filter_matrix(
            grid_out,
            grid_in,
            self.kernel_shape,
            quadrature_weights,
            basis_type=basis_type,
            radius_cutoff=radius_cutoff,
            periodic=periodic,
            transpose_normalization=True,
        )

        # to improve performance, we make psi a matrix by merging the first two dimensions
        # This has to be accounted for in the forward pass
        idx = torch.stack([idx[0] * self.n_out + idx[2], idx[1]], dim=0)

        self.register_buffer("psi_idx", idx.contiguous(), persistent=False)
        self.register_buffer("psi_vals", vals.contiguous(), persistent=False)

    def get_local_filter_matrix(self):
        """
        Returns the precomputed local convolution filter matrix Psi.
        Psi parameterizes the kernel function as triangular basis functions
        evaluated on pairs of points on the convolution's input and output grids,
        such that Psi[l, i, j] is the l-th basis function evaluated on point i in
        the output grid and point j in the input grid.
        """

        psi = torch.sparse_coo_tensor(
            self.psi_idx, self.psi_vals, size=(self.kernel_size * self.n_out, self.n_in)
        )
        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call. Expects an input of shape batch_size x in_channels x n_in.
        """

        # pre-multiply x with the quadrature weights
        x = self.quadrature_weights * x

        psi = self.get_local_filter_matrix()

        # extract shape
        B, C, _ = x.shape

        # bring x into the right shape for the bmm (batch_size x channels, n_in) and pre-apply psi to x
        x = x.reshape(B * C, self.n_in).permute(1, 0).contiguous()
        x = torch.mm(psi, x)

        x = x.permute(1, 0).reshape(B, C, self.kernel_size, self.n_out)
        x = x.reshape(B, self.groups, self.groupsize, self.kernel_size, self.n_out)

        # do weight multiplication
        out = torch.einsum(
            "bgckx,gock->bgox",
            x,
            self.weight.reshape(
                self.groups, -1, self.weight.shape[1], self.weight.shape[2]
            ),
        )
        out = out.reshape(out.shape[0], -1, out.shape[-1])

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1)

        return out


class EquidistantDiscreteContinuousConv2d(DiscreteContinuousConv):
    """
    Discrete-continuous convolutions (DISCO) on equidistant 2d grids
    as implemented for [1]_. This implementation maps to 2d convolution
    kernels which makes it more efficient than the unstructured implementation
    above. Due to the mapping to an equidistant grid, the domain lengths need
    to be specified in order to compute the effective resolution and the
    corresponding cutoff radius. Forward call expects an input of shape
    (batch_size, in_channels, in_shape[0], in_shape[1]).

    Parameters
    ----------
    in_channels: int
        input channels to DISCO convolution
    out_channels: int
        output channels of DISCO convolution
    in_shape: Tuple[int]
        shape of the (regular) input grid.
    out_shape: torch.Tensor or str
        shape of the (regular) output grid. Note that the side lengths
        of out_shape must be less than or equal to the side lengths
        of in_shape, and must be integer divisions of the corresponding
        in_shape side lengths.
    kernel_shape : ``Union[int, List[int]]``
        Dimensions of the convolution kernel, either one int or a two-int tuple.
        * If one int k, the kernel will be a square of shape (k,k), meaning the convolution
        will be 'isotropic': both directions are equally scaled in feature space.

        * If two ints (k1,k2), the kernel will be a rectangle of shape (k1,k2), meaning the convolution
        will  be 'anisotropic': one direction will be compressed or stretched in feature space.
    basis_type: str literal ``{'piecewise_linear', 'morlet', 'zernike'}``
        choice of basis functions to use for convolution filter tensor.
    domain_length: torch.Tensor, optional
        extent/length of the physical domain. Assumes square domain [-1, 1]^2 by default
    periodic: bool, optional
        whether the domain is periodic, by default False
    groups: int, optional
        number of groups in the convolution, by default 1
    bias: bool, optional
        whether to use a bias, by default True
    radius_cutoff: float, optional
        cutoff radius for the kernel. For a point ``x`` on the input grid,
        every point ``y`` on the output grid with ``||x - y|| <= radius_cutoff``
        will be affected by the value at ``x``.
        By default, set to 2 / sqrt(# of output points)

    References
    ----------
    .. [1] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.;
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: tuple[int],
        out_shape: tuple[int],
        kernel_shape: Union[int, list[int]],
        basis_type: str = "piecewise_linear",
        domain_length: Optional[tuple[float]] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            groups=groups,
            bias=bias,
        )

        # to ensure compatibility with the unstructured code, only constant zero and periodic padding are supported currently
        self.padding_mode = "circular" if periodic else "zeros"

        # if domain length is not specified we use
        self.domain_length = [2, 2] if domain_length is None else domain_length

        # compute the cutoff radius based on the assumption that the grid is [-1, 1]^2
        # this still assumes a quadratic domain
        if radius_cutoff is None:
            radius_cutoff = max(
                [self.domain_length[i] / float(out_shape[i]) for i in (0, 1)]
            )

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # compute how big the discrete kernel needs to be for the 2d convolution kernel to work
        self.psi_local_h = (
            math.floor(2 * radius_cutoff * in_shape[0] / self.domain_length[0]) + 1
        )
        self.psi_local_w = (
            math.floor(2 * radius_cutoff * in_shape[1] / self.domain_length[1]) + 1
        )

        # compute the scale_factor
        assert (in_shape[0] >= out_shape[0]) and (in_shape[0] % out_shape[0] == 0)
        self.scale_h = in_shape[0] // out_shape[0]
        assert (in_shape[1] >= out_shape[1]) and (in_shape[1] % out_shape[1] == 0)
        self.scale_w = in_shape[1] // out_shape[1]

        # psi_local is essentially the support of the hat functions evaluated locally
        x = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_h)
        y = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_w)
        x, y = torch.meshgrid(x, y, indexing="ij")
        grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])

        # compute quadrature weights on the incoming grid
        self.q_weight = (
            self.domain_length[0] * self.domain_length[1] / in_shape[0] / in_shape[1]
        )
        quadrature_weights = self.q_weight * torch.ones(
            self.psi_local_h * self.psi_local_w
        )
        grid_out = torch.Tensor([[0.0], [0.0]])

        # precompute psi using conventional routines onto the local grid
        idx, vals = _precompute_convolution_filter_matrix(
            grid_in,
            grid_out,
            self.kernel_shape,
            quadrature_weights,
            basis_type=basis_type,
            radius_cutoff=radius_cutoff,
            periodic=False,
            normalize=True,
        )

        # extract the local psi as a dense representation
        local_filter_matrix = torch.zeros(
            self.kernel_size, self.psi_local_h * self.psi_local_w
        )
        for ie in range(len(vals)):
            f = idx[0, ie]
            j = idx[2, ie]
            v = vals[ie]
            local_filter_matrix[f, j] = v

        # compute local version of the filter matrix
        local_filter_matrix = local_filter_matrix.reshape(
            self.kernel_size, self.psi_local_h, self.psi_local_w
        )

        self.register_buffer(
            "local_filter_matrix", local_filter_matrix, persistent=False
        )

    def get_local_filter_matrix(self):
        """
        Returns the precomputed local convolution filter matrix Psi.
        Psi parameterizes the kernel function as triangular basis functions
        evaluated on pairs of points on the convolution's input and output grids,
        such that Psi[l, i, j] is the l-th basis function evaluated on point i in
        the output grid and point j in the input grid.
        """

        return self.local_filter_matrix.permute(0, 2, 1).flip(dims=(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call. Expects an input of shape batch_size x in_channels x in_shape[0] x in_shape[1].
        """

        kernel = torch.einsum(
            "kxy,ogk->ogxy", self.get_local_filter_matrix(), self.weight
        )
        # padding is rounded down to give the right result when even kernels are applied
        # Check https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for output shape math
        h_pad = (self.psi_local_h + 1) // 2 - 1
        w_pad = (self.psi_local_w + 1) // 2 - 1
        out = nn.functional.conv2d(
            self.q_weight * x,
            kernel,
            self.bias,
            stride=[self.scale_h, self.scale_w],
            dilation=1,
            padding=[h_pad, w_pad],
            groups=self.groups,
        )

        return out


class EquidistantDiscreteContinuousConvTranspose2d(DiscreteContinuousConv):
    """
    Transpose Discrete-continuous convolutions (DISCO) on equidistant 2d grids
    as implemented for [1]_. This implementation maps to 2d convolution kernels
    which makes it more efficient than the unstructured implementation above.
    Due to the mapping to an equidistant grid, the domain lengths need to be
    specified in order to compute the effective resolution and the corresponding
    cutoff radius. Forward call expects an input of shape
    (batch_size, in_channels, in_shape[0], in_shape[1]).

    Parameters
    ----------
    in_channels: int
        input channels to DISCO convolution
    out_channels: int
        output channels of DISCO convolution
    in_shape: Tuple[int]
        shape of the (regular) input grid.
    out_shape: torch.Tensor or str
        shape of the (regular) output grid. Note that the side lengths
        of out_shape must be greater than or equal to the side lengths
        of in_shape, and must be integer multiples of the corresponding
        in_shape side lengths.
    kernel_shape : ``Union[int, List[int]]``
        Dimensions of the convolution kernel, either one int or a two-int tuple.
        * If one int k, the kernel will be a square of shape (k,k), meaning the convolution
        will be 'isotropic': both directions are equally scaled in feature space.

        * If two ints (k1,k2), the kernel will be a rectangle of shape (k1,k2), meaning the convolution
        will  be 'anisotropic': one direction will be compressed or stretched in feature space.
    basis_type: str literal ``{'piecewise_linear', 'morlet', 'zernike'}``
        choice of basis functions to use for convolution filter tensor.
    domain_length: torch.Tensor, optional
        extent/length of the physical domain. Assumes square domain [-1, 1]^2 by default
    periodic: bool, optional
        whether the domain is periodic, by default False
    groups: int, optional
        number of groups in the convolution, by default 1
    bias: bool, optional
        whether to use a bias, by default True
    radius_cutoff: float, optional
        cutoff radius for the kernel. For a point ``x`` on the input grid,
        every point ``y`` on the output grid with ``||x - y|| <= radius_cutoff``
        will be affected by the value at ``x``.
        By default, set to 2 / sqrt(# of output points)

    References
    ----------
    .. [1] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.;
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: tuple[int],
        out_shape: tuple[int],
        kernel_shape: Union[int, list[int]],
        basis_type: str = "piecewise_linear",
        domain_length: Optional[tuple[float]] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            groups=groups,
            bias=bias,
        )
        # torch ConvTranspose2d expects grouped weights stacked along the out_channels
        # shape (in_channels, out_channels/groups, h, w)
        self.weight = nn.Parameter(
            self.weight.permute(1, 0, 2).reshape(
                self.groupsize * self.groups, -1, self.weight.shape[-1]
            )
        )

        # to ensure compatibility with the unstructured code, only constant zero and periodic padding are supported currently
        self.padding_mode = "circular" if periodic else "zeros"

        # if domain length is not specified we use
        self.domain_length = [2, 2] if domain_length is None else domain_length

        # compute the cutoff radius based on the assumption that the grid is [-1, 1]^2
        # this still assumes a quadratic domain
        if radius_cutoff is None:
            radius_cutoff = max(
                [self.domain_length[i] / float(in_shape[i]) for i in (0, 1)]
            )

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # compute how big the discrete kernel needs to be for the 2d convolution kernel to work
        self.psi_local_h = (
            math.floor(2 * radius_cutoff * out_shape[0] / self.domain_length[0]) + 1
        )
        self.psi_local_w = (
            math.floor(2 * radius_cutoff * out_shape[1] / self.domain_length[1]) + 1
        )

        # compute the scale_factor
        assert (in_shape[0] <= out_shape[0]) and (out_shape[0] % in_shape[0] == 0)
        self.scale_h = out_shape[0] // in_shape[0]
        assert (in_shape[1] <= out_shape[1]) and (out_shape[1] % in_shape[1] == 0)
        self.scale_w = out_shape[1] // in_shape[1]

        # psi_local is essentially the support of the hat functions evaluated locally
        x = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_h)
        y = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_w)
        x, y = torch.meshgrid(x, y, indexing="ij")
        grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
        grid_out = torch.Tensor([[0.0], [0.0]])

        # compute quadrature weights on the incoming grid
        self.q_weight = (
            self.domain_length[0] * self.domain_length[1] / out_shape[0] / out_shape[1]
        )
        quadrature_weights = self.q_weight * torch.ones(
            self.psi_local_h * self.psi_local_w
        )

        # precompute psi using conventional routines onto the local grid
        idx, vals = _precompute_convolution_filter_matrix(
            grid_in,
            grid_out,
            self.kernel_shape,
            quadrature_weights,
            basis_type=basis_type,
            radius_cutoff=radius_cutoff,
            periodic=False,
            normalize=True,
            transpose_normalization=False,
        )

        # extract the local psi as a dense representation
        local_filter_matrix = torch.zeros(
            self.kernel_size, self.psi_local_h * self.psi_local_w
        )
        for ie in range(len(vals)):
            f = idx[0, ie]
            j = idx[2, ie]
            v = vals[ie]
            local_filter_matrix[f, j] = v

        # compute local version of the filter matrix
        local_filter_matrix = local_filter_matrix.reshape(
            self.kernel_size, self.psi_local_h, self.psi_local_w
        )

        self.register_buffer(
            "local_filter_matrix", local_filter_matrix, persistent=False
        )

    def get_local_filter_matrix(self):
        """
        Returns the precomputed local convolution filter matrix Psi.
        Psi parameterizes the kernel function as triangular basis functions
        evaluated on pairs of points on the convolution's input and output grids,
        such that Psi[l, i, j] is the l-th basis function evaluated on point i in
        the output grid and point j in the input grid.
        """

        return self.local_filter_matrix.permute(0, 2, 1).flip(dims=(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call. Expects an input of shape batch_size x in_channels x in_shape[0] x in_shape[1].
        """
        kernel = torch.einsum(
            "kxy,ogk->ogxy", self.get_local_filter_matrix(), self.weight
        )

        # padding is rounded down to give the right result when even kernels are applied
        # Check https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for output shape math
        h_pad = (self.psi_local_h + 1) // 2 - 1
        w_pad = (self.psi_local_w + 1) // 2 - 1
        # additional one-sided padding. See https://discuss.pytorch.org/t/question-of-2d-transpose-convolution/99419
        h_pad_out = self.scale_h - (self.psi_local_h // 2 - h_pad) - 1
        w_pad_out = self.scale_w - (self.psi_local_w // 2 - w_pad) - 1

        out = nn.functional.conv_transpose2d(
            self.q_weight * x,
            kernel,
            self.bias,
            stride=[self.scale_h, self.scale_w],
            dilation=[1, 1],
            padding=[h_pad, w_pad],
            output_padding=[h_pad_out, w_pad_out],
            groups=self.groups,
        )

        return out


class EquidistantDiscreteContinuousConv3d(DiscreteContinuousConv):
    """
    Discrete-continuous convolutions (DISCO) on equidistant 3d grids.
    This implementation maps to 3d convolution kernels which makes it more efficient
    than an unstructured implementation. Due to the mapping to an equidistant grid,
    the domain lengths need to be specified in order to compute the effective resolution
    and the corresponding cutoff radius. Forward call expects an input of shape
    (batch_size, in_channels, in_shape[0], in_shape[1], in_shape[2]).

    Parameters
    ----------
    in_channels: int
        input channels to DISCO convolution
    out_channels: int
        output channels of DISCO convolution
    in_shape: Tuple[int]
        shape of the (regular) input grid.
    out_shape: Tuple[int]
        shape of the (regular) output grid. Note that the side lengths
        of out_shape must be less than or equal to the side lengths
        of in_shape, and must be integer divisions of the corresponding
        in_shape side lengths.
    kernel_shape : ``Union[int, List[int]]``
        Dimensions of the convolution kernel, either one int or a three-int tuple.
        * If one int k, the kernel will be a cube of shape (k,k,k), meaning the convolution
        will be 'isotropic': all directions are equally scaled in feature space.

        * If three ints (k1,k2,k3), the kernel will have shape (k1,k2,k3), meaning the convolution
        will be 'anisotropic': directions can be compressed or stretched in feature space.
    basis_type: str literal, must be 'piecewise_linear3d' or 'morlet3d'
        choice of basis functions to use for convolution filter tensor.
    domain_length: torch.Tensor, optional
        extent/length of the physical domain. Assumes cube domain [-1, 1]^3 by default
    periodic: bool, optional
        whether the domain is periodic, by default False
    groups: int, optional
        number of groups in the convolution, by default 1
    bias: bool, optional
        whether to use a bias, by default True
    radius_cutoff: float, optional
        cutoff radius for the kernel. For a point ``x`` on the input grid,
        every point ``y`` on the output grid with ``||x - y|| <= radius_cutoff``
        will be affected by the value at ``x``.
        By default, set to 2 / sqrt(# of output points)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: tuple[int, int, int],
        out_shape: tuple[int, int, int],
        kernel_shape: Union[int, list[int]],
        basis_type: str = "piecewise_linear3d",
        domain_length: Optional[tuple[float, float, float]] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_shape=kernel_shape,
            basis_type=basis_type,
            groups=groups,
            bias=bias,
        )

        # Consistent with 2D version
        self.padding_mode = "circular" if periodic else "zeros"

        # Domain lengths: default cube [-1, 1]^3 mapped from length 2
        self.domain_length = [2, 2, 2] if domain_length is None else domain_length

        # Cutoff radius: analogous to 2D logic
        if radius_cutoff is None:
            radius_cutoff = max([
                self.domain_length[i] / float(out_shape[i])
                for i in (0, 1, 2)
            ])

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # Compute discrete support window sizes (3D analogue of psi_local_h/w)
        self.psi_local_d = (
            math.floor(2 * radius_cutoff * in_shape[0] / self.domain_length[0]) + 1
        )
        self.psi_local_h = (
            math.floor(2 * radius_cutoff * in_shape[1] / self.domain_length[1]) + 1
        )
        self.psi_local_w = (
            math.floor(2 * radius_cutoff * in_shape[2] / self.domain_length[2]) + 1
        )

        # Compute scale factors just like 2D version
        assert (in_shape[0] >= out_shape[0]) and (in_shape[0] % out_shape[0] == 0)
        self.scale_d = in_shape[0] // out_shape[0]

        assert (in_shape[1] >= out_shape[1]) and (in_shape[1] % out_shape[1] == 0)
        self.scale_h = in_shape[1] // out_shape[1]

        assert (in_shape[2] >= out_shape[2]) and (in_shape[2] % out_shape[2] == 0)
        self.scale_w = in_shape[2] // out_shape[2]

        # Build local grid, analogous to 2D:
        # linspace(-radius_cutoff, radius_cutoff, psi_local_dim)
        xd = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_d)
        yh = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_h)
        zw = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_w)

        D, H, W = torch.meshgrid(xd, yh, zw, indexing="ij")
        grid_in = torch.stack([
            D.reshape(-1),
            H.reshape(-1),
            W.reshape(-1)
        ])  # Shape [3, N]

        # Quadrature weights: perfect 3D match to 2D logic
        self.q_weight = (
            self.domain_length[0]
            * self.domain_length[1]
            * self.domain_length[2]
            / (in_shape[0] * in_shape[1] * in_shape[2])
        )
        quadrature_weights = self.q_weight * torch.ones(
            self.psi_local_d * self.psi_local_h * self.psi_local_w
        )

        # Single output point at the origin, as in 2D
        grid_out = torch.zeros(3, 1)

        # Precompute psi on this local grid using the 3D version
        idx, vals = _precompute_convolution_filter_matrix_3d(
            grid_in,
            grid_out,
            self.kernel_shape,
            quadrature_weights,
            basis_type=basis_type,
            radius_cutoff=radius_cutoff,
            periodic=False,
            normalize=True,
        )

        # Assemble dense local filter matrix, directly parallel to 2D version
        num_points = self.psi_local_d * self.psi_local_h * self.psi_local_w
        local_filter_matrix = torch.zeros(self.kernel_size, num_points)

        # idx: [4, nnz] -> The format from compute_support_vals is [basis_idx, m, n, ?]
        # where m is the output index (0 in our case, since we have 1 output point)
        # and n is the linear input index into the flattened input grid
        # We need to convert the linear index n to z, y, x coordinates
        for ie in range(len(vals)):
            f = idx[0, ie].item() if torch.is_tensor(idx[0, ie]) else int(idx[0, ie])
            
            # idx[2, ie] is likely the linear input index n into the [M, N] grid
            # Since M=1 (single output point), n is the input point index (0 to N-1)
            linear_input_idx = idx[2, ie].item() if torch.is_tensor(idx[2, ie]) else int(idx[2, ie])
            
            # Bounds check for linear input index
            if linear_input_idx < 0 or linear_input_idx >= num_points:
                continue  # Skip out-of-bounds indices
            
            # Convert linear index to z, y, x coordinates
            # grid_in is flattened as: D.reshape(-1), H.reshape(-1), W.reshape(-1)
            # The meshgrid uses indexing="ij", so the order is (d, h, w) = (z, y, x)
            z = linear_input_idx // (self.psi_local_h * self.psi_local_w)
            remainder = linear_input_idx % (self.psi_local_h * self.psi_local_w)
            y = remainder // self.psi_local_w
            x = remainder % self.psi_local_w
            
            # Bounds check for z, y, x (should always pass if linear_input_idx is valid)
            if (z < 0 or z >= self.psi_local_d or 
                y < 0 or y >= self.psi_local_h or 
                x < 0 or x >= self.psi_local_w):
                continue  # Skip out-of-bounds indices
            
            # Bounds check for f (basis index)
            if f < 0 or f >= self.kernel_size:
                continue  # Skip out-of-bounds basis index
            
            # Compute linear index j from z, y, x (should equal linear_input_idx)
            j = (
                z * (self.psi_local_h * self.psi_local_w)
                + y * self.psi_local_w
                + x
            )
            
            # Final bounds check for j
            if j < 0 or j >= num_points:
                continue  # Skip out-of-bounds indices
                
            local_filter_matrix[f, j] = vals[ie]

        # Reshape into a 3D window
        local_filter_matrix = local_filter_matrix.reshape(
            self.kernel_size, self.psi_local_d, self.psi_local_h, self.psi_local_w
        )

        self.register_buffer(
            "local_filter_matrix", local_filter_matrix, persistent=False
        )

    def get_local_filter_matrix(self):
        """
        Returns the precomputed local convolution filter matrix Psi.
        Psi parameterizes the kernel function as basis functions
        evaluated on the 3D grid.
        """
        # Permute to match the expected 3D convolution kernel format
        # return self.local_filter_matrix.permute(0, 3, 2, 1).flip(dims=(-1, -2, -3))
        return self.local_filter_matrix.flip(dims=(-1, -2, -3))
    
    def compile_kernel(self, use_einsum: bool = True) -> torch.Tensor:
        """Combines the local filter matrix (basis patterns) with the weights to create the final kernel."""
        if use_einsum:
            return torch.einsum(
                "kxyz,ogk->ogxyz", self.get_local_filter_matrix(), self.weight
            )
        else:
            raise NotImplementedError("Non-einsum implementation for faster execution is not implemented yet")
            # Non-einsum implementation for faster execution
            local_filter = self.get_local_filter_matrix()  # Shape: (k, x, y, z)
            weight = self.weight  # Shape: (o, g, k)
            
            # Reshape local_filter to (k, xyz) for matrix multiplication
            k, x, y, z = local_filter.shape
            local_filter_flat = local_filter.view(k, x * y * z)  # Shape: (k, xyz)
            
            # Perform matrix multiplication: (o, g, k) @ (k, xyz) -> (o, g, xyz)
            result = torch.matmul(weight, local_filter_flat)  # Shape: (o, g, xyz)
            
            # Reshape back to (o, g, x, y, z)
            return result.view(weight.shape[0], weight.shape[1], x, y, z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call. Expects an input of shape batch_size x in_channels x in_shape[0] x in_shape[1] x in_shape[2].
        """
        kernel = self.compile_kernel()

        # padding is rounded down to give the right result when even kernels are applied
        d_pad = (self.psi_local_d + 1) // 2 - 1
        h_pad = (self.psi_local_h + 1) // 2 - 1
        w_pad = (self.psi_local_w + 1) // 2 - 1

        out = nn.functional.conv3d(
            self.q_weight * x,
            kernel,
            self.bias,
            stride=[self.scale_d, self.scale_h, self.scale_w],
            dilation=1,
            padding=[d_pad, h_pad, w_pad],
            groups=self.groups,
        )

        return out


### for testing purposes ###
if __name__ == "__main__":
    N = 9
    x = torch.randn(1, 1, N, N, N)
    K = 5
    r = 1
    # layer = EquidistantDiscreteContinuousConv3d(
    #     1,
    #     1,
    #     in_shape=(N, N, N),
    #     out_shape=(N, N, N),
    #     kernel_shape=[K,K,K],
    #     radius_cutoff=r, 
    # )
    layer = EquidistantDiscreteContinuousConv3d(
    1, 1,
    in_shape=(5, 20, 10),    # Depth=5, Height=20, Width=10
    out_shape=(5, 20, 10),
    kernel_shape=[3, 3, 3],  # 这里的 kernel_shape 对应物理尺寸
    radius_cutoff=1.0,
)
    y = layer(x)
    print(y.shape)
    y = layer(x)
    print(y.shape)

    # INSERT_YOUR_CODE
    import matplotlib.pyplot as plt

    # Get the kernel from the layer
    kernel = layer.compile_kernel().detach().cpu().numpy()  # shape: (out_channels, in_channels/groups, D, H, W)

    # Plot and save every slice of the kernel from all 3 points of view (depth, height, width)
    out_ch = 0
    in_ch = 0
    D, H, W = kernel.shape[2], kernel.shape[3], kernel.shape[4]

    # Plot all depth slices (axis 2)
    fig, axs = plt.subplots(1, D, figsize=(3*D, 3))
    if D == 1:
        axs = [axs]
    for d in range(D):
        axs[d].imshow(kernel[out_ch, in_ch, d, :, :], cmap='viridis')
        axs[d].set_title(f'depth={d}')
        axs[d].axis('off')
    # plt.tight_layout()
    plt.savefig("kernel_slices_depth.png")
    plt.close(fig)

    # Plot all height slices (axis 3)
    fig, axs = plt.subplots(1, H, figsize=(3*H, 3))
    if H == 1:
        axs = [axs]
    for h in range(H):
        axs[h].imshow(kernel[out_ch, in_ch, :, h, :], cmap='viridis')
        axs[h].set_title(f'height={h}')
        axs[h].axis('off')
    # plt.tight_layout()
    plt.savefig("kernel_slices_height.png")
    plt.close(fig)

    # Plot all width slices (axis 4)
    fig, axs = plt.subplots(1, W, figsize=(3*W, 3))
    if W == 1:
        axs = [axs]
    for w in range(W):
        axs[w].imshow(kernel[out_ch, in_ch, :, :, w], cmap='viridis')
        axs[w].set_title(f'width={w}')
        axs[w].axis('off')
    # plt.tight_layout()
    plt.savefig("kernel_slices_width.png")
    plt.close(fig)
