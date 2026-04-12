import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.spectral_convolution import SpectralConv
from ..layers.local_no_block import LocalNOBlocks
from ..layers.complex import cselu


class LocalRNOCell(nn.Module):
    """N-Dimensional Recurrent Neural Operator cell using LocalNO blocks.

    Identical to RNOCell but replaces FNOBlocks with LocalNOBlocks that
    combine spectral convolutions with differential and local integral kernels.

    The cell implements the GRU-like recurrence relation:
        z_t = σ(f1(x_t) + f2(h_{t-1}) + b1)            [update gate]
        r_t = σ(f3(x_t) + f4(h_{t-1}) + b2)            [reset gate]
        h̃_t = selu(f5(x_t) + f6(r_t ⊙ h_{t-1}) + b3)  [candidate state]
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t         [next state]

    where f1-f6 are LocalNO blocks.

    Parameters
    ----------
    n_modes : tuple of int
        Number of modes to keep in Fourier Layer along each dimension.
        The dimensionality is inferred from ``len(n_modes)``.
    hidden_channels : int
        Number of hidden channels in the RNO cell.
    default_in_shape : tuple of int
        Default input spatial shape for DISCO convolutions (input gate shape).

    Other Parameters
    ----------------
    resolution_scaling_factor : float or None, optional
        Factor by which to scale spatial resolution. If set, input gates scale
        from ``default_in_shape`` to the hidden state shape. Default: None
    max_n_modes : tuple of int or None, optional
        Maximum number of modes during training. Default: None
    local_no_block_precision : str, optional
        Precision for spectral convolution. Default: "full"
    use_channel_mlp : bool, optional
        Whether to use MLP after each LocalNO block. Default: False
    channel_mlp_dropout : float, optional
        Dropout in channel MLP. Default: 0
    channel_mlp_expansion : float, optional
        Expansion factor for channel MLP. Default: 0.5
    non_linearity : callable, optional
        Nonlinear activation. Default: F.gelu
    stabilizer : str or None, optional
        Stabilizer ("tanh" or None). Default: None
    norm : str or None, optional
        Normalization layer. Default: None
    preactivation : bool, optional
        Whether to use pre-activation. Default: False
    local_no_skip : str or None, optional
        Skip connection type for LocalNO layers. Default: "linear"
    channel_mlp_skip : str or None, optional
        Skip connection type for channel MLP. Default: "soft-gating"
    complex_data : bool, optional
        Whether data is complex-valued. Default: False
    separable : bool, optional
        Separable spectral convolution. Default: False
    factorization : str or None, optional
        Tensor factorization method. Default: None
    rank : float, optional
        Tensor rank for factorization. Default: 1.0
    conv_module : nn.Module, optional
        Convolution module. Default: SpectralConv
    fixed_rank_modes : bool, optional
        Whether to fix certain rank modes. Default: False
    implementation : str, optional
        Implementation for factorized tensors. Default: "factorized"
    decomposition_kwargs : dict, optional
        Extra kwargs for tensor decomposition. Default: {}
    enforce_hermitian_symmetry : bool, optional
        Enforce Hermitian symmetry in iFFT. Default: True
    disco_layers : bool or list of bool, optional
        Whether to use DISCO local integral layers. Default: True
    disco_kernel_shape : int or list of int, optional
        Kernel shape for DISCO convolutions. Default: [2, 4]
    radius_cutoff : float or None, optional
        Cutoff radius for DISCO kernel. Default: None
    domain_length : list of float or None, optional
        Physical domain length. Default: None (uses [2]*n_dim)
    disco_groups : int, optional
        Number of groups in DISCO convolution. Default: 1
    disco_bias : bool, optional
        Whether to use bias in DISCO kernel. Default: True
    diff_layers : bool or list of bool, optional
        Whether to use differential kernel layers. Default: True
    conv_padding_mode : str, optional
        Padding mode for spatial convolutions. Default: "periodic"
    fin_diff_kernel_size : int, optional
        Kernel size for finite difference convolution. Default: 3
    mix_derivatives : bool, optional
        Whether to mix derivatives across channels. Default: True
    """

    def __init__(
        self,
        n_modes,
        hidden_channels,
        default_in_shape,
        resolution_scaling_factor=None,
        max_n_modes=None,
        local_no_block_precision="full",
        use_channel_mlp=False,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        preactivation=False,
        local_no_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=None,
        enforce_hermitian_symmetry=True,
        disco_layers=True,
        disco_kernel_shape=None,
        radius_cutoff=None,
        domain_length=None,
        disco_groups=1,
        disco_bias=True,
        diff_layers=True,
        conv_padding_mode="periodic",
        fin_diff_kernel_size=3,
        mix_derivatives=True,
    ):
        super().__init__()
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        if disco_kernel_shape is None:
            disco_kernel_shape = [2, 4]

        self.hidden_channels = hidden_channels
        n_dim = len(n_modes)

        if domain_length is None:
            domain_length = [2] * n_dim

        # Compute the hidden state spatial shape after optional scaling
        if resolution_scaling_factor is not None:
            h_default_in_shape = tuple(
                int(round(resolution_scaling_factor * s)) for s in default_in_shape
            )
            # resolution_scaling_factor for input gates: [factor] (list for LocalNOBlocks n_layers=1)
            input_scaling = [resolution_scaling_factor]
        else:
            h_default_in_shape = default_in_shape
            input_scaling = None

        local_no_kwargs = dict(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=n_modes,
            n_layers=1,
            max_n_modes=max_n_modes,
            local_no_block_precision=local_no_block_precision,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            local_no_skip=local_no_skip,
            channel_mlp_skip=channel_mlp_skip,
            separable=separable,
            factorization=factorization,
            rank=rank,
            conv_module=conv_module,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
            disco_layers=disco_layers,
            disco_kernel_shape=disco_kernel_shape,
            radius_cutoff=radius_cutoff,
            domain_length=domain_length,
            disco_groups=disco_groups,
            disco_bias=disco_bias,
            diff_layers=diff_layers,
            conv_padding_mode=conv_padding_mode,
            fin_diff_kernel_size=fin_diff_kernel_size,
            mix_derivatives=mix_derivatives,
        )

        self.input_gates = nn.ModuleList()
        self.hidden_gates = nn.ModuleList()
        self.biases = nn.ParameterList()

        # 3 gates: Update (z), Reset (r), Candidate (h_tilde)
        # Each gate: one LocalNOBlocks processing x (input_gate, with scaling),
        #            one LocalNOBlocks processing h (hidden_gate, no scaling)
        for _ in range(3):
            self.input_gates.append(
                LocalNOBlocks(
                    default_in_shape=default_in_shape,
                    resolution_scaling_factor=input_scaling,
                    **local_no_kwargs,
                )
            )
            self.hidden_gates.append(
                LocalNOBlocks(
                    default_in_shape=h_default_in_shape,
                    resolution_scaling_factor=None,
                    **local_no_kwargs,
                )
            )
            if complex_data:
                self.biases.append(nn.Parameter(torch.randn(()) + 1j * torch.randn(())))
            else:
                self.biases.append(nn.Parameter(torch.randn(())))

    def forward(self, x, h):
        """Forward pass for LocalRNO cell.

        Parameters
        ----------
        x : torch.Tensor
            Input function at current timestep, shape (batch, hidden_channels, *spatial_dims)
        h : torch.Tensor
            Hidden state from previous timestep, shape (batch, hidden_channels, *spatial_dims_h)

        Returns
        -------
        torch.Tensor
            Updated hidden state, shape (batch, hidden_channels, *spatial_dims_h)
        """
        # Update gate
        update_gate = torch.sigmoid(
            self.input_gates[0](x) + self.hidden_gates[0](h) + self.biases[0]
        )

        # Reset gate
        reset_gate = torch.sigmoid(
            self.input_gates[1](x) + self.hidden_gates[1](h) + self.biases[1]
        )

        # Candidate state
        h_combined = (
            self.input_gates[2](x)
            + self.hidden_gates[2](reset_gate * h)
            + self.biases[2]
        )

        if x.dtype == torch.cfloat:
            candidate_state = cselu(h_combined)
        else:
            candidate_state = F.selu(h_combined)

        h_next = (1.0 - update_gate) * h + update_gate * candidate_state

        return h_next


class LocalRNOBlock(nn.Module):
    """N-Dimensional Recurrent Neural Operator layer using LocalNO blocks.

    Extends LocalRNOCell to process a full sequence of time steps:
        For t = 1 to T:
            h_t = LocalRNOCell(x_t, h_{t-1})

    Parameters
    ----------
    n_modes : tuple of int
        Number of Fourier modes along each dimension.
    hidden_channels : int
        Number of hidden channels.
    default_in_shape : tuple of int
        Default spatial input shape for DISCO convolutions.
    return_sequences : bool, optional
        Whether to return hidden states for all timesteps. Default: False

    Other Parameters
    ----------------
    resolution_scaling_factor : float or None, optional
        Spatial resolution scaling factor. Default: None
    max_n_modes : tuple of int or None, optional
        Maximum number of modes. Default: None
    local_no_block_precision : str, optional
        Precision for spectral convolution. Default: "full"
    use_channel_mlp : bool, optional
        Whether to use MLP after each LocalNO block. Default: False
    channel_mlp_dropout : float, optional
        Dropout in channel MLP. Default: 0
    channel_mlp_expansion : float, optional
        Expansion factor for channel MLP. Default: 0.5
    non_linearity : callable, optional
        Nonlinear activation. Default: F.gelu
    stabilizer : str or None, optional
        Stabilizer. Default: None
    norm : str or None, optional
        Normalization layer. Default: None
    preactivation : bool, optional
        Pre-activation mode. Default: False
    local_no_skip : str or None, optional
        Skip connection type for LocalNO layers. Default: "linear"
    channel_mlp_skip : str or None, optional
        Skip connection type for channel MLP. Default: "soft-gating"
    complex_data : bool, optional
        Complex-valued data. Default: False
    separable : bool, optional
        Separable spectral convolution. Default: False
    factorization : str or None, optional
        Tensor factorization. Default: None
    rank : float, optional
        Tensor rank. Default: 1.0
    conv_module : nn.Module, optional
        Convolution module. Default: SpectralConv
    fixed_rank_modes : bool, optional
        Fixed rank modes. Default: False
    implementation : str, optional
        Factorization implementation. Default: "factorized"
    decomposition_kwargs : dict, optional
        Extra decomposition kwargs. Default: {}
    enforce_hermitian_symmetry : bool, optional
        Enforce Hermitian symmetry in iFFT. Default: True
    disco_layers : bool or list of bool, optional
        Whether to use DISCO layers. Default: True
    disco_kernel_shape : int or list of int, optional
        DISCO kernel shape. Default: [2, 4]
    radius_cutoff : float or None, optional
        DISCO radius cutoff. Default: None
    domain_length : list of float or None, optional
        Physical domain length. Default: None
    disco_groups : int, optional
        DISCO convolution groups. Default: 1
    disco_bias : bool, optional
        DISCO bias. Default: True
    diff_layers : bool or list of bool, optional
        Whether to use differential layers. Default: True
    conv_padding_mode : str, optional
        Padding mode. Default: "periodic"
    fin_diff_kernel_size : int, optional
        Finite difference kernel size. Default: 3
    mix_derivatives : bool, optional
        Mix derivatives across channels. Default: True
    """

    def __init__(
        self,
        n_modes,
        hidden_channels,
        default_in_shape,
        return_sequences=False,
        resolution_scaling_factor=None,
        max_n_modes=None,
        local_no_block_precision="full",
        use_channel_mlp=False,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        preactivation=False,
        local_no_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=None,
        enforce_hermitian_symmetry=True,
        disco_layers=True,
        disco_kernel_shape=None,
        radius_cutoff=None,
        domain_length=None,
        disco_groups=1,
        disco_bias=True,
        diff_layers=True,
        conv_padding_mode="periodic",
        fin_diff_kernel_size=3,
        mix_derivatives=True,
    ):
        super().__init__()
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        if disco_kernel_shape is None:
            disco_kernel_shape = [2, 4]

        self.hidden_channels = hidden_channels
        self.return_sequences = return_sequences
        self.resolution_scaling_factor = resolution_scaling_factor

        self.cell = LocalRNOCell(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            default_in_shape=default_in_shape,
            resolution_scaling_factor=resolution_scaling_factor,
            max_n_modes=max_n_modes,
            local_no_block_precision=local_no_block_precision,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            local_no_skip=local_no_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            separable=separable,
            factorization=factorization,
            rank=rank,
            conv_module=conv_module,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
            disco_layers=disco_layers,
            disco_kernel_shape=disco_kernel_shape,
            radius_cutoff=radius_cutoff,
            domain_length=domain_length,
            disco_groups=disco_groups,
            disco_bias=disco_bias,
            diff_layers=diff_layers,
            conv_padding_mode=conv_padding_mode,
            fin_diff_kernel_size=fin_diff_kernel_size,
            mix_derivatives=mix_derivatives,
        )

        if complex_data:
            self.bias_h = nn.Parameter(torch.randn(()) + 1j * torch.randn(()))
        else:
            self.bias_h = nn.Parameter(torch.randn(()))

    def forward(self, x, h=None):
        """Forward pass for LocalRNO layer.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (batch, timesteps, hidden_channels, *spatial_dims)
        h : torch.Tensor or None, optional
            Initial hidden state, shape (batch, hidden_channels, *spatial_dims_h).
            If None, initialized to zeros plus bias. Default: None

        Returns
        -------
        torch.Tensor
            If return_sequences=True: all hidden states,
                shape (batch, timesteps, hidden_channels, *spatial_dims_h)
            If return_sequences=False: final hidden state,
                shape (batch, hidden_channels, *spatial_dims_h)
        """
        batch_size, timesteps = x.shape[:2]
        dom_sizes = x.shape[3:]

        if h is None:
            if not self.resolution_scaling_factor:
                h_shape = (batch_size, self.hidden_channels, *dom_sizes)
            else:
                scaled_sizes = tuple(
                    int(round(self.resolution_scaling_factor * s)) for s in dom_sizes
                )
                h_shape = (batch_size, self.hidden_channels, *scaled_sizes)
            h = torch.zeros(h_shape, dtype=x.dtype, device=x.device)
            h = h + self.bias_h

        outputs = []
        for i in range(timesteps):
            h = self.cell(x[:, i], h)
            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return h
