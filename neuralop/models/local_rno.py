from typing import Tuple, List, Union, Literal

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

warnings.filterwarnings("once", category=UserWarning)

from ..layers.local_rno_block import LocalRNOBlock
from ..layers.padding import DomainPadding
from ..layers.channel_mlp import ChannelMLP
from ..layers.spectral_convolution import SpectralConv
from ..layers.complex import ComplexValued
from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from .base_model import BaseModel


class LocalRNO(BaseModel, name="LocalRNO"):
    """
    N-Dimensional Local Recurrent Neural Operator.

    Identical architecture to RNO but replaces FNO blocks with LocalNO blocks
    that combine spectral convolutions with differential and local integral kernels
    as described in [LocalNO]_.

    The GRU-like recurrence relation is preserved:
        z_t = sigmoid(W_z x + U_z h_{t-1} + b_z)
        r_t = sigmoid(W_r x + U_r h_{t-1} + b_r)
        hat_h_t = selu(W_h x_t + U_h (r_t * h_{t-1}) + b_h)
        h_t = (1 - z_t) * h_{t-1} + z_t * hat_h_t,

    where W_i and U_i are LocalNO blocks instead of FNO blocks.

    References
    ----------
    .. [RNO] Liu-Schiaffini, M., Singer, C. E., Kovachki, N., Schneider, T.,
        Azizzadenesheli, K., & Anandkumar, A. (2023). Tipping point forecasting
        in non-stationary dynamics on function spaces. arXiv preprint
        arXiv:2308.08794.
    .. [LocalNO] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K.,
        Anandkumar A.; "Neural Operators with Localized Integral and Differential
        Kernels" (2024). ICML 2024, https://arxiv.org/pdf/2402.16845.

    Parameters
    ----------
    n_modes : Tuple[int, ...]
        Number of modes to keep in Fourier Layer, along each dimension.
        The dimensionality of the LocalRNO is inferred from ``len(n_modes)``.
    in_channels : int
        Number of input channels in input function.
    out_channels : int
        Number of output channels in output function.
    hidden_channels : int
        Width of the LocalRNO (i.e. number of channels).
    default_in_shape : Tuple[int, ...]
        Default spatial input shape for DISCO convolutions in LocalNO blocks.
        Must match ``len(n_modes)``.
    n_layers : int, optional
        Number of LocalRNO layers. Default: 4
    rno_skip : bool, optional
        Whether to use skip connections between LocalRNO layers. Default: False

    Other Parameters
    ----------------
    lifting_channel_ratio : Number, optional
        Ratio of lifting channels to hidden_channels. Default: 2
    projection_channel_ratio : Number, optional
        Ratio of projection channels to hidden_channels. Default: 2
    positional_embedding : Union[str, nn.Module, None], optional
        Positional embedding type. Options: "grid", GridEmbeddingND, GridEmbedding2D,
        None. Default: "grid"
    non_linearity : nn.Module, optional
        Non-linear activation function. Default: F.gelu
    norm : str or None, optional
        Normalization layer. Options: "ada_in", "group_norm", "instance_norm", None.
        Default: None
    complex_data : bool, optional
        Whether data is complex-valued. Default: False
    use_channel_mlp : bool, optional
        Whether to use MLP after each LocalNO block. Default: False
    channel_mlp_dropout : float, optional
        Dropout in channel MLP. Default: 0
    channel_mlp_expansion : float, optional
        Expansion factor for channel MLP. Default: 0.5
    channel_mlp_skip : str or None, optional
        Skip connection type in channel MLP. Default: "soft-gating"
    local_no_skip : str or None, optional
        Skip connection type in LocalNO layers. Default: "linear"
    return_sequences : bool, optional
        Whether the final layer returns the full sequence. Default: False
    resolution_scaling_factor : Union[Number, List[Number]], optional
        Layer-wise resolution scaling factor. Default: None
    domain_padding : Union[Number, List[Number]], optional
        Percentage of domain padding. Default: None
    local_no_block_precision : str, optional
        Precision for spectral convolution. Default: "full"
    stabilizer : str or None, optional
        Stabilizer in LocalNO block. Default: None
    max_n_modes : Tuple[int, ...], optional
        Maximum number of modes during training. Default: None
    factorization : str or None, optional
        Tensor factorization method. Default: None
    rank : float, optional
        Tensor rank for factorization. Default: 1.0
    fixed_rank_modes : bool, optional
        Whether to fix certain rank modes. Default: False
    implementation : str, optional
        Factorization implementation. Default: "factorized"
    decomposition_kwargs : dict, optional
        Extra kwargs for tensor decomposition. Default: {}
    separable : bool, optional
        Separable spectral convolution. Default: False
    preactivation : bool, optional
        ResNet-style preactivation. Default: False
    conv_module : nn.Module, optional
        Convolution module for LocalNO blocks. Default: SpectralConv
    enforce_hermitian_symmetry : bool, optional
        Enforce Hermitian symmetry in iFFT. Default: True
    disco_layers : Union[bool, List[bool]], optional
        Whether to use DISCO local integral layers. Default: True
    disco_kernel_shape : Union[int, List[int]], optional
        DISCO kernel shape. Default: [2, 4]
    radius_cutoff : float or None, optional
        DISCO radius cutoff. Default: None
    domain_length : List[float] or None, optional
        Physical domain length. Default: None (uses [2]*n_dim)
    disco_groups : int, optional
        DISCO convolution groups. Default: 1
    disco_bias : bool, optional
        DISCO bias. Default: True
    diff_layers : Union[bool, List[bool]], optional
        Whether to use differential kernel layers. Default: True
    conv_padding_mode : str, optional
        Padding mode for spatial convolutions. Default: "periodic"
    fin_diff_kernel_size : int, optional
        Kernel size for finite difference convolution. Default: 3
    mix_derivatives : bool, optional
        Mix derivatives across channels. Default: True
    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        default_in_shape: Tuple[int, ...],
        n_layers: int = 4,
        rno_skip: bool = False,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = False,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        local_no_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        return_sequences: bool = False,
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        local_no_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        enforce_hermitian_symmetry: bool = True,
        disco_layers: Union[bool, List[bool]] = True,
        disco_kernel_shape: Union[int, List[int]] = None,
        radius_cutoff: float = None,
        domain_length: List[float] = None,
        disco_groups: int = 1,
        disco_bias: bool = True,
        diff_layers: Union[bool, List[bool]] = True,
        conv_padding_mode: str = "periodic",
        fin_diff_kernel_size: int = 3,
        mix_derivatives: bool = True,
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        if disco_kernel_shape is None:
            disco_kernel_shape = [2, 4]

        super().__init__()
        self.n_dim = len(n_modes)

        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.rno_skip = rno_skip

        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * self.hidden_channels)

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * self.hidden_channels)

        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.local_no_skip = (local_no_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.local_no_block_precision = local_no_block_precision

        ## Domain padding
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        ## Positional embedding
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(
                    f"Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}"
                )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(
                f"Error: tried to instantiate LocalRNO positional embedding with {positional_embedding}, "
                f"expected one of 'grid', GridEmbeddingND, GridEmbedding2D, or None"
            )

        ## Resolution scaling factor
        if resolution_scaling_factor:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        else:
            resolution_scaling_factor = [None] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        ## Return sequences configuration
        return_sequences_list = [True] * (self.n_layers - 1) + [return_sequences]
        self.return_sequences = return_sequences_list

        if domain_length is None:
            domain_length = [2] * self.n_dim

        module_list = [
            LocalRNOBlock(
                n_modes=self.n_modes,
                hidden_channels=hidden_channels,
                default_in_shape=default_in_shape,
                return_sequences=return_sequences_list[i],
                resolution_scaling_factor=self.resolution_scaling_factor[i],
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
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(module_list)

        ## Lifting layer
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim

        if self.lifting_channels:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        else:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        ## Projection layer
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def forward(
        self,
        x,
        init_hidden_states=None,
        return_hidden_states=False,
        keep_states_padded=False,
    ):
        """
        Forward pass for the Local Recurrent Neural Operator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, timesteps, in_channels, *spatial_dims).
            Channel dim MUST be at index 2, time dim MUST be at index 1.
        init_hidden_states : list[torch.Tensor] or None, optional
            Per-layer initial hidden states, each shaped
            (batch, hidden_channels, *spatial_dims_h). Default: None
        return_hidden_states : bool, optional
            Whether to also return final hidden states. Default: False
        keep_states_padded : bool, optional
            If True, treats init_hidden_states as already padded and returns
            final_hidden_states in padded form. Default: False

        Returns
        -------
        pred : torch.Tensor
            Output tensor, shape (batch, out_channels, *spatial_dims_out).
        final_hidden_states : list[torch.Tensor], optional
            Returned only if return_hidden_states=True.
        """
        expected_rank = 3 + self.n_dim
        if x.ndim != expected_rank:
            raise ValueError(
                f"LocalRNO.forward expected input of rank {expected_rank} = "
                f"(batch, timesteps, channels, {self.n_dim} spatial dims), "
                f"got rank {x.ndim} with shape {tuple(x.shape)}"
            )
        if x.shape[2] != self.in_channels:
            raise ValueError(
                f"LocalRNO.forward expected x.shape[2] == in_channels ({self.in_channels}); "
                f"got {x.shape[2]}. Input must be shaped as (batch, timesteps, in_channels, *spatial_dims)."
            )

        batch_size, timesteps = x.shape[:2]

        if init_hidden_states is None:
            init_hidden_states = [None] * self.n_layers
        else:
            if self.domain_padding and not keep_states_padded:
                padded_hidden_states = []
                for h in init_hidden_states:
                    if h is not None:
                        h = self.domain_padding.pad(h)
                    padded_hidden_states.append(h)
                init_hidden_states = padded_hidden_states

        # Reshape for processing: (batch*timesteps, channels, *spatial_dims)
        x = x.reshape(batch_size * timesteps, *x.shape[2:])

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)
        x = x.reshape(batch_size, timesteps, *x.shape[1:])

        if self.domain_padding:
            x = x.reshape(batch_size * timesteps, *x.shape[2:])
            x = self.domain_padding.pad(x)
            x = x.reshape(batch_size, timesteps, *x.shape[1:])

        final_hidden_states = []
        for i in range(self.n_layers):
            pred_x = self.layers[i](x, init_hidden_states[i])
            if i < self.n_layers - 1:
                if self.rno_skip:
                    x = x + pred_x
                else:
                    x = pred_x
                final_hidden_states.append(x[:, -1])
            else:
                x = pred_x
                final_hidden_states.append(x)
        h = final_hidden_states[-1]

        if self.domain_padding:
            h = self.domain_padding.unpad(h)

            if return_hidden_states and not keep_states_padded:
                unpadded_states = []
                for state in final_hidden_states:
                    unpadded_states.append(self.domain_padding.unpad(state))
                final_hidden_states = unpadded_states

        pred = self.projection(h)

        if return_hidden_states:
            return pred, final_hidden_states
        else:
            return pred

    def predict(self, x, num_steps, grid_function=None):
        """Autoregressively predict future time steps.

        Parameters
        ----------
        x : torch.Tensor
            Initial input sequence, shape (batch, timesteps, in_channels, *spatial_dims).
        num_steps : int
            Number of future time steps to predict.
        grid_function : callable, optional
            Function generating positional embeddings to concatenate with predictions.
            Signature: grid_function(shape, device) -> torch.Tensor. Default: None

        Returns
        -------
        torch.Tensor
            Predicted sequence, shape (batch, num_steps, out_channels, *spatial_dims).
        """
        output = []
        states = [None] * self.n_layers

        for _ in range(num_steps):
            pred, states = self.forward(
                x, states, return_hidden_states=True, keep_states_padded=True
            )

            output.append(pred)
            x = pred.reshape((pred.shape[0], 1, *pred.shape[1:]))
            if grid_function:
                grid = grid_function(
                    (x.shape[0], x.shape[1], 1, x.shape[-2], x.shape[-1]), x.device
                )
                x = torch.cat((x, grid), dim=2)

        return torch.stack(output, dim=1)
