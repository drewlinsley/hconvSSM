# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------

from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import he_normal, normal, uniform
from jax.numpy.linalg import eigh
from jax.scipy.linalg import block_diag

# from . import diagonal_scans
from . import diagonal_scans_conv as diagonal_scans
# from . import diagonal_scans_fft as diagonal_scans
from .conv_ops import VmapResnetBlock, VmapDiag_CD_Conv, VmapDiagResnetBlock, Half_GLU


def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def sample_u(H, W, N, key, dtype, c=0.9999, s=0.001):
    # return (c - uniform(s)(key, shape=[H, W, N, N], dtype=dtype)) ** 2
    return (c - uniform(s)(key, shape=[N, N, H, W], dtype=dtype)) ** 2


def make_LRU(N, H, W, key, dtype=np.float32):
    A = sample_u(H, W, N, key=key, dtype=dtype)
    V = -0.5 * np.log(A + A.min())
    theta = 2 * np.pi * sample_u(H, W, N, key=key, dtype=dtype)
    C = np.exp(jax.lax.complex(-V, theta))
    _, V = eigh(C)
    return C, V


def make_Normal_HiPPO(N):
    """normal approximation to the HiPPO-LegS matrix"""
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)
    nhippo = hippo + P[:, np.newaxis] * P[np.newaxis, :]

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)

    return nhippo, P, B


def make_NPLR_init(N, key=None, H=7, W=7, init="LRU"):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
        init (str): LRU or HiPPO
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    if init.lower() == "hippo":
        wi = make_HiPPO(N)
    elif init.lower() == "lru":
        wi = make_LRU(N, H, W, key)
    else:
        raise NotImplementedError(init)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return wi, P, B


def make_DPLR_init(N, key=None, H=7, W=7, init="LRU"):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_init(N, key=key, H=H, W=W, init=init)
    if init.lower() == "hippo":
        S = A + P[:, np.newaxis] * P[np.newaxis, :]
    elif init.lower() == "lru":
        wi = make_LRU(N, H, W, key)
        raise RuntimeError("Stop this path")
    else:
        raise NotImplementedError(init)

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    # Identity = np.eye(Lambda.shape[0])
    Identity = np.ones(Lambda.shape[0])
    # Lambda_bar = np.exp(Lambda * Delta[:, None, None, None])
    Lambda_bar = Lambda * Delta[:, None, None, None]
    # B_bar = (1/Lambda * (Lambda_bar-Identity[...,  None, None])) * B_tilde
    B_bar = ((1/Lambda) * Lambda_bar) * B_tilde
    return Lambda_bar, B_bar


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    U, dt_min, dt_max = input
    log_steps = []
    for i in range(U):
        key, skey = jax.random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)
    return np.array(log_steps)


def initialize_C_kernel(key, shape):
    """For general kernels, e.g. C,D, encoding/decoding"""
    out_dim, in_dim, k = shape
    fan_in = in_dim*(k**2)

    # Note in_axes should be the first by default:
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling
    return he_normal()(key,
                       (fan_in, out_dim)).reshape(out_dim,
                                                  in_dim,
                                                  k, k).transpose(0, 2, 3, 1).reshape(-1, in_dim)


def initialize_B_kernel(key, shape):
    """We will store the B kernel as a matrix,
    returns shape: (out_dim, in_dim*k*k)"""
    out_dim, in_dim, k = shape
    fan_in = in_dim*(k**2)

    # Note in_axes should be the first by default:
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling
    return he_normal()(key,
                       (fan_in, out_dim)).T


def init_VinvB(key, shape, Vinv):
    # B = initialize_B_kernel(key, shape)
    pre_shape = Vinv.shape
    Vinv = Vinv.reshape(pre_shape[0], pre_shape[1], pre_shape[2] * pre_shape[3])
    B = he_normal()(key, Vinv.shape)
    VinvB = np.einsum("ABC,ABC->ABC", Vinv, B)
    VinvB = VinvB.reshape(pre_shape)
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def init_CV(key, shape, V):
    pre_shape = V.shape
    V = V.reshape(pre_shape[0], pre_shape[1], pre_shape[2] * pre_shape[3])
    C = he_normal()(key, V.shape)
    CV = np.einsum("ABC,ABC->ABC", C, V)
    CV = CV.reshape(pre_shape)
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


# def init_CV(key, shape, V):
#     out_dim, in_dim, k = shape
#     C = initialize_C_kernel(key, shape)
#     CV = C @ V
#     CV = CV.reshape(out_dim, k, k, in_dim//2).transpose(1, 2, 3, 0)
#     CV_real = CV.real
#     CV_imag = CV.imag
#     return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


class ConvS5SSM(nn.Module):
    Lambda_re_init: jax.Array
    Lambda_im_init: jax.Array
    V: jax.Array
    Vinv: jax.Array
    clip_eigs: bool
    parallel: bool  # Compute scan in parallel
    activation: nn.module
    num_groups: int

    U: int    # Number of SSM input and output features
    P: int    # Number of state features of SSM
    k_B: int  # B kernel width/height
    k_C: int  # C kernel width/height
    k_D: int  # D kernel width/height

    dt_min: float  # for initializing discretization step
    dt_max: float
    C_D_config: str = "standard"
    squeeze_excite: bool = False

    def setup(self):
        # Initialize diagonal state to state transition kernel Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) and output to state (C) kernels
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(rng,
                                                          shape,
                                                          self.Vinv),
                            (2*self.P, self.U, self.k_B))
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        self.C = self.param("C",
                            lambda rng, shape: init_CV(rng, shape, self.V),
                            (self.U, 2*self.P, self.k_C))
        self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        if self.C_D_config == "standard":
            self.C_D_conv = VmapDiag_CD_Conv(activation=self.activation,
                                             k_D=self.k_D,
                                             out_channels=self.U,
                                             num_groups=self.num_groups,
                                             squeeze_excite=self.squeeze_excite)
        elif self.C_D_config == "resnet":
            self.C_D_conv = VmapResnetBlock(activation=self.activation,
                                            k_size=self.k_D,
                                            out_channels=self.U,
                                            num_groups=self.num_groups,
                                            squeeze_excite=self.squeeze_excite)
        elif self.C_D_config == "diag_resnet":
            self.C_D_conv = VmapDiagResnetBlock(activation=self.activation,
                                                k_size=self.k_D,
                                                out_channels=self.U,
                                                num_groups=self.num_groups,
                                                squeeze_excite=self.squeeze_excite)

        elif self.C_D_config == "half_glu":
            self.C_D_conv = Half_GLU(dim=self.U)
        else:
            raise NotImplementedError(self.C_D_config)

        # Initialize learnable discretization steps
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = np.exp(self.log_step[:, 0])

        if self.parallel:
            # Discretize
            self.A_bar, self.B_bar = discretize_zoh(self.Lambda,
                                                    B_tilde,
                                                    step)
            init_spatial_kernels = False  # False
            if init_spatial_kernels:
                # self.A_bar = self.A_bar.reshape(-1, 1).repeat(9, -1).reshape(len(self.A_bar), 3, 3)
                # self.A_bar = he_normal()(jax.random.PRNGKey(42), shape=self.A_bar.shape, dtype=self.A_bar.dtype)
                self.A_bar = he_normal(in_axis=1, out_axis=0)(jax.random.PRNGKey(42), shape=self.A_bar.shape, dtype=self.A_bar.dtype)
                self.B_bar = he_normal(in_axis=1, out_axis=0)(jax.random.PRNGKey(42), shape=self.B_bar.shape, dtype=self.B_bar.dtype)
                # self.B_bar = self.B_bar.reshape(self.P, self.U, self.k_B, self.k_B).transpose(2, 3, 1, 0)
        else:
            # trick to cache the discretization for step-by-step
            # generation
            def init_discrete():
                A_bar, B_bar = discretize_zoh(self.Lambda,
                                              B_tilde,
                                              step)
                B_bar = B_bar.reshape(self.P, self.U, self.k_B, self.k_B).transpose(2, 3, 1, 0)
                return A_bar, B_bar
            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

    def __call__(self, input_sequence, x0):
        """
        input sequence is shape (L, bsz, H, W, U)
        x0 is (bsz, H, W, U)
        Returns:
            x_L (float32): the last state of the SSM  (bsz, H, W, P)
            ys (float32): the conv SSM outputs       (L,bsz, H, W, U)
        """
        if self.parallel:
            # TODO: right now parallel version assumes x_init is zeros
            x_last, ys = diagonal_scans.apply_convSSM_parallel(self.A_bar,
                                                               self.B_bar,
                                                               self.C_tilde,
                                                               input_sequence,
                                                               x0)

        else:
            # For sequential generation (e.g. autoregressive decoding)
            x_last, ys = diagonal_scans.apply_convSSM_sequential(*self.ssm,
                                                                 self.C_tilde,
                                                                 input_sequence,
                                                                 x0)
        if self.C_D_config == "standard":
            ys = self.C_D_conv(ys, input_sequence)
        elif self.C_D_config == "resnet":
            ys = self.C_D_conv(ys)
        elif self.C_D_config in ["half_glu"]:
            ys = jax.vmap(self.C_D_conv)(ys)
        return x_last, ys


def hippo_initializer(ssm_size, blocks, H, W, init, key):
    block_size = int(ssm_size/blocks)
    if init.lower() == "hippo":
        Lambda, _, _, V, _ = make_DPLR_init(block_size, H=H, W=W, init=init, key=key)
    elif init.lower() == "lru":
        Lambda, V = make_LRU(block_size, H=H, W=W, key=key)        
    else:
        not NotImplementedError(init)

    # ssm_size = ssm_size // 2
    # block_size = block_size // 2

    if init.lower() == "hippo":
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))
    elif init.lower() == "lru":
        Lambda = Lambda[..., :block_size, :block_size]
        Vc = V.conj()  # .T

        # Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
        # V = block_diag(*([V] * blocks))
        # Vinv = block_diag(*([Vc] * blocks))
        Vinv = Vc
    return Lambda.real, Lambda.imag, V, Vinv, ssm_size


def init_ConvS5SSM(ssm_size,
                   blocks,
                   clip_eigs,
                   U,
                   k_B,
                   k_C,
                   k_D,
                   dt_min,
                   dt_max,
                   C_D_config,
                   H=7,
                   W=7,
                   init="LRU"):  # LRU hippo
    key = jax.random.PRNGKey(42)
    Lambda_re_init, Lambda_im_init,\
        V, Vinv, ssm_size = hippo_initializer(ssm_size, blocks, H, W, key=key, init=init)

    return partial(ConvS5SSM,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   clip_eigs=clip_eigs,
                   U=U,
                   P=ssm_size,
                   k_B=k_B,
                   k_C=k_C,
                   k_D=k_D,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   C_D_config=C_D_config)
