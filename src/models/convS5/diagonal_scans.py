# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------

import jax
from jax import lax, numpy as np
from .conv_ops import vmap_conv


# Scan functions
@jax.vmap
def conv_binary_operator_old(q_i, q_j):
    """Assumes 1x1 kernels
       :inputs q_i an q_j are tuples containing (A_i, BU_i) and (A_j, BU_j)
       :inputs A_i and A_j are (P,)
       :inputs BU_i and BU_j are bszxH_UxW_UxP
       :returns tuple where first entry AA is (P,)
                and second entry is bszxH_UxW_UxP"""

    A_i, BU_i = q_i
    A_j, BU_j = q_j

    # AA = convolve_1x1_kernels(A_j, A_i)
    AA = A_j * A_i
    A_jBU_i = np.expand_dims(A_j, (0, 1, 2)) * BU_i

    return AA, A_jBU_i + BU_j


@jax.vmap
def conv_binary_operator(q_i, q_j):
    """Assumes 1x1 kernels
       :inputs q_i an q_j are tuples containing (A_i, BU_i) and (A_j, BU_j)
       :inputs A_i and A_j are (P,)
       :inputs BU_i and BU_j are bszxH_UxW_UxP
       :returns tuple where first entry AA is (P,)
                and second entry is bszxH_UxW_UxP"""

    A_i, BU_i = q_i
    A_j, BU_j = q_j

    # AA = convolve_HxW_kernels(A_j, A_i)
    AA = lax.conv_general_dilated(A_i, A_j, (1, 1), "SAME", dimension_numbers=('HWNC', 'HWIO', 'HWNC'))
    A_jBU_i = lax.conv_general_dilated(BU_i, A_j, (1, 1), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

    return AA, A_jBU_i + BU_j


def apply_convSSM_parallel(A, B, C, us, x0):
    """Compute the output sequence of the convolutional SSM
        given the input sequence using a parallel scan.
        Computes x_k = A * x_{k-1} + B * u_k
                 y_k = C * x_k     + D * U_k
        where * is a convolution operator.
    Args:
        A (complex64): Conv kernel A                (P,)
        B (complex64): input-to-state conv kernel   (k_B,k_B,U,P)
        C (complex64): state-to-output conv kernel  (k_c,k_c, P, U)
        us (float32): input sequence of features  (L,bsz,H, W, U)
        x0 (complex64): initial state               (bsz, H, W, P)
    Returns:
        x_L (complex64): the last state of the SSM  (bsz, H, W, P)
        ys (float32): the conv SSM outputs        (L,bsz, H, W, U)
    """
    L = us.shape[0]
    Bus = vmap_conv(B, np.complex64(us))
    if len(A.shape) == 1:
        As = (np.eye(len(A)) * A)[None, None, None].repeat(L, 0)
        Bus = Bus.at[0].add(np.expand_dims(A, (0, 1, 2)) * x0)
    else:
        Ae = (np.eye(len(A))[..., None, None] * A[None])
        Ae = Ae.transpose(2, 3, 0, 1)
        As = np.ones((L,)+Ae.shape) * Ae[None]
        Ax = lax.conv_general_dilated(x0.astype(A.dtype), A[None].repeat(x0.shape[-1], 0), (1, 1), "SAME", dimension_numbers=('NHWC', 'IOHW', 'NHWC'))
        Bus = Bus.at[0].add(Ax)
    # As = A * np.ones((L,)+A.shape)
    # Bus = Bus.at[0].add(np.expand_dims(A, (0, 1, 2)) * x0)

    _, xs = lax.associative_scan(conv_binary_operator, (As, Bus))

    ys = 2 * vmap_conv(C, xs).real

    return xs[-1], ys


def apply_convSSM_sequential(A, B, C, us, x0):
    """Compute the output sequence of the convolutional SSM
        given the input sequence sequentially. For testing purposes.
    Args:
        A (complex64): Conv kernel A                (P,)
        B (complex64): input-to-state conv kernel   (k_B,k_B,U,P)
        C (complex64): state-to-output conv kernel  (k_c,k_c, P, U)
        us (float32): input sequence of features  (L,bsz,H, W, U)
        x0 (complex64): initial state               (bsz, H, W, P)
    Returns:
        x_L (complex64): the last state of the SSM  (bsz, H, W, P)
        ys (float32): the conv SSM outputs        (L,bsz, H, W, U)
    """
    def step(x_k_1, u_k):
        Bu = lax.conv_general_dilated(np.complex64(u_k), B, (1, 1),
                                      'SAME',
                                      dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        x_k = np.expand_dims(A, (0, 1, 2)) * x_k_1 + Bu
        y_k = 2 * lax.conv_general_dilated(x_k, C, (1, 1),
                                           'SAME',
                                           dimension_numbers=('NHWC', 'HWIO', 'NHWC')).real
        return x_k, y_k
    return lax.scan(step, np.complex64(x0), us)
