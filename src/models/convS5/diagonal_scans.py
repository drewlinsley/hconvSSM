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
from src.models.convS5.fft_conv import fft_conv


# Jimmy's version of merging kernels
def merge_conv_kernels(k1, k2):
    """
    Should work for convolving general kernels
    (though has not been rigorously tested)

    :input k1: A tensor of shape ``(out1, in1, s1, s1)``
    :input k2: A tensor of shape ``(out2, in2, s2, s2)``
    :returns: A tensor of shape ``(out2, in1, s1+s2-1, s1+s2-1)``
      so that convolving an image with it equals convolving with k1 and
      then with k2.

    Current implementation has the following assumptions:
       - Full padding is used for the sequential case
       - stride equals 1 (in both directions)
       - no dilation
    """
    # Note that first we will transpose k1 to adapt to BCHW format, i.e. we will
    # dim for the k2 kernel (since the latter is what happens when applying
    # sequentially)

    # We will also flip the k2's height and width dimensions to account for
    # the fact that this is actually cross-correlation

    # padding = (k2.shape[-1] // 2) + 1
    padding = k2.shape[-1] - 1
    k3 = lax.conv_with_general_padding(k1.transpose(1,0,2,3), # lhs = BCHW image tensor
                                       np.flip(k2, axis=(-1,-2) ),  # rhs = OIWH conv kernel tensor
                                       (1, 1),  # window strides
                                        ((padding,padding),(padding,padding)),
                                       None, #LHS/image dilation
                                       None #RHS/kernel dilation
                                       )
    print(k3.shape)
    return k3.transpose(1,0,2,3)  #permute to adapt to OIHW


def merge_convfft_kernels(k1, k2, signal_size=24):
    """
    Should work for convolving general kernels
    (though has not been rigorously tested)

    :input k1: A tensor of shape ``(out1, in1, s1, s1)``
    :input k2: A tensor of shape ``(out2, in2, s2, s2)``
    :returns: A tensor of shape ``(out2, in1, s1+s2-1, s1+s2-1)``
      so that convolving an image with it equals convolving with k1 and
      then with k2.

    Current implementation has the following assumptions:
       - Full padding is used for the sequential case
       - stride equals 1 (in both directions)
       - no dilation
    """
    # Note that first we will transpose k1 to adapt to BCHW format, i.e. we will
    # dim for the k2 kernel (since the latter is what happens when applying
    # sequentially)

    # We will also flip the k2's height and width dimensions to account for
    # the fact that this is actually cross-correlation

    # padding = (k2.shape[-1] // 2) + 1
    padding = k2.shape[-1] - 1
    k3 = fft_conv(
        signal=k1.transpose(1,0,2,3),
        kernel=np.flip(k2, axis=(-1,-2) ),
        stride=(1, 1),
        padding=padding,  # "same",  # ((padding,padding),(padding,padding)),
        signal_size=signal_size,
        crop_output=False)
    print(k3.shape)
    return k3.transpose(1,0,2,3)  #permute to adapt to OIHW


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
def conv_binary_operator_fft(q_i, q_j):
    """Assumes 1x1 kernels
       :inputs q_i an q_j are tuples containing (A_i, BU_i) and (A_j, BU_j)
       :inputs A_i and A_j are (P,)
       :inputs BU_i and BU_j are bszxH_UxW_UxP
       :returns tuple where first entry AA is (P,)
                and second entry is bszxH_UxW_UxP"""

    A_i, BU_i = q_i
    A_j, BU_j = q_j

    # AA = merge_convfft_kernels(A_i, A_j)
    import pdb;pdb.set_trace()
    AA = merge_conv_kernels(A_i, A_j)
    A_jBU_i = lax.conv_general_dilated(BU_i, A_j, (1, 1), 'SAME', dimension_numbers=('NHWC', 'OIHW', 'NHWC'))

    return AA, A_jBU_i + BU_j


def apply_convSSM_parallel(A, B, C, us, x0, use_fft=False):
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
        # Ae = (np.eye(len(A))[..., None, None] * A[None])
        # Ae = Ae.transpose(2, 3, 0, 1)
        # As = np.ones((L,)+Ae.shape) * Ae[None]
        # Ax = lax.conv_general_dilated(x0.astype(A.dtype), A[None].repeat(x0.shape[-1], 0), (1, 1), "SAME", dimension_numbers=('NHWC', 'IOHW', 'NHWC'))
        As = (np.ones((L,)+A.shape) * A)
        Ax = lax.conv_general_dilated(x0.astype(A.dtype), A, (1, 1), "SAME", dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
        Bus = Bus.at[0].add(Ax)
    # As = A * np.ones((L,)+A.shape)
    # Bus = Bus.at[0].add(np.expand_dims(A, (0, 1, 2)) * x0)

    # As = As.transpose(0, 3, 4, 1, 2)
    import pdb;pdb.set_trace()

    if use_fft:
        # import pdb;pdb.set_trace()
        # As = np.stack([np.fft.fft2(a) for a in As], 0)
        # As = fft(As)
        # Bus = fft(Bus)
        _, xs = lax.associative_scan(conv_binary_operator_fft, (As, Bus))
    else:
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
