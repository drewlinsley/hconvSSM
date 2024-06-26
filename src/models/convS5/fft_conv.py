import os

from functools import partial
from typing import Iterable, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from skimage import io
from skimage.transform import rescale, resize
from jax import lax, vmap


def complex_matmul(a: jnp.ndarray, b: jnp.ndarray, groups: int = 1) -> jnp.ndarray:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    # a = a.reshape(a.shape[0], groups, -1, *a.shape[2:])
    # b = b.reshape(groups, -1, *b.shape[1:])
    # a = jnp.moveaxis(a, 2, a.ndim - 1)[..., None, :]
    # b = jnp.moveaxis(b, (1, 2), (b.ndim - 1, b.ndim - 2))

    # complex value matrix multiplication
    real = jnp.einsum("COHW,NCHW->NOHW", b.real, a.real) - jnp.einsum("COHW,NCHW->NOHW", b.imag, a.imag)
    imag = jnp.einsum("COHW,NCHW->NOHW", b.imag, a.real) + jnp.einsum("COHW,NCHW->NOHW", b.real, a.imag)
    # real = jnp.einsum("NCHW,COHW->NOHW", a.real, b.real) - jnp.einsum("NCHW,COHW->NOHW", a.imag, b.imag)
    # imag = jnp.einsum("NCHW,COHW->NOHW", a.imag, b.real) + jnp.einsum("NCHW,COHW->NOHW", a.real, b.imag)

    # real = a.real @ b.real - a.imag @ b.imag
    # imag = a.imag @ b.real + a.real @ b.imag
    # real = jnp.moveaxis(real, real.ndim - 1, 2).squeeze(-1)
    # imag = jnp.moveaxis(imag, imag.ndim - 1, 2).squeeze(-1)
    # c = jnp.zeros(real.shape, dtype=jnp.complex64)
    # c.real, c.imag = real, imag
    c = lax.complex(real, imag).astype(jnp.complex64)
    return c

    # return c.reshape(c.shape[0], -1, *c.shape[3:])


def complex_matmul_split_expensive(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, groups: int = 1) -> jnp.ndarray:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.reshape(a.shape[0], groups, -1, *a.shape[2:])
    b = b.reshape(groups, -1, *b.shape[1:])
    c = c.reshape(groups, -1, *c.shape[1:])

    a = a.transpose(0, 1, 3, 4, 2)[..., None, :]
    # a = jnp.moveaxis(a, 2, a.ndim - 1)[..., None, :]
    # b = jnp.moveaxis(b, (1, 2), (b.ndim - 1, b.ndim - 2))
    # c = jnp.moveaxis(c, (1, 2), (c.ndim - 1, c.ndim - 2))
    b = b.transpose(0, 3, 4, 1, 2)
    c = c.transpose(0, 3, 4, 1, 2)

    # complex value matrix multiplication
    real = a.real @ b - a.imag @ c
    imag = a.imag @ b + a.real @ c
    real = real.squeeze().transpose(0, 3, 1, 2)[:, None]
    imag = imag.squeeze().transpose(0, 3, 1, 2)[:, None]
    # real = jnp.moveaxis(real, real.ndim - 1, 2).squeeze(-1)
    # imag = jnp.moveaxis(imag, imag.ndim - 1, 2).squeeze(-1)
    # c = lax.complex(real, imag).astype(jnp.complex64)
    c = real + imag*1j
    return c.reshape(c.shape[0], -1, *c.shape[3:])


def complex_matmul_cheap(a: jnp.ndarray, b: jnp.ndarray, groups: int = 1) -> jnp.ndarray:
    # Reshape inputs
    # a = a.reshape(a.shape[0], groups, -1, *a.shape[2:])
    # b = b.reshape(groups, -1, *b.shape[1:])

    # Complex matrix multiplication using einsum
    # real = jnp.einsum('NSCHW,SNCHW->NSHWC', a.real, b.real) - jnp.einsum('NSCHW,SNCHW->NSHWC', a.imag, -1 * b.imag)
    # imag = jnp.einsum('NSCHW,SNCHW->NSHWC', a.imag, b.real) - jnp.einsum('NSCHW,SNCHW->NSHWC', a.real, -1 * b.imag)
    real = jnp.einsum("NCHW,OCHW->NCHW", a.real, b.real) - jnp.einsum("NCHW,OCHW->NCHW", a.imag, -1 * b.imag)
    imag = jnp.einsum("NCHW,OCHW->NCHW", a.imag, b.real) + jnp.einsum("NCHW,OCHW->NCHW", a.real, -1 * b.imag)
    # real = jnp.einsum("ABCD,NBDC->ABCD", a.real, b.real) - jnp.einsum("ABCD,NBDC->ABCD", a.imag, -1 * b.imag)
    # imag = jnp.einsum("ABCD,NBDC->ABCD", a.imag, b.real) - jnp.einsum("ABCD,NBDC->ABCD", a.real, -1 * b.imag)

    # Reshape and combine real and imaginary parts
    # c = (real + 1j * imag).transpose(0, 2, 3, 1).reshape(real.shape[0], -1, *real.shape[4:])
    # c = (real + 1j * imag).squeeze(1).transpose(0, 3, 1, 2)
    c = real + 1j * imag
    return c


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_transform(
        signal: jnp.ndarray,
        kernel: jnp.ndarray,
        signal_dims: str = "NCHW",
        kernel_dims: str = "IOHW",
        bias: jnp.ndarray = None,
        padding: Union[int, Iterable[int], str] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        signal_size = None,
        crop_output = True
        ) -> jnp.ndarray:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (jnp.ndarray) Input tensor to be convolved with the kernel.
        kernel: (jnp.ndarray) Convolution kernel.
        bias: (jnp.ndarray) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int], str) If int, Number of zero samples to pad then
            input on the last dimension. If str, "same" supported to pad input for size preservation.
        padding_mode: (str) Padding mode to use from {constant, reflection, replication}.
                      reflection not available for 3d.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
        dilation: (Union[int, Iterable[int]) Dilation rate for the kernel.
        groups: (int) Number of groups for the convolution.

    Returns:
        (jnp.ndarray) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    if isinstance(padding, str):
        if padding == "same":
            padding_ = [(k - 1) / 2 for k in kernel.shape[2:]]
        else:
            raise ValueError(f"Padding mode {padding} not supported.")
    else:
        padding_ = to_ntuple(padding, n=n)

    # internal dilation offsets
    offset = jnp.zeros((1, 1, *dilation_), dtype=signal.dtype)
    offset = offset.at[(slice(None), slice(None), *((0,) * n))].set(1.0)

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = jnp.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # # Pad the input signal & kernel tensors (round to support even sized convolutions)
    # signal_padding = [r(p).astype(int) for p in padding_[::-1] for r in (np.floor, np.ceil)]
    # signal_padding = [[0, 0], [0, 0], [signal_padding[0], signal_padding[1]], [signal_padding[2], signal_padding[3]]]
    # signal = jnp.pad(signal, signal_padding, mode=padding_mode)

    signal_padding = [(0, 0)] * 2 + [(int(np.ceil(p)), int(np.floor(p))) for p in padding_[::-1]]
    signal = jnp.pad(signal, signal_padding, mode=padding_mode)

    # signal = jnp.pad(signal, [[0, 0], [0, 0], [padding, padding], [padding, padding]], mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    original_size = signal.shape  # original signal size without padding to even
    # if signal.shape[-1] % 2 != 0:
    #     signal = jnp.pad(signal, [(0, 0)] * (signal.ndim - 1) + [(0, 1)])

    kernel_hw = [kernel_dims.index("H"), kernel_dims.index("W")]
    signal_hw = [signal_dims.index("H"), signal_dims.index("W")]
    kernel_padding = [
        pad
        for i in range(2)  # reversed(range(2, signal.ndim))
        for pad in [0, signal.shape[signal_hw[i]] - kernel.shape[kernel_hw[i]]]
    ]
    kernel_padding = [[0, 0], [0, 0], [kernel_padding[0], kernel_padding[1]], [kernel_padding[2], kernel_padding[3]]]
    # ONE POSSIBILITY __ CENTER PAD KERNEL
    padded_kernel = jnp.pad(kernel, kernel_padding)

    return signal, padded_kernel, kernel, original_size, stride_


def fft_compute(
        signal: jnp.ndarray,
        padded_kernel: jnp.ndarray,
        kernel,
        original_size,
        stride_,
        signal_size,
        bias=None,
        crop_output=True,
        signal_dims: str = "NCHW",
        kernel_dims: str = "IOHW",
        ) -> jnp.ndarray:

    kernel_hw = [kernel_dims.index("H"), kernel_dims.index("W")]
    signal_hw = [signal_dims.index("H"), signal_dims.index("W")]

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    if signal_size is not None:
        signal_fr = jnp.fft.fftn(signal, (signal_size, signal_size), axes=tuple(signal_hw))
        kernel_fr = jnp.fft.fftn(padded_kernel, (signal_size, signal_size), axes=tuple(kernel_hw))
    else:
        signal_fr = jnp.fft.fftn(signal, axes=tuple(signal_hw))
        kernel_fr = jnp.fft.fftn(padded_kernel, axes=tuple(kernel_hw))

    output_fr = complex_matmul_cheap(signal_fr, kernel_fr)  # (signal_fr, kernel_fr.real, kernel_fr.imag * -1, groups=groups)

    # kernel_fr.imag = kernel_fr.imag * -1
    # kernel_fr = lax.complex(kernel_fr.real, kernel_fr.imag * -1)
    # output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    # output_fr = complex_matmul_split(signal_fr, kernel_fr.real, kernel_fr.imag * -1, groups=groups)
    output_fr = complex_matmul_cheap(signal_fr, kernel_fr)  # (signal_fr, kernel_fr.real, kernel_fr.imag * -1, groups=groups)
    if signal_size is not None:
        output = jnp.fft.ifftn(output_fr, (signal_size, signal_size), axes=tuple(range(2, signal.ndim)))
    else:
        output = jnp.fft.ifftn(output_fr, axes=tuple(range(2, signal.ndim)))


    return output


def fft_preproc(
        signal: jnp.ndarray,
        kernel: jnp.ndarray,
        signal_dims: str = "NCHW",
        kernel_dims: str = "IOHW",
        bias: jnp.ndarray = None,
        padding: Union[int, Iterable[int], str] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        signal_size = None,
        crop_output = True
        ) -> jnp.ndarray:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    TESTED FOR SIGNAL: NCHW KERNEL: OIHW

    Args:
        signal: (jnp.ndarray) Input tensor to be convolved with the kernel.
        kernel: (jnp.ndarray) Convolution kernel.
        bias: (jnp.ndarray) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int], str) If int, Number of zero samples to pad then
            input on the last dimension. If str, "same" supported to pad input for size preservation.
        padding_mode: (str) Padding mode to use from {constant, reflection, replication}.
                      reflection not available for 3d.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
        dilation: (Union[int, Iterable[int]) Dilation rate for the kernel.
        groups: (int) Number of groups for the convolution.

    Returns:
        (jnp.ndarray) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    if isinstance(padding, str):
        if padding == "same":
            padding_ = [(k - 1) / 2 for k in kernel.shape[2:]]
        else:
            raise ValueError(f"Padding mode {padding} not supported.")
    else:
        padding_ = to_ntuple(padding, n=n)

    # internal dilation offsets
    offset = jnp.zeros((1, 1, *dilation_), dtype=signal.dtype)
    offset = offset.at[(slice(None), slice(None), *((0,) * n))].set(1.0)

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = jnp.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # # Pad the input signal & kernel tensors (round to support even sized convolutions)
    # signal_padding = [r(p).astype(int) for p in padding_[::-1] for r in (np.floor, np.ceil)]
    # signal_padding = [[0, 0], [0, 0], [signal_padding[0], signal_padding[1]], [signal_padding[2], signal_padding[3]]]
    # signal = jnp.pad(signal, signal_padding, mode=padding_mode)

    signal_padding = [(0, 0)] * 2 + [(int(np.ceil(p)), int(np.floor(p))) for p in padding_[::-1]]
    signal = jnp.pad(signal, signal_padding, mode=padding_mode)

    # signal = jnp.pad(signal, [[0, 0], [0, 0], [padding, padding], [padding, padding]], mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    original_size = signal.shape  # original signal size without padding to even
    # if signal.shape[-1] % 2 != 0:
    #     signal = jnp.pad(signal, [(0, 0)] * (signal.ndim - 1) + [(0, 1)])

    kernel_hw = [kernel_dims.index("H"), kernel_dims.index("W")]
    signal_hw = [signal_dims.index("H"), signal_dims.index("W")]
    kernel_padding = [
        pad
        for i in range(2)  # reversed(range(2, signal.ndim))
        for pad in [0, signal.shape[signal_hw[i]] - kernel.shape[kernel_hw[i]]]
    ]
    kernel_padding = [[0, 0], [0, 0], [kernel_padding[0], kernel_padding[1]], [kernel_padding[2], kernel_padding[3]]]
    padded_kernel = jnp.pad(kernel, kernel_padding)
    return signal, padded_kernel, signal_size, signal_hw, kernel_hw, original_size, stride_


def fft_proc(
        signal,
        padded_kernel,
        signal_size,
        signal_hw,
        kernel_hw,
        crop_output=True
        ) -> jnp.ndarray:
    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    if signal_size is not None:
        signal_fr = jnp.fft.fftn(signal, (signal_size, signal_size), axes=tuple(signal_hw))
        kernel_fr = jnp.fft.fftn(padded_kernel, (signal_size, signal_size), axes=tuple(kernel_hw))
    else:
        signal_fr = jnp.fft.fftn(signal, axes=tuple(signal_hw))
        kernel_fr = jnp.fft.fftn(padded_kernel, axes=tuple(kernel_hw))

    # output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output_fr = complex_matmul_cheap(signal_fr, kernel_fr)  # (signal_fr, kernel_fr.real, kernel_fr.imag * -1, groups=groups)
    if signal_size is not None:
        output = jnp.fft.ifftn(output_fr, (signal_size, signal_size), axes=tuple(range(2, signal.ndim)))
    else:
        output = jnp.fft.ifftn(output_fr, axes=tuple(range(2, signal.ndim)))
    return output


def fft_postproc(
        output,
        kernel,
        signal,
        original_size,
        stride_,
        bias=None,
        crop_output=True
        ) -> jnp.ndarray:

    # Remove extra padded values
    if crop_output:
        crop_slices = [slice(None), slice(None)] + [
            slice(0, (original_size[i] - kernel.shape[i] + 1), stride_[i - 2])
            for i in range(2, signal.ndim)
        ]

        output = output[tuple(crop_slices)]

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output = output + bias.reshape(bias_shape)

    return output


def fft_conv(
        signal: jnp.ndarray,
        kernel: jnp.ndarray,
        signal_dims: str = "NCHW",
        kernel_dims: str = "IOHW",
        bias: jnp.ndarray = None,
        padding: Union[int, Iterable[int], str] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        signal_size = None,
        crop_output = True,
        crop_original = False
        ) -> jnp.ndarray:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    TESTED FOR SIGNAL: NCHW KERNEL: OIHW

    Args:
        signal: (jnp.ndarray) Input tensor to be convolved with the kernel.
        kernel: (jnp.ndarray) Convolution kernel.
        bias: (jnp.ndarray) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int], str) If int, Number of zero samples to pad then
            input on the last dimension. If str, "same" supported to pad input for size preservation.
        padding_mode: (str) Padding mode to use from {constant, reflection, replication}.
                      reflection not available for 3d.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
        dilation: (Union[int, Iterable[int]) Dilation rate for the kernel.
        groups: (int) Number of groups for the convolution.

    Returns:
        (jnp.ndarray) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    if isinstance(signal_size, int):
        signal_size = (signal_size, signal_size)

    if isinstance(padding, str):
        if padding == "same":
            padding_ = [(k - 1) / 2 for k in kernel.shape[2:]]
        else:
            raise ValueError(f"Padding mode {padding} not supported.")
    else:
        padding_ = to_ntuple(padding, n=n)

    # internal dilation offsets
    offset = jnp.zeros((1, 1, *dilation_), dtype=signal.dtype)
    offset = offset.at[(slice(None), slice(None), *((0,) * n))].set(1.0)

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = jnp.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # # Pad the input signal & kernel tensors (round to support even sized convolutions)
    # signal_padding = [r(p).astype(int) for p in padding_[::-1] for r in (np.floor, np.ceil)]
    # signal_padding = [[0, 0], [0, 0], [signal_padding[0], signal_padding[1]], [signal_padding[2], signal_padding[3]]]
    # signal = jnp.pad(signal, signal_padding, mode=padding_mode)
    signal_padding = [(0, 0)] * 2 + [(int(np.ceil(p)), int(np.floor(p))) for p in padding_[::-1]]
    original_size = signal.shape  # original signal size without padding to even

    # signal = jnp.pad(signal, signal_padding, mode=padding_mode)
    padded_signal = jnp.pad(signal, signal_padding, mode=padding_mode)

    # signal = jnp.pad(signal, [[0, 0], [0, 0], [padding, padding], [padding, padding]], mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    new_original_size = padded_signal.shape  # original signal size without padding to even

    # if signal.shape[-1] % 2 != 0:
    #     signal = jnp.pad(signal, [(0, 0)] * (signal.ndim - 1) + [(0, 1)])

    kernel_hw = [kernel_dims.index("H"), kernel_dims.index("W")]
    signal_hw = [signal_dims.index("H"), signal_dims.index("W")]
    kernel_padding = [
        pad
        for i in range(2)  # reversed(range(2, signal.ndim))
        for pad in [0, padded_signal.shape[signal_hw[i]] - kernel.shape[kernel_hw[i]]]
    ]
    kernel_padding = [[0, 0], [0, 0], [kernel_padding[0], kernel_padding[1]], [kernel_padding[2], kernel_padding[3]]]
    padded_kernel = jnp.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    if signal_size is not None:
        signal_fr = jnp.fft.fftn(padded_signal, signal_size, axes=tuple(signal_hw))
        kernel_fr = jnp.fft.fftn(padded_kernel, signal_size, axes=tuple(kernel_hw))
    else:
        signal_fr = jnp.fft.fftn(padded_signal, axes=tuple(signal_hw))
        kernel_fr = jnp.fft.fftn(padded_kernel, axes=tuple(kernel_hw))

    # output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output_fr = complex_matmul_cheap(signal_fr, kernel_fr)  # (signal_fr, kernel_fr.real, kernel_fr.imag * -1, groups=groups)
    if signal_size is not None:
        output = jnp.fft.ifftn(output_fr, signal_size, axes=tuple(range(2, padded_signal.ndim)))
    else:
        output = jnp.fft.ifftn(output_fr, axes=tuple(range(2, padded_signal.ndim)))

    # Remove extra padded values
    if crop_output:
        # crop_slices = [slice(None), slice(None)] + [
        #     slice(0, (original_size[i] - kernel.shape[i] + 1), stride_[i - 2])
        #     for i in range(2, signal.ndim)
        # ]
        crop_slices = [slice(None), slice(None), slice(signal_padding[2][0] - 1, padded_signal.shape[2] - signal_padding[2][0], stride_[0]), slice(signal_padding[3][0] - 1, padded_signal.shape[3] - signal_padding[3][0], stride_[1])]
        output = output[tuple(crop_slices)]

    if crop_original:
        crop_slices = [slice(None), slice(None)] + [
            slice(0, (new_original_size[i] - kernel.shape[i] + 1), stride_[i - 2])
            for i in range(2, padded_signal.ndim)
        ]
        output = output[tuple(crop_slices)]

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (padded_signal.ndim - 2) * [1])
        output = output + bias.reshape(bias_shape)

    return output


class FFTConv:
    """Base class for JAX FFT convolution layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Iterable[int]],
        padding: Union[int, Iterable[int]] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        ndim: int = 1,
        signal_size = None,
    ):
        """
        Args:
            in_channels: (int) Number of channels in input tensors
            out_channels: (int) Number of channels in output tensors
            kernel_size: (Union[int, Iterable[int]) Square radius of the kernel
            padding: (Union[int, Iterable[int]) Number of zero samples to pad the
                input on the last dimension. If str, "same" supported to pad input for size preservation.
            padding_mode: (str) Padding mode to use from {constant, reflection, replication}.
                          reflection not available for 3d.
            stride: (Union[int, Iterable[int]) Stride size for computing output values.
            dilation: (Union[int, Iterable[int]) Dilation rate for the kernel.
            groups: (int) Number of groups for the convolution.
            bias: (bool) If True, includes bias, which is added after convolution
            ndim: (int) Number of dimensions of the input tensor.
        """
        if in_channels % groups != 0:
            raise ValueError(
                "'in_channels' must be divisible by 'groups'."
                f"Found: in_channels={in_channels}, groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "'out_channels' must be divisible by 'groups'."
                f"Found: out_channels={out_channels}, groups={groups}."
            )

        kernel_size = to_ntuple(kernel_size, ndim)
        weight = jnp.randn(out_channels, in_channels // groups, *kernel_size)

        self.weight = weight
        self.bias = jnp.randn(out_channels) if bias else None
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.signal_size = signal_size

    def __call__(self, signal):
        return fft_conv(
            signal,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            padding_mode=self.padding_mode,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            signal_size=self.signal_size
        )

# @jax.jit
def vmap_conv(us, Bs, dns, padding="SAME", activities=True):
    """Performs a convolution at each timestep of a sequence using vmap
       to vectorize across the sequence length.
       Args:
            B (float32):   conv kernel            (k_B, k_B, U, P)
            us (float 32): input sequence         (L, bsz, h_u, w_u, U)
       Returns:
            Sequence of convolved inputs Bu (float32)  (L, bsz, h_u, w_u, P)
            )
    """
    def act_input_to_state_conv(u, B):
        # Performs the input to state convolution for a single timestep
        return lax.conv_general_dilated(u, B, (1, 1), padding, dimension_numbers=('NHWC', 'IOHW', 'NHWC'))

    def kernel_input_to_state_conv(u, B):
        # Performs the input to state convolution for a single timestep
        return lax.conv_general_dilated(u, B, (1, 1), padding, dimension_numbers=('NCHW', 'IOHW', 'NCHW'))

    if activities:
        return vmap(act_input_to_state_conv, in_axes=(0, 0))(us, Bs)
    else:
        return vmap(kernel_input_to_state_conv, in_axes=(0, 0))(us, Bs)


@jax.jit
def vmap_fft_conv(us, Bs, kdo="IOHW", padding="SAME"):
    """Performs a convolution at each timestep of a sequence using vmap
       to vectorize across the sequence length.
       Args:
            B (float32):   conv kernel            (k_B, k_B, U, P)
            us (float 32): input sequence         (L, bsz, h_u, w_u, U)
       Returns:
            Sequence of convolved inputs Bu (float32)  (L, bsz, h_u, w_u, P)
            )
    """
    def input_to_state_conv(u, B):
        # Performs the input to state convolution for a single timestep
        return fft_conv(u, B, padding="same", crop_original=True, crop_output=False)

    return vmap(input_to_state_conv, in_axes=(0, 0))(us, Bs)


@jax.jit
def conv_binary_operator(q_i, q_j):
    """Assumes 1x1 kernels
       :inputs q_i an q_j are tuples containing (A_i, BU_i) and (A_j, BU_j)
       :inputs A_i and A_j are (P,)
       :inputs BU_i and BU_j are bszxH_UxW_UxP
       :returns tuple where first entry AA is (P,)
                and second entry is bszxH_UxW_UxP"""

    A_i, BU_i = q_i
    A_j, BU_j = q_j
    # padding = (A_j.shape[-2] // 2) + 1  # A_j.shape[-2] - 1  # [6, 6]

    # A_jBU_i = lax.conv_general_dilated(BU_i, A_j, (1, 1, 1), "SAME", dimension_numbers=('TNHWC', 'TIOHW', 'TNHWC'))
    # AA = lax.conv_general_dilated(A_i, A_j, (1, 1, 1), "SAME", dimension_numbers=('TNCHW', 'TIOHW', 'TNCHW'))

    A_jBU_i = vmap_conv(BU_i, A_j, dns=('NHWC', 'IOHW', 'NHWC'), padding="SAME", activities=True)
    AA = vmap_conv(A_i, A_j, dns=('NCHW', 'IOHW', 'NCHW'), padding="SAME", activities=False)

    # plt.subplot(141);plt.imshow(BU_i[0].squeeze());plt.subplot(142);plt.imshow(A_j[0].squeeze());plt.subplot(143);plt.imshow(A_i[0].squeeze());plt.subplot(144);plt.imshow(A_jBU_i[0].squeeze().real);plt.show()
    # return AA, (A_jBU_i + BU_j).real
    return AA, A_jBU_i + BU_j


@jax.jit
def conv_fft_binary_operator(q_i, q_j):
    """Assumes 1x1 kernels
       :inputs q_i an q_j are tuples containing (A_i, BU_i) and (A_j, BU_j)
       :inputs A_i and A_j are (P,)
       :inputs BU_i and BU_j are bszxH_UxW_UxP
       :returns tuple where first entry AA is (P,)
                and second entry is bszxH_UxW_UxP"""

    A_i, BU_i = q_i
    A_j, BU_j = q_j

    # A_jBU_i = fft_conv(BU_i, A_j, padding="same", crop_original=True, crop_output=False)
    A_jBU_i = vmap_fft_conv(BU_i, A_j)
    AA = vmap_fft_conv(A_i, A_j)  # Cross correlation
    # AA = fft_conv(A_i, A_j, padding="same", crop_original=True, crop_output=False)
    return AA.real, (A_jBU_i + BU_j).real


# @jax.jit
def run_conv_control(x_conv, kernel, timesteps):
    res = []
    for t in range(timesteps):
        x_conv = lax.conv_general_dilated(x_conv, kernel, (1, 1), padding="SAME", dimension_numbers=["NHWC", "OIHW", "NHWC"])

        # FFT convolution
        res.append(x_conv)
        # xt, fk, kes, ors, st = fft_transform(x_fft, k_fft, padding="same")
        # xt = fft_proc(xt, fk, ss, shw, khw)
    return x_conv, res


# @jax.jit
def run_fft_conv_control(x_fft, k_fft, timesteps):
    res = []
    for t in range(timesteps):
        # FFT convolution
        x_fft = fft_conv(x_fft, k_fft, padding="same", crop_original=True, crop_output=False)  # Combined function
        res.append(x_fft)
        # xt, fk, kes, ors, st = fft_transform(x_fft, k_fft, padding="same")
        # xt = fft_proc(xt, fk, ss, shw, khw)
    return x_fft, res


if __name__ == '__main__':
    kernel = np.load(os.path.join("weights", "gabors_for_contours_7.npy"), allow_pickle=True, encoding="latin1").item()["s1"][0]
    ks = kernel.shape
    kernel = kernel.transpose(3, 2, 0, 1)
    kernel = kernel[:-1]  # Reduce to 24 channels

    # Make the kernel into an identity NxN kernel
    kernel = (np.eye(24)[..., None, None] * np.ones((1, 1, kernel.shape[2], kernel.shape[3]))) * kernel
    # kernel = np.ascontiguousarray(kernel)

    image = io.imread(os.path.join("data", "test.png")) 
    x = image[..., [0]] / 255.
    x = resize(x, (x.shape[0] // 4, x.shape[1] // 4), anti_aliasing=True) 
    x = x[:, 50:100]
    x = x[None].astype(np.float32)

    # Inflate channel dim
    x = x.repeat(kernel.shape[0], -1)

    from matplotlib import pyplot as plt
    from timeit import default_timer as timer

    # For recurrence, use a single kernel
    timesteps = 120

    # kernel = kernel[[0]]

    # Regular kernel convolution
    # x_conv = x.copy()
    x_fft = x.copy().transpose(0, 3, 1, 2)
    k_fft = kernel.transpose(1, 0, 2, 3)
    x_fft_a = x_fft.copy()
    k_fft_a = k_fft.copy()
    # x_fft, fk, kes, ors = fft_transform(x_fft, k_fft)
    # xt, fk, ss, shw, khw, ors, s_ = fft_preproc(x_fft, k_fft, padding="same")

    k_a = kernel[None].repeat(timesteps, 0)
    x_a = x[None].repeat(timesteps, 0)
    scan_time = timer()
    a_k_fft, a_x_fft = lax.associative_scan(conv_binary_operator, (k_a, x_a), axis=0)
    scan_time = timer() - scan_time
    afout = a_x_fft[-1]  # .real[-1]

    # FFT conv
    fft_start = timer()
    x_fft, fft_res = run_fft_conv_control(x_fft, k_fft, timesteps)
    fout = x_fft.real
    fft_time = timer() - fft_start

    # Sequential conv
    conv_start = timer()
    out, res = run_conv_control(x, kernel, timesteps)
    conv_time = timer() - conv_start

    # Now do an associative scan version
    # rAs, rBus = [], []
    # padding = 3
    # osz = (50, 125)
    # osz = x_fft_a.shape[-2:]  # (60, 131)

    # k_fft_a = fft_conv(k_fft_a, np.ones_like(k_fft_a[..., [0], [0]][None]), padding="same", signal_size=osz, crop_output=False).real
    # A_jBU_i = fft_conv(x_fft_a, k_fft_a, padding="same", crop_output=False)
    # plt.imshow(A_jBU_i.squeeze().real);plt.show()
    # AA = fft_conv(k_fft_a, k_fft_a, padding=padding, signal_size=osz, crop_output=False).real
    print("Timing")
    print("Conv: {}".format(conv_time))
    print("FFT: {}".format(fft_time))
    print("Scan: {}".format(scan_time))
    import pdb;pdb.set_trace()
    plt.subplot(221);plt.imshow(out[0, ..., 0]);plt.subplot(222);plt.imshow(kernel[0, 0]);plt.subplot(223);plt.imshow(fout[0, 0]);plt.subplot(224);plt.imshow(afout.squeeze());plt.show()

    os._exit(1)

    for i in range(len(a_x_fft)):
        plt.subplot(2, len(a_x_fft), i + 1)
        plt.imshow(a_x_fft[i].squeeze())
        if i < len(res):
            plt.subplot(2, len(a_x_fft), i + 1 + len(a_x_fft))
            plt.imshow(res[i].squeeze())
    plt.show()


