import os

from functools import partial
from typing import Iterable, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from skimage import io
from skimage.transform import rescale, resize


def complex_matmul(a: jnp.ndarray, b: jnp.ndarray, groups: int = 1) -> jnp.ndarray:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.reshape(a.shape[0], groups, -1, *a.shape[2:])
    b = b.reshape(groups, -1, *b.shape[1:])

    a = jnp.moveaxis(a, 2, a.ndim - 1)[..., None, :]
    b = jnp.moveaxis(b, (1, 2), (b.ndim - 1, b.ndim - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = jnp.moveaxis(real, real.ndim - 1, 2).squeeze(-1)
    imag = jnp.moveaxis(imag, imag.ndim - 1, 2).squeeze(-1)
    # c = jnp.zeros(real.shape, dtype=jnp.complex64)
    # c.real, c.imag = real, imag
    c = lax.complex(real, imag).astype(jnp.complex64)

    return c.reshape(c.shape[0], -1, *c.shape[3:])


def complex_matmul_split_expensive(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, groups: int = 1) -> jnp.ndarray:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    import pdb;pdb.set_trace()
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
    real = jnp.einsum("ABCD,NBDC->ABCD", a.real, b.real) - jnp.einsum("ABCD,NBDC->ABCD", a.imag, -1 * b.imag)
    imag = jnp.einsum("ABCD,NBDC->ABCD", a.imag, b.real) - jnp.einsum("ABCD,NBDC->ABCD", a.real, -1 * b.imag)

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

    kernel_padding = [
        pad
        for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.shape[i] - kernel.shape[i]]
    ]
    kernel_padding = [[0, 0], [0, 0], [kernel_padding[0], kernel_padding[1]], [kernel_padding[2], kernel_padding[3]]]
    padded_kernel = jnp.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    if signal_size is not None:
        signal_fr = jnp.fft.fftn(signal, (signal_size, signal_size), axes=tuple(range(2, signal.ndim)))
        kernel_fr = jnp.fft.fftn(padded_kernel, (signal_size, signal_size), axes=tuple(range(2, signal.ndim)))
    else:
        signal_fr = jnp.fft.fftn(signal, axes=tuple(range(2, signal.ndim)))
        kernel_fr = jnp.fft.fftn(padded_kernel, axes=tuple(range(2, signal.ndim)))

    return signal_fr, kernel_fr


def fft_compute(
        signal: jnp.ndarray,
        kernel: jnp.ndarray,
        groups: int = 1,
        ) -> jnp.ndarray:
    # kernel_fr.imag = kernel_fr.imag * -1
    # kernel_fr = lax.complex(kernel_fr.real, kernel_fr.imag * -1)
    # output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output_fr = complex_matmul_cheap(signal, kernel, groups=groups)
    return output_fr


def fft_conv(
        signal: jnp.ndarray,
        kernel: jnp.ndarray,
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

    kernel_padding = [
        pad
        for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.shape[i] - kernel.shape[i]]
    ]
    kernel_padding = [[0, 0], [0, 0], [kernel_padding[0], kernel_padding[1]], [kernel_padding[2], kernel_padding[3]]]
    padded_kernel = jnp.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    import pdb;pdb.set_trace()
    if signal_size is not None:
        signal_fr = jnp.fft.fftn(signal, (signal_size, signal_size), axes=tuple(range(2, signal.ndim)))
        kernel_fr = jnp.fft.fftn(padded_kernel, (signal_size, signal_size), axes=tuple(range(2, signal.ndim)))
    else:
        signal_fr = jnp.fft.fftn(signal, axes=tuple(range(2, signal.ndim)))
        kernel_fr = jnp.fft.fftn(padded_kernel, axes=tuple(range(2, signal.ndim)))

    # kernel_fr.imag = kernel_fr.imag * -1
    # kernel_fr = lax.complex(kernel_fr.real, kernel_fr.imag * -1)
    # output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    # output_fr = complex_matmul_split(signal_fr, kernel_fr.real, kernel_fr.imag * -1, groups=groups)
    output_fr = complex_matmul_split_expensive(signal_fr, kernel_fr.real, kernel_fr.imag * -1, groups=groups)
    if signal_size is not None:
        output = jnp.fft.ifftn(output_fr, (signal_size, signal_size), axes=tuple(range(2, signal.ndim)))
    else:
        output = jnp.fft.ifftn(output_fr, axes=tuple(range(2, signal.ndim)))

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


if __name__ == '__main__':
    kernel = np.load(os.path.join("weights", "gabors_for_contours_11.npy"), allow_pickle=True, encoding="latin1").item()["s1"][0]
    ks = kernel.shape
    import pdb;pdb.set_trace()
    kernel = kernel.reshape(ks[3], ks[2], ks[0], ks[1])
    kernel = kernel[1:]  # Reduce to 24 channels
    # kernel = np.ascontiguousarray(kernel)

    image = io.imread(os.path.join("data", "test.png")) 
    x = image[..., [0]] / 255.
    x = resize(x, (x.shape[0] // 4, x.shape[1] // 4), anti_aliasing=True) 
    x = x[None]
    import pdb;pdb.set_trace()
    from matplotlib import pyplot as plt
    out = lax.conv_general_dilated(x, kernel, (1, 1), padding="SAME", dimension_numbers=["NHWC", "OIHW", "NHWC"])
    plt.subplot(121);plt.imshow(out[0, ..., 0]);plt.subplot(122);plt.imshow(kernel[0, 0]);plt.show()


