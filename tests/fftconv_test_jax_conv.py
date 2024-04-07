import os
import numpy as np

from functools import partial
from typing import Iterable, Tuple, Union

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.fft import irfftn, rfftn, ifftn, fftn
from math import ceil, floor

import jax
from jax import numpy as jnp
from jax import lax

from skimage import io
from skimage.transform import rescale, resize


def complex_matmul_old(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def complex_matmul(signal_fr: Tensor, kernel_fr: Tensor, groups: int = 1) -> Tensor:
    signal_fr = signal_fr.view(signal_fr.size(0), groups, -1, *signal_fr.shape[2:])
    kernel_fr = kernel_fr.view(groups, -1, *kernel_fr.shape[1:])

    signal_fr = torch.movedim(signal_fr, 2, signal_fr.dim() - 1).unsqueeze(-2)
    kernel_fr = torch.movedim(kernel_fr, (1, 2), (kernel_fr.dim() - 1, kernel_fr.dim() - 2))

    # complex value matrix multiplication
    real = signal_fr.real @ kernel_fr.real - signal_fr.imag @ kernel_fr.imag
    imag = signal_fr.imag @ kernel_fr.real + signal_fr.real @ kernel_fr.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    output_fr = torch.zeros(real.shape, dtype=torch.complex64, device=signal_fr.device)
    output_fr.real, output_fr.imag = real, imag

    return output_fr.view(output_fr.size(0), -1, *output_fr.shape[3:])


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


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int], str] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int], str) If int, Number of zero samples to pad then
            input on the last dimension. If str, "same" supported to pad input for size preservation.
        padding_mode: (str) Padding mode to use from {constant, reflection, replication}.
                      reflection not available for 3d.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
        dilation: (Union[int, Iterable[int]) Dilation rate for the kernel.
        groups: (int) Number of groups for the convolution.

    Returns:
        (Tensor) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    if isinstance(padding, str):
        if padding == "same":
            if stride != 1 or dilation != 1:
                raise ValueError("stride must be 1 for padding='same'.")
            padding_ = [(k - 1) / 2 for k in kernel.shape[2:]]
        else:
            raise ValueError(f"Padding mode {padding} not supported.")
    else:
        padding_ = to_ntuple(padding, n=n)

    # internal dilation offsets
    # offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset = np.zeros((1, 1, *dilation_)).astype(np.float32)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = jnp.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # Pad the input signal & kernel tensors (round to support even sized convolutions)
    signal_padding = [r(p) for p in padding_[::-1] for r in (floor, ceil)]
    signal_padding = [(0, x) for x in signal_padding]
    # signal_padding = [(x, 0) for x in signal_padding]
    signal = jnp.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    signal_size = signal.shape  # original signal size without padding to even
    if signal.shape[-1] % 2 != 0:
        signal = jnp.pad(signal, [(0, 0), (0, 0), (0, 0), (0, 1)])

    kernel_padding = [
        pad
        for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.shape[i] - kernel.shape[i]]
    ]
    kernel_padding = [(0, x) for x in kernel_padding]
    # kernel_padding = [(x, 0) for x in kernel_padding]
    padded_kernel = jnp.pad(kernel, kernel_padding)

    # # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_fr = rfftn(signal.float(), dim=tuple(range(2, signal.ndim)))
    # kernel_fr = rfftn(padded_kernel.float(), dim=tuple(range(2, signal.ndim)))
    import pdb;pdb.set_trace()
    # signal_fr = jnp.fftn(signal, dim=tuple(range(2, signal.ndim)))
    signal_fr = jnp.fft.fftn(signal, axes=tuple(range(2, signal.ndim)))
    # kernel_fr = jnp.fftn(padded_kernel,  dim=tuple(range(2, signal.ndim)))
    kernel_fr = jnp.fft.fftn(padded_kernel, axes=tuple(range(2, kernel.ndim)))

    # kernel_fr.imag = kernel_fr.imag * -1
    kernel_fr = lax.complex(kernel_fr.real, kernel_fr.imag * -1)
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)

    # output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))
    output = ifftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    crop_slices = [slice(None), slice(None)] + [
        slice(0, (signal_size[i] - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output = output + bias.view(bias_shape)

    return output


class _FFTConv(nn.Module):
    """Base class for PyTorch FFT convolution layers."""

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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

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
        weight = torch.randn(out_channels, in_channels // groups, *kernel_size)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, signal):
        return fft_conv(
            signal,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            padding_mode=self.padding_mode,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )


FFTConv1d = partial(_FFTConv, ndim=1)
FFTConv2d = partial(_FFTConv, ndim=2)
FFTConv3d = partial(_FFTConv, ndim=3)


def _gcd(x: int, y: int) -> int:
    while y:
        x, y = y, x % y
    return x


def _assert_almost_equal(x: Tensor, y: Tensor) -> bool:
    abs_error = torch.abs(x - y)
    assert abs_error.mean().item() < 5e-5
    assert abs_error.max().item() < 1e-4
    return True


def test_fft_conv_functional(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Iterable[int]],
    padding: Union[int, Iterable[int]],
    stride: Union[int, Iterable[int]],
    dilation: Union[int, Iterable[int]],
    groups: int,
    bias: bool,
    ndim: int,
    input_size: int,
):
    if padding == "same" and (stride != 1 or dilation != 1):
        # padding='same' is not compatible with strided convolutions
        return

    torch_conv = getattr(f, f"conv{ndim}d")
    groups = _gcd(in_channels, _gcd(out_channels, groups))

    batch_size = 2  # TODO: Make this non-constant?
    dims = ndim * [input_size]
    # signal = torch.randn(batch_size, in_channels, *dims)
    signal = io.imread(os.path.join("data", "test.png"))
    signal = signal[..., [0]] / 255.
    signal = resize(signal, (signal.shape[0] // 4, signal.shape[1] // 4), anti_aliasing=True)
    signal = signal[None].astype(np.float32)
    signal = signal.transpose(0, 3, 1, 2)
    # signal = torch.from_numpy(signal)
    in_channels = 1
    kwargs = dict(
        bias=torch.randn(out_channels) if bias else None,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    kernel_size = to_ntuple(kernel_size, n=signal.ndim - 2)
    # w0 = torch.randn(
    #     out_channels, in_channels // groups, *kernel_size, requires_grad=True
    # )

    kernel = np.load(os.path.join("weights", "gabors_for_contours_7.npy"), allow_pickle=True, encoding="latin1").item()["s1"][0]
    ks = kernel.shape
    kernel = kernel.transpose(3, 2, 0, 1)
    kernel = kernel[:-1]  # Reduce to 24 channels

    w0 = kernel  # torch.from_numpy(kernel)
    w1 = w0  # w0.detach().clone().requires_grad_()

    b0 = torch.randn(out_channels, requires_grad=True) if bias else None
    b1 = b0.detach().clone().requires_grad_() if bias else None

    kwargs = dict(
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    # signal = signal.astype(torch.complex64)
    signal = signal  # .to(torch.complex64)
    w0 = w0  # .to(torch.complex64)
    w1 = w1  # .to(torch.complex64)
    b1 = b1  # .to(torch.complex64)

    # y0 = fft_conv(signal, w0, bias=b0, **kwargs)
    # y1 = torch_conv(signal, w1, bias=b1, **kwargs)
    y0 = fft_conv(signal, w0, **kwargs)
    y1 = torch_conv(signal, w1, **kwargs)

    print((y0 - y1).sum())
    from matplotlib import pyplot as plt
    y0 = y0.detach().cpu().real
    y1 = y1.detach().cpu()
    plt.subplot(121);plt.imshow(y0[0, 1]);plt.subplot(122);plt.imshow(y1[0, 1]);plt.show()
    _assert_almost_equal(y0, y1)


test_fft_conv_functional(1, 24, 3, 3//2, 1, 1, 1, True, 2, 128)

