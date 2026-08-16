#!POPCORN leaderboard grayscale_v2
#!POPCORN gpu A100

"""v1: fuse RGB weighting and reduction into one native CUDA kernel."""

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


CPP_SOURCE = r"""
torch::Tensor launch_grayscale(torch::Tensor image, torch::Tensor output);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void grayscale_kernel(
    const float* __restrict__ image,
    float* __restrict__ output,
    int pixels) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) {
        return;
    }

    const int rgb = pixel * 3;
    const float red = image[rgb];
    const float green = image[rgb + 1];
    const float blue = image[rgb + 2];

    output[pixel] =
        red * 0.2989f + green * 0.5870f + blue * 0.1140f;
}

torch::Tensor launch_grayscale(
    torch::Tensor image,
    torch::Tensor output) {
    TORCH_CHECK(image.is_cuda() && output.is_cuda(),
                "image and output must be CUDA tensors");
    TORCH_CHECK(image.scalar_type() == at::ScalarType::Float &&
                output.scalar_type() == at::ScalarType::Float,
                "image and output must use float32");
    TORCH_CHECK(image.is_contiguous() && output.is_contiguous(),
                "image and output must be contiguous");
    TORCH_CHECK(image.dim() == 3 && image.size(2) == 3,
                "image must have shape [height, width, 3]");
    TORCH_CHECK(output.dim() == 2 &&
                output.size(0) == image.size(0) &&
                output.size(1) == image.size(1),
                "output must have shape [height, width]");

    const int pixels = static_cast<int>(output.numel());
    constexpr int threads = 256;
    const int blocks = (pixels + threads - 1) / threads;

    grayscale_kernel<<<blocks, threads>>>(
        image.data_ptr<float>(),
        output.data_ptr<float>(),
        pixels);

    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "grayscale_kernel launch failed: ",
                cudaGetErrorString(error));
    return output;
}
"""


_module = load_inline(
    name="pmpp_grayscale_v1_fused_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=["launch_grayscale"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    image, output = data
    return _module.launch_grayscale(image, output)
