#!POPCORN leaderboard grayscale_v2
#!POPCORN gpu A100

"""v11: test v10 with 64-thread blocks while retaining its exact-grid fast path."""

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


CPP_SOURCE = r"""
torch::Tensor launch_grayscale_exact_grid(
    torch::Tensor image,
    torch::Tensor output);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float4 grayscale_group(
    const float4* __restrict__ image,
    int group) {
    const float4 values0 = image[group * 3];
    const float4 values1 = image[group * 3 + 1];
    const float4 values2 = image[group * 3 + 2];

    float4 grayscale;
    grayscale.x =
        values0.x * 0.2989f + values0.y * 0.5870f + values0.z * 0.1140f;
    grayscale.y =
        values0.w * 0.2989f + values1.x * 0.5870f + values1.y * 0.1140f;
    grayscale.z =
        values1.z * 0.2989f + values1.w * 0.5870f + values2.x * 0.1140f;
    grayscale.w =
        values2.y * 0.2989f + values2.z * 0.5870f + values2.w * 0.1140f;
    return grayscale;
}

__global__ void grayscale_exact_grid_kernel(
    const float4* __restrict__ image,
    float4* __restrict__ output) {
    const int group = (blockIdx.x << 6) + threadIdx.x;
    output[group] = grayscale_group(image, group);
}

__global__ void grayscale_guarded_kernel(
    const float4* __restrict__ image,
    float4* __restrict__ output,
    int pixel_groups) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group < pixel_groups) {
        output[group] = grayscale_group(image, group);
    }
}

torch::Tensor launch_grayscale_exact_grid(
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
    TORCH_CHECK(pixels % 4 == 0,
                "the even square image must contain a multiple of four pixels");

    const int pixel_groups = pixels / 4;
    constexpr int threads = 64;
    const auto* input_ptr =
        reinterpret_cast<const float4*>(image.data_ptr<float>());
    auto* output_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());

    if ((pixel_groups & (threads - 1)) == 0) {
        grayscale_exact_grid_kernel<<<pixel_groups >> 6, threads>>>(
            input_ptr, output_ptr);
    } else {
        grayscale_guarded_kernel<<<
            (pixel_groups + threads - 1) / threads, threads>>>(
            input_ptr, output_ptr, pixel_groups);
    }

    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "grayscale exact-grid kernel launch failed: ",
                cudaGetErrorString(error));
    return output;
}
"""


_module = load_inline(
    name="pmpp_grayscale_v11_block64_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=["launch_grayscale_exact_grid"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    image, output = data
    return _module.launch_grayscale_exact_grid(image, output)
