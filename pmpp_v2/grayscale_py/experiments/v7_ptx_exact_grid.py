#!POPCORN leaderboard grayscale_v2
#!POPCORN gpu A100

"""v7: explicit PTX vector I/O with a branch-free exact-grid fast path.

The vector-I/O technique is informed by public GPU Mode submission 230431.
"""

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


CPP_SOURCE = r"""
torch::Tensor launch_grayscale_ptx_exact_grid(
    torch::Tensor image,
    torch::Tensor output);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float4 load_float4_ptx(const float* pointer) {
    float4 value;
    asm volatile(
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
        : "l"(pointer));
    return value;
}

__device__ __forceinline__ void store_float4_ptx(
    float* pointer,
    const float4& value) {
    asm volatile(
        "st.global.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(pointer),
          "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
        : "memory");
}

__device__ __forceinline__ void grayscale_group(
    const float* __restrict__ image,
    float* __restrict__ output,
    int group) {
    const float* values = image + group * 12;
    const float4 values0 = load_float4_ptx(values);
    const float4 values1 = load_float4_ptx(values + 4);
    const float4 values2 = load_float4_ptx(values + 8);

    float4 grayscale;
    grayscale.x =
        values0.x * 0.2989f + values0.y * 0.5870f + values0.z * 0.1140f;
    grayscale.y =
        values0.w * 0.2989f + values1.x * 0.5870f + values1.y * 0.1140f;
    grayscale.z =
        values1.z * 0.2989f + values1.w * 0.5870f + values2.x * 0.1140f;
    grayscale.w =
        values2.y * 0.2989f + values2.z * 0.5870f + values2.w * 0.1140f;

    store_float4_ptx(output + group * 4, grayscale);
}

__global__ void grayscale_ptx_exact_grid_kernel(
    const float* __restrict__ image,
    float* __restrict__ output) {
    const int group = (blockIdx.x << 8) + threadIdx.x;
    grayscale_group(image, output, group);
}

__global__ void grayscale_ptx_guarded_kernel(
    const float* __restrict__ image,
    float* __restrict__ output,
    int pixel_groups) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group < pixel_groups) {
        grayscale_group(image, output, group);
    }
}

torch::Tensor launch_grayscale_ptx_exact_grid(
    torch::Tensor image,
    torch::Tensor output) {
    const int pixel_groups = static_cast<int>(output.numel()) >> 2;
    constexpr int threads = 256;

    const float* input_ptr = image.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    if ((pixel_groups & (threads - 1)) == 0) {
        grayscale_ptx_exact_grid_kernel<<<pixel_groups >> 8, threads>>>(
            input_ptr, output_ptr);
    } else {
        grayscale_ptx_guarded_kernel<<<
            (pixel_groups + threads - 1) / threads, threads>>>(
            input_ptr, output_ptr, pixel_groups);
    }
    return output;
}
"""


_module = load_inline(
    name="pmpp_grayscale_v7_ptx_exact_grid_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=["launch_grayscale_ptx_exact_grid"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    image, output = data
    return _module.launch_grayscale_ptx_exact_grid(image, output)

