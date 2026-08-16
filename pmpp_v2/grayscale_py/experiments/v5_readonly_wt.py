#!POPCORN leaderboard grayscale_v2
#!POPCORN gpu A100

"""v5: combine read-only float4 loads with a write-through vector store."""

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


CPP_SOURCE = r"""
torch::Tensor launch_grayscale_readonly_wt(
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
    const float4* values = image + group * 3;
    const float4 values0 = __ldg(values);
    const float4 values1 = __ldg(values + 1);
    const float4 values2 = __ldg(values + 2);

    float4 grayscale;
    grayscale.x = __fmaf_rn(
        values0.x, 0.2989f,
        __fmaf_rn(values0.y, 0.5870f, values0.z * 0.1140f));
    grayscale.y = __fmaf_rn(
        values0.w, 0.2989f,
        __fmaf_rn(values1.x, 0.5870f, values1.y * 0.1140f));
    grayscale.z = __fmaf_rn(
        values1.z, 0.2989f,
        __fmaf_rn(values1.w, 0.5870f, values2.x * 0.1140f));
    grayscale.w = __fmaf_rn(
        values2.y, 0.2989f,
        __fmaf_rn(values2.z, 0.5870f, values2.w * 0.1140f));
    return grayscale;
}

__device__ __forceinline__ void store_write_through(
    float4* output,
    int group,
    const float4& value) {
    asm volatile(
        "st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(output + group),
          "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
        : "memory");
}

__global__ __launch_bounds__(256) void grayscale_exact_kernel(
    const float4* __restrict__ image,
    float4* __restrict__ output) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    const float4 grayscale = grayscale_group(image, group);
    store_write_through(output, group, grayscale);
}

__global__ __launch_bounds__(256) void grayscale_guarded_kernel(
    const float4* __restrict__ image,
    float4* __restrict__ output,
    int pixel_groups) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group < pixel_groups) {
        output[group] = grayscale_group(image, group);
    }
}

torch::Tensor launch_grayscale_readonly_wt(
    torch::Tensor image,
    torch::Tensor output) {
    const int pixel_groups = static_cast<int>(output.numel()) / 4;
    constexpr int threads = 256;

    const auto* input_ptr =
        reinterpret_cast<const float4*>(image.data_ptr<float>());
    auto* output_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());

    if (pixel_groups % threads == 0) {
        grayscale_exact_kernel<<<pixel_groups / threads, threads>>>(
            input_ptr, output_ptr);
    } else {
        grayscale_guarded_kernel<<<
            (pixel_groups + threads - 1) / threads, threads>>>(
            input_ptr, output_ptr, pixel_groups);
    }

    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "grayscale read-only/write-through kernel launch failed: ",
                cudaGetErrorString(error));
    return output;
}
"""


_module = load_inline(
    name="pmpp_grayscale_v5_readonly_wt_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=["launch_grayscale_readonly_wt"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    image, output = data
    return _module.launch_grayscale_readonly_wt(image, output)

