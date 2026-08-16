#!POPCORN leaderboard matmul_v2
#!POPCORN gpu A100

"""v0: one CUDA thread computes one FP16 matrix-multiplication output."""

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


CPP_SOURCE = r"""
torch::Tensor launch_naive(torch::Tensor a, torch::Tensor b, torch::Tensor c);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k_size) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < k_size; ++k) {
        sum += __half2float(a[row * k_size + k])
             * __half2float(b[k * n + col]);
    }
    c[row * n + col] = __float2half_rn(sum);
}

torch::Tensor launch_naive(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(),
                "a, b, and c must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half &&
                b.scalar_type() == at::ScalarType::Half &&
                c.scalar_type() == at::ScalarType::Half,
                "a, b, and c must use float16");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
                "a, b, and c must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2,
                "a, b, and c must be matrices");

    const int m = static_cast<int>(a.size(0));
    const int k_size = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));
    TORCH_CHECK(b.size(0) == k_size, "inner dimensions must match");
    TORCH_CHECK(c.size(0) == m && c.size(1) == n,
                "c must have shape [m, n]");

    const dim3 threads(16, 16);
    const dim3 blocks(
        (n + threads.x - 1) / threads.x,
        (m + threads.y - 1) / threads.y);

    matmul_naive_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        m,
        n,
        k_size);

    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "matmul_naive_kernel launch failed: ",
                cudaGetErrorString(error));
    return c;
}
"""


_module = load_inline(
    name="pmpp_matmul_v0_naive_ext",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=["launch_naive"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    return _module.launch_naive(a, b, c)
