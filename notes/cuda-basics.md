# CUDA basics used by this project

## CPU and GPU

A CPU has relatively few sophisticated cores optimized for low-latency,
branch-heavy work. A GPU has many arithmetic lanes optimized for running a very
large number of similar operations at once. Matrix multiplication exposes that
parallel work: its output elements can be computed independently.

## Host, device, and kernels

- **Host:** the CPU and Python/C++ code that prepares tensors and launches work.
- **Device:** the GPU and its memory.
- **Kernel:** a function executed by many GPU threads. CUDA marks a host-callable
  device kernel with `__global__`.

The C++ host wrapper launches a kernel with CUDA's triple-chevron syntax:

```cpp
grayscale_kernel<<<blocks, threads>>>(input, output, pixels);
```

`blocks` and `threads` configure the grid; every launched thread begins at the
same kernel entry point but sees its own `blockIdx` and `threadIdx` values.

## Thread, block, grid, and warp

- A **thread** is one logical execution instance. In the matmul attempt, it owns
  one output element `C[row, col]`.
- A **block** is a cooperating group of threads. The matmul attempt uses
  `16 x 16 = 256` threads per block.
- A **grid** is all blocks in one launch. Its dimensions are rounded up so every
  output row and column is covered.
- NVIDIA schedules threads in **warps** of 32. If threads in a warp take
  different branches, both paths may need to execute; this is warp divergence.

The 1D indexing expression

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

adds the block's global starting offset to the thread's local position. The
matmul kernel uses this expression in x for `col` and in y for `row`.

## Memory hierarchy

- **Registers:** private per-thread storage; the matmul attempt's accumulator
  normally lives here. Fastest, but limited.
- **Shared memory:** small, fast, explicitly managed storage shared by one
  block. Later tiling can use it to reuse A and B values.
- **L1/L2 caches:** hardware-managed caches between the cores and VRAM.
- **Global memory / VRAM:** large and accessible by every block, but expensive
  compared with registers/shared memory.

When neighboring threads access neighboring addresses, the GPU can combine
their requests into fewer memory transactions. This is **coalescing**. In the
matmul attempt, adjacent x-threads read neighboring `B[k, col]` values and store
neighboring `C[row, col]` values.

## Grayscale mapping used in the measured kernel

For v1, `idx = blockIdx.x * blockDim.x + threadIdx.x` is a pixel index.
With 256 threads per block, each full block covers 256 pixels and contains eight
warps. A bounds check protects the final partially filled block.

For v2, the same expression is a group-of-four index. Each thread reads twelve
contiguous floats through three `float4` values and writes four results through
one `float4`. This reduces instruction overhead without changing the required
input/output byte count.

Example: `blockIdx.x=2`, `blockDim.x=16`, `threadIdx.x=5` gives global index
`2*16+5 = 37`.

## Matrix multiplication refresher

For `A[M,K]` and `B[K,N]`, the result has shape `C[M,N]`:

```text
C[row, col] = sum(A[row, k] * B[k, col]) for k = 0..K-1
```

A naive CPU implementation has three loops: output row, output column, and the
inner `k` reduction. A simple GPU mapping launches many threads so every
`(row, col)` output has one owner; each owner still runs its own `k` loop.

That mapping is parallel and correct in real arithmetic, but it does not
explicitly share operands. Each `A[row,k]` is useful to many output columns, and
each `B[k,col]` is useful to many output rows. Re-fetching those values from
global memory wastes bandwidth. Floating-point addition is also not associative,
so changing the reduction order can change the rounded resultâ€”as the strict
PMPP FP16 check demonstrated.

## Tiling intuition

Every output is a dot product, so many neighboring outputs reuse the same rows
of A and columns/regions of B. A tiled kernel has a block cooperatively load a
small A tile and B tile into shared memory, synchronize, and reuse those values
for several multiply-adds before loading the next tiles.

