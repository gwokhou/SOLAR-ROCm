// SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
// SPDX-License-Identifier: Apache-2.0

#include <c10/hip/HIPStream.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kTile = 16;

__global__ void MatmulKernel(const __half* a, const __half* b, __half* output,
                             int n) {
  __shared__ __half tile_a[kTile][kTile];
  __shared__ __half tile_b[kTile][kTile];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int column = blockIdx.x * kTile + threadIdx.x;
  float accumulator = 0.0F;

  for (int tile = 0; tile < (n + kTile - 1) / kTile; ++tile) {
    const int a_column = tile * kTile + threadIdx.x;
    const int b_row = tile * kTile + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] =
        row < n && a_column < n ? a[row * n + a_column] : __float2half(0.0F);
    tile_b[threadIdx.y][threadIdx.x] =
        b_row < n && column < n ? b[b_row * n + column] : __float2half(0.0F);
    __syncthreads();
    for (int k = 0; k < kTile; ++k) {
      accumulator += __half2float(tile_a[threadIdx.y][k]) *
                     __half2float(tile_b[k][threadIdx.x]);
    }
    __syncthreads();
  }
  if (row < n && column < n) {
    output[row * n + column] = __float2half(accumulator);
  }
}

}  // namespace

torch::Tensor run(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be HIP tensors");
  TORCH_CHECK(a.scalar_type() == torch::kFloat16 &&
                  b.scalar_type() == torch::kFloat16,
              "inputs must use float16");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.sizes() == b.sizes() &&
                  a.size(0) == a.size(1),
              "inputs must be equally sized square matrices");
  auto output = torch::empty_like(a);
  const int n = static_cast<int>(a.size(0));
  const dim3 threads(kTile, kTile);
  const dim3 blocks((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
  hipLaunchKernelGGL(
      MatmulKernel, blocks, threads, 0, c10::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), n);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "HIP kernel launch failed");
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
