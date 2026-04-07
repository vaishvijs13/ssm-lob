#pragma once

#include <cuda_runtime.h>

#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif

namespace ssm_cuda {

#ifdef __CUDACC__
constexpr int MAX_D = 256;
constexpr int THREADS = 128;
constexpr int ITEMS = 4;

template<int kThreads, int kItems, int kD, typename T = float>
struct ScanTraits {
  using input_t = T;
  using scan_t = float;

  static constexpr int threads = kThreads;
  static constexpr int items = kItems;
  static constexpr int D = kD;
  static constexpr int block_size = kThreads * kItems;

  using BlockLoad = cub::BlockLoad<T, kThreads, kItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStore = cub::BlockStore<T, kThreads, kItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockScan = cub::BlockScan<scan_t, kThreads, cub::BLOCK_SCAN_WARP_SCANS>;

  static constexpr int smem_size =
    sizeof(typename BlockLoad::TempStorage) +
    sizeof(typename BlockStore::TempStorage) +
    sizeof(typename BlockScan::TempStorage) +
    kD * sizeof(scan_t);
};

using Traits64 = ScanTraits<128, 4, 64>;
#endif

#ifdef __CUDACC__
//device math
__device__ __forceinline__ float softplus_d(float x) {
  return x > 20.0f ? x : logf(1.0f + expf(x));
}

__device__ __forceinline__ float sigmoid_d(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float clamp_d(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}
#endif

#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

struct ScanParams {
  const float* x;
  const float* A_log;
  const float* log_dt;
  const float* B;
  const float* C;
  const float* W_in;
  const float* b_in;
  float* y;
  float* h_out;
  float* h_all;

  int T;
  int D;
  int D_in;

  float log_dt_lo;
  float log_dt_hi;
};

}
