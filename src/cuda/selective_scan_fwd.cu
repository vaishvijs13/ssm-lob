#include <cassert>
#include "kernels.cuh"

namespace ssm_cuda {

__global__ void selective_scan_fwd_kernel(ScanParams p) {
  int d = threadIdx.x;
  if (d >= p.D) return;

  float A = -softplus_d(p.A_log[d]);
  float ld = clamp_d(p.log_dt[d], p.log_dt_lo, p.log_dt_hi);
  float dt = expf(ld);
  float decay = expf(A * dt);
  float B_d = p.B[d];
  float C_d = p.C[d];

  float h = 0.0f;

  //shared mem for output reduction
  __shared__ float y_shared[256];

  for (int t = 0; t < p.T; t++) {
    float u = p.b_in[d];
    for (int j = 0; j < p.D_in; j++) {
      u += p.W_in[d * p.D_in + j] * p.x[t * p.D_in + j];
    }

    float bu = B_d * u;
    h = decay * h + (1.0f - decay) * bu;

    //save for backward
    if (p.h_all) {
      p.h_all[t * p.D + d] = h;
    }

    //out contribution
    y_shared[d] = C_d * h;
    __syncthreads();

    if (d == 0) {
      float y_t = 0.0f;
      for (int i = 0; i < p.D; i++) {
        y_t += y_shared[i];
      }
      p.y[t] = y_t;
    }
    __syncthreads();
  }

  p.h_out[d] = h;
}

void selective_scan_fwd(ScanParams& p, cudaStream_t stream) {
  int threads = (p.D + 31) / 32 * 32;
  if (threads > 256) threads = 256;
  selective_scan_fwd_kernel<<<1, threads, 0, stream>>>(p);
}

//change a: W_in staged into shared memory once at kernel entry, serial reduction
__global__ void selective_scan_fwd_sharedw_kernel(ScanParams p) {
  int d = threadIdx.x;
  if (d >= p.D) return;

  //each thread loads its own row of W_in; D threads * D_in floats = full W_in
  extern __shared__ float smem[];
  float* smem_W   = smem;                //[D * D_in]
  float* y_shared = smem + p.D * p.D_in; //[D]

  for (int j = 0; j < p.D_in; j++)
    smem_W[d * p.D_in + j] = p.W_in[d * p.D_in + j];
  __syncthreads();

  float A = -softplus_d(p.A_log[d]);
  float ld = clamp_d(p.log_dt[d], p.log_dt_lo, p.log_dt_hi);
  float dt = expf(ld);
  float decay = expf(A * dt);
  float B_d = p.B[d];
  float C_d = p.C[d];
  float h = 0.0f;

  for (int t = 0; t < p.T; t++) {
    float u = p.b_in[d];
    for (int j = 0; j < p.D_in; j++)
      u += smem_W[d * p.D_in + j] * p.x[t * p.D_in + j];

    float bu = B_d * u;
    h = decay * h + (1.0f - decay) * bu;

    if (p.h_all)
      p.h_all[t * p.D + d] = h;

    y_shared[d] = C_d * h;
    __syncthreads();

    if (d == 0) {
      float y_t = 0.0f;
      for (int i = 0; i < p.D; i++)
        y_t += y_shared[i];
      p.y[t] = y_t;
    }
    __syncthreads();
  }

  p.h_out[d] = h;
}

void selective_scan_fwd_sharedw(ScanParams& p, cudaStream_t stream) {
  int threads = (p.D + 31) / 32 * 32;
  if (threads > 256) threads = 256;
  int smem = (p.D * p.D_in + p.D) * (int)sizeof(float);
  selective_scan_fwd_sharedw_kernel<<<1, threads, smem, stream>>>(p);
}

//change a+b: W_in in shared memory + warp shuffle reduction
//requires D == 32 (exactly one warp); asserts at runtime, static at launch site
__global__ void selective_scan_fwd_opt_kernel(ScanParams p) {
  int d = threadIdx.x;

  extern __shared__ float smem_W[]; //[D * D_in]

  for (int j = 0; j < p.D_in; j++)
    smem_W[d * p.D_in + j] = p.W_in[d * p.D_in + j];
  __syncthreads();

  float A = -softplus_d(p.A_log[d]);
  float ld = clamp_d(p.log_dt[d], p.log_dt_lo, p.log_dt_hi);
  float dt = expf(ld);
  float decay = expf(A * dt);
  float B_d = p.B[d];
  float C_d = p.C[d];
  float h = 0.0f;

  for (int t = 0; t < p.T; t++) {
    float u = p.b_in[d];
    for (int j = 0; j < p.D_in; j++)
      u += smem_W[d * p.D_in + j] * p.x[t * p.D_in + j];

    float bu = B_d * u;
    h = decay * h + (1.0f - decay) * bu;

    if (p.h_all)
      p.h_all[t * p.D + d] = h;

    //warp shuffle tree reduction: 5 instructions, no shared mem, no sync
    float val = C_d * h;
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (d == 0) p.y[t] = val;
    //no __syncthreads() needed: warp-internal, all 32 lanes stay live
  }

  p.h_out[d] = h;
}

void selective_scan_fwd_opt(ScanParams& p, cudaStream_t stream) {
  //D == 32 required: exactly one warp, full mask shuffle
  assert(p.D == 32);
  int smem = p.D * p.D_in * (int)sizeof(float);
  selective_scan_fwd_opt_kernel<<<1, 32, smem, stream>>>(p);
}

}
