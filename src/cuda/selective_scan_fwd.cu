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

    //ssm update
    float bu = B_d * u;
    h = decay * h + (1.0f - decay) * bu;

    //out contribution
    y_shared[d] = C_d * h;
    __syncthreads();

    //reduce (thread 0 sums)
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
  int threads = (p.D + 31) / 32 * 32; //round up to warp
  if (threads > 256) threads = 256;

  selective_scan_fwd_kernel<<<1, threads, 0, stream>>>(p);
}

}
