#include "kernels.cuh"

namespace ssm_cuda {

struct ScanBwdParams {
  const float* x;
  const float* A_log;
  const float* log_dt;
  const float* B;
  const float* C;
  const float* W_in;
  const float* b_in;

  //saved from fwd
  const float* h_all;

  //grad output
  const float* gy;

  //grad inputs
  float* gx;
  float* gA_log;
  float* glog_dt;
  float* gB;
  float* gC;
  float* gW_in;
  float* gb_in;

  int T;
  int D;
  int D_in;

  float log_dt_lo;
  float log_dt_hi;
};

__global__ void selective_scan_bwd_kernel(ScanBwdParams p) {
  int d = threadIdx.x;
  if (d >= p.D) return;
}

void selective_scan_bwd(ScanBwdParams& p, cudaStream_t stream) {
  int threads = (p.D + 31) / 32 * 32;
  if (threads > 256) threads = 256;

  selective_scan_bwd_kernel<<<1, threads, 0, stream>>>(p);
}

}
