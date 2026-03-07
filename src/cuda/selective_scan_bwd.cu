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

  float A = -softplus_d(p.A_log[d]);
  float ld_raw = p.log_dt[d];
  float ld = clamp_d(ld_raw, p.log_dt_lo, p.log_dt_hi);
  float dt = expf(ld);
  float decay = expf(A * dt);
  float B_d = p.B[d];
  float C_d = p.C[d];

  //local grad accumulators
  float gA_log_local = 0.0f;
  float glog_dt_local = 0.0f;
  float gB_local = 0.0f;
  float gC_local = 0.0f;
  float gb_in_local = 0.0f;

  float dh = 0.0f;

  for (int t = p.T - 1; t >= 0; t--) {
    float h_t = p.h_all[t * p.D + d];
    float h_prev = (t > 0) ? p.h_all[(t - 1) * p.D + d] : 0.0f;

    //grad from output
    dh += p.gy[t] * C_d;
    gC_local += p.gy[t] * h_t;

    float u = p.b_in[d];
    for (int j = 0; j < p.D_in; j++) {
      u += p.W_in[d * p.D_in + j] * p.x[t * p.D_in + j];
    }

    float bu = B_d * u;

    //grads
    float du = dh * (1.0f - decay) * B_d;
    gB_local += dh * (1.0f - decay) * u;

    float d_decay = dh * (h_prev - bu);
    float d_dt = d_decay * A * decay;

    if (ld_raw >= p.log_dt_lo && ld_raw <= p.log_dt_hi) {
      glog_dt_local += d_dt * dt;
    }

    float dA = d_decay * dt * decay;
    float sig = sigmoid_d(p.A_log[d]);
    gA_log_local += dA * (-sig);

    dh = dh * decay;
    gb_in_local += du;

    for (int j = 0; j < p.D_in; j++) {
      atomicAdd(&p.gW_in[d * p.D_in + j], du * p.x[t * p.D_in + j]);
      atomicAdd(&p.gx[t * p.D_in + j], du * p.W_in[d * p.D_in + j]);
    }
  }

  p.gA_log[d] = gA_log_local;
  p.glog_dt[d] = glog_dt_local;
  p.gB[d] = gB_local;
  p.gC[d] = gC_local;
  p.gb_in[d] = gb_in_local;
}

void selective_scan_bwd(ScanBwdParams& p, cudaStream_t stream) {
  int threads = (p.D + 31) / 32 * 32;
  if (threads > 256) threads = 256;

  selective_scan_bwd_kernel<<<1, threads, 0, stream>>>(p);
}

}
