#pragma once

#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include "kernels.cuh"

namespace ssm_cuda {

void selective_scan_fwd(ScanParams& p, cudaStream_t stream = 0);
void selective_scan_fwd_sharedw(ScanParams& p, cudaStream_t stream = 0);
void selective_scan_fwd_opt(ScanParams& p, cudaStream_t stream = 0);

class StreamingSSMCuda {
public:
  StreamingSSMCuda(int D, int D_in)
    : D_(D), D_in_(D_in), stream_(0)
  {
    //alloc device memory
    CUDA_CHECK(cudaMalloc(&d_A_log_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_log_dt_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_in_, D * D_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_in_, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_, D * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_h_, 0, D * sizeof(float)));

    h_A_log_.resize(D, 0.0f);
    h_log_dt_.resize(D, -5.0f);
    h_B_.resize(D, 1.0f);
    h_C_.resize(D, 1.0f);
    h_W_in_.resize(D * D_in, 0.0f);
    h_b_in_.resize(D, 0.0f);
  }

  ~StreamingSSMCuda() {
    cudaFree(d_A_log_);
    cudaFree(d_log_dt_);
    cudaFree(d_B_);
    cudaFree(d_C_);
    cudaFree(d_W_in_);
    cudaFree(d_b_in_);
    cudaFree(d_h_);
    if (d_x_) cudaFree(d_x_);
    if (d_y_) cudaFree(d_y_);
    if (d_h_all_) cudaFree(d_h_all_);
  }

  void upload_params() {
    CUDA_CHECK(cudaMemcpy(d_A_log_, h_A_log_.data(), D_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_log_dt_, h_log_dt_.data(), D_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_, h_B_.data(), D_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_, h_C_.data(), D_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_in_, h_W_in_.data(), D_ * D_in_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_in_, h_b_in_.data(), D_ * sizeof(float), cudaMemcpyHostToDevice));
  }

  void reset_state() {
    CUDA_CHECK(cudaMemset(d_h_, 0, D_ * sizeof(float)));
  }

  //run fwd pass on sequence
  //variant: 0=baseline, 1=shared_w, 2=shared_w+warp_reduce (D must==32 for variant 2)
  void forward(const float* x, int T, float* y, int variant = 0) {
    ensure_buffers(T);

    CUDA_CHECK(cudaMemcpy(d_x_, x, T * D_in_ * sizeof(float), cudaMemcpyHostToDevice));

    ScanParams p;
    p.x = d_x_;
    p.A_log = d_A_log_;
    p.log_dt = d_log_dt_;
    p.B = d_B_;
    p.C = d_C_;
    p.W_in = d_W_in_;
    p.b_in = d_b_in_;
    p.y = d_y_;
    p.h_out = d_h_;
    p.h_all = d_h_all_;
    p.T = T;
    p.D = D_;
    p.D_in = D_in_;
    p.log_dt_lo = -20.0f;
    p.log_dt_hi = 5.0f;

    if (variant == 1)      selective_scan_fwd_sharedw(p, stream_);
    else if (variant == 2) selective_scan_fwd_opt(p, stream_);
    else                   selective_scan_fwd(p, stream_);

    CUDA_CHECK(cudaMemcpy(y, d_y_, T * sizeof(float), cudaMemcpyDeviceToHost));
  }

  //param accessors
  std::vector<float>& A_log() { return h_A_log_; }
  std::vector<float>& log_dt() { return h_log_dt_; }
  std::vector<float>& B() { return h_B_; }
  std::vector<float>& C() { return h_C_; }
  std::vector<float>& W_in() { return h_W_in_; }
  std::vector<float>& b_in() { return h_b_in_; }

private:
  int D_, D_in_;
  cudaStream_t stream_;

  //device ptrs
  float* d_A_log_ = nullptr;
  float* d_log_dt_ = nullptr;
  float* d_B_ = nullptr;
  float* d_C_ = nullptr;
  float* d_W_in_ = nullptr;
  float* d_b_in_ = nullptr;
  float* d_h_ = nullptr;
  float* d_x_ = nullptr;
  float* d_y_ = nullptr;
  float* d_h_all_ = nullptr;
  int buf_T_ = 0;

  //host copies
  std::vector<float> h_A_log_;
  std::vector<float> h_log_dt_;
  std::vector<float> h_B_;
  std::vector<float> h_C_;
  std::vector<float> h_W_in_;
  std::vector<float> h_b_in_;

  void ensure_buffers(int T) {
    if (T <= buf_T_) return;
    if (d_x_) cudaFree(d_x_);
    if (d_y_) cudaFree(d_y_);
    if (d_h_all_) cudaFree(d_h_all_);
    CUDA_CHECK(cudaMalloc(&d_x_, T * D_in_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_, T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_all_, T * D_ * sizeof(float)));
    buf_T_ = T;
  }
};

}
