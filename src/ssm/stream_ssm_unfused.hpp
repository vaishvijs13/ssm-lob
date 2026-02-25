#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "../framework/math_utils.hpp"
#include "types.hpp"

//making this cache unfriendly for proving o(1) per step
class StreamingSSMUnfused {
public:
  explicit StreamingSSMUnfused(int D, int in_dim)
    : D_(D), in_dim_(in_dim),
      h_(D, 0.0f),
      A_log_(D, 0.0f),
      log_dt_(D, -5.0f),
      B_(D, 1.0f),
      W_in_(D * in_dim, 0.0f),
      b_in_(D, 0.0f),
      W_out_(D, 0.0f),
      b_out_(0.0f),
      //unfused intermediates
      u_(D, 0.0f),
      A_discrete_(D, 0.0f),
      dt_vals_(D, 0.0f),
      decay_vals_(D, 0.0f),
      tmp_x_(in_dim, 0.0f)
  {}

  void reset_state() {
    std::fill(h_.begin(), h_.end(), 0.0f);
  }

  std::vector<float>& A_log() { return A_log_; }
  std::vector<float>& log_dt() { return log_dt_; }
  std::vector<float>& B() { return B_; }
  std::vector<float>& W_in() { return W_in_; }
  std::vector<float>& b_in() { return b_in_; }
  std::vector<float>& W_out() { return W_out_; }
  float& b_out() { return b_out_; }

  void featurize(const TickFeatures& t, std::vector<float>& x) const {
    if ((int)x.size() != in_dim_) x.assign(in_dim_, 0.0f);
    int idx = 0;
    x[idx++] = t.bid_ask_spread;
    x[idx++] = t.order_flow_imbalance;
    x[idx++] = t.mid_price_change;
    for (int i = 0; i < 10; i++) x[idx++] = t.book_depth[i];
  }

  __attribute__((noinline))
  void compute_input_projection() {
    for (int d = 0; d < D_; d++) {
      float acc = b_in_[d];
      const int row = d * in_dim_;
      for (int j = 0; j < in_dim_; j++) {
        acc += W_in_[row + j] * tmp_x_[j];
      }
      u_[d] = acc;
    }
  }
  
  __attribute__((noinline))
  void compute_discretization() {
    constexpr float LOGDT_LO = -20.0f;
    constexpr float LOGDT_HI = 5.0f;
    for (int d = 0; d < D_; d++) {
      A_discrete_[d] = -softplus_f(A_log_[d]);
      float ld = clamp_f(log_dt_[d], LOGDT_LO, LOGDT_HI);
      dt_vals_[d] = std::exp(ld);
      decay_vals_[d] = std::exp(A_discrete_[d] * dt_vals_[d]);
    }
  }
  
  __attribute__((noinline))
  void update_state() {
    for (int d = 0; d < D_; d++) {
      float decay = decay_vals_[d];
      float bu = B_[d] * u_[d];
      h_[d] = decay * h_[d] + (1.0f - decay) * bu;
    }
  }
  
  __attribute__((noinline))
  float compute_output() {
    float y = b_out_;
    for (int d = 0; d < D_; d++) {
      y += W_out_[d] * h_[d];
    }
    return y;
  }

  float forward_step(const TickFeatures& tick) {
    featurize(tick, tmp_x_);
    compute_input_projection();
    compute_discretization();
    update_state();
    return compute_output();
  }

  const std::vector<float>& hidden_state() const { return h_; }

private:
  int D_, in_dim_;
  
  std::vector<float> h_;
  std::vector<float> A_log_, log_dt_, B_;
  std::vector<float> W_in_, b_in_;
  std::vector<float> W_out_;
  float b_out_;
  
  std::vector<float> u_;
  std::vector<float> A_discrete_;
  std::vector<float> dt_vals_;
  std::vector<float> decay_vals_;
  std::vector<float> tmp_x_;
};
