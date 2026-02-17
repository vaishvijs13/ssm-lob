#pragma once
#include <vector>
#include <cmath>
#include "../framework/math_utils.hpp"
#include "types.hpp"

class MultiScaleSSM {
public:
  MultiScaleSSM(int D_per_scale, int in_dim, int n_scales = 3)
    : n_scales_(n_scales), D_per_(D_per_scale), in_dim_(in_dim),
      D_total_(D_per_scale * n_scales),
      h_(D_per_scale * n_scales, 0.0f),

      A_log_(D_per_scale * n_scales, 0.0f),
      log_dt_(D_per_scale * n_scales, 0.0f),
      B_(D_per_scale * n_scales, 1.0f),

      W_in_(D_per_scale * n_scales * in_dim, 0.0f),
      b_in_(D_per_scale * n_scales, 0.0f),

      //learned cross-scale combiner
      W_combine_(D_per_scale * n_scales, 0.0f),
      b_combine_(0.0f),

      tmp_x_(in_dim, 0.0f),
      tmp_u_(D_per_scale * n_scales, 0.0f)
  {
    for (int s = 0; s < n_scales_; s++) {
      float base_dt = -7.0f + (float)s * 3.0f;
      for (int d = 0; d < D_per_; d++) {
        int idx = s * D_per_ + d;
        log_dt_[idx] = base_dt;
        A_log_[idx] = -1.0f;
        W_combine_[idx] = 1.0f / (float)D_total_;
      }
    }
  }

  void reset_state() { std::fill(h_.begin(), h_.end(), 0.0f); }

  std::vector<float>& A_log() { return A_log_; }
  std::vector<float>& log_dt() { return log_dt_; }
  std::vector<float>& B() { return B_; }
  std::vector<float>& W_in() { return W_in_; }
  std::vector<float>& b_in() { return b_in_; }
  std::vector<float>& W_combine() { return W_combine_; }
  float& b_combine() { return b_combine_; }
  int D_total() const { return D_total_; }
  int n_scales() const { return n_scales_; }
  int D_per() const { return D_per_; }

  const std::vector<float>& hidden_state() const { return h_; }

  float forward_step(const TickFeatures& tick) {
    featurize(tick, tmp_x_);

    for (int d = 0; d < D_total_; d++) {
      float acc = b_in_[d];
      const int row = d * in_dim_;
      for (int j = 0; j < in_dim_; j++)
        acc += W_in_[row + j] * tmp_x_[j];
      tmp_u_[d] = acc;
    }

    //parallel SSM updates at different timescales
    float y = b_combine_;

    constexpr float LOGDT_LO = -20.0f;
    constexpr float LOGDT_HI = 5.0f;

    for (int d = 0; d < D_total_; d++) {
      float A = -softplus_f(A_log_[d]);
      float ld = clamp_f(log_dt_[d], LOGDT_LO, LOGDT_HI);
      float dt = std::exp(ld);
      float decay = std::exp(A * dt);
      float bu = B_[d] * tmp_u_[d];
      float h_new = decay * h_[d] + (1.0f - decay) * bu;
      h_[d] = h_new;
      y += W_combine_[d] * h_new;
    }

    return y;
  }

  //per-scale hidden state energy
  void scale_norms(float* norms) const {
    for (int s = 0; s < n_scales_; s++) {
      float ss = 0.0f;
      for (int d = 0; d < D_per_; d++) {
        float v = h_[s * D_per_ + d];
        ss += v * v;
      }
      norms[s] = std::sqrt(ss);
    }
  }

private:
  int n_scales_, D_per_, in_dim_, D_total_;

  std::vector<float> h_;
  std::vector<float> A_log_, log_dt_, B_;
  std::vector<float> W_in_, b_in_;
  std::vector<float> W_combine_;
  float b_combine_;

  std::vector<float> tmp_x_, tmp_u_;

  void featurize(const TickFeatures& t, std::vector<float>& x) const {
    if ((int)x.size() != in_dim_) x.assign(in_dim_, 0.0f);
    int i = 0;
    x[i++] = t.bid_ask_spread;
    x[i++] = t.order_flow_imbalance;
    x[i++] = t.mid_price_change;
    for (int k = 0; k < 10; k++) x[i++] = t.book_depth[k];
  }
};
