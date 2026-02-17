#pragma once
#include <vector>
#include <cmath>
#include "../framework/math_utils.hpp"
#include "types.hpp"

class SelectiveStreamSSM {
public:
  explicit SelectiveStreamSSM(int D, int in_dim, int conv_k = 4)
    : D_(D), in_dim_(in_dim), conv_k_(conv_k),
      h_(D, 0.0f),
      A_log_(D, 0.0f),

      W_proj_(5 * D * in_dim, 0.0f),
      b_proj_(5 * D, 0.0f),

      W_out_(D, 0.0f),
      b_out_(0.0f),

      //causal conv circular buffer
      conv_buf_(conv_k > 1 ? conv_k * in_dim : 0, 0.0f),
      conv_w_(conv_k > 1 ? conv_k : 0, 0.0f),
      conv_ptr_(0),

      //volatility tracker
      vol_ema_(1e-4f),
      vol_alpha_(0.05f),
      vol_baseline_(0.01f),
      vol_sens_(2.0f),

      proj_(5 * D, 0.0f),
      tmp_x_(in_dim, 0.0f)
  {
    if (conv_k_ > 1) {
      for (int k = 0; k < conv_k_; k++)
        conv_w_[k] = 1.0f / (float)conv_k_;
    }
  }

  void reset_state() {
    std::fill(h_.begin(), h_.end(), 0.0f);
    if (conv_k_ > 1) {
      std::fill(conv_buf_.begin(), conv_buf_.end(), 0.0f);
      conv_ptr_ = 0;
    }
    vol_ema_ = 1e-4f;
  }

  std::vector<float>& A_log() { return A_log_; }
  std::vector<float>& W_proj() { return W_proj_; }
  std::vector<float>& b_proj() { return b_proj_; }
  std::vector<float>& W_out() { return W_out_; }
  float& b_out() { return b_out_; }
  std::vector<float>& conv_w() { return conv_w_; }
  float& vol_alpha() { return vol_alpha_; }
  float& vol_sens() { return vol_sens_; }
  int D() const { return D_; }
  int in_dim() const { return in_dim_; }

  const std::vector<float>& hidden_state() const { return h_; }
  float vol_estimate() const { return std::sqrt(vol_ema_); }

  float forward_step(const TickFeatures& tick) {
    featurize(tick, tmp_x_);

    //update vol ema
    float pc = tick.mid_price_change;
    vol_ema_ = vol_alpha_ * (pc * pc) + (1.0f - vol_alpha_) * vol_ema_;

    if (conv_k_ > 1) {
      for (int j = 0; j < in_dim_; j++)
        conv_buf_[conv_ptr_ * in_dim_ + j] = tmp_x_[j];

      for (int j = 0; j < in_dim_; j++) {
        float acc = 0.0f;
        for (int k = 0; k < conv_k_; k++) {
          int idx = ((conv_ptr_ - k + conv_k_) % conv_k_) * in_dim_ + j;
          acc += conv_w_[k] * conv_buf_[idx];
        }
        tmp_x_[j] = acc;
      }
      conv_ptr_ = (conv_ptr_ + 1) % conv_k_;
    }

    //fused 5-head projection: W_proj * x + b_proj → [5*D]
    const int pd = 5 * D_;
    for (int i = 0; i < pd; i++) {
      float acc = b_proj_[i];
      const int row = i * in_dim_;
      for (int j = 0; j < in_dim_; j++)
        acc += W_proj_[row + j] * tmp_x_[j];
      proj_[i] = acc;
    }

    float vol_scale = 1.0f + vol_sens_ * (std::sqrt(vol_ema_) - vol_baseline_);
    vol_scale = clamp_f(vol_scale, 0.5f, 2.0f);

    float y = b_out_;

    for (int d = 0; d < D_; d++) {
      float u = proj_[d];
      float B_t = proj_[D_ + d];
      float C_t = proj_[2 * D_ + d];
      float dt_raw = proj_[3 * D_ + d];
      float gate_r = proj_[4 * D_ + d];

      float A = -softplus_f(A_log_[d]);
      float dt = softplus_f(dt_raw) * vol_scale;
      float decay = std::exp(A * clamp_f(dt, 1e-9f, 10.0f));

      //selective state update
      float h_new = decay * h_[d] + (1.0f - decay) * B_t * u;
      h_[d] = h_new;

      //gated output
      float gate = silu_f(gate_r);
      y += W_out_[d] * gate * C_t * h_new;
    }

    return y;
  }

private:
  int D_, in_dim_, conv_k_;

  std::vector<float> h_;
  std::vector<float> A_log_;
  std::vector<float> W_proj_, b_proj_;
  std::vector<float> W_out_;
  float b_out_;

  std::vector<float> conv_buf_, conv_w_;
  int conv_ptr_;

  float vol_ema_, vol_alpha_, vol_baseline_, vol_sens_;

  std::vector<float> proj_, tmp_x_;

  void featurize(const TickFeatures& t, std::vector<float>& x) const {
    if ((int)x.size() != in_dim_) x.assign(in_dim_, 0.0f);
    int i = 0;
    x[i++] = t.bid_ask_spread;
    x[i++] = t.order_flow_imbalance;
    x[i++] = t.mid_price_change;
    for (int k = 0; k < 10; k++) x[i++] = t.book_depth[k];
  }
};
