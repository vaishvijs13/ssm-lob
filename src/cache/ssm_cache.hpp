#pragma once
#include <vector>
#include <cmath>
#include "../framework/math_utils.hpp"

class SSMKVCache {
public:
  explicit SSMKVCache(int n_heads, int d_head, int d_state = 64, int n_recent = 128)
    : nh_(n_heads), dh_(d_head), ds_(d_state), nr_(n_recent), seq_(0),
      state_k_(n_heads * d_state * d_head, 0.0f),
      state_v_(n_heads * d_state * d_head, 0.0f),
      recent_k_(n_heads * n_recent * d_head, 0.0f),
      recent_v_(n_heads * n_recent * d_head, 0.0f),
      rptr_(0), rlen_(0),
      A_log_(n_heads * d_state, 0.0f),
      W_B_(n_heads * d_state * d_head, 0.0f),
      W_dt_(n_heads * d_state, 0.1f),
      b_dt_(n_heads * d_state, -4.0f)
  {
    init_params();
  }

  void reset() {
    std::fill(state_k_.begin(), state_k_.end(), 0.0f);
    std::fill(state_v_.begin(), state_v_.end(), 0.0f);
    std::fill(recent_k_.begin(), recent_k_.end(), 0.0f);
    std::fill(recent_v_.begin(), recent_v_.end(), 0.0f);
    rptr_ = rlen_ = seq_ = 0;
  }

  //append kv pair, compresses oldest if buffer full
  void append(const std::vector<float>& k, const std::vector<float>& v) {
    if (rlen_ >= nr_) compress_oldest();

    int wpos = (rptr_ + rlen_) % nr_;
    for (int h = 0; h < nh_; h++) {
      int roff = h * nr_ * dh_ + wpos * dh_;
      int koff = h * dh_;
      for (int d = 0; d < dh_; d++) {
        recent_k_[roff + d] = k[koff + d];
        recent_v_[roff + d] = v[koff + d];
      }
    }
    if (rlen_ < nr_) rlen_++;
    seq_++;
  }

  //attend: combines compressed state + recent exact
  float attend(const std::vector<float>& q, std::vector<float>& out) {
    out.assign(nh_ * dh_, 0.0f);
    float scale = 1.0f / std::sqrt((float)dh_);
    float max_s = -1e9f;

    for (int h = 0; h < nh_; h++) {
      std::vector<float> scores(rlen_ + ds_, 0.0f);
      float local_max = -1e9f;

      //recent tokens
      for (int t = 0; t < rlen_; t++) {
        int pos = (rptr_ + t) % nr_;
        int koff = h * nr_ * dh_ + pos * dh_;
        int qoff = h * dh_;
        float s = 0.0f;
        for (int d = 0; d < dh_; d++)
          s += q[qoff + d] * recent_k_[koff + d];
        scores[t] = s * scale;
        local_max = std::max(local_max, scores[t]);
      }

      //compressed state
      for (int i = 0; i < ds_; i++) {
        int soff = h * ds_ * dh_ + i * dh_;
        int qoff = h * dh_;
        float s = 0.0f;
        for (int d = 0; d < dh_; d++)
          s += q[qoff + d] * state_k_[soff + d];
        scores[rlen_ + i] = s * scale * 0.5f; //downweight compressed
        local_max = std::max(local_max, scores[rlen_ + i]);
      }

      max_s = std::max(max_s, local_max);

      //softmax + weighted sum
      float sum_exp = 0.0f;
      for (int i = 0; i < rlen_ + ds_; i++)
        sum_exp += std::exp(scores[i] - local_max);

      int ooff = h * dh_;

      for (int t = 0; t < rlen_; t++) {
        float w = std::exp(scores[t] - local_max) / (sum_exp + 1e-9f);
        int pos = (rptr_ + t) % nr_;
        int voff = h * nr_ * dh_ + pos * dh_;
        for (int d = 0; d < dh_; d++)
          out[ooff + d] += w * recent_v_[voff + d];
      }

      for (int i = 0; i < ds_; i++) {
        float w = std::exp(scores[rlen_ + i] - local_max) / (sum_exp + 1e-9f);
        int soff = h * ds_ * dh_ + i * dh_;
        for (int d = 0; d < dh_; d++)
          out[ooff + d] += w * state_v_[soff + d];
      }
    }
    return max_s;
  }

  size_t memory_bytes() const {
    return 2 * nh_ * (ds_ + nr_) * dh_ * sizeof(float);
  }

  size_t baseline_bytes() const {
    return 2 * nh_ * seq_ * dh_ * sizeof(float);
  }

  float compression_ratio() const {
    if (seq_ <= nr_) return 1.0f;
    return (float)baseline_bytes() / (float)memory_bytes();
  }

  int seq_len() const { return seq_; }

private:
  int nh_, dh_, ds_, nr_, seq_;

  std::vector<float> state_k_, state_v_;
  std::vector<float> recent_k_, recent_v_;
  int rptr_, rlen_;

  std::vector<float> A_log_, W_B_, W_dt_, b_dt_;

  void init_params() {
    float sb = 1.0f / std::sqrt((float)dh_);
    for (int i = 0; i < nh_ * ds_; i++)
      A_log_[i] = 0.5f * ((float)(i % ds_) / ds_ - 0.5f);
    for (size_t i = 0; i < W_B_.size(); i++)
      W_B_[i] = sb * ((float)(i % 1000) / 500.0f - 1.0f);
  }

  void compress_oldest() {
    for (int h = 0; h < nh_; h++) {
      int old_off = h * nr_ * dh_ + rptr_ * dh_;

      for (int s = 0; s < ds_; s++) {
        //input projection
        float inp_k = 0.0f, inp_v = 0.0f;
        int wb_off = h * ds_ * dh_ + s * dh_;
        for (int d = 0; d < dh_; d++) {
          inp_k += W_B_[wb_off + d] * recent_k_[old_off + d];
          inp_v += W_B_[wb_off + d] * recent_v_[old_off + d];
        }

        //discretization
        int dt_idx = h * ds_ + s;
        float dt = softplus_f(W_dt_[dt_idx] + b_dt_[dt_idx]);
        dt = clamp_f(dt, 1e-4f, 1.0f);

        float A = -softplus_f(A_log_[dt_idx]);
        float decay = std::exp(A * dt);

        //state update
        int st_off = h * ds_ * dh_ + s * dh_;
        for (int d = 0; d < dh_; d++) {
          state_k_[st_off + d] = decay * state_k_[st_off + d] + (1.0f - decay) * inp_k * recent_k_[old_off + d];
          state_v_[st_off + d] = decay * state_v_[st_off + d] + (1.0f - decay) * inp_v * recent_v_[old_off + d];
        }
      }
    }
    rptr_ = (rptr_ + 1) % nr_;
  }
};
