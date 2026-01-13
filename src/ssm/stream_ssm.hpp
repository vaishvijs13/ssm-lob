#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

//exx:
struct TickFeatures {
  float bid_ask_spread;
  float order_flow_imbalance;
  float mid_price_change;
  float book_depth[10];
};

//NO autograd here
inline float softplus_f(float x, float threshold = 20.0f) {
  if (x > threshold) return x;
  return std::log1p(std::exp(x));
}

inline float sigmoid_f(float x) {
  if (x >= 0.0f) {
    float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  } else {
    float z = std::exp(x);
    return z / (1.0f + z);
  }
}

inline float clamp_f(float x, float lo, float hi) {
  return std::max(lo, std::min(hi, x));
}

class StreamingSSM {
public:
  explicit StreamingSSM(int D, int in_dim)
    : D_(D), in_dim_(in_dim),
      h_(D, 0.0f),

      //pararm
      A_log_(D, 0.0f),
      log_dt_(D, -5.0f),
      B_(D, 1.0f),

      //input projection
      W_in_(D * in_dim, 0.0f),
      b_in_(D, 0.0f),

      //output head
      W_out_(D, 0.0f),
      b_out_(0.0f)
  {}

  //reset hidden state, aka start of day
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

  //convert tickfeatures struct to input dim
  void featurize(const TickFeatures& t, std::vector<float>& x) const {
    if ((int)x.size() != in_dim_) x.assign(in_dim_, 0.0f);

    int idx = 0;
    x[idx++] = t.bid_ask_spread;
    x[idx++] = t.order_flow_imbalance;
    x[idx++] = t.mid_price_change;
    for (int i = 0; i < 10; i++) x[idx++] = t.book_depth[i];
  }

  //O(1) per tick update, returns scalar prediction
  float forward_step(const TickFeatures& tick) {
    tmp_x_.resize(in_dim_);
    featurize(tick, tmp_x_);
    tmp_u_.assign(D_, 0.0f);
    for (int d = 0; d < D_; d++) {
      float acc = b_in_[d];
      const int row = d * in_dim_;
      for (int j = 0; j < in_dim_; j++) {
        acc += W_in_[row + j] * tmp_x_[j];
      }
      tmp_u_[d] = acc;
    }

    //fused discretization, state update, and output accumulation
    float y = b_out_;

    //clamp range
    constexpr float LOGDT_LO = -20.0f;
    constexpr float LOGDT_HI =  5.0f;

    for (int d = 0; d < D_; d++) {
      float A = -softplus_f(A_log_[d]);
      float ld = clamp_f(log_dt_[d], LOGDT_LO, LOGDT_HI);
      float dt = std::exp(ld);
      float decay = std::exp(A * dt);
      float bu = B_[d] * tmp_u_[d];
      float h_new = decay * h_[d] + (1.0f - decay) * bu;
      h_[d] = h_new;

      y += W_out_[d] * h_new; //out
    }

    return y;
  }

  const std::vector<float>& hidden_state() const { return h_; }

private:
  int D_;
  int in_dim_;

  //persistent hidden state
  std::vector<float> h_;

  //param
  std::vector<float> A_log_;
  std::vector<float> log_dt_;
  std::vector<float> B_;

  std::vector<float> W_in_;
  std::vector<float> b_in_;

  std::vector<float> W_out_;
  float b_out_;

  //scratch buffers
  std::vector<float> tmp_x_;
  std::vector<float> tmp_u_;
};
