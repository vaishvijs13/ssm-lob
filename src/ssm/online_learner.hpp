#pragma once
#include <vector>
#include <cmath>
#include "../framework/math_utils.hpp"

class OnlineSGD {
public:
  OnlineSGD(int D, float lr = 0.01f, float momentum = 0.9f)
    : D_(D), lr_(lr), mom_(momentum),
      v_w_(D, 0.0f),
      v_b_(0.0f),
      ema_err_(0.0f)
  {}

  //update W_out, b_out given hidden state + prediction error
  void step(std::vector<float>& W_out, float& b_out,
            const std::vector<float>& h, float error) {
    for (int d = 0; d < D_; d++) {
      v_w_[d] = mom_ * v_w_[d] + (1.0f - mom_) * error * h[d];
      W_out[d] -= lr_ * v_w_[d];
    }
    v_b_ = mom_ * v_b_ + (1.0f - mom_) * error;
    b_out -= lr_ * v_b_;

    ema_err_ = 0.99f * ema_err_ + 0.01f * std::abs(error);
  }

  float ema_error() const { return ema_err_; }
  float learning_rate() const { return lr_; }
  void set_lr(float lr) { lr_ = lr; }

private:
  int D_;
  float lr_, mom_;
  std::vector<float> v_w_;
  float v_b_;
  float ema_err_;
};
