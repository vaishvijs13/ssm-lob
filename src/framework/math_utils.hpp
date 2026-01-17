#pragma once
#include <cmath>

//softplus branch stable
inline float softplus_f(float x, float threshold = 20.0f) {
  if (x > threshold) return x;
  return std::log1p(std::exp(x));
}

//sigmoid
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
  return (x < lo) ? lo : (x > hi ? hi : x);
}
