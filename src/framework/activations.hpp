#pragma once
#include "tensor.hpp"

inline std::shared_ptr<Tensor> unary_op_contig(
  const std::shared_ptr<Tensor>& x,
  const char*,
  std::function<float(float)> fwd,
  std::function<float(float, float)> df_dx //deriv
) {
  if (!x->is_contiguous()) throw std::runtime_error("unary op requires contig tensor");

  vector_float out(x->numel());
  for (Size i = 0; i < out.size(); i++) {
    out[i] = fwd(x->flat(i));
  }

  bool req = track_grad(x);
  auto y = std::make_shared<Tensor>(std::move(out), x->shape, req);

  if (req) {
    y->node = std::make_shared<Node>();
    y->node->parents = {x};

    //backward
    y->node->backward = [x, y, df_dx](const vector_float& gout) {
      vector_float gx(x->numel(), 0.0f);
      if (x->req_grad) {
        for (Size i = 0; i < gx.size(); i++) {
          float xv = x->flat(i);
          float yv = y->flat(i);
          gx[i] = gout[i] * df_dx(xv, yv);
        }
      }
      return std::vector<vector_float>{ gx };
    };
  }

  return y;
}

//stable sigmoid
inline std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& x) {
  auto fwd = [](float v) -> float {
    if (v >= 0.0f) {
      float z = std::exp(-v);
      return 1.0f / (1.0f + z);
    } else {
      float z = std::exp(v);
      return z / (1.0f + z);
    }
  };
  auto df = [](float, float y) -> float {
    //deriv
    return y * (1.0f - y);
  };
  return unary_op_contig(x, "sigmoid", fwd, df);
}

//softplus
inline std::shared_ptr<Tensor> softplus(
  const std::shared_ptr<Tensor>& x,
  float threshold = 20.0f
) {
  auto fwd = [threshold](float v) -> float {
    if (v > threshold) return v;
    return std::log1p(std::exp(v));
  };
  auto df = [](float v, float) -> float {
    //sigmoid same thing
    if (v >= 0.0f) {
      float z = std::exp(-v);
      return 1.0f / (1.0f + z);
    } else {
      float z = std::exp(v);
      return z / (1.0f + z);
    }
  };
  return unary_op_contig(x, "softplus", fwd, df);
}

inline std::shared_ptr<Tensor> clamp(const std::shared_ptr<Tensor>& x, float lo, float hi) {
  auto fwd = [lo, hi](float v) -> float {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
  };
  auto df = [lo, hi](float v, float) -> float {
    return (v >= lo && v <= hi) ? 1.0f : 0.0f;
  };
  return unary_op_contig(x, "clamp", fwd, df);
}

//w/ input clamp to prevent overflow
inline std::shared_ptr<Tensor> exp_clamp(const std::shared_ptr<Tensor>& x, float lo, float hi) {
  auto fwd = [lo, hi](float v) -> float {
    float c = v;
    if (c < lo) c = lo;
    if (c > hi) c = hi;
    return std::exp(c);
  };
  auto df = [lo, hi](float v, float y) -> float {
    //if x was clamped, deriv "is 0" for clamp nonlinearity
    if (v < lo || v > hi) return 0.0f;
    return y; //derivative of exp is exp
  };
  return unary_op_contig(x, "exp_clamp", fwd, df);
}
