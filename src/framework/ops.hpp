#pragma once
#include "tensor.hpp"

inline void assert_same_shape(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) throw std::runtime_error("shape mismatch");
  if (!a.is_contiguous() || !b.is_contiguous()) throw std::runtime_error("ops require contiguous tensors");
}

inline std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  assert_same_shape(*a, *b);

  vector_float out(a->numel());
  for (Size i = 0; i < out.size(); i++) out[i] = a->flat(i) + b->flat(i);

  bool req = track_grad(a, b);
  auto y = std::make_shared<Tensor>(std::move(out), a->shape, req);

  if (req) {
    y->node = std::make_shared<Node>();
    y->node->parents = {a, b};

    //d(a+b)/da = 1, d(a+b)/db = 1
    y->node->backward = [a, b](const vector_float& gout) {
      std::vector<vector_float> grads(2);
      grads[0] = a->req_grad ? gout : vector_float(a->numel(), 0.0f);
      grads[1] = b->req_grad ? gout : vector_float(b->numel(), 0.0f);
      return grads;
    };
  }
  return y;
}

inline std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  assert_same_shape(*a, *b);

  vector_float out(a->numel());
  for (Size i = 0; i < out.size(); i++) out[i] = a->flat(i) * b->flat(i);

  bool req = track_grad(a, b);
  auto y = std::make_shared<Tensor>(std::move(out), a->shape, req);

  if (req) {
    y->node = std::make_shared<Node>();
    y->node->parents = {a, b};

    //d(a*b)/da = b, d(a*b)/db = a
    y->node->backward = [a, b](const vector_float& gout) {
      std::vector<vector_float> grads(2);
      grads[0].assign(a->numel(), 0.0f);
      grads[1].assign(b->numel(), 0.0f);

      if (a->req_grad) {
        for (Size i = 0; i < a->numel(); i++) grads[0][i] = gout[i] * b->flat(i);
      }
      if (b->req_grad) {
        for (Size i = 0; i < b->numel(); i++) grads[1][i] = gout[i] * a->flat(i);
      }
      return grads;
    };
  }
  return y;
}

inline std::shared_ptr<Tensor> sum_all(const std::shared_ptr<Tensor>& x) {
  if (!x->is_contiguous()) throw std::runtime_error("sum_all requires contiguous tensor");

  float s = 0.0f;
  for (Size i = 0; i < x->numel(); i++) s += x->flat(i);

  bool req = track_grad(x);
  auto y = std::make_shared<Tensor>(vector_float{ s }, vector_int{ 1 }, req);

  if (req) {
    y->node = std::make_shared<Node>();
    y->node->parents = {x};

    //d/dx sum(x) = 1
    y->node->backward = [x](const vector_float& gout) {
      vector_float gx(x->numel(), 0.0f);
      if (x->req_grad) {
        for (Size i = 0; i < gx.size(); i++) gx[i] = gout[0];
      }
      return std::vector<vector_float>{ gx };
    };
  }
  return y;
}

inline std::shared_ptr<Tensor> matmul2d(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B) {
  if (!A->is_contiguous() || !B->is_contiguous()) throw std::runtime_error("matmul2d requires contiguous inputs");
  if (A->shape.size() != 2 || B->shape.size() != 2) throw std::runtime_error("matmul2d expects 2D tensors");

  int M = A->shape[0];
  int K = A->shape[1];
  int K2 = B->shape[0];
  int N = B->shape[1];
  if (K != K2) throw std::runtime_error("matmul2d shape mismatch: A is (M,K), B is (K,N)");

  vector_float out((Size)M * (Size)N, 0.0f);

  //forward matmul
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      float a_ik = A->flat((Size)i * K + k);
      for (int j = 0; j < N; j++) {
        out[(Size)i * N + j] += a_ik * B->flat((Size)k * N + j);
      }
    }
  }

  bool req = track_grad(A, B);
  auto Y = std::make_shared<Tensor>(std::move(out), vector_int{M, N}, req);

  if (req) {
    Y->node = std::make_shared<Node>();
    Y->node->parents = {A, B};

    Y->node->backward = [A, B, M, K, N](const vector_float& gY) {
      std::vector<vector_float> grads(2);

      grads[0].assign((Size)M * (Size)K, 0.0f);
      grads[1].assign((Size)K * (Size)N, 0.0f);

      if (A->req_grad) {
        for (int i = 0; i < M; i++) {
          for (int k = 0; k < K; k++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++) {
              acc += gY[(Size)i * N + j] * B->flat((Size)k * N + j);
            }
            grads[0][(Size)i * K + k] = acc;
          }
        }
      }

      if (B->req_grad) {
        for (int k = 0; k < K; k++) {
          for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int i = 0; i < M; i++) {
              acc += A->flat((Size)i * K + k) * gY[(Size)i * N + j];
            }
            grads[1][(Size)k * N + j] = acc;
          }
        }
      }

      return grads;
    };
  }

  return Y;
}
