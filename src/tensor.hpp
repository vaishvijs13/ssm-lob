#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <algorithm>
#include <utility>
#include <cstddef>

using vector_float = std::vector<float>;
using vector_int = std::vector<int>;
using Size = std::size_t;

struct Tensor;

struct Node {
  std::vector<std::shared_ptr<Tensor>> parents;
  std::function<std::vector<vector_float>(const vector_float& grad_out)> backward;
};

inline bool& grad_enabled() {
  static bool enabled = true;
  return enabled;
}

struct NoGradGuard {
  bool old;
  NoGradGuard() : old(grad_enabled()) { grad_enabled() = false; }
  ~NoGradGuard() { grad_enabled() = old; }
};

inline Size prod(const vector_int& shape) {
  Size n = 1;
  for (int x : shape) n *= static_cast<Size>(x);
  return n;
}

inline vector_int contiguous_strides(const vector_int& shape) {
  vector_int s(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i) {
    s[i] = s[i + 1] * shape[i + 1];
  }
  return s;
}

struct Tensor : std::enable_shared_from_this<Tensor> {
  //shared contig storage
  std::shared_ptr<vector_float> storage;

  //offset into storage
  Size offset = 0;

  vector_int shape;
  vector_int strides;

  //gradient buffer
  vector_float grad;

  bool req_grad = false;
  std::shared_ptr<Node> node;

  Tensor() = default;

  //contiguous tensor
  Tensor(vector_float data, vector_int shp, bool req=false)
    : storage(std::make_shared<vector_float>(std::move(data))),
      offset(0),
      shape(std::move(shp)),
      strides(contiguous_strides(shape)),
      req_grad(req) {

    if (storage->size() != prod(shape)) {
      throw std::runtime_error("tensor ctor: data.size() != prod(shape)");
    }

    if (req_grad) grad.assign(numel(), 0.0f);
  }

  //view tensor
  Tensor(std::shared_ptr<vector_float> stor, Size off, vector_int shp, vector_int str, bool req=false)
    : storage(std::move(stor)),
      offset(off),
      shape(std::move(shp)),
      strides(std::move(str)),
      req_grad(req) {

    if (req_grad) grad.assign(numel(), 0.0f);
  }

  Size numel() const { return prod(shape); }

  //row-major contiguous or not
  bool is_contiguous() const {
    return strides == contiguous_strides(shape);
  }

  //flat indexing
  float& flat(Size i) {
    if (!is_contiguous()) throw std::runtime_error("flat() requires contiguous tensor");
    return (*storage)[offset + i];
  }

  const float& flat(Size i) const {
    if (!is_contiguous()) throw std::runtime_error("flat() requires contiguous tensor");
    return (*storage)[offset + i];
  }

  //zeros tensor
  static std::shared_ptr<Tensor> zeros(const vector_int& shape, bool req=false) {
    return std::make_shared<Tensor>(vector_float(prod(shape), 0.0f), shape, req);
  }

  //zero grads buffer
  void zero_grad() {
    if (req_grad) std::fill(grad.begin(), grad.end(), 0.0f);
  }

  //numels must match and must be contiguous
  std::shared_ptr<Tensor> reshape_view(const vector_int& new_shape) {
    if (!is_contiguous()) throw std::runtime_error("reshape_view requires contiguous tensor");
    if (prod(new_shape) != numel()) throw std::runtime_error("reshape_view numel mismatch");
    return std::make_shared<Tensor>(storage, offset, new_shape, contiguous_strides(new_shape), req_grad);
  }

  //backprop from a scalar output
  void backward_scalar(float grad_seed = 1.0f) {
    if (!req_grad) throw std::runtime_error("backward called on req_grad=false");
    if (numel() != 1) throw std::runtime_error("backward_scalar expects scalar output");

    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;

    std::function<void(std::shared_ptr<Tensor>)> dfs =
      [&](std::shared_ptr<Tensor> t) {
        if (!t) return;
        if (visited.count(t.get())) return;
        visited.insert(t.get());

        if (t->node) {
          for (auto& p : t->node->parents) dfs(p);
        }
        topo.push_back(t);
      };

    dfs(shared_from_this());

    //seed grad for the scalar output
    grad[0] += grad_seed;

    //reverse pass
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      auto& v = *it;
      if (!v->node) continue; //leaf tensor

      auto parent_grads = v->node->backward(v->grad);
      if (parent_grads.size() != v->node->parents.size())
        throw std::runtime_error("backward returned wrong number of parent grads");

      for (Size i = 0; i < v->node->parents.size(); i++) {
        auto& p = v->node->parents[i];
        if (!p->req_grad) continue;

        auto& g = parent_grads[i];
        if (g.size() != p->numel())
          throw std::runtime_error("parent grad size mismatch");

        for (Size k = 0; k < g.size(); k++) {
          p->grad[k] += g[k];
        }
      }
    }
  }
};

inline bool track_grad(const std::shared_ptr<Tensor>& a) {
  return grad_enabled() && a->req_grad;
}
inline bool track_grad(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  return grad_enabled() && (a->req_grad || b->req_grad);
}