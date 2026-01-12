#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <algorithm>
#include <utility>

using vector_float = std::vector<float>;
using vector_int = std::vector<int>;

struct Tensor; //stored contiguously

struct Node {
  std::vector<std::shared_ptr<Tensor>> parents;
  std::function<std::vector<vector_float>(const vector_float& grad_out)> backward;
};

struct Tensor : std::enable_shared_from_this<Tensor> {
  vector_float data;
  vector_int shape; //dimensions

  vector_float grad;
  bool req_grad = false; //boolean that decides if the tensor participates in autograd

  std::shared_ptr<Node> node;

  Tensor() = default;

  Tensor(vector_float d, vector_int s, bool req=false) //d: contig storage, s:shape, req:req_grad bool
    : data(std::move(d)), shape(std::move(s)), req_grad(req) {

    //gradients are made the same size as data
    if (req_grad) grad.assign(data.size(), 0.0f);
  }

  size_t numel() const { return data.size(); }

  static std::shared_ptr<Tensor> zeros(const vector_int& shape, bool req=false) {
    size_t n = 1;
    for (int x : shape) n *= static_cast<size_t>(x);
    return std::make_shared<Tensor>(vector_float(n, 0.0f), shape, req);
  }

  void zero_grad() {
    if (req_grad) std::fill(grad.begin(), grad.end(), 0.0f);
  }

  void backward_scalar(float grad_seed = 1.0f) {
    //can't backprop if tensor doesnt track gradients
    if (!req_grad) throw std::runtime_error("backward called on req_grad=false");

    if (numel() != 1) throw std::runtime_error("backward_scalar expects scalar output");

    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;

    //visit all parents recursively
    std::function<void(std::shared_ptr<Tensor>)> dfs =
      [&](std::shared_ptr<Tensor> t) {
        if (!t) return;
        if (visited.count(t.get())) return;

        visited.insert(t.get());

        //if op made this tensor, it has node + parents
        if (t->node) {
          for (auto& p : t->node->parents) dfs(p);
        }

        topo.push_back(t);
      };

    dfs(shared_from_this());
    grad[0] += grad_seed;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      auto& v = *it;
      if (!v->node) continue; //leaf tensor

      auto parent_grads = v->node->backward(v->grad);

      if (parent_grads.size() != v->node->parents.size())
        throw std::runtime_error("backward returned wrong number of parent grads");

      //include in each parent's gradient buffer
      for (size_t i = 0; i < v->node->parents.size(); i++) {
        auto& p = v->node->parents[i];

        if (!p->req_grad) continue;

        auto& g = parent_grads[i];

        if (g.size() != p->numel())
          throw std::runtime_error("parent grad size mismatch");

        //gradient accumulation
        for (size_t k = 0; k < g.size(); k++) {
          p->grad[k] += g[k];
        }
      }
    }
  }
};

inline size_t prod(const vector_int& shape) {
  size_t n = 1;
  for (int x : shape) n *= static_cast<size_t>(x);
  return n;
}

inline void assert_same_shape(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape)
    throw std::runtime_error("shape mismatch");
}