#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "cuda/ssm_cuda.hpp"
#include "ssm/stream_ssm.hpp"

using Clock = std::chrono::high_resolution_clock;

template<typename F>
double bench(F&& f, int iters) {
  for (int i = 0; i < 10; i++) f();

  auto t0 = Clock::now();
  for (int i = 0; i < iters; i++) f();
  auto t1 = Clock::now();

  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

int main() {
  constexpr int D = 64;
  constexpr int D_in = 13;
  constexpr int T = 1000;
  constexpr int iters = 100;

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 0.1f);

  //init input
  std::vector<float> x(T * D_in);
  for (auto& v : x) v = dist(rng);

  //cpu model
  StreamingSSM cpu_ssm(D, D_in);
  for (auto& v : cpu_ssm.A_log()) v = dist(rng);
  for (auto& v : cpu_ssm.log_dt()) v = -5.0f + dist(rng);
  for (auto& v : cpu_ssm.B()) v = 1.0f + dist(rng);
  for (auto& v : cpu_ssm.W_in()) v = dist(rng);
  for (auto& v : cpu_ssm.b_in()) v = dist(rng);

  //gpu model
  ssm_cuda::StreamingSSMCuda gpu_ssm(D, D_in);
  gpu_ssm.A_log() = cpu_ssm.A_log();
  gpu_ssm.log_dt() = cpu_ssm.log_dt();
  gpu_ssm.B() = cpu_ssm.B();
  gpu_ssm.C() = cpu_ssm.W_out();
  gpu_ssm.W_in() = cpu_ssm.W_in();
  gpu_ssm.b_in() = cpu_ssm.b_in();
  gpu_ssm.upload_params();

  //outputs
  std::vector<float> y_cpu(T);
  std::vector<float> y_gpu(T);

  //bench cpu (tick by tick)
  TickFeatures tick;
  auto cpu_time = bench([&]() {
    cpu_ssm.reset_state();
    for (int t = 0; t < T; t++) {
      tick.bid_ask_spread = x[t * D_in + 0];
      tick.order_flow_imbalance = x[t * D_in + 1];
      tick.mid_price_change = x[t * D_in + 2];
      for (int i = 0; i < 10; i++) tick.book_depth[i] = x[t * D_in + 3 + i];
      y_cpu[t] = cpu_ssm.forward_step(tick);
    }
  }, iters);

  //bench gpu (batch)
  auto gpu_time = bench([&]() {
    gpu_ssm.reset_state();
    gpu_ssm.forward(x.data(), T, y_gpu.data());
  }, iters);

  float max_diff = 0.0f;
  for (int t = 0; t < T; t++) {
    float diff = std::abs(y_cpu[t] - y_gpu[t]);
    if (diff > max_diff) max_diff = diff;
  }

  std::cout << "T=" << T << " D=" << D << " D_in=" << D_in << "\n";
  std::cout << "cpu: " << cpu_time << " us/seq\n";
  std::cout << "gpu: " << gpu_time << " us/seq\n";
  std::cout << "speedup: " << cpu_time / gpu_time << "x\n";
  std::cout << "max diff: " << max_diff << "\n";

  return 0;
}
