#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "cuda/ssm_cuda.hpp"
#include "ssm/stream_ssm.hpp"
#include "ssm/benchmark.hpp"

int main() {
  constexpr int D    = 32;  //variant 2 requires exactly one warp
  constexpr int D_in = 13;
  constexpr int T    = 1000;
  constexpr int warmup = 100;
  constexpr int iters  = 1000;

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 0.1f);

  std::vector<float> x(T * D_in);
  for (auto& v : x) v = dist(rng);

  //cpu reference
  StreamingSSM cpu_ssm(D, D_in);
  for (auto& v : cpu_ssm.A_log())  v = dist(rng);
  for (auto& v : cpu_ssm.log_dt()) v = -5.0f + dist(rng);
  for (auto& v : cpu_ssm.B())      v = 1.0f + dist(rng);
  for (auto& v : cpu_ssm.W_in())   v = dist(rng);
  for (auto& v : cpu_ssm.b_in())   v = dist(rng);

  //gpu model, same params
  ssm_cuda::StreamingSSMCuda gpu_ssm(D, D_in);
  gpu_ssm.A_log()  = cpu_ssm.A_log();
  gpu_ssm.log_dt() = cpu_ssm.log_dt();
  gpu_ssm.B()      = cpu_ssm.B();
  gpu_ssm.C()      = cpu_ssm.W_out();
  gpu_ssm.W_in()   = cpu_ssm.W_in();
  gpu_ssm.b_in()   = cpu_ssm.b_in();
  gpu_ssm.upload_params();

  std::vector<float> y_cpu(T), y_gpu(T);

  //cpu reference output for correctness checks
  TickFeatures tick;
  cpu_ssm.reset_state();
  for (int t = 0; t < T; t++) {
    tick.bid_ask_spread      = x[t * D_in + 0];
    tick.order_flow_imbalance = x[t * D_in + 1];
    tick.mid_price_change    = x[t * D_in + 2];
    for (int i = 0; i < 10; i++) tick.book_depth[i] = x[t * D_in + 3 + i];
    y_cpu[t] = cpu_ssm.forward_step(tick);
  }

  auto check_diff = [&](const char* name) {
    float mx = 0.0f;
    for (int t = 0; t < T; t++) {
      float d = std::fabs(y_cpu[t] - y_gpu[t]);
      if (d > mx) mx = d;
    }
    std::cout << "  [correctness] " << name << " max|delta|=" << mx
              << (mx < 1e-4f ? "  ok\n" : "  FAIL\n");
  };

  std::cout << "T=" << T << " D=" << D << " D_in=" << D_in
            << " warmup=" << warmup << " iters=" << iters << "\n\n";

  //variant 0: baseline
  gpu_ssm.reset_state();
  gpu_ssm.forward(x.data(), T, y_gpu.data(), 0);
  check_diff("baseline");
  bench_microseconds("gpu_baseline", warmup, iters, [&]() {
    gpu_ssm.reset_state();
    gpu_ssm.forward(x.data(), T, y_gpu.data(), 0);
  });
  std::cout << "  per-step: ~" << "see p50/T above" << "\n\n";

  //variant 1: +shared W_in
  gpu_ssm.reset_state();
  gpu_ssm.forward(x.data(), T, y_gpu.data(), 1);
  check_diff("shared_w");
  bench_microseconds("gpu_shared_w", warmup, iters, [&]() {
    gpu_ssm.reset_state();
    gpu_ssm.forward(x.data(), T, y_gpu.data(), 1);
  });
  std::cout << "\n";

  //variant 2: +shared W_in +warp shuffle (D==32 only)
  gpu_ssm.reset_state();
  gpu_ssm.forward(x.data(), T, y_gpu.data(), 2);
  check_diff("shared_w+warp_reduce");
  bench_microseconds("gpu_opt", warmup, iters, [&]() {
    gpu_ssm.reset_state();
    gpu_ssm.forward(x.data(), T, y_gpu.data(), 2);
  });
  std::cout << "\n";

  //cpu reference timing for context
  bench_microseconds("cpu_basic", warmup, iters, [&]() {
    cpu_ssm.reset_state();
    for (int t = 0; t < T; t++) {
      tick.bid_ask_spread       = x[t * D_in + 0];
      tick.order_flow_imbalance = x[t * D_in + 1];
      tick.mid_price_change     = x[t * D_in + 2];
      for (int i = 0; i < 10; i++) tick.book_depth[i] = x[t * D_in + 3 + i];
      y_cpu[t] = cpu_ssm.forward_step(tick);
    }
  });
  std::cout << "(cpu numbers include T=" << T << " sequential steps; gpu numbers include H2D+D2H)\n";

  return 0;
}
