#pragma once
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <functional>

struct BenchStats {
  double mean_us = 0.0;
  double p50_us = 0.0;
  double p90_us = 0.0;
  double p99_us = 0.0;
  double ops_per_sec = 0.0;
};

inline double now_ns() {
  using clock = std::chrono::steady_clock;
  auto t = clock::now().time_since_epoch();
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

inline BenchStats bench_microseconds(
  const char* name,
  int warmup_iters,
  int iters,
  const std::function<void()>& fn
) {
  for (int i = 0; i < warmup_iters; i++) fn();

  std::vector<double> times_us;
  times_us.reserve((size_t)iters);

  double t0 = now_ns();
  for (int i = 0; i < iters; i++) {
    double s = now_ns();
    fn();
    double e = now_ns();
    times_us.push_back((e - s) / 1000.0);
  }
  double t1 = now_ns();

  //stats
  double sum = 0.0;
  for (double x : times_us) sum += x;

  std::sort(times_us.begin(), times_us.end());

  auto pct = [&](double p) -> double {
    //p in [0,1]
    if (times_us.empty()) return 0.0;
    double idx = p * (times_us.size() - 1);
    size_t i = (size_t)idx;
    return times_us[i];
  };

  BenchStats st;
  st.mean_us = sum / (double)times_us.size();
  st.p50_us = pct(0.50);
  st.p90_us = pct(0.90);
  st.p99_us = pct(0.99);

  double total_s = (t1 - t0) / 1e9;
  st.ops_per_sec = (total_s > 0.0) ? ((double)iters / total_s) : 0.0;

  std::cout << "[bench] " << name
            << "  mean=" << st.mean_us << "us"
            << "  p50=" << st.p50_us << "us"
            << "  p90=" << st.p90_us << "us"
            << "  p99=" << st.p99_us << "us"
            << "  ops/s=" << st.ops_per_sec
            << "\n";

  return st;
}


inline void print_bytes_estimate_per_step(int D, int D_in, bool store_h_all) {
  const int bytes_per_float = 4;

  //input row
  long long read_x = (long long)D_in * bytes_per_float;

  //params
  long long read_params = (long long)(5LL * D) * bytes_per_float;

  long long read_W = (long long)D * D_in * bytes_per_float;

  long long rw_h = (long long)(2LL * D) * bytes_per_float;

  //output write
  long long write_y = 1LL * bytes_per_float;

  long long write_h_all = store_h_all ? ((long long)D * bytes_per_float) : 0LL;

  long long total = read_x + read_params + read_W + rw_h + write_y + write_h_all;

  std::cout << "[bytes/tick] approx=" << total
            << " bytes  (D=" << D << ", D_in=" << D_in
            << ", store_h_all=" << (store_h_all ? "yes" : "no") << ")\n";
}
