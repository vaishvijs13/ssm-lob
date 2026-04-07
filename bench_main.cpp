#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <random>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#include "src/ssm/stream_ssm.hpp"
#include "src/ssm/stream_ssm_unfused.hpp"
#include "src/ssm/selective_ssm.hpp"
#include "src/ssm/multi_scale.hpp"
#include "src/ssm/benchmark.hpp"

struct benchconfig {
  const char* kernel = "selective";
  int d_model = 32;
  int d_in = 13;
  int conv_k = 4;
  int seq_len = 10000;
  int warmup = 1000;
  int steps = 10000;
  int seed = 42;
  const char* output = "bench_results.jsonl";
};

void print_usage() {
  std::cout << "usage: ./bench [options]\n"
            << "  --kernel <basic|unfused|selective|multiscale|proj_only>  (default: selective)\n"
            << "  --d_model <int>                             (default: 32)\n"
            << "  --d_in <int>                                (default: 13)\n"
            << "  --conv_k <int>                              (default: 4)\n"
            << "  --seq_len <int>                             (default: 10000)\n"
            << "  --warmup <int>                              (default: 1000)\n"
            << "  --steps <int>                               (default: 10000)\n"
            << "  --seed <int>                                (default: 42)\n"
            << "  --output <file>                             (default: bench_results.jsonl)\n";
}

bool parse_args(int argc, char** argv, benchconfig& cfg) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_usage();
      return false;
    }
    if (i + 1 >= argc) {
      std::cerr << "error: " << argv[i] << " requires argument\n";
      return false;
    }
    
    if (strcmp(argv[i], "--kernel") == 0) cfg.kernel = argv[++i];
    else if (strcmp(argv[i], "--d_model") == 0) cfg.d_model = atoi(argv[++i]);
    else if (strcmp(argv[i], "--d_in") == 0) cfg.d_in = atoi(argv[++i]);
    else if (strcmp(argv[i], "--conv_k") == 0) cfg.conv_k = atoi(argv[++i]);
    else if (strcmp(argv[i], "--seq_len") == 0) cfg.seq_len = atoi(argv[++i]);
    else if (strcmp(argv[i], "--warmup") == 0) cfg.warmup = atoi(argv[++i]);
    else if (strcmp(argv[i], "--steps") == 0) cfg.steps = atoi(argv[++i]);
    else if (strcmp(argv[i], "--seed") == 0) cfg.seed = atoi(argv[++i]);
    else if (strcmp(argv[i], "--output") == 0) cfg.output = argv[++i];
    else {
      std::cerr << "error: unknown option " << argv[i] << "\n";
      return false;
    }
  }
  return true;
}

//simple lob sim for bench
class lobsim {
public:
  lobsim(unsigned seed) : gen_(seed), norm_(0.0f, 0.01f), unif_(-1.0f, 1.0f),
    mid_(100.0f), vol_(0.01f), imbal_(0.0f) {}
  
  TickFeatures next() {
    imbal_ = 0.7f * imbal_ + 0.3f * unif_(gen_);
    float signal = 0.02f * imbal_;
    float noise = vol_ * norm_(gen_);
    float change = signal + noise - 0.0001f * (mid_ - 100.0f);
    mid_ += change;
    vol_ = 0.95f * vol_ + 0.05f * std::abs(change) + 0.001f;
    
    TickFeatures t;
    t.mid_price_change = change;
    t.bid_ask_spread = 0.01f + vol_ * 0.5f;
    t.order_flow_imbalance = imbal_;
    for (int i = 0; i < 10; i++)
      t.book_depth[i] = std::exp(-0.3f * i) * (1.0f + 0.05f * unif_(gen_));
    return t;
  }
  
private:
  std::mt19937 gen_;
  std::normal_distribution<float> norm_;
  std::uniform_real_distribution<float> unif_;
  float mid_, vol_, imbal_;
};

void init_basic(StreamingSSM& m, std::mt19937& g) {
  int D = m.A_log().size();
  int D_in = m.W_in().size() / D;
  for (int i = 0; i < D; i++) {
    m.A_log()[i] = -1.0f;
    m.log_dt()[i] = -5.0f;
    m.B()[i] = 1.0f;
  }
  std::normal_distribution<float> d_in(0.0f, 0.1f / std::sqrt((float)D_in));
  std::normal_distribution<float> d_out(0.0f, 0.01f);
  for (float& v : m.W_in()) v = d_in(g);
  for (float& v : m.W_out()) v = d_out(g);
  m.b_out() = 0.0f;
}

void init_selective(SelectiveStreamSSM& m, std::mt19937& g) {
  int D = m.D(), D_in = m.in_dim();
  for (int i = 0; i < D; i++) m.A_log()[i] = -1.0f;
  
  float s_u  = 0.1f / std::sqrt((float)D_in);
  float s_bc = 0.02f / std::sqrt((float)D_in);
  float s_dt = 0.001f;
  float s_g  = 0.1f / std::sqrt((float)D_in);
  
  std::normal_distribution<float> du(0, s_u), dbc(0, s_bc), ddt(0, s_dt), dg(0, s_g);
  
  for (int i = 0; i < D; i++) {
    for (int j = 0; j < D_in; j++) {
      m.W_proj()[(0*D+i)*D_in+j] = du(g);
      m.W_proj()[(1*D+i)*D_in+j] = dbc(g);
      m.W_proj()[(2*D+i)*D_in+j] = dbc(g);
      m.W_proj()[(3*D+i)*D_in+j] = ddt(g);
      m.W_proj()[(4*D+i)*D_in+j] = dg(g);
    }
  }
  
  std::fill(m.b_proj().begin(), m.b_proj().end(), 0.0f);
  for (int d = 0; d < D; d++) {
    m.b_proj()[1*D+d] = 1.0f;
    m.b_proj()[2*D+d] = 1.0f;
    m.b_proj()[3*D+d] = -5.0f;
    m.b_proj()[4*D+d] = 1.0f;
  }
  
  std::normal_distribution<float> d_out(0.0f, 0.05f);
  for (float& v : m.W_out()) v = d_out(g);
  m.b_out() = 0.0f;
}

void init_multiscale(MultiScaleSSM& m, std::mt19937& g) {
  int D = m.D_total(), D_in = m.W_in().size() / D;
  std::normal_distribution<float> d_in(0.0f, 0.1f / std::sqrt((float)D_in));
  std::normal_distribution<float> d_out(0.0f, 0.01f);
  for (float& v : m.W_in()) v = d_in(g);
  for (float& v : m.W_combine()) v = d_out(g);
  m.b_combine() = 0.0f;
}

//hw info for tagging bench rows
struct hwinfo {
  std::string cpu;
  std::string gpu;
  std::string build;
};

hwinfo get_hw_info() {
  hwinfo h;

#ifndef BUILD_GIT_HASH
#define BUILD_GIT_HASH "unknown"
#endif
  h.build = BUILD_GIT_HASH;

#ifdef __APPLE__
  char cbuf[256] = {};
  size_t csz = sizeof(cbuf);
  sysctlbyname("machdep.cpu.brand_string", cbuf, &csz, nullptr, 0);
  h.cpu = cbuf;
#elif defined(__linux__)
  std::ifstream cpuf("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuf, line)) {
    if (line.rfind("model name", 0) == 0) {
      auto p = line.find(':');
      if (p != std::string::npos) { h.cpu = line.substr(p + 2); break; }
    }
  }
#endif

  FILE* ns = popen("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null", "r");
  if (ns) {
    char nbuf[256] = {};
    if (fgets(nbuf, sizeof(nbuf), ns)) {
      h.gpu = nbuf;
      if (!h.gpu.empty() && h.gpu.back() == '\n') h.gpu.pop_back();
    }
    pclose(ns);
  }

  return h;
}

//get timestamp
std::string timestamp_iso() {
  std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return std::string(buf);
}

//write json line
void write_jsonl(const char* path, const benchconfig& cfg, const BenchStats& st, const hwinfo& hw) {
  std::ofstream out(path, std::ios::app);
  out << "{"
      << "\"kernel\":\"" << cfg.kernel << "\","
      << "\"d_model\":" << cfg.d_model << ","
      << "\"d_in\":" << cfg.d_in << ","
      << "\"conv_k\":" << cfg.conv_k << ","
      << "\"seq_len\":" << cfg.seq_len << ","
      << "\"warmup\":" << cfg.warmup << ","
      << "\"steps\":" << cfg.steps << ","
      << "\"seed\":" << cfg.seed << ","
      << "\"mean_us\":" << st.mean_us << ","
      << "\"p50_us\":" << st.p50_us << ","
      << "\"p90_us\":" << st.p90_us << ","
      << "\"p95_us\":" << st.p95_us << ","
      << "\"p99_us\":" << st.p99_us << ","
      << "\"p999_us\":" << st.p999_us << ","
      << "\"ops_per_sec\":" << st.ops_per_sec << ","
      << "\"cpu\":\"" << hw.cpu << "\","
      << "\"gpu\":\"" << hw.gpu << "\","
      << "\"build\":\"" << hw.build << "\","
      << "\"timestamp\":\"" << timestamp_iso() << "\""
      << "}\n";
}

int main(int argc, char** argv) {
  benchconfig cfg;
  if (!parse_args(argc, argv, cfg)) return 1;
  
  std::cout << "bench config:\n"
            << "  kernel=" << cfg.kernel << " d_model=" << cfg.d_model
            << " d_in=" << cfg.d_in << " conv_k=" << cfg.conv_k << "\n"
            << "  seq_len=" << cfg.seq_len << " warmup=" << cfg.warmup
            << " steps=" << cfg.steps << " seed=" << cfg.seed << "\n"
            << "  output=" << cfg.output << "\n\n";
  
  hwinfo hw = get_hw_info();
  std::cout << "hw: cpu=\"" << hw.cpu << "\"  gpu=\"" << hw.gpu << "\"  build=" << hw.build << "\n\n";

  std::mt19937 gen(cfg.seed);
  lobsim sim(cfg.seed + 1);
  
  //generate ticks upfront
  std::vector<TickFeatures> ticks;
  ticks.reserve(cfg.seq_len);
  for (int i = 0; i < cfg.seq_len; i++) ticks.push_back(sim.next());
  
  BenchStats stats;
  
  if (strcmp(cfg.kernel, "selective") == 0) {
    SelectiveStreamSSM model(cfg.d_model, cfg.d_in, cfg.conv_k);
    init_selective(model, gen);

    stats = bench_microseconds("selective", cfg.warmup, cfg.steps, [&]() {
      static int idx = 0;
      model.forward_step(ticks[idx % cfg.seq_len]);
      idx++;
    });

  } else if (strcmp(cfg.kernel, "proj_only") == 0) {
    //bare 5-head GEMV: isolates projection cost so scan cost = selective - proj_only
    int pd = 5 * cfg.d_model;
    std::vector<float> W((size_t)pd * cfg.d_in, 0.0f);
    std::vector<float> b((size_t)pd, 0.0f);
    std::vector<float> proj((size_t)pd, 0.0f);
    std::vector<float> x((size_t)cfg.d_in, 0.0f);
    std::normal_distribution<float> dn(0.0f, 0.1f);
    for (float& v : W) v = dn(gen);
    for (float& v : x) v = dn(gen);

    stats = bench_microseconds("proj_only", cfg.warmup, cfg.steps, [&]() {
      for (int i = 0; i < pd; i++) {
        float acc = b[i];
        int row = i * cfg.d_in;
        for (int j = 0; j < cfg.d_in; j++)
          acc += W[row + j] * x[j];
        proj[i] = acc;
      }
    });

  } else if (strcmp(cfg.kernel, "basic") == 0) {
    StreamingSSM model(cfg.d_model, cfg.d_in);
    init_basic(model, gen);
    
    stats = bench_microseconds("basic", cfg.warmup, cfg.steps, [&]() {
      static int idx = 0;
      model.forward_step(ticks[idx % cfg.seq_len]);
      idx++;
    });
    
  } else if (strcmp(cfg.kernel, "unfused") == 0) {
    StreamingSSMUnfused model(cfg.d_model, cfg.d_in);
    for (int i = 0; i < cfg.d_model; i++) {
      model.A_log()[i] = -1.0f;
      model.log_dt()[i] = -5.0f;
      model.B()[i] = 1.0f;
    }
    std::normal_distribution<float> d_in(0.0f, 0.1f / std::sqrt((float)cfg.d_in));
    std::normal_distribution<float> d_out(0.0f, 0.01f);
    for (float& v : model.W_in()) v = d_in(gen);
    for (float& v : model.W_out()) v = d_out(gen);
    model.b_out() = 0.0f;
    
    stats = bench_microseconds("unfused", cfg.warmup, cfg.steps, [&]() {
      static int idx = 0;
      model.forward_step(ticks[idx % cfg.seq_len]);
      idx++;
    });
    
  } else if (strcmp(cfg.kernel, "multiscale") == 0) {
    MultiScaleSSM model(cfg.d_model / 3, cfg.d_in, 3);
    init_multiscale(model, gen);
    
    stats = bench_microseconds("multiscale", cfg.warmup, cfg.steps, [&]() {
      static int idx = 0;
      model.forward_step(ticks[idx % cfg.seq_len]);
      idx++;
    });
    
  } else {
    std::cerr << "error: unknown kernel " << cfg.kernel << "\n";
    return 1;
  }
  
  write_jsonl(cfg.output, cfg, stats, hw);
  std::cout << "\nwrote results to " << cfg.output << "\n";
  
  return 0;
}
