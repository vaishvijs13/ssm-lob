#pragma once

struct TickFeatures {
  float bid_ask_spread;
  float order_flow_imbalance;
  float mid_price_change;
  float book_depth[10];
};

constexpr int TICK_FEAT_DIM = 13;