# Selective State Space Models for Limit Order Books

**Microsecond-latency C++ implementation of selective SSMs for tick-level order flow prediction**

## Abstract

This is a C++ implementation of selective state space models (SSMs) optimized for microsecond-latency order flow prediction in limit order books. While SSMs have shown promise in sequence modeling, existing implementations are limited to Python/PyTorch and operate on daily or minute-level data. We demonstrate that SSMs achieve **0.45–1.33 microseconds** per-tick inference latency, validating the theoretical O(1) recurrence property in a production setting. This implementation includes three novel architectures: (1) a selective SSM with input-dependent B, C, dt parameters and volatility-adaptive discretization, (2) a multi-timescale SSM capturing micro/milli/second dynamics simultaneously, and (3) an online learning framework for real-time regime adaptation. All models achieve **>500K ops/s** throughput while maintaining cache-friendly memory access patterns (1–3KB per tick). 

image.png