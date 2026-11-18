// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hash and time the exact gfx12 FP8/BF8 WMMA entry points over deterministic tiles, half of
// them arbitrary bit patterns. Build it against two librj_arith.a builds and diff the output:
// the hashes must match, which is how the vectorised FP8 kernels were shown to reproduce the
// scalar model through the public API. See wmma_gfx12_fp8_exact.md.
//
//   amdclang++ -O2 -std=gnu++20 -I<lib/arith/include> \
//       wmma_gfx12_fp8_exact_bench.cpp <build>/lib/arith/src/librj_arith.a -lm -o bench
#include "rj_arith/rj_arith.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdint.h>

namespace {
uint64_t mix(uint64_t h, uint32_t v) { return (h ^ v) * 0x100000001b3ull; }
using Entry = int (*)(uint64_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                      const uint32_t *const *, const uint32_t *const *);
struct Form { const char *name; Entry fn; int regs; };

template <typename F> double median_ns(F &&body) {
  using clock = std::chrono::steady_clock;
  long iters = 1;
  for (auto t0 = clock::now(); clock::now() - t0 < std::chrono::milliseconds(20); iters *= 2) {
    t0 = clock::now();
    for (long i = 0; i < iters; ++i) body();
  }
  double s[7];
  for (double &x : s) {
    auto t0 = clock::now();
    for (long i = 0; i < iters; ++i) body();
    x = std::chrono::duration<double, std::nano>(clock::now() - t0).count() / iters;
  }
  std::sort(s, s + 7);
  return s[3];
}
} // namespace

int main() {
  const uint64_t cpu = rj_arith_init_cpu_flags() & RJ_ARITH_CPU_MASK;
  const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY;
  const Form forms[] = {
      {"fp8_fp8_w32", rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12, 32},
      {"fp8_bf8_w32", rj_arith_v_wmma_f32_16x16x16_fp8_bf8_w32_gfx12, 32},
      {"bf8_fp8_w32", rj_arith_v_wmma_f32_16x16x16_bf8_fp8_w32_gfx12, 32},
      {"bf8_bf8_w32", rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w32_gfx12, 32},
      {"fp8_fp8_w64", rj_arith_v_wmma_f32_16x16x16_fp8_fp8_w64_gfx12, 64},
      {"bf8_bf8_w64", rj_arith_v_wmma_f32_16x16x16_bf8_bf8_w64_gfx12, 64},
  };
  for (const Form &form : forms) {
    std::mt19937 rng(11);
    uint64_t hash = 0xcbf29ce484222325ull;
    alignas(64) uint32_t regs[24][64];
    const uint32_t *a[2], *b[2], *c[8];
    uint32_t *d[8];
    for (int r = 0; r < 2; ++r) { a[r] = regs[r]; b[r] = regs[2 + r]; }
    for (int r = 0; r < 8; ++r) { c[r] = regs[4 + r]; d[r] = regs[12 + r]; }
    for (int tile = 0; tile < 3000; ++tile) {
      const bool any = tile % 2;
      for (int r = 0; r < 4; ++r)
        for (int l = 0; l < form.regs; ++l) {
          auto byte = [&]() -> uint32_t {
            return any ? (rng() & 0xff)
                       : (rng() % 10 ? (rng() & 0x80) | (4 + rng() % 6) << 3 | (rng() & 7)
                                     : rng() & 0x80);
          };
          regs[r][l] = byte() | byte() << 8 | byte() << 16 | byte() << 24;
        }
      for (int r = 4; r < 20; ++r)
        for (int l = 0; l < form.regs; ++l) {
          float f = std::uniform_real_distribution<float>(-4, 4)(rng);
          uint32_t bits; memcpy(&bits, &f, 4);
          regs[r][l] = any ? uint32_t(rng()) : bits;
        }
      if (form.fn(flags, 0, d, a, b, c)) { printf("%s ERROR\n", form.name); return 1; }
      for (int r = 0; r < 8; ++r)
        for (int l = 0; l < form.regs; ++l) hash = mix(hash, d[r][l]);
    }
    const double ns = median_ns([&] { form.fn(flags, 0, d, a, b, c); });
    printf("%-12s hash %016llx  %9.1f ns\n", form.name, (unsigned long long)hash, ns);
  }
  return 0;
}
