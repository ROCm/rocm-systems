// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file util_simd_test.cpp
/// @brief Generic-core unit tests for util SIMD primitives. Header-only,
/// no rocjitsu Operand/Wavefront fixture.

#include "util/simd.h"

#include "util/data_types.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>

namespace {

// Toolchain-guard sweep length. The bit-exactness guards below were validated
// over a 250k-iteration full-range sweep, but running that on every CI build
// adds significant wall time (5 sweeps x 250k x SIMD width). Default to a
// shorter deterministic sweep that still covers NaN/Inf/denormal/negative bit
// patterns; set ROCJITSU_SIMD_FULL_SWEEP=1 for the exhaustive run (e.g. when
// qualifying a new toolchain).
inline int sweep_iters() {
  static const int n = [] {
    const char *full = std::getenv("ROCJITSU_SIMD_FULL_SWEEP");
    return (full && full[0] == '1') ? 250000 : 20000;
  }();
  return n;
}

// Each test skips at runtime when `<experimental/simd>` is unavailable;
// the SIMD body is type-checked but compiles out via `if constexpr`.
#define SKIP_IF_NO_SIMD()                                                                          \
  if constexpr (!util::has_stdx_simd) {                                                            \
    GTEST_SKIP() << "<experimental/simd> unavailable; util SIMD helpers skipped.";                 \
    return;                                                                                        \
  }

constexpr std::size_t kW = util::native_width_v<uint32_t>;

TEST(UtilSimd, Load_Contiguous_U32) {
  SKIP_IF_NO_SIMD();
  alignas(util::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i)
    src[i] = static_cast<uint32_t>(0xDEAD'0000u + i);
  auto v = util::load<uint32_t>(src);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], src[i]) << "lane " << i;
}

TEST(UtilSimd, Load_Contiguous_F32) {
  SKIP_IF_NO_SIMD();
  alignas(util::native<float>) uint32_t src[kW];
  std::array<float, kW> expected{};
  for (std::size_t i = 0; i < kW; ++i) {
    expected[i] = 1.5f * static_cast<float>(i) - 7.25f;
    src[i] = std::bit_cast<uint32_t>(expected[i]);
  }
  auto v = util::load<float>(src);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], expected[i]) << "lane " << i;
}

TEST(UtilSimd, Broadcast_U32) {
  SKIP_IF_NO_SIMD();
  constexpr uint32_t kBits = 0xCAFEBABEu;
  auto v = util::broadcast<uint32_t>(kBits);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], kBits) << "lane " << i;
}

TEST(UtilSimd, Broadcast_F32) {
  SKIP_IF_NO_SIMD();
  constexpr float kVal = -3.14159f;
  const uint32_t bits = std::bit_cast<uint32_t>(kVal);
  auto v = util::broadcast<float>(bits);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], kVal) << "lane " << i;
}

TEST(UtilSimd, MaskedStore_FullMask) {
  SKIP_IF_NO_SIMD();
  alignas(util::native<uint32_t>) uint32_t dst[kW] = {};
  alignas(util::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i)
    src[i] = 0xA5A5'0000u + i;
  auto v = util::load<uint32_t>(src);
  const uint64_t full = util::mask<uint64_t>(static_cast<int>(kW));
  util::masked_store<uint32_t>(dst, v, full);
  EXPECT_EQ(0, std::memcmp(dst, src, sizeof(dst)));
}

TEST(UtilSimd, MaskedStore_PartialMask) {
  SKIP_IF_NO_SIMD();
  alignas(util::native<uint32_t>) uint32_t dst[kW];
  alignas(util::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    dst[i] = 0x1111'1111u;
    src[i] = 0x2222'0000u + i;
  }
  auto v = util::load<uint32_t>(src);
  // 0xA... pattern: enable every other lane starting from bit 1.
  uint64_t mask = 0;
  for (std::size_t i = 1; i < kW; i += 2)
    mask |= (1ULL << i);
  util::masked_store<uint32_t>(dst, v, mask);
  for (std::size_t i = 0; i < kW; ++i) {
    if (mask & (1ULL << i))
      EXPECT_EQ(dst[i], src[i]) << "lane " << i << " should be written";
    else
      EXPECT_EQ(dst[i], 0x1111'1111u) << "lane " << i << " should be preserved";
  }
}

TEST(UtilSimd, MaskedStore_EmptyMask) {
  SKIP_IF_NO_SIMD();
  alignas(util::native<uint32_t>) uint32_t dst[kW];
  alignas(util::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    dst[i] = 0xDEAD'BEEFu;
    src[i] = 0x0BAD'F00Du;
  }
  auto v = util::load<uint32_t>(src);
  util::masked_store<uint32_t>(dst, v, 0ULL);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(dst[i], 0xDEAD'BEEFu) << "lane " << i;
}

TEST(UtilSimd, BlitToBuffer_F32) {
  SKIP_IF_NO_SIMD();
  alignas(util::native<float>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i)
    src[i] = std::bit_cast<uint32_t>(2.0f * static_cast<float>(i) + 0.5f);
  auto v = util::load<float>(src);
  alignas(util::native<float>) uint32_t buf[util::native<float>::size()] = {};
  util::blit_to_buffer<float>(buf, v);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(buf[i], src[i]) << "lane " << i;
}

// --- narrow32 prims (native_width64-wide 32-bit lanes) for the f64<->32-bit
// conversion glue. Round-trips and the double<->narrow32 bridge cast. ---

TEST(UtilSimd, LoadNarrow_RoundTrip_F32) {
  SKIP_IF_NO_SIMD();
  constexpr std::size_t W = util::native_width64;
  alignas(util::narrow32<float>) uint32_t src[W];
  std::array<float, W> expected{};
  for (std::size_t i = 0; i < W; ++i) {
    expected[i] = 1.5f * static_cast<float>(i) - 7.25f;
    src[i] = std::bit_cast<uint32_t>(expected[i]);
  }
  auto v = util::load_narrow<float>(src);
  for (std::size_t i = 0; i < W; ++i)
    EXPECT_EQ(v[i], expected[i]) << "lane " << i;
}

TEST(UtilSimd, BroadcastNarrow_I32) {
  SKIP_IF_NO_SIMD();
  constexpr int32_t kVal = -123456789;
  auto v = util::broadcast_narrow<int32_t>(std::bit_cast<uint32_t>(kVal));
  for (std::size_t i = 0; i < util::native_width64; ++i)
    EXPECT_EQ(v[i], kVal) << "lane " << i;
}

TEST(UtilSimd, MaskedStoreNarrow_FullAndPartial) {
  SKIP_IF_NO_SIMD();
  constexpr std::size_t W = util::native_width64;
  alignas(util::narrow32<int32_t>) int32_t vals[W];
  for (std::size_t i = 0; i < W; ++i)
    vals[i] = static_cast<int32_t>(i) - 3;
  util::narrow32<int32_t> v(vals, util::stdx::vector_aligned);

  uint32_t dst_full[W] = {};
  util::masked_store_narrow<int32_t>(dst_full, v, util::mask<uint64_t>(static_cast<int>(W)));
  for (std::size_t i = 0; i < W; ++i)
    EXPECT_EQ(dst_full[i], std::bit_cast<uint32_t>(vals[i])) << "full lane " << i;

  uint32_t dst_part[W];
  for (std::size_t i = 0; i < W; ++i)
    dst_part[i] = 0x1111'1111u;
  uint64_t mask = 0;
  for (std::size_t i = 1; i < W; i += 2)
    mask |= (1ULL << i);
  util::masked_store_narrow<int32_t>(dst_part, v, mask);
  for (std::size_t i = 0; i < W; ++i) {
    if (mask & (1ULL << i))
      EXPECT_EQ(dst_part[i], std::bit_cast<uint32_t>(vals[i])) << "part write lane " << i;
    else
      EXPECT_EQ(dst_part[i], 0x1111'1111u) << "part preserve lane " << i;
  }
}

TEST(UtilSimd, NarrowBridgeCast_DoubleToFromB32) {
  SKIP_IF_NO_SIMD();
  constexpr std::size_t W = util::native_width64;
  // f64 -> f32: a direct static_simd_cast between native<double> and
  // narrow32<float> is the cvt_f32_f64 bridge; matches a per-lane (float)cast.
  alignas(util::native<double>) double in[W];
  for (std::size_t i = 0; i < W; ++i)
    in[i] = 0.1 * static_cast<double>(i) - 1.3 + 1e30;
  util::native<double> d(in, util::stdx::vector_aligned);
  auto f = util::stdx::static_simd_cast<util::narrow32<float>>(d);
  for (std::size_t i = 0; i < W; ++i)
    EXPECT_EQ(f[i], static_cast<float>(in[i])) << "f64->f32 lane " << i;
  // i32 -> f64: the cvt_f64_i32 bridge (exact widening).
  alignas(util::narrow32<int32_t>) int32_t iv[W];
  for (std::size_t i = 0; i < W; ++i)
    iv[i] = static_cast<int32_t>(i) * -1000003;
  util::narrow32<int32_t> ni(iv, util::stdx::vector_aligned);
  auto dd = util::stdx::static_simd_cast<util::native<double>>(ni);
  for (std::size_t i = 0; i < W; ++i)
    EXPECT_EQ(dd[i], static_cast<double>(iv[i])) << "i32->f64 lane " << i;
}

TEST(UtilSimd, ForceScalar_ImmutableProcessWide) {
  // force_scalar() is seeded once from RJ_FORCE_SCALAR at startup. Absent the
  // test-only setter (util/simd_test_hooks.h), it is stable across calls and
  // identical on every thread; this test exercises that steady-state behaviour
  // without flipping the gate.
  const bool v = util::force_scalar();
  EXPECT_EQ(util::force_scalar(), v) << "force_scalar() must be stable within a process";

  std::atomic<bool> other{!v};
  std::thread t([&]() { other.store(util::force_scalar()); });
  t.join();
  EXPECT_EQ(other.load(), v) << "force_scalar() must be process-wide (identical on all threads)";

  // Value reflects the env parse: unset/empty/"0" => false, else true.
  const char *e = std::getenv("RJ_FORCE_SCALAR");
  const bool expected = e && e[0] && !(e[0] == '0' && e[1] == '\0');
  EXPECT_EQ(v, expected);
}

// Toolchain guard for the SIMD fast path of v_exp_f32 (stdx::exp2) and
// v_log_f32 (stdx::log2). Those VOP1 kernels take the SIMD path whenever
// `util::has_stdx_simd`, with no runtime value check — so the vector libm
// MUST be bit-identical to the scalar std::* used by the generated body, or
// the fast path silently returns wrong results. These tests sweep full-range
// random bit patterns (covering NaN/Inf/denormal/negative) and fail loudly if
// any lane diverges, blocking a divergent toolchain at CI time. If one fails,
// drop the corresponding row from SIMD_VOP1_UNARY rather than ship wrong math.
template <class ScalarFn, class VectorFn>
void expect_simd_bit_exact(ScalarFn scalar_fn, VectorFn vector_fn) {
  using V = util::native<float>;
  constexpr std::size_t W = util::native_width_v<float>;
  std::mt19937 rng(0xA17C0DEu);
  for (int iter = 0, n = sweep_iters(); iter < n; ++iter) {
    alignas(V) float in[W];
    for (std::size_t i = 0; i < W; ++i)
      in[i] = std::bit_cast<float>(static_cast<uint32_t>(rng()));
    V v(in, util::stdx::element_aligned);
    alignas(V) float out[W];
    vector_fn(v).copy_to(out, util::stdx::element_aligned);
    for (std::size_t i = 0; i < W; ++i) {
      const float s = scalar_fn(in[i]);
      ASSERT_EQ(std::bit_cast<uint32_t>(s), std::bit_cast<uint32_t>(out[i]))
          << "divergent at iter " << iter << " lane " << i << std::hex << " in=0x"
          << std::bit_cast<uint32_t>(in[i]);
    }
  }
}

TEST(UtilSimd, Exp2_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_simd_bit_exact([](float x) { return std::exp2(x); },
                        [](util::native<float> v) { return util::stdx::exp2(v); });
}

TEST(UtilSimd, Log2_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_simd_bit_exact([](float x) { return std::log2(x); },
                        [](util::native<float> v) { return util::stdx::log2(v); });
}

// Toolchain guard for the ternary VOP2 FMA/MAC/MAD SIMD fast path
// (SIMD_VOP2_TERNARY). Those kernels use util::stdx::fma unconditionally under
// has_stdx_simd, so it MUST be the single-rounded fused operation matching the
// scalar std::fma in the generated bodies — a separate mul+add (two roundings)
// would diverge silently. This sweeps full-range random bit patterns for all
// three operands and asserts bit-exactness, EXCLUDING lanes with a NaN input:
// for NaN inputs the packed and scalar FMA may propagate a different NaN
// payload (toolchain-dependent, e.g. g++-13 picks the second multiplicand while
// the packed form picks the first), and that NaN-payload divergence is an
// accepted difference (the result is a NaN either way). Inf and Inf*0→NaN cases
// ARE checked (they do match). If a finite/Inf lane diverges, drop
// SIMD_VOP2_TERNARY rather than ship wrong math. The f16 forms inherit this
// guard plus the f16<->f32 ones.
TEST(UtilSimd, Fma_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  using V = util::native<float>;
  constexpr std::size_t W = util::native_width_v<float>;
  auto is_nan_bits = [](uint32_t u) {
    return ((u >> 23) & 0xFFu) == 0xFFu && (u & 0x7FFFFFu) != 0u;
  };
  std::mt19937 rng(0xF1AC0DEu);
  for (int iter = 0, n = sweep_iters(); iter < n; ++iter) {
    alignas(V) float a[W], b[W], c[W];
    for (std::size_t i = 0; i < W; ++i) {
      a[i] = std::bit_cast<float>(static_cast<uint32_t>(rng()));
      b[i] = std::bit_cast<float>(static_cast<uint32_t>(rng()));
      c[i] = std::bit_cast<float>(static_cast<uint32_t>(rng()));
    }
    V va(a, util::stdx::element_aligned), vb(b, util::stdx::element_aligned),
        vc(c, util::stdx::element_aligned);
    alignas(V) float out[W];
    util::stdx::fma(va, vb, vc).copy_to(out, util::stdx::element_aligned);
    for (std::size_t i = 0; i < W; ++i) {
      if (is_nan_bits(std::bit_cast<uint32_t>(a[i])) ||
          is_nan_bits(std::bit_cast<uint32_t>(b[i])) || is_nan_bits(std::bit_cast<uint32_t>(c[i])))
        continue; // NaN-input payload divergence is accepted
      const float s = std::fma(a[i], b[i], c[i]);
      ASSERT_EQ(std::bit_cast<uint32_t>(s), std::bit_cast<uint32_t>(out[i]))
          << "divergent at iter " << iter << " lane " << i << std::hex << " a=0x"
          << std::bit_cast<uint32_t>(a[i]) << " b=0x" << std::bit_cast<uint32_t>(b[i]) << " c=0x"
          << std::bit_cast<uint32_t>(c[i]);
    }
  }
}

// Toolchain guard for the 64-bit-lane VOP2 FMA SIMD fast path (v_fmac_f64,
// SIMD_VOP2_FMA_F64). Same contract as the f32 Fma guard above, over
// native<double>: util::stdx::fma must be the single-rounded fused operation
// matching the scalar std::fma in the generated body. Sweeps full-range random
// 64-bit patterns for all three operands, EXCLUDING NaN-input lanes (accepted
// payload divergence). If a finite/Inf lane diverges, drop SIMD_VOP2_FMA_F64.
TEST(UtilSimd, FmaF64_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  using V = util::native<double>;
  constexpr std::size_t W = util::native_width64;
  auto is_nan_bits = [](uint64_t u) {
    return ((u >> 52) & 0x7FFu) == 0x7FFu && (u & 0xFFFFFFFFFFFFFull) != 0u;
  };
  std::mt19937_64 rng(0xF1AC0DE64ull);
  for (int iter = 0, n = sweep_iters(); iter < n; ++iter) {
    alignas(V) double a[W], b[W], c[W];
    for (std::size_t i = 0; i < W; ++i) {
      a[i] = std::bit_cast<double>(rng());
      b[i] = std::bit_cast<double>(rng());
      c[i] = std::bit_cast<double>(rng());
    }
    V va(a, util::stdx::element_aligned), vb(b, util::stdx::element_aligned),
        vc(c, util::stdx::element_aligned);
    alignas(V) double out[W];
    util::stdx::fma(va, vb, vc).copy_to(out, util::stdx::element_aligned);
    for (std::size_t i = 0; i < W; ++i) {
      if (is_nan_bits(std::bit_cast<uint64_t>(a[i])) ||
          is_nan_bits(std::bit_cast<uint64_t>(b[i])) || is_nan_bits(std::bit_cast<uint64_t>(c[i])))
        continue; // NaN-input payload divergence is accepted
      const double s = std::fma(a[i], b[i], c[i]);
      ASSERT_EQ(std::bit_cast<uint64_t>(s), std::bit_cast<uint64_t>(out[i]))
          << "divergent at iter " << iter << " lane " << i << std::hex << " a=0x"
          << std::bit_cast<uint64_t>(a[i]) << " b=0x" << std::bit_cast<uint64_t>(b[i]) << " c=0x"
          << std::bit_cast<uint64_t>(c[i]);
    }
  }
}

// native<double>: the f64 unary VOP1 fast path (v_ceil/floor/trunc/rndne/fract/
// rcp/rsq/sqrt_f64) relies on util::stdx::{ceil,floor,trunc,nearbyint,sqrt} and
// 1.0/x being bit-identical to the scalar std::* used in the generated bodies.
// Sweeps full-range random 64-bit patterns, skipping lanes whose result is NaN
// (accepted payload divergence). If a finite/Inf lane diverges, drop the
// corresponding SIMD_VOP1_UNARY_F64 row.
template <class ScalarFn, class VectorFn>
void expect_f64_unary_bit_exact(ScalarFn scalar_fn, VectorFn vector_fn) {
  using V = util::native<double>;
  constexpr std::size_t W = util::native_width64;
  auto is_nan_bits = [](uint64_t u) {
    return ((u >> 52) & 0x7FFu) == 0x7FFu && (u & 0xFFFFFFFFFFFFFull) != 0u;
  };
  std::mt19937_64 rng(0xD06F64ull);
  for (int iter = 0, n = sweep_iters(); iter < n; ++iter) {
    alignas(V) double a[W];
    for (std::size_t i = 0; i < W; ++i)
      a[i] = std::bit_cast<double>(rng());
    V va(a, util::stdx::element_aligned);
    alignas(V) double out[W];
    vector_fn(va).copy_to(out, util::stdx::element_aligned);
    for (std::size_t i = 0; i < W; ++i) {
      const double s = scalar_fn(a[i]);
      const uint64_t su = std::bit_cast<uint64_t>(s), ou = std::bit_cast<uint64_t>(out[i]);
      if (is_nan_bits(su) || is_nan_bits(ou))
        continue; // NaN-result payload divergence is accepted
      ASSERT_EQ(su, ou) << "divergent at iter " << iter << " lane " << i << std::hex << " in=0x"
                        << std::bit_cast<uint64_t>(a[i]);
    }
  }
}

TEST(UtilSimd, CeilF64_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_f64_unary_bit_exact([](double x) { return std::ceil(x); },
                             [](util::native<double> x) { return util::ceil_simd(x); });
}
TEST(UtilSimd, FloorF64_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_f64_unary_bit_exact([](double x) { return std::floor(x); },
                             [](util::native<double> x) { return util::floor_simd(x); });
}
TEST(UtilSimd, TruncF64_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_f64_unary_bit_exact([](double x) { return std::trunc(x); },
                             [](util::native<double> x) { return util::trunc_simd(x); });
}
TEST(UtilSimd, RndneF64_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_f64_unary_bit_exact([](double x) { return std::nearbyint(x); },
                             [](util::native<double> x) { return util::rndne_simd(x); });
}
TEST(UtilSimd, FractF64_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_f64_unary_bit_exact([](double x) { return x - std::floor(x); },
                             [](util::native<double> x) { return x - util::floor_simd(x); });
}
TEST(UtilSimd, RcpF64_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_f64_unary_bit_exact([](double x) { return 1.0f / x; },
                             [](util::native<double> x) { return util::native<double>(1.0) / x; });
}
TEST(UtilSimd, RsqF64_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_f64_unary_bit_exact(
      [](double x) { return 1.0f / std::sqrt(x); },
      [](util::native<double> x) { return util::native<double>(1.0) / util::stdx::sqrt(x); });
}
TEST(UtilSimd, SqrtF64_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_f64_unary_bit_exact([](double x) { return std::sqrt(x); },
                             [](util::native<double> x) { return util::stdx::sqrt(x); });
}

// Toolchain guard for the float min/max SIMD fast path (v_max/min_f32 and the
// f16 forms). util::stdx::fmax/fmin must match the scalar std::fmax/fmin used by
// the generated bodies. This sweeps full-range random bit patterns, biasing one
// in eight operands to a signed zero so the -0/+0 tie cases (both orders) are
// hit heavily, and asserts bit-exactness for all finite/Inf inputs. Two classes
// of divergence are accepted (and skipped): a NaN input (NaN payload differs)
// and a signed-zero tie (scalar std::fmax/fmin returns the first operand, the
// packed vmaxps/vminps the second — a -0 vs +0 sign difference, numerically
// equal). If any OTHER lane diverges, drop the float min/max rows.
template <class ScalarFn, class VectorFn>
void expect_minmax_bit_exact(ScalarFn scalar_fn, VectorFn vector_fn) {
  using V = util::native<float>;
  constexpr std::size_t W = util::native_width_v<float>;
  auto is_nan_bits = [](uint32_t u) {
    return ((u >> 23) & 0xFFu) == 0xFFu && (u & 0x7FFFFFu) != 0u;
  };
  auto is_zero_bits = [](uint32_t u) { return (u & 0x7FFFFFFFu) == 0u; };
  std::mt19937 rng(0xFA00C0DEu);
  for (int iter = 0, n = sweep_iters(); iter < n; ++iter) {
    alignas(V) float a[W], b[W];
    for (std::size_t i = 0; i < W; ++i) {
      uint32_t ra = rng(), rb = rng();
      if ((rng() & 7u) == 0u)
        ra &= 0x80000000u; // force +/-0
      if ((rng() & 7u) == 0u)
        rb &= 0x80000000u;
      a[i] = std::bit_cast<float>(ra);
      b[i] = std::bit_cast<float>(rb);
    }
    V va(a, util::stdx::element_aligned), vb(b, util::stdx::element_aligned);
    alignas(V) float out[W];
    vector_fn(va, vb).copy_to(out, util::stdx::element_aligned);
    for (std::size_t i = 0; i < W; ++i) {
      const uint32_t ua = std::bit_cast<uint32_t>(a[i]), ub = std::bit_cast<uint32_t>(b[i]);
      if (is_nan_bits(ua) || is_nan_bits(ub))
        continue; // NaN-input payload divergence is accepted
      if (is_zero_bits(ua) && is_zero_bits(ub))
        continue; // signed-zero tie: -0/+0 sign difference is accepted
      const float s = scalar_fn(a[i], b[i]);
      ASSERT_EQ(std::bit_cast<uint32_t>(s), std::bit_cast<uint32_t>(out[i]))
          << "divergent at iter " << iter << " lane " << i << std::hex << " a=0x" << ua << " b=0x"
          << ub;
    }
  }
}

TEST(UtilSimd, Fmax_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_minmax_bit_exact(
      [](float x, float y) { return std::fmax(x, y); },
      [](util::native<float> x, util::native<float> y) { return util::stdx::fmax(x, y); });
}

TEST(UtilSimd, Fmin_VectorMatchesScalar_BitExact) {
  SKIP_IF_NO_SIMD();
  expect_minmax_bit_exact(
      [](float x, float y) { return std::fmin(x, y); },
      [](util::native<float> x, util::native<float> y) { return util::stdx::fmin(x, y); });
}

// Bit-exactness guards for the f16<->f32 vector conversions. The 16-bit VOP1
// SIMD functors rely on these matching the scalar util::f16_to_f32 /
// util::f32_to_f16 for every input (the SIMD path is unconditional under
// has_stdx_simd). f16->f32 is verified EXHAUSTIVELY over all 65536 f16 bit
// patterns; f32->f16 over a heavy full-range sweep. Both fail CI on divergence.
TEST(UtilSimd, F16ToF32_VectorMatchesScalar_Exhaustive) {
  SKIP_IF_NO_SIMD();
  using V = util::native<uint32_t>;
  constexpr std::size_t W = util::native_width_v<uint32_t>;
  for (uint32_t base = 0; base < 65536u; base += static_cast<uint32_t>(W)) {
    alignas(V) uint32_t in[W];
    for (std::size_t i = 0; i < W; ++i)
      in[i] = base + static_cast<uint32_t>(i); // f16 bits 0..65535 (all)
    V v(in, util::stdx::element_aligned);
    util::native<float> rv = util::f16_to_f32_simd(v);
    alignas(util::native<float>) float out[W];
    rv.copy_to(out, util::stdx::element_aligned);
    for (std::size_t i = 0; i < W; ++i) {
      const float s = util::f16_to_f32(static_cast<uint16_t>(in[i]));
      ASSERT_EQ(std::bit_cast<uint32_t>(s), std::bit_cast<uint32_t>(out[i]))
          << "f16=0x" << std::hex << in[i];
    }
  }
}

TEST(UtilSimd, F32ToF16_VectorMatchesScalar_Sweep) {
  SKIP_IF_NO_SIMD();
  using V = util::native<float>;
  constexpr std::size_t W = util::native_width_v<float>;
  // Stride-prime sweep across the full 2^32 f32 bit space (~4.3M samples),
  // hitting every exponent and a spread of mantissas, incl. NaN/Inf/denormal.
  for (uint64_t u = 0; u < 0x100000000ULL; u += 997ULL * W) {
    alignas(V) float in[W];
    for (std::size_t i = 0; i < W; ++i)
      in[i] = std::bit_cast<float>(static_cast<uint32_t>(u + 997ULL * i));
    V v(in, util::stdx::element_aligned);
    util::native<uint32_t> rv = util::f32_to_f16_simd(v);
    alignas(util::native<uint32_t>) uint32_t out[W];
    rv.copy_to(out, util::stdx::element_aligned);
    for (std::size_t i = 0; i < W; ++i) {
      const uint32_t s = util::f32_to_f16(in[i]);
      ASSERT_EQ(s, out[i]) << "f32=0x" << std::hex << std::bit_cast<uint32_t>(in[i]);
    }
  }
}

// Guard for util::flush_denorm_f32_simd: denormals flush to sign-preserving
// zero (FTZ); every other class (±0, normal, ±Inf, NaN payloads) passes through
// bit-for-bit. The f32 rcp/rsq/exp/log SIMD ports funnel through this helper.
TEST(UtilSimd, FlushDenormF32) {
  SKIP_IF_NO_SIMD();
  using V = util::native<float>;
  constexpr std::size_t W = util::native_width_v<float>;
  struct Case {
    uint32_t in, out;
  };
  const Case cases[] = {
      {0x00000001u, 0x00000000u}, // +smallest denormal -> +0
      {0x80000001u, 0x80000000u}, // -smallest denormal -> -0
      {0x007FFFFFu, 0x00000000u}, // +largest denormal  -> +0
      {0x807FFFFFu, 0x80000000u}, // -largest denormal  -> -0
      {0x00000000u, 0x00000000u}, // +0 untouched
      {0x80000000u, 0x80000000u}, // -0 untouched
      {0x00800000u, 0x00800000u}, // smallest normal untouched
      {0x3F800000u, 0x3F800000u}, // 1.0 untouched
      {0x7F800000u, 0x7F800000u}, // +Inf untouched
      {0xFF800000u, 0xFF800000u}, // -Inf untouched
      {0x7FC00000u, 0x7FC00000u}, // qNaN payload untouched
      {0x7F800001u, 0x7F800001u}, // sNaN payload untouched
  };
  for (const auto &c : cases) {
    V r = util::flush_denorm_f32_simd(util::broadcast<float>(c.in));
    alignas(V) float out[W];
    r.copy_to(out, util::stdx::element_aligned);
    for (std::size_t i = 0; i < W; ++i)
      EXPECT_EQ(std::bit_cast<uint32_t>(out[i]), c.out) << "in=0x" << std::hex << c.in;
  }
}

// --- IEEE-2019 maximum / minimum (NaN-propagating, signed-zero-ordered) ------
//
// util::ieee_{maximum,minimum}_simd back the v_maximum_*/v_minimum_* gap ops.
// They must be bit-identical to the scalar bodies the generator emits, which is
// the exact expression below. The grid pairs every IEEE class against every
// other so ±0 ties, NaN-in-either, Inf, and ordinary compares are all covered.

// Bit pattern of a float/double as an unsigned integer, for exact lane-value
// comparison (EXPECT_EQ on the raw bits distinguishes NaN payloads and ±0,
// which == on the float values would not).
//
// Both overloads share a single uint64_t return type so the helper is
// type-generic at the call sites. The float overload therefore zero-extends
// its 32-bit pattern into the 64-bit result; this is intentional and safe
// because only same-type lanes are ever compared (float-vs-float or
// double-vs-double), so the high 32 zero bits are present on both sides of any
// float comparison and never change the result.
inline uint64_t bits_of(float v) { return std::bit_cast<uint32_t>(v); }
inline uint64_t bits_of(double v) { return std::bit_cast<uint64_t>(v); }

// Scalar references — literally the generated VOP3 body's inner expression.
template <typename T> T scalar_ieee_maximum(T a, T b) {
  if (std::isnan(a) || std::isnan(b))
    return std::numeric_limits<T>::quiet_NaN();
  if (a == b)
    return std::signbit(a) ? b : a;
  return a > b ? a : b;
}
template <typename T> T scalar_ieee_minimum(T a, T b) {
  if (std::isnan(a) || std::isnan(b))
    return std::numeric_limits<T>::quiet_NaN();
  if (a == b)
    return std::signbit(a) ? a : b;
  return a < b ? a : b;
}

template <typename T> std::array<T, 13> ieee_minmax_grid() {
  const T inf = std::numeric_limits<T>::infinity();
  const T qnan = std::numeric_limits<T>::quiet_NaN();
  const T den = std::numeric_limits<T>::denorm_min();
  return {T(0), -T(0), T(1), -T(1), T(2), -T(2), T(0.5), inf, -inf, qnan, -qnan, den, -den};
}

template <typename V, typename T, typename SimdFn, typename ScalarFn>
void check_ieee_minmax(SimdFn simd_fn, ScalarFn scalar_fn) {
  constexpr std::size_t W = V::size();
  const auto grid = ieee_minmax_grid<T>();
  for (T bv : grid) {
    // Fill a vector with `bv` in every lane and sweep `av` across the grid so
    // both same-class (ties) and cross-class pairs are exercised per lane.
    for (T av : grid) {
      alignas(V) T abuf[W];
      alignas(V) T bbuf[W];
      for (std::size_t i = 0; i < W; ++i) {
        abuf[i] = av;
        bbuf[i] = bv;
      }
      V a(abuf, util::stdx::vector_aligned);
      V b(bbuf, util::stdx::vector_aligned);
      V r = simd_fn(a, b);
      const T expect = scalar_fn(av, bv);
      for (std::size_t i = 0; i < W; ++i) {
        const T rv = r[i];
        EXPECT_EQ(bits_of(rv), bits_of(expect)) << "av=" << av << " bv=" << bv << " lane=" << i;
      }
    }
  }
}

TEST(UtilSimd, IeeeMaximum_F32_BitExact) {
  SKIP_IF_NO_SIMD();
  check_ieee_minmax<util::native<float>, float>(
      [](auto a, auto b) { return util::ieee_maximum_simd(a, b); }, scalar_ieee_maximum<float>);
}
TEST(UtilSimd, IeeeMinimum_F32_BitExact) {
  SKIP_IF_NO_SIMD();
  check_ieee_minmax<util::native<float>, float>(
      [](auto a, auto b) { return util::ieee_minimum_simd(a, b); }, scalar_ieee_minimum<float>);
}
TEST(UtilSimd, IeeeMaximum_F64_BitExact) {
  SKIP_IF_NO_SIMD();
  check_ieee_minmax<util::native<double>, double>(
      [](auto a, auto b) { return util::ieee_maximum_simd(a, b); }, scalar_ieee_maximum<double>);
}
TEST(UtilSimd, IeeeMinimum_F64_BitExact) {
  SKIP_IF_NO_SIMD();
  check_ieee_minmax<util::native<double>, double>(
      [](auto a, auto b) { return util::ieee_minimum_simd(a, b); }, scalar_ieee_minimum<double>);
}

// The 3-input / combined gap ops are exact nested compositions of the binary
// helper; verify the composition matches the nested scalar reference too.
TEST(UtilSimd, IeeeMaximum3_F32_BitExact) {
  SKIP_IF_NO_SIMD();
  using V = util::native<float>;
  constexpr std::size_t W = V::size();
  const auto grid = ieee_minmax_grid<float>();
  for (float cv : grid)
    for (float bv : grid)
      for (float av : grid) {
        alignas(V) float a_[W], b_[W], c_[W];
        for (std::size_t i = 0; i < W; ++i) {
          a_[i] = av;
          b_[i] = bv;
          c_[i] = cv;
        }
        V a(a_, util::stdx::vector_aligned), b(b_, util::stdx::vector_aligned),
            c(c_, util::stdx::vector_aligned);
        // maximumminimum = minimum(maximum(a,b), c) — exercises both helpers.
        V r = util::ieee_minimum_simd(util::ieee_maximum_simd(a, b), c);
        float expect = scalar_ieee_minimum(scalar_ieee_maximum(av, bv), cv);
        for (std::size_t i = 0; i < W; ++i) {
          const float rv = r[i];
          EXPECT_EQ(bits_of(rv), bits_of(expect))
              << "av=" << av << " bv=" << bv << " cv=" << cv << " lane=" << i;
        }
      }
}

// --- Cubemap face ops (v_cube{id,ma,sc,tc}_f32) ------------------------------
//
// util::cube_{id,sc,tc}_f32_simd back the v_cube* gap ops; they must be
// bit-identical to the generated scalar bodies (transcribed verbatim below).
// Sweep every sign/magnitude/tie/zero/Inf/NaN triple.

float scalar_cubeid(float x, float y, float z) {
  float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
  if (ax >= ay && ax >= az)
    return x >= 0 ? 0.0f : 1.0f;
  if (ay >= ax && ay >= az)
    return y >= 0 ? 2.0f : 3.0f;
  return z >= 0 ? 4.0f : 5.0f;
}
float scalar_cubesc(float x, float y, float z) {
  float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
  if (ax >= ay && ax >= az)
    return x >= 0 ? z : -z;
  if (ay >= ax && ay >= az)
    return x;
  return z >= 0 ? -x : x;
}
float scalar_cubetc(float x, float y, float z) {
  float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
  if (ax >= ay && ax >= az)
    return -y;
  if (ay >= ax && ay >= az)
    return y >= 0 ? -z : z;
  return -y;
}
float scalar_cubema(float x, float y, float z) {
  float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
  return 2.0f * std::fmax(ax, std::fmax(ay, az));
}

template <typename SimdFn, typename ScalarFn> void check_cube(SimdFn simd_fn, ScalarFn scalar_fn) {
  using V = util::native<float>;
  constexpr std::size_t W = V::size();
  const std::array<float, 9> grid = {{-2.0f, -1.0f, -0.0f, 0.0f, 0.5f, 1.0f, 2.0f,
                                      std::numeric_limits<float>::infinity(),
                                      std::numeric_limits<float>::quiet_NaN()}};
  for (float zv : grid)
    for (float yv : grid)
      for (float xv : grid) {
        alignas(V) float xb[W], yb[W], zb[W];
        for (std::size_t i = 0; i < W; ++i) {
          xb[i] = xv;
          yb[i] = yv;
          zb[i] = zv;
        }
        V x(xb, util::stdx::vector_aligned), y(yb, util::stdx::vector_aligned),
            z(zb, util::stdx::vector_aligned);
        V r = simd_fn(x, y, z);
        const float expect = scalar_fn(xv, yv, zv);
        for (std::size_t i = 0; i < W; ++i) {
          const float rv = r[i];
          EXPECT_EQ(bits_of(rv), bits_of(expect))
              << "x=" << xv << " y=" << yv << " z=" << zv << " lane=" << i;
        }
      }
}

TEST(UtilSimd, CubeId_F32_BitExact) {
  SKIP_IF_NO_SIMD();
  check_cube([](auto x, auto y, auto z) { return util::cube_id_f32_simd(x, y, z); }, scalar_cubeid);
}
TEST(UtilSimd, CubeSc_F32_BitExact) {
  SKIP_IF_NO_SIMD();
  check_cube([](auto x, auto y, auto z) { return util::cube_sc_f32_simd(x, y, z); }, scalar_cubesc);
}
TEST(UtilSimd, CubeTc_F32_BitExact) {
  SKIP_IF_NO_SIMD();
  check_cube([](auto x, auto y, auto z) { return util::cube_tc_f32_simd(x, y, z); }, scalar_cubetc);
}
TEST(UtilSimd, CubeMa_F32_BitExact) {
  SKIP_IF_NO_SIMD();
  check_cube(
      [](auto x, auto y, auto z) {
        return 2.0f * util::stdx::fmax(util::stdx::abs(x),
                                       util::stdx::fmax(util::stdx::abs(y), util::stdx::abs(z)));
      },
      scalar_cubema);
}

// --- Normalized pack-convert lanes (v_cvt_pk[_]norm_{i16,u16}_*) -------------

int16_t scalar_cvt_pknorm_i16(float f) {
  if (std::isnan(f))
    return 0;
  return static_cast<int16_t>(std::clamp(f * 32767.0f, -32768.0f, 32767.0f));
}
uint16_t scalar_cvt_pknorm_u16(float f) {
  if (std::isnan(f))
    return 0;
  return static_cast<uint16_t>(std::clamp(f * 65535.0f, 0.0f, 65535.0f));
}

TEST(UtilSimd, CvtPkNormI16_F32_BitExact) {
  SKIP_IF_NO_SIMD();
  using V = util::native<float>;
  constexpr std::size_t W = V::size();
  const std::array<float, 14> grid = {{0.0f, -0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 0.99999f,
                                       1e30f, -1e30f, std::numeric_limits<float>::infinity(),
                                       -std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::quiet_NaN()}};
  for (float fv : grid) {
    alignas(V) float fb[W];
    for (std::size_t i = 0; i < W; ++i)
      fb[i] = fv;
    V f(fb, util::stdx::vector_aligned);
    auto ri = util::cvt_pknorm_i16_f32_simd(f);
    auto ru = util::cvt_pknorm_u16_f32_simd(f);
    for (std::size_t i = 0; i < W; ++i) {
      EXPECT_EQ(static_cast<uint16_t>(static_cast<int32_t>(ri[i])),
                static_cast<uint16_t>(scalar_cvt_pknorm_i16(fv)))
          << "i16 f=" << fv << " lane=" << i;
      EXPECT_EQ(static_cast<uint16_t>(ru[i]), scalar_cvt_pknorm_u16(fv)) << "u16 f=" << fv;
    }
  }
}

#undef SKIP_IF_NO_SIMD

} // namespace
