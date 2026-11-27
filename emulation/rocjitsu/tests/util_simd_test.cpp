// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Check raw SIMD register transport, lane-mask bridges and the VM dispatch
// override. Numerical instruction behavior is tested through rj-arith.

#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include "util/host_target.h"

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {

// Each test skips at runtime when `<experimental/simd>` is unavailable;
// the SIMD body is type-checked but compiles out via `if constexpr`.
#define SKIP_IF_NO_SIMD()                                                                          \
  if constexpr (!util::has_stdx_simd) {                                                            \
    GTEST_SKIP() << "<experimental/simd> unavailable; util SIMD helpers skipped.";                 \
    return;                                                                                        \
  }

constexpr std::size_t kW = util::native_width_v<uint32_t>;

struct FallbackMask {
  static constexpr std::size_t kSize = 5;
  explicit FallbackMask(bool value) { lanes.fill(value); }
  static constexpr std::size_t size() { return kSize; }
  bool operator[](std::size_t i) const { return lanes[i]; }
  bool &operator[](std::size_t i) { return lanes[i]; }

  std::array<bool, kSize> lanes{};
};

struct FallbackSimd {
  using mask_type = FallbackMask;
  static constexpr std::size_t size() { return mask_type::size(); }
};

TEST(UtilSimd, Native64MaskWorkaroundMacroIsBooleanOverridePoint) {
  constexpr int workaround = RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS;
  constexpr int user_override = RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS_USER_OVERRIDE;
  static_assert(workaround == 0 || workaround == 1);
  static_assert(user_override == 0 || user_override == 1);

  if constexpr (!user_override) {
    // Keep the expected policy independent of host_target.h. This reference
    // checks detection; it does not select a production implementation.
#if defined(__clang__) && defined(__GLIBCXX__) && defined(__AVX512F__)
    constexpr int kExpectedDefaultMaskFallback = 1;
#else
    constexpr int kExpectedDefaultMaskFallback = 0;
#endif
    EXPECT_EQ(workaround, kExpectedDefaultMaskFallback);
  }
}

TEST(UtilSimd, Load_Contiguous_U32) {
  SKIP_IF_NO_SIMD();
  alignas(util::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    src[i] = static_cast<uint32_t>(0xDEAD'0000u + i);
  }
  auto v = util::load<uint32_t>(src);
  for (std::size_t i = 0; i < kW; ++i) {
    EXPECT_EQ(v[i], src[i]) << "lane " << i;
  }
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
  for (std::size_t i = 0; i < kW; ++i) {
    EXPECT_EQ(v[i], expected[i]) << "lane " << i;
  }
}

TEST(UtilSimd, Broadcast_U32) {
  SKIP_IF_NO_SIMD();
  constexpr uint32_t kBits = 0xCAFEBABEu;
  auto v = util::broadcast<uint32_t>(kBits);
  for (std::size_t i = 0; i < kW; ++i) {
    EXPECT_EQ(v[i], kBits) << "lane " << i;
  }
}

TEST(UtilSimd, Broadcast_F32) {
  SKIP_IF_NO_SIMD();
  constexpr float kVal = -3.14159f;
  const uint32_t bits = std::bit_cast<uint32_t>(kVal);
  auto v = util::broadcast<float>(bits);
  for (std::size_t i = 0; i < kW; ++i) {
    EXPECT_EQ(v[i], kVal) << "lane " << i;
  }
}

TEST(UtilSimd, MaskBitBridgeRoundTripsEveryPattern) {
  SKIP_IF_NO_SIMD();
#if RJ_HOST_HAS_STDX_SIMD
  const auto check = []<typename Simd>() {
    constexpr std::size_t W = Simd::size();
    ASSERT_LE(W, 16u) << "exhaustive mask sweep is intentionally bounded";
    const uint64_t full = util::mask<uint64_t>(static_cast<int>(W));
    for (uint64_t bits = 0; bits <= full; ++bits) {
      const auto mask = util::simd_mask_from_bits<Simd>(bits);
      for (std::size_t i = 0; i < W; ++i) {
        ASSERT_EQ(static_cast<bool>(mask[i]), ((bits >> i) & 1u) != 0)
            << "width " << W << ", bits " << bits << ", lane " << i;
      }
      ASSERT_EQ(util::simd_mask_to_bits(mask), bits) << "width " << W << ", bits " << bits;
    }
    EXPECT_EQ(util::simd_mask_to_bits(util::simd_mask_from_bits<Simd>(~uint64_t{0})), full)
        << "input bits above the SIMD width must be ignored";
    EXPECT_EQ(util::simd_mask_to_bits(util::simd_mask_from_bits<Simd>(~full)), 0u)
        << "high-only input bits must not set SIMD lanes";
  };

  check.template operator()<util::native<uint32_t>>();
#if !RJ_SIMD_BROKEN_NATIVE_64BIT_MASKS
  check.template operator()<util::native<uint64_t>>();
  check.template operator()<util::narrow32<uint32_t>>();
#endif
#endif
}

TEST(UtilSimd, MaskBitBridgePortableFallbackRoundTripsEveryPattern) {
  SKIP_IF_NO_SIMD();
#if RJ_HOST_HAS_STDX_SIMD
  const uint64_t full = util::mask<uint64_t>(FallbackSimd::size());
  for (uint64_t bits = 0; bits <= full; ++bits) {
    const auto mask = util::simd_mask_from_bits<FallbackSimd>(bits);
    for (std::size_t i = 0; i < FallbackSimd::size(); ++i) {
      ASSERT_EQ(mask[i], ((bits >> i) & 1u) != 0) << "bits " << bits << ", lane " << i;
    }
    ASSERT_EQ(util::simd_mask_to_bits(mask), bits);
  }
  EXPECT_EQ(util::simd_mask_to_bits(util::simd_mask_from_bits<FallbackSimd>(~uint64_t{0})), full);
  EXPECT_EQ(util::simd_mask_to_bits(util::simd_mask_from_bits<FallbackSimd>(~full)), 0u);
#endif
}

TEST(UtilSimd, MaskedStore_FullMask) {
  SKIP_IF_NO_SIMD();
  alignas(util::native<uint32_t>) uint32_t dst[kW] = {};
  alignas(util::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    src[i] = 0xA5A5'0000u + i;
  }
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
  for (std::size_t i = 1; i < kW; i += 2) {
    mask |= (1ULL << i);
  }
  util::masked_store<uint32_t>(dst, v, mask);
  for (std::size_t i = 0; i < kW; ++i) {
    if (mask & (1ULL << i)) {
      EXPECT_EQ(dst[i], src[i]) << "lane " << i << " should be written";
    } else {
      EXPECT_EQ(dst[i], 0x1111'1111u) << "lane " << i << " should be preserved";
    }
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
  for (std::size_t i = 0; i < kW; ++i) {
    EXPECT_EQ(dst[i], 0xDEAD'BEEFu) << "lane " << i;
  }
}

TEST(UtilSimd, BlitToBuffer_F32) {
  SKIP_IF_NO_SIMD();
  alignas(util::native<float>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    src[i] = std::bit_cast<uint32_t>(2.0f * static_cast<float>(i) + 0.5f);
  }
  auto v = util::load<float>(src);
  alignas(util::native<float>) uint32_t buf[util::native<float>::size()] = {};
  util::blit_to_buffer<float>(buf, v);
  for (std::size_t i = 0; i < kW; ++i) {
    EXPECT_EQ(buf[i], src[i]) << "lane " << i;
  }
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
  for (std::size_t i = 0; i < W; ++i) {
    EXPECT_EQ(v[i], expected[i]) << "lane " << i;
  }
}

TEST(UtilSimd, BroadcastNarrow_I32) {
  SKIP_IF_NO_SIMD();
  constexpr int32_t kVal = -123456789;
  auto v = util::broadcast_narrow<int32_t>(std::bit_cast<uint32_t>(kVal));
  for (std::size_t i = 0; i < util::native_width64; ++i) {
    EXPECT_EQ(v[i], kVal) << "lane " << i;
  }
}

TEST(UtilSimd, MaskedStoreNarrow_FullAndPartial) {
  SKIP_IF_NO_SIMD();
  constexpr std::size_t W = util::native_width64;
  alignas(util::narrow32<int32_t>) int32_t vals[W];
  for (std::size_t i = 0; i < W; ++i) {
    vals[i] = static_cast<int32_t>(i) - 3;
  }
  util::narrow32<int32_t> v(vals, util::stdx::vector_aligned);

  uint32_t dst_full[W] = {};
  util::masked_store_narrow<int32_t>(dst_full, v, util::mask<uint64_t>(static_cast<int>(W)));
  for (std::size_t i = 0; i < W; ++i) {
    EXPECT_EQ(dst_full[i], std::bit_cast<uint32_t>(vals[i])) << "full lane " << i;
  }

  uint32_t dst_part[W];
  for (std::size_t i = 0; i < W; ++i) {
    dst_part[i] = 0x1111'1111u;
  }
  uint64_t mask = 0;
  for (std::size_t i = 1; i < W; i += 2) {
    mask |= (1ULL << i);
  }
  util::masked_store_narrow<int32_t>(dst_part, v, mask);
  for (std::size_t i = 0; i < W; ++i) {
    if (mask & (1ULL << i)) {
      EXPECT_EQ(dst_part[i], std::bit_cast<uint32_t>(vals[i])) << "part write lane " << i;
    } else {
      EXPECT_EQ(dst_part[i], 0x1111'1111u) << "part preserve lane " << i;
    }
  }
}

TEST(UtilSimd, NarrowBridgeCast_DoubleToFromB32) {
  SKIP_IF_NO_SIMD();
  constexpr std::size_t W = util::native_width64;
  // f64 -> f32: a direct static_simd_cast between native<double> and
  // narrow32<float> is the cvt_f32_f64 bridge; matches a per-lane (float)cast.
  alignas(util::native<double>) double in[W];
  for (std::size_t i = 0; i < W; ++i) {
    in[i] = 0.1 * static_cast<double>(i) - 1.3 + 1e30;
  }
  util::native<double> d(in, util::stdx::vector_aligned);
  auto f = util::stdx::static_simd_cast<util::narrow32<float>>(d);
  for (std::size_t i = 0; i < W; ++i) {
    EXPECT_EQ(f[i], static_cast<float>(in[i])) << "f64->f32 lane " << i;
  }
  // i32 -> f64: the cvt_f64_i32 bridge (exact widening).
  alignas(util::narrow32<int32_t>) int32_t iv[W];
  for (std::size_t i = 0; i < W; ++i) {
    iv[i] = static_cast<int32_t>(i) * -1000003;
  }
  util::narrow32<int32_t> ni(iv, util::stdx::vector_aligned);
  auto dd = util::stdx::static_simd_cast<util::native<double>>(ni);
  for (std::size_t i = 0; i < W; ++i) {
    EXPECT_EQ(dd[i], static_cast<double>(iv[i])) << "i32->f64 lane " << i;
  }
}

TEST(UtilSimd, ForceScalar_StableAcrossThreadsWithinImage) {
  // force_scalar() is seeded once from RJ_FORCE_SCALAR at image load. Absent
  // the test-only setter (util/simd_test_hooks.h), it is stable across calls
  // and identical on every thread; this test exercises that steady-state
  // behaviour without flipping the gate.
  //
  // Scope note: the gate has local binding per shared object (see
  // util::detail::g_force_scalar), so this only covers the copy linked into
  // this test executable. It deliberately does NOT assert process-wide
  // behaviour -- a same-image thread check cannot observe the DSO boundary,
  // where each image carries its own copy.
  const bool v = util::force_scalar();
  EXPECT_EQ(util::force_scalar(), v) << "force_scalar() must be stable within an image";

  std::atomic<bool> other{!v};
  std::thread t([&]() { other.store(util::force_scalar()); });
  t.join();
  EXPECT_EQ(other.load(), v) << "force_scalar() must be identical on all threads in this image";
}

// The documented contract for RJ_FORCE_SCALAR, driven through the parser with
// literal expectations. Restating the parser to compute the expected value
// cannot fail, so these cases name the answers instead.
TEST(UtilSimd, ForceScalar_EnvContract) {
  struct EnvCase {
    const char *value; // nullptr means unset.
    bool expected;
  };
  static constexpr EnvCase kCases[] = {
      {nullptr, false}, {"", false}, {"0", false}, {"1", true}, {"00", true}, {"false", true},
  };

  const char *saved = std::getenv("RJ_FORCE_SCALAR");
  const bool was_set = saved != nullptr;
  const std::string saved_value = was_set ? saved : std::string();

  for (const EnvCase &test_case : kCases) {
    SCOPED_TRACE(test_case.value ? test_case.value : "<unset>");
    if (test_case.value == nullptr) {
      ASSERT_EQ(::unsetenv("RJ_FORCE_SCALAR"), 0);
    } else {
      ASSERT_EQ(::setenv("RJ_FORCE_SCALAR", test_case.value, 1), 0);
    }
    EXPECT_EQ(util::detail::init_force_scalar(), test_case.expected);
  }

  if (was_set) {
    ASSERT_EQ(::setenv("RJ_FORCE_SCALAR", saved_value.c_str(), 1), 0);
  } else {
    ASSERT_EQ(::unsetenv("RJ_FORCE_SCALAR"), 0);
  }
}

// Positive control for the test seam. Without this, a setter that did nothing
// would leave every scalar/SIMD comparison in this suite passing, because both
// halves of each comparison would run the same path.
TEST(UtilSimd, ForceScalar_SeamActuallyMovesTheGate) {
  const bool original = util::force_scalar();

  util::set_force_scalar_for_testing(!original);
  EXPECT_EQ(util::force_scalar(), !original) << "the seam must move the gate";

  util::set_force_scalar_for_testing(original);
  EXPECT_EQ(util::force_scalar(), original) << "the seam must restore the gate";
}

// The gate is an inline variable with hidden visibility, so a module the process
// loads carries its own copy and the seam cannot reach it. This is the scope the
// headers document; pinning it here keeps a later test from flipping the gate and
// expecting a loaded module to follow, which would silently assert nothing.
TEST(UtilSimd, ForceScalarOverrideDoesNotReachADlopenedModule) {
  void *module = ::dlopen(RJ_FORCE_SCALAR_PROBE_MODULE, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(module, nullptr) << ::dlerror();
  auto *probe = reinterpret_cast<bool (*)()>(::dlsym(module, "rj_probe_force_scalar"));
  ASSERT_NE(probe, nullptr) << ::dlerror();

  const bool original = util::force_scalar();
  const bool module_before = probe();

  util::set_force_scalar_for_testing(!original);
  EXPECT_EQ(util::force_scalar(), !original) << "the host gate must have moved";
  EXPECT_EQ(probe(), module_before) << "a loaded module must keep its own gate";

  util::set_force_scalar_for_testing(original);
  ::dlclose(module);
}

#undef SKIP_IF_NO_SIMD

} // namespace
