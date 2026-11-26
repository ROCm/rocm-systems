// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Check raw register transport, masked stores and the VM dispatch override.
// Numerical instruction behavior is tested through rj-arith.

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

constexpr std::size_t kW = util::register_block_lanes_v<uint32_t>;

TEST(UtilSimd, Load_Contiguous_U32) {
  alignas(util::RegisterBlock<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    src[i] = static_cast<uint32_t>(0xDEAD'0000u + i);
  }
  auto v = util::load<uint32_t>(src);
  for (std::size_t i = 0; i < kW; ++i) {
    EXPECT_EQ(v[i], src[i]) << "lane " << i;
  }
}

TEST(UtilSimd, Load_Contiguous_F32) {
  alignas(util::RegisterBlock<float>) uint32_t src[kW];
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
  constexpr uint32_t kBits = 0xCAFEBABEu;
  auto v = util::broadcast<uint32_t>(kBits);
  for (std::size_t i = 0; i < kW; ++i) {
    EXPECT_EQ(v[i], kBits) << "lane " << i;
  }
}

TEST(UtilSimd, Broadcast_F32) {
  constexpr float kVal = -3.14159f;
  const uint32_t bits = std::bit_cast<uint32_t>(kVal);
  auto v = util::broadcast<float>(bits);
  for (std::size_t i = 0; i < kW; ++i) {
    EXPECT_EQ(v[i], kVal) << "lane " << i;
  }
}

TEST(UtilSimd, MaskedStore_FullMask) {
  alignas(util::RegisterBlock<uint32_t>) uint32_t dst[kW] = {};
  alignas(util::RegisterBlock<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    src[i] = 0xA5A5'0000u + i;
  }
  auto v = util::load<uint32_t>(src);
  const uint64_t full = util::mask<uint64_t>(static_cast<int>(kW));
  util::masked_store<uint32_t>(dst, v, full);
  EXPECT_EQ(0, std::memcmp(dst, src, sizeof(dst)));
}

TEST(UtilSimd, MaskedStore_PartialMask) {
  alignas(util::RegisterBlock<uint32_t>) uint32_t dst[kW];
  alignas(util::RegisterBlock<uint32_t>) uint32_t src[kW];
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
  alignas(util::RegisterBlock<uint32_t>) uint32_t dst[kW];
  alignas(util::RegisterBlock<uint32_t>) uint32_t src[kW];
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
  alignas(util::RegisterBlock<float>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    src[i] = std::bit_cast<uint32_t>(2.0f * static_cast<float>(i) + 0.5f);
  }
  auto v = util::load<float>(src);
  alignas(util::RegisterBlock<float>) uint32_t buf[util::RegisterBlock<float>::size()] = {};
  util::blit_to_buffer<float>(buf, v);
  for (std::size_t i = 0; i < kW; ++i) {
    EXPECT_EQ(buf[i], src[i]) << "lane " << i;
  }
}

// Transport must preserve payloads and signaling bits without FP arithmetic.
TEST(UtilSimd, RawFloatingEncodings) {
  const std::array<uint32_t, 8> patterns = {0x7f812345u, 0xffc54321u, 0x80000000u, 0x7f800000u,
                                            1u,          0x7f7fffffu, 0xff800000u, 0u};
  std::array<uint32_t, kW> words;
  for (std::size_t lane = 0; lane < kW; ++lane) {
    words[lane] = patterns[lane % patterns.size()] ^ uint32_t(lane / patterns.size()) * 0x1234567u;
  }
  const uint64_t full_mask = util::mask<uint64_t>(kW);
  std::array<uint32_t, kW> output;
  // All lane masks, including ignored high bits, preserve the other words.
  for (uint64_t exec_mask = 0; exec_mask <= full_mask; ++exec_mask) {
    output.fill(0xdeadbeefu);
    util::masked_store(output.data(), util::load<float>(words.data()), exec_mask | ~full_mask);
    for (std::size_t lane = 0; lane < kW; ++lane) {
      EXPECT_EQ(output[lane], ((exec_mask >> lane) & 1) ? words[lane] : 0xdeadbeefu)
          << "exec_mask=" << exec_mask << " lane=" << lane;
    }
  }
  for (uint32_t word : words) {
    uint32_t broadcast[kW];
    util::blit_to_buffer<float>(broadcast, util::broadcast<float>(word));
    for (uint32_t value : broadcast) {
      EXPECT_EQ(value, word);
    }
  }

  const std::array<uint32_t, util::register_block_lanes64> low = {0x12345678u, 0u, 1u, 0u};
  const std::array<uint32_t, util::register_block_lanes64> high = {0x7ff00000u, 0x80000000u, 0u,
                                                                   0xfff00000u};
  std::array<uint32_t, util::register_block_lanes64> out_low, out_high;
  const uint64_t full_mask64 = util::mask<uint64_t>(low.size());
  for (uint64_t exec_mask = 0; exec_mask <= full_mask64; ++exec_mask) {
    out_low.fill(0xdeadbeefu);
    out_high.fill(0xdeadbeefu);
    util::masked_store64(out_low.data(), out_high.data(),
                         util::load64<double>(low.data(), high.data()), exec_mask | ~full_mask64);
    for (std::size_t lane = 0; lane < low.size(); ++lane) {
      const bool active = (exec_mask >> lane) & 1;
      EXPECT_EQ(out_low[lane], active ? low[lane] : 0xdeadbeefu);
      EXPECT_EQ(out_high[lane], active ? high[lane] : 0xdeadbeefu);
    }
  }
  // A repeated register destination retains the high word of each active lane.
  out_low.fill(0xdeadbeefu);
  util::masked_store64(out_low.data(), out_low.data(),
                       util::load64<double>(low.data(), high.data()), 0x5);
  std::array<uint32_t, util::register_block_lanes64> expected;
  expected.fill(0xdeadbeefu);
  expected[0] = high[0];
  expected[2] = high[2];
  EXPECT_EQ(out_low, expected);
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

} // namespace
