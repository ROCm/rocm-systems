// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hwfloat_mul_f32_test.cpp
/// @brief Integer binary32 multiply facts, measured cause mapping, and backends.

#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/mul_f32.h"

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <vector>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif
#if defined(__x86_64__) && defined(__GLIBC__)
#include <fpu_control.h>
#define RJ_TEST_HAS_X87_CONTROL 1
#endif
#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

using namespace rocjitsu::amdgpu::hwfloat;
namespace fact = mul_f32_fact;
namespace cause = mul_f32_cause;

constexpr MulF32Policy policy(uint32_t round, uint32_t denorm, bool quiet_nan = false) {
  return {.round = round, .denorm = denorm, .quiet_nan = quiet_nan};
}

constexpr bool produces(MulF32Result result, uint32_t bits, uint32_t facts) {
  return result.bits == bits && result.facts == facts;
}

constexpr uint32_t causes_of(uint32_t a, uint32_t b, MulF32Policy p) {
  return plain_mul_f32_causes(multiply_f32(a, b, p), p);
}

constexpr uint32_t kTiny = fact::kTinyBeforeRounding;
constexpr uint32_t kTinyAfter = fact::kTinyAfterRounding;
constexpr uint32_t kTasr = fact::kTinyAfterSignificandRounding;

// Exact and rounded normal products.
static_assert(produces(multiply_f32(0x3fc00000u, 0x3fc00000u, policy(0, 0)), 0x40100000u, 0));
static_assert(produces(multiply_f32(0x3f8ccccdu, 0x3f8ccccdu, policy(0, 0)), 0x3f9ae148u,
                       fact::kInexact));
static_assert(produces(multiply_f32(0x3f8ccccdu, 0x3f8ccccdu, policy(1, 0)), 0x3f9ae149u,
                       fact::kInexact));
static_assert(produces(multiply_f32(0x00000000u, 0xbf800000u, policy(0, 0)), 0x80000000u, 0));
static_assert(produces(multiply_f32(0x7f800000u, 0xc0000000u, policy(0, 0)), 0xff800000u, 0));

// Overflow is directed by the guest rounding mode and is always inexact.
static_assert(produces(multiply_f32(0x7f7fffffu, 0x40000000u, policy(0, 0)), 0x7f800000u,
                       fact::kOverflow | fact::kInexact));
static_assert(produces(multiply_f32(0x7f7fffffu, 0x40000000u, policy(3, 0)), 0x7f7fffffu,
                       fact::kOverflow | fact::kInexact));
static_assert(produces(multiply_f32(0xff7fffffu, 0x40000000u, policy(1, 0)), 0xff7fffffu,
                       fact::kOverflow | fact::kInexact));
static_assert(produces(multiply_f32(0xff7fffffu, 0x40000000u, policy(2, 0)), 0xff800000u,
                       fact::kOverflow | fact::kInexact));

// Invalid operations and NaN propagation. The first NaN source wins.
static_assert(produces(multiply_f32(0x00000000u, 0x7f800000u, policy(0, 3)), 0xffc00000u,
                       fact::kInvalid));
static_assert(produces(multiply_f32(0x7fa00000u, 0x3f800000u, policy(0, 0)), 0x7fa00000u,
                       fact::kInvalid));
static_assert(produces(multiply_f32(0x7fa00000u, 0x3f800000u, policy(0, 0, true)), 0x7fe00000u,
                       fact::kInvalid));
static_assert(produces(multiply_f32(0x7fc00001u, 0xffa00000u, policy(0, 0)), 0x7fc00001u,
                       fact::kInvalid));
static_assert(produces(multiply_f32(0x3f800000u, 0xff800001u, policy(0, 0, true)), 0xffc00001u,
                       fact::kInvalid));

// Subnormal sources: a flushed source can make zero times infinity invalid.
static_assert(produces(multiply_f32(0x00000001u, 0x7f800000u, policy(0, 0)), 0xffc00000u,
                       fact::kInputDenormal | fact::kInputFlushed | fact::kInvalid));
static_assert(produces(multiply_f32(0x00000001u, 0x7f800000u, policy(0, 3)), 0x7f800000u,
                       fact::kInputDenormal));
static_assert(produces(multiply_f32(0x00000001u, 0x3f800000u, policy(0, 3)), 0x00000001u,
                       fact::kInputDenormal | kTiny | kTinyAfter | kTasr));
static_assert(produces(multiply_f32(0x00000001u, 0x3f800000u, policy(0, 1)), 0x00000000u,
                       fact::kInputDenormal | kTiny | kTinyAfter | fact::kOutputFlushed | kTasr));
static_assert(produces(multiply_f32(0x00000001u, 0x3f800000u, policy(0, 0)), 0x00000000u,
                       fact::kInputDenormal | fact::kInputFlushed));
static_assert(produces(multiply_f32(0x00000001u, 0x3f800000u, policy(0, 2)), 0x00000000u,
                       fact::kInputDenormal | fact::kInputFlushed));

// (1 + 2^-23) * 2^-64 times (2 - 2^-22) * 2^-63 is (2 - 2^-45) * 2^-127:
// tiny before rounding, but its 24-bit rounding carries to 2^-126.
static_assert(produces(multiply_f32(0x1f800001u, 0x207ffffeu, policy(0, 0)), 0x00800000u,
                       fact::kInexact | kTiny));
static_assert(produces(multiply_f32(0x1f800001u, 0x207ffffeu, policy(0, 3)), 0x00800000u,
                       fact::kInexact | kTiny));
static_assert(produces(multiply_f32(0x1f800001u, 0x207ffffeu, policy(3, 0)), 0x00000000u,
                       fact::kInexact | kTiny | kTinyAfter | fact::kOutputFlushed | kTasr));
static_assert(produces(multiply_f32(0x1f800001u, 0x207ffffeu, policy(3, 3)), 0x007fffffu,
                       fact::kInexact | kTiny | kTinyAfter | kTasr));

// (2 - 2^-23) * 2^-64 times 2^-63 is exact in 24 bits below 2^-126, while
// subnormal precision rounds it up to the smallest normal value.
static_assert(produces(multiply_f32(0x1fffffffu, 0x20000000u, policy(0, 3)), 0x00800000u,
                       fact::kInexact | kTiny | kTasr));
static_assert(produces(multiply_f32(0x1fffffffu, 0x20000000u, policy(0, 0)), 0x00000000u,
                       kTiny | kTinyAfter | fact::kOutputFlushed | kTasr));
static_assert(produces(multiply_f32(0x9fffffffu, 0x20000000u, policy(2, 3)), 0x80800000u,
                       fact::kInexact | kTiny | kTasr));

// Measured plain-form causes.
static_assert(causes_of(0x3fc00000u, 0x3fc00000u, policy(0, 0)) == 0);
static_assert(causes_of(0x3f8ccccdu, 0x3f8ccccdu, policy(0, 0)) == cause::kInexact);
static_assert(causes_of(0x7f7fffffu, 0x40000000u, policy(3, 0)) ==
              (cause::kOverflow | cause::kInexact));
static_assert(causes_of(0x00000000u, 0x7f800000u, policy(0, 0)) == cause::kInvalid);
static_assert(causes_of(0x7fa00000u, 0x3f800000u, policy(0, 3)) == cause::kInvalid);
// Input denormal requires a preserved source and a non-NaN result.
static_assert(causes_of(0x00000001u, 0x7f800000u, policy(0, 3)) == cause::kInputDenormal);
static_assert(causes_of(0x00000001u, 0x7f800000u, policy(0, 0)) == cause::kInvalid);
static_assert(causes_of(0x00000001u, 0x7fc00000u, policy(0, 3)) == 0);
static_assert(causes_of(0x00000001u, 0x7fa00000u, policy(0, 3)) == cause::kInvalid);
static_assert(causes_of(0x00000001u, 0x3f800000u, policy(0, 3)) == cause::kInputDenormal);
static_assert(causes_of(0x00000001u, 0x3f800000u, policy(0, 0)) == 0);
// Output flushing is inexact and, with tininess, underflow.
static_assert(causes_of(0x00000001u, 0x3f800000u, policy(0, 1)) ==
              (cause::kInputDenormal | cause::kUnderflow | cause::kInexact));
// Underflow follows tininess after significand rounding, not before rounding.
static_assert(causes_of(0x1f800001u, 0x207ffffeu, policy(0, 0)) == cause::kInexact);
static_assert(causes_of(0x1f800001u, 0x207ffffeu, policy(0, 3)) == cause::kInexact);
static_assert(causes_of(0x1f800001u, 0x207ffffeu, policy(3, 3)) ==
              (cause::kUnderflow | cause::kInexact));
static_assert(causes_of(0x1fffffffu, 0x20000000u, policy(0, 3)) ==
              (cause::kUnderflow | cause::kInexact));
static_assert(causes_of(0x1fffffffu, 0x20000000u, policy(0, 0)) ==
              (cause::kUnderflow | cause::kInexact));

// Boundary inputs, including exponent sums on both sides of the AVX-512
// backend's normal-result range (128..380).
constexpr std::array<uint32_t, 40> kBoundaryValues = {
    0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x00400000u, 0x007fffffu, 0x00800000u,
    0x00800001u, 0x3f000000u, 0x3f7fffffu, 0x3f800000u, 0xbf800000u, 0x3f800001u, 0x3fffffffu,
    0x40000000u, 0x7f000000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u,
    0x7fa00000u, 0xffc00001u, 0xffa00001u, 0x1f800001u, 0x207ffffeu, 0x1fffffffu, 0x20000000u,
    0x9fffffffu, 0x3f8ccccdu, 0x3fc00000u, 0x5f800000u, 0x5f7fffffu, 0x1f000000u, 0x60000000u,
    0x0d800000u, 0x71800000u, 0x3effffffu, 0x00ffffffu, 0x3e800001u,
};

uint32_t next_random(uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return static_cast<uint32_t>(state >> 16);
}

/// Deterministic operand stream: boundary values, random bit patterns, and
/// random significands with exponents near the normal-result limits.
std::vector<uint32_t> operand_stream(size_t count, uint64_t seed) {
  std::vector<uint32_t> values(kBoundaryValues.begin(), kBoundaryValues.end());
  uint64_t state = seed;
  while (values.size() < count) {
    const uint32_t bits = next_random(state);
    switch (bits & 3) {
    case 0:
      values.push_back(bits);
      break;
    case 1:
      values.push_back((bits & 0x807fffffu) | ((next_random(state) % 24) << 23));
      break;
    case 2:
      values.push_back((bits & 0x807fffffu) | ((230 + next_random(state) % 25) << 23));
      break;
    default:
      values.push_back((bits & 0x807fffffu) | ((52 + next_random(state) % 24) << 23));
      break;
    }
  }
  return values;
}

std::vector<MulF32Policy> all_policies() {
  std::vector<MulF32Policy> policies;
  for (uint32_t round = 0; round < 4; ++round)
    for (uint32_t denorm = 0; denorm < 4; ++denorm)
      for (bool quiet_nan : {false, true})
        policies.push_back(policy(round, denorm, quiet_nan));
  return policies;
}

constexpr uint64_t kWaveMasks[] = {
    ~uint64_t{0},
    0xaaaa'aaaa'aaaa'aaaaull,
    uint64_t{1} << 33,
    0x8000'0002'0000'0001ull,
    0x0000'0000'ffff'0000ull,
};

constexpr uint32_t kUnwritten = 0xdeadbeefu;

struct WaveRun {
  std::array<uint32_t, kMulF32MaxLanes> result;
  uint32_t causes;
};

template <typename Backend>
WaveRun run_wave(Backend backend, const uint32_t *lhs, const uint32_t *rhs, uint64_t active,
                 uint32_t lanes, MulF32Policy p) {
  WaveRun run;
  run.result.fill(kUnwritten);
  run.causes = backend(lhs, rhs, run.result.data(), active, lanes, p);
  return run;
}

TEST(HwfloatMulF32Test, WaveBackendsMatchPerLaneModel) {
  RecordProperty("avx512f", mul_f32_avx512_available() ? "available" : "unavailable");
  const std::vector<uint32_t> lhs_values = operand_stream(2048, 0x9e3779b97f4a7c15ull);
  const std::vector<uint32_t> rhs_values = operand_stream(2048, 0xd1b54a32d192ed03ull);
  for (const MulF32Policy p : all_policies()) {
    for (uint32_t lanes : {32u, 64u}) {
      for (uint64_t mask : kWaveMasks) {
        const uint64_t active = lanes == 64 ? mask : mask & 0xffff'ffffull;
        for (size_t base = 0; base + lanes <= lhs_values.size(); base += lanes) {
          const uint32_t *lhs = lhs_values.data() + base;
          // Rotate one side so boundary values meet each other.
          const uint32_t *rhs = rhs_values.data() + (base * 7 % (rhs_values.size() - lanes));
          SCOPED_TRACE(testing::Message()
                       << "round " << p.round << " denorm " << p.denorm << " quiet " << p.quiet_nan
                       << " lanes " << lanes << " mask " << std::hex << active << " base " << base);
          uint32_t expected_causes = 0;
          std::array<uint32_t, kMulF32MaxLanes> expected;
          expected.fill(kUnwritten);
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            if (!(active & (uint64_t{1} << lane)))
              continue;
            const MulF32Result r = multiply_f32(lhs[lane], rhs[lane], p);
            expected[lane] = r.bits;
            expected_causes |= plain_mul_f32_causes(r, p);
          }
          const WaveRun integer = run_wave(multiply_f32_wave_integer, lhs, rhs, active, lanes, p);
          const WaveRun avx512 = run_wave(multiply_f32_wave_avx512, lhs, rhs, active, lanes, p);
          const WaveRun dispatched = run_wave(multiply_f32_wave, lhs, rhs, active, lanes, p);
          EXPECT_EQ(integer.causes, expected_causes);
          EXPECT_EQ(avx512.causes, expected_causes);
          EXPECT_EQ(dispatched.causes, expected_causes);
          for (uint32_t lane = 0; lane < kMulF32MaxLanes; ++lane) {
            EXPECT_EQ(integer.result[lane], expected[lane]) << "lane " << lane;
            EXPECT_EQ(avx512.result[lane], expected[lane]) << "lane " << lane;
            EXPECT_EQ(dispatched.result[lane], expected[lane]) << "lane " << lane;
          }
          if (testing::Test::HasFailure())
            return;
        }
      }
    }
  }
}

TEST(HwfloatMulF32Test, InactiveExceptionalLanesDoNotContribute) {
  std::array<uint32_t, 64> lhs;
  std::array<uint32_t, 64> rhs;
  lhs.fill(0x7fa00000u); // signaling NaN
  rhs.fill(0x00000000u);
  lhs[33] = 0x3fc00000u;
  rhs[33] = 0x3fc00000u;
  for (const MulF32Policy p : all_policies()) {
    for (auto backend : {multiply_f32_wave_integer, multiply_f32_wave_avx512, multiply_f32_wave}) {
      const WaveRun run = run_wave(backend, lhs.data(), rhs.data(), uint64_t{1} << 33, 64, p);
      EXPECT_EQ(run.causes, 0u);
      EXPECT_EQ(run.result[33], 0x40100000u);
      EXPECT_EQ(run.result[32], kUnwritten);
      EXPECT_EQ(run.result[34], kUnwritten);
    }
  }
}

// Host floating-point state that a host-FP implementation would read or change.
// On x86-64 the exception flags cover both the x87 status word and MXCSR.
struct HostFpState {
  int rounding = 0;
  int flags = 0;
  int error_number = 0;
  uint32_t mxcsr = 0;
  uint32_t x87_control = 0;

  static HostFpState capture() {
    HostFpState state;
    state.error_number = errno;
    state.rounding = std::fegetround();
    state.flags = std::fetestexcept(FE_ALL_EXCEPT);
#if defined(__x86_64__)
    state.mxcsr = _mm_getcsr();
#endif
#if defined(RJ_TEST_HAS_X87_CONTROL)
    fpu_control_t control = 0;
    _FPU_GETCW(control);
    state.x87_control = control;
#endif
    return state;
  }

  bool operator==(const HostFpState &) const = default;
};

/// Runs every backend under upward host rounding, preset sticky flags, DAZ,
/// FTZ, single x87 precision, unmasked traps, and a sentinel errno. Returns
/// zero when all results and causes equal the default-environment reference
/// and the host state is unchanged.
int run_under_hostile_host_state() {
  const std::vector<uint32_t> lhs = operand_stream(1024, 0x2545f4914f6cdd1dull);
  const std::vector<uint32_t> rhs = operand_stream(1024, 0x9e3779b97f4a7c15ull);
  const std::vector<MulF32Policy> policies = all_policies();
  std::vector<WaveRun> reference;
  for (const MulF32Policy p : policies)
    for (size_t base = 0; base + 64 <= lhs.size(); base += 64)
      reference.push_back(run_wave(multiply_f32_wave_integer, lhs.data() + base, rhs.data() + base,
                                   ~uint64_t{0}, 64, p));
  std::vector<WaveRun> observed;
  observed.reserve(reference.size() * 3);

  if (std::fesetround(FE_UPWARD) != 0)
    return 10;
  std::feclearexcept(FE_ALL_EXCEPT);
  std::feraiseexcept(FE_INEXACT);
#if defined(__GLIBC__)
  if (feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW) == -1)
    return 11;
#endif
#if defined(__x86_64__)
  constexpr uint32_t kMxcsrDaz = 1u << 6;
  constexpr uint32_t kMxcsrFtz = 1u << 15;
  _mm_setcsr(_mm_getcsr() | kMxcsrDaz | kMxcsrFtz);
#endif
#if defined(RJ_TEST_HAS_X87_CONTROL)
  fpu_control_t control = 0;
  _FPU_GETCW(control);
  control = (control & ~static_cast<fpu_control_t>(_FPU_EXTENDED)) | _FPU_SINGLE;
  _FPU_SETCW(control);
#endif
  errno = EDOM;
  const HostFpState before = HostFpState::capture();

  for (const MulF32Policy p : policies) {
    for (size_t base = 0; base + 64 <= lhs.size(); base += 64) {
      for (auto backend :
           {multiply_f32_wave_integer, multiply_f32_wave_avx512, multiply_f32_wave}) {
        observed.push_back(
            run_wave(backend, lhs.data() + base, rhs.data() + base, ~uint64_t{0}, 64, p));
        (void)mul_f32_avx512_available();
      }
    }
  }

  const HostFpState after = HostFpState::capture();
  if (!(after == before))
    return 12;
  for (size_t i = 0; i < observed.size(); ++i) {
    const WaveRun &expected = reference[i / 3];
    if (observed[i].causes != expected.causes || observed[i].result != expected.result)
      return 13;
  }
  return 0;
}

#if GTEST_HAS_DEATH_TEST && defined(__linux__)
TEST(HwfloatMulF32DeathTest, HostFloatingPointStateIsNeitherReadNorChanged) {
  // Unmasked host traps turn any host FP exception into SIGFPE, so the check
  // runs in a child process.
  EXPECT_EXIT(_exit(run_under_hostile_host_state()), testing::ExitedWithCode(0), "");
}
#endif

} // namespace
