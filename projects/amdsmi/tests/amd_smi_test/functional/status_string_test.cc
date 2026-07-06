/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Driver-free unit test for amdsmi status-code handling: no GPU, no
// amdsmi_init().

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_common.h"

namespace {

// ---------------------------------------------------------------------------
// Compile-time enum reflection (stopgap until C++26 static reflection / P2996).
// C++17 cannot enumerate enum members, so this uses the magic_enum trick:
// instantiate a template on a value and parse its name from __PRETTY_FUNCTION__.
// Real enumerators render as names, whereas unused values render as numbers or casts.
//
// Normal status codes are discovered by scanning a low numeric range. The two
// high-value codes, AMDSMI_STATUS_MAP_ERROR and AMDSMI_STATUS_UNKNOWN_ERROR, are
// appended explicitly because their values are near UINT_MAX.
// ---------------------------------------------------------------------------

// Reflection needs enumerator names in __PRETTY_FUNCTION__. Clang does this at any version,
// whereas GCC does it starting in GCC 9. When an older GCC is detected,
// the test automatically skips the reflection-based test.
#if defined(__clang__)
#define AMDSMI_ENUM_REFLECTION_AVAILABLE 1
#elif defined(__GNUC__) && (__GNUC__ >= 9)
#define AMDSMI_ENUM_REFLECTION_AVAILABLE 1
#else
#define AMDSMI_ENUM_REFLECTION_AVAILABLE 0
#endif

namespace enum_reflect {

constexpr bool kAvailable = AMDSMI_ENUM_REFLECTION_AVAILABLE;

// Compiler name and version, included in failure/skip diagnostics.
inline const char* CompilerId() {
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#else
  return "unknown compiler";
#endif
}

#if AMDSMI_ENUM_REFLECTION_AVAILABLE

template <amdsmi_status_t V>
constexpr const char* Signature() {
  // Return type is a plain `const char*` so the signature has no stray "= "
  // from typedef expansion to confuse the parse below.
  return __PRETTY_FUNCTION__;
}

// Enumerator name for V, e.g. "AMDSMI_STATUS_TIMEOUT".
template <amdsmi_status_t V>
constexpr std::string_view EnumName() {
  std::string_view s = Signature<V>();
  const auto start = s.find("V = ") + 4;  // anchor on the value parameter
  const auto end = s.find_first_of(";]", start);
  return s.substr(start, end - start);
}

// Real enumerators start like C identifiers. Unused values start with '(' on
// GCC or a digit on Clang, so this rejects both renderings.
template <amdsmi_status_t V>
constexpr bool IsEnumerator() {
  const auto name = EnumName<V>();
  if (name.empty()) return false;
  const char c = name.front();
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

template <uint32_t... Is>
void Collect(std::vector<std::pair<amdsmi_status_t, std::string>>& out,
             std::integer_sequence<uint32_t, Is...>) {
  ((IsEnumerator<static_cast<amdsmi_status_t>(Is)>()
        ? out.push_back({static_cast<amdsmi_status_t>(Is),
                         std::string(EnumName<static_cast<amdsmi_status_t>(Is)>())})
        : void()),
   ...);
}

// Scan the full range status codes can occupy (normal codes stay well below
// this; the two high-value codes near UINT_MAX are appended separately).
constexpr uint32_t kScanLimit = 256;

// {value, "AMDSMI_STATUS_..."} for every amdsmi_status_t enumerator.
std::vector<std::pair<amdsmi_status_t, std::string>> AllStatusCodes() {
  std::vector<std::pair<amdsmi_status_t, std::string>> out;
  Collect(out, std::make_integer_sequence<uint32_t, kScanLimit>{});
  // Append the two high-value codes the scan cannot reach.
  out.push_back({AMDSMI_STATUS_MAP_ERROR, std::string(EnumName<AMDSMI_STATUS_MAP_ERROR>())});
  out.push_back(
      {AMDSMI_STATUS_UNKNOWN_ERROR, std::string(EnumName<AMDSMI_STATUS_UNKNOWN_ERROR>())});
  return out;
}

// Include a raw signature in failures so compiler-format changes are visible.
inline const char* SampleSignature() { return Signature<AMDSMI_STATUS_TIMEOUT>(); }

#else  // !AMDSMI_ENUM_REFLECTION_AVAILABLE

// Unsupported compiler: no codes discovered; the positive test skips itself.
std::vector<std::pair<amdsmi_status_t, std::string>> AllStatusCodes() { return {}; }
inline const char* SampleSignature() { return "<enum reflection unavailable>"; }

#endif  // AMDSMI_ENUM_REFLECTION_AVAILABLE

}  // namespace enum_reflect

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

// The Python unit test (test_status_code_to_string) covers the same ground, but
// it needs an initialized library and drivers; this C++ test is driver-free and
// runs in CI without a GPU.
TEST(AmdSmiStatusStringTest, EveryStatusCodeResolvesToItsOwnName) {
  if (!enum_reflect::kAvailable) {
    GTEST_SKIP() << "enum reflection unavailable on this compiler (" << enum_reflect::CompilerId()
                 << "). This test needs Clang or GCC >= 9 "
                 << "(older GCC renders enum template args as numeric casts). "
                 << "The matching Python test covers this case elsewhere.";
  }

  const auto all_status_codes = enum_reflect::AllStatusCodes();

  // Guard against an empty or incomplete reflection result: if these known
  // codes aren't discovered, the __PRETTY_FUNCTION__ parser or kScanLimit broke.
  // The floor is a loose lower bound (well below the current enumerator count)
  // to catch a broken parser without needing an update for every new code.
  ASSERT_GE(all_status_codes.size(), 40u)
      << "enum reflection discovered only " << all_status_codes.size()
      << " status codes; the __PRETTY_FUNCTION__ parser or kScanLimit is broken. "
      << "Compiler: " << enum_reflect::CompilerId() << "; sample signature: \""
      << enum_reflect::SampleSignature() << "\".";

  auto contains = [&](amdsmi_status_t code) {
    for (const auto& [value, name] : all_status_codes) {
      if (value == code) return true;
    }
    return false;
  };
  for (amdsmi_status_t code : {AMDSMI_STATUS_TIMEOUT, AMDSMI_STATUS_MORE_DATA,
                               AMDSMI_STATUS_MAP_ERROR, AMDSMI_STATUS_UNKNOWN_ERROR}) {
    ASSERT_TRUE(contains(code))
        << "enum reflection did not discover known status code " << code
        << "; the __PRETTY_FUNCTION__ parser in enum_reflect likely broke due to "
        << "a compiler change (or kScanLimit is too low). Compiler: " << enum_reflect::CompilerId()
        << "; sample signature: \"" << enum_reflect::SampleSignature()
        << "\". This is a test-harness "
        << "problem, not an amdsmi library bug.";
  }

  for (const auto& [code, name] : all_status_codes) {
    const char* status_string = nullptr;
    amdsmi_status_t ret = amdsmi_status_code_to_string(code, &status_string);

    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS)
        << name << " (" << code << ") did not resolve, this status is missing from "
        << "amdsmi_status_code_to_string().";
    ASSERT_NE(status_string, nullptr) << name;

    const std::string description(status_string);
    EXPECT_TRUE(StartsWith(description, name))
        << name << " (" << code << ") resolved to \"" << description
        << "\". Expected the description to start with its own enum name. A "
        << "missing case falls through to the UNKNOWN_ERROR default.";
  }
}

// Reflection-free spot check for the regressed codes + a few other common returns.
TEST(AmdSmiStatusStringTest, KnownStatusCodesResolveToTheirOwnName) {
  const std::vector<std::pair<amdsmi_status_t, const char*>> spot_checks = {
      // The two codes whose missing cases prompted this fix.
      {AMDSMI_STATUS_TIMEOUT, "AMDSMI_STATUS_TIMEOUT"},
      {AMDSMI_STATUS_MORE_DATA, "AMDSMI_STATUS_MORE_DATA"},
      // Controls that were already handled, including both catch-all codes.
      {AMDSMI_STATUS_SUCCESS, "AMDSMI_STATUS_SUCCESS"},
      {AMDSMI_STATUS_INVAL, "AMDSMI_STATUS_INVAL"},
      {AMDSMI_STATUS_MAP_ERROR, "AMDSMI_STATUS_MAP_ERROR"},
      {AMDSMI_STATUS_UNKNOWN_ERROR, "AMDSMI_STATUS_UNKNOWN_ERROR"},
  };

  for (const auto& [code, name] : spot_checks) {
    const char* status_string = nullptr;
    amdsmi_status_t ret = amdsmi_status_code_to_string(code, &status_string);

    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS)
        << name << " (" << code << ") did not resolve, this case is missing from "
        << "amdsmi_status_code_to_string(). Must return AMDSMI_STATUS_SUCCESS "
        << "for a valid status code.";
    ASSERT_NE(status_string, nullptr) << name;
    EXPECT_TRUE(StartsWith(status_string, name))
        << name << " (" << code << ") resolved to \"" << status_string
        << "\". Expected the description to start with its own enum name.";
  }
}

// Negative: a lower-level (rocm_smi) status that amdsmi has no mapping for must
// surface AMDSMI_STATUS_MAP_ERROR. This guards against rocm_smi adding a new
// status that silently goes untranslated through rsmi_to_amdsmi_status().
TEST(AmdSmiStatusStringTest, UnmappedLowerLevelStatusYieldsMapError) {
  // A value that is intentionally not present in amd::smi::rsmi_status_map.
  const auto unmapped_rsmi_status = static_cast<rsmi_status_t>(0x0BADF00D);
  EXPECT_EQ(amd::smi::rsmi_to_amdsmi_status(unmapped_rsmi_status), AMDSMI_STATUS_MAP_ERROR)
      << "An unmapped rocm_smi status must translate to AMDSMI_STATUS_MAP_ERROR; "
      << "a new rsmi status is missing from amd::smi::rsmi_status_map.";

  // Sanity: known mappings must NOT produce MAP_ERROR.
  EXPECT_EQ(amd::smi::rsmi_to_amdsmi_status(RSMI_STATUS_SUCCESS), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(amd::smi::rsmi_to_amdsmi_status(RSMI_STATUS_INVALID_ARGS), AMDSMI_STATUS_INVAL);
  EXPECT_NE(amd::smi::rsmi_to_amdsmi_status(RSMI_STATUS_NOT_SUPPORTED), AMDSMI_STATUS_MAP_ERROR);
}

// A value that is not a defined status code (an enum gap) must hit the default
// path and report AMDSMI_STATUS_UNKNOWN_ERROR rather than a stale string.
TEST(AmdSmiStatusStringTest, UndefinedStatusCodeYieldsUnknownError) {
  const char* status_string = nullptr;
  // 100 sits in an enum gap (above the highest normal code, below the two
  // high-value error codes).
  const auto undefined_code = static_cast<amdsmi_status_t>(100);
  EXPECT_EQ(amdsmi_status_code_to_string(undefined_code, &status_string),
            AMDSMI_STATUS_UNKNOWN_ERROR);
  ASSERT_NE(status_string, nullptr);
}

// A null output pointer must be rejected, not dereferenced.
TEST(AmdSmiStatusStringTest, NullOutputPointerYieldsInval) {
  EXPECT_EQ(amdsmi_status_code_to_string(AMDSMI_STATUS_SUCCESS, nullptr), AMDSMI_STATUS_INVAL);
}
