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
//
// Two checks:
//   1. Positive: every amdsmi_status_t enumerator resolves, via
//      amdsmi_status_code_to_string(), to a description starting with its own
//      enum name. A missing `case` falls through to the UNKNOWN_ERROR default,
//      which this catches. (Mirrors the Python test_status_code_to_string.)
//   2. Negative: a lower-level rocm_smi status with no amdsmi mapping must
//      surface AMDSMI_STATUS_MAP_ERROR via rsmi_to_amdsmi_status(). The Python
//      test cannot reach this, since it only feeds existing amdsmi codes.

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
//
// C++17 cannot enumerate an enum's members or recover their names, so unlike
// the Python test we cannot just loop over amdsmi.AmdSmiStatus. We use the
// magic_enum trick: instantiate a template on a value and read the enumerator
// name the compiler bakes into __PRETTY_FUNCTION__. A real enumerator renders
// as `V = AMDSMI_STATUS_TIMEOUT`; an unused value renders as a cast on GCC
// (`V = (amdsmi_status_t)28`) or a bare integer on Clang (`V = 28`). A name
// that starts like a C identifier is therefore a real enumerator, which lets
// us auto-discover every member by scanning a bounded range of values.
//
// The two sentinels sit at the top of the value range: AMDSMI_STATUS_MAP_ERROR
// (UINT_MAX - 1) and AMDSMI_STATUS_UNKNOWN_ERROR (UINT_MAX). A scan cannot
// reach those, so the low range is scanned for normal codes and the two
// sentinels are seeded explicitly (names still come from reflection). Adding a
// normal status code in range needs no change here.
// ---------------------------------------------------------------------------

// Reflection needs the compiler to render enumerator *names* (not numeric
// casts) in __PRETTY_FUNCTION__. Clang does this at any version; GCC only since
// GCC 9. On older GCC the positive test GTEST_SKIPs instead of failing the
// build; the negative test needs no reflection and always runs.
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

// Enumerator name for V, e.g. "AMDSMI_STATUS_TIMEOUT". For an integer with no
// corresponding enumerator the compilers differ: GCC renders a cast,
// "(amdsmi_status_t)28", while Clang renders a bare integer, "28".
template <amdsmi_status_t V>
constexpr std::string_view EnumName() {
  std::string_view s = Signature<V>();
  const auto start = s.find("V = ") + 4;  // anchor on the value parameter
  const auto end = s.find_first_of(";]", start);
  return s.substr(start, end - start);
}

// True when V names a real enumerator. A real name starts like a C identifier
// (letter or '_'); an unused value starts with '(' on GCC or a digit on Clang,
// so requiring an identifier-start character rejects both.
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

// The highest normal enumerator is well under 100; scan past it for headroom.
constexpr uint32_t kScanLimit = 128;

// {value, "AMDSMI_STATUS_..."} for every amdsmi_status_t enumerator.
std::vector<std::pair<amdsmi_status_t, std::string>> AllStatusCodes() {
  std::vector<std::pair<amdsmi_status_t, std::string>> out;
  Collect(out, std::make_integer_sequence<uint32_t, kScanLimit>{});
  // Append the two sentinels the scan can't reach (see note above).
  out.push_back({AMDSMI_STATUS_MAP_ERROR, std::string(EnumName<AMDSMI_STATUS_MAP_ERROR>())});
  out.push_back(
      {AMDSMI_STATUS_UNKNOWN_ERROR, std::string(EnumName<AMDSMI_STATUS_UNKNOWN_ERROR>())});
  return out;
}

// A sample raw __PRETTY_FUNCTION__ rendering, surfaced in failure diagnostics so
// a compiler change to the signature format is immediately visible in the log.
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

// Positive: every status code resolves to a description starting with its own
// enum name. A missing `case` (e.g. AMDSMI_STATUS_TIMEOUT / MORE_DATA) makes
// amdsmi_status_code_to_string() return AMDSMI_STATUS_UNKNOWN_ERROR with the
// "unknown error" string, which fails both assertions below.
TEST(AmdSmiStatusStringTest, EveryStatusCodeResolvesToItsOwnName) {
  if (!enum_reflect::kAvailable) {
    GTEST_SKIP() << "enum reflection unavailable on this compiler (" << enum_reflect::CompilerId()
                 << "); it needs Clang or GCC >= 9 "
                 << "(older GCC renders enum template args as numeric casts). "
                 << "The matching Python test covers this case elsewhere.";
  }

  const auto all_status_codes = enum_reflect::AllStatusCodes();

  // Guard against a vacuous pass without pinning a magic count (which would need
  // bumping for every new status code). Reflection must have found the codes
  // whose missing cases prompted this test (TIMEOUT, MORE_DATA) plus the two
  // sentinels; if the __PRETTY_FUNCTION__ parser broke they would drop out and
  // fail the check below -- a test-harness problem, not an amdsmi library bug.
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
        << name << " (" << code << ") did not resolve; a case is missing from "
        << "amdsmi_status_code_to_string().";
    ASSERT_NE(status_string, nullptr) << name;

    const std::string description(status_string);
    EXPECT_TRUE(StartsWith(description, name))
        << name << " (" << code << ") resolved to \"" << description
        << "\"; expected the description to start with its own enum name. A "
        << "missing case falls through to the UNKNOWN_ERROR default.";
  }
}

// Simpler companion to the reflection test above: no reflection, no scan, no
// compiler dependency. It spot-checks the codes that regressed (TIMEOUT,
// MORE_DATA) plus a few controls. It will not catch a missing case for some
// other code -- the reflection test and the Python test cover that breadth.
TEST(AmdSmiStatusStringTest, KnownStatusCodesResolveToTheirOwnName) {
  const std::vector<std::pair<amdsmi_status_t, const char*>> spot_checks = {
      // The two codes whose missing cases prompted this fix.
      {AMDSMI_STATUS_TIMEOUT, "AMDSMI_STATUS_TIMEOUT"},
      {AMDSMI_STATUS_MORE_DATA, "AMDSMI_STATUS_MORE_DATA"},
      // Controls that were already handled, including both sentinels.
      {AMDSMI_STATUS_SUCCESS, "AMDSMI_STATUS_SUCCESS"},
      {AMDSMI_STATUS_INVAL, "AMDSMI_STATUS_INVAL"},
      {AMDSMI_STATUS_MAP_ERROR, "AMDSMI_STATUS_MAP_ERROR"},
      {AMDSMI_STATUS_UNKNOWN_ERROR, "AMDSMI_STATUS_UNKNOWN_ERROR"},
  };

  for (const auto& [code, name] : spot_checks) {
    const char* status_string = nullptr;
    amdsmi_status_t ret = amdsmi_status_code_to_string(code, &status_string);

    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS)
        << name << " (" << code << ") did not resolve; a case is missing from "
        << "amdsmi_status_code_to_string().";
    ASSERT_NE(status_string, nullptr) << name;
    EXPECT_TRUE(StartsWith(status_string, name))
        << name << " (" << code << ") resolved to \"" << status_string
        << "\"; expected the description to start with its own enum name.";
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
