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

// Driver-free unit test for amdsmi status-code handling. It needs no GPU and
// performs no amdsmi_init(), so it can move into a dedicated unit-test target
// during the test refactor.
//
// Two complementary checks:
//   1. Positive (mirror of the Python test_status_code_to_string): every
//      amdsmi_status_t enumerator must resolve, via amdsmi_status_code_to_string(),
//      to a description that begins with its own enum name. A missing `case`
//      makes the code fall through to the UNKNOWN_ERROR default, which this
//      catches.
//   2. Negative: simulate a lower-level (rocm_smi) status that the amdsmi layer
//      has no mapping for and confirm rsmi_to_amdsmi_status() surfaces
//      AMDSMI_STATUS_MAP_ERROR rather than silently mistranslating it. This is
//      the case the Python test cannot reach, because it can only feed codes
//      that already exist in the amdsmi layer.

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
// C++17 erases enums to plain integers: there is no built-in way to list an
// enum's members or recover their names, which is why the matching Python test
// can loop `for status in amdsmi.AmdSmiStatus` but C++ cannot. We recover the
// names the same way the magic_enum library does: instantiate a template on the
// value and read the enumerator name the compiler bakes into __PRETTY_FUNCTION__
// (or __FUNCSIG__ on MSVC). For a real enumerator GCC/Clang render
// `... V = AMDSMI_STATUS_TIMEOUT]`; for an unused slot GCC renders a cast,
// `... V = (amdsmi_status_t)28]`, and Clang renders a bare integer,
// `... V = 28]`. A real enumerator name is therefore the one that starts like a
// C identifier, which lets us auto-discover every member in a bounded range.
//
// amdsmi_status_t is sparse and ends in two ~4-billion sentinels
// (AMDSMI_STATUS_MAP_ERROR = 0xFFFFFFFE, AMDSMI_STATUS_UNKNOWN_ERROR =
// 0xFFFFFFFF). A value scan cannot practically reach those, so the normal codes
// are discovered by scanning the low range and the two stable sentinels are
// seeded explicitly (their names still come from reflection, not hand-typing).
// Adding a normal status code in range needs no change here.
// ---------------------------------------------------------------------------

// The reflection trick relies on __PRETTY_FUNCTION__ AND on the compiler
// rendering enumerator *names* (not numeric casts) for enum template arguments.
// Clang does this at any version; GCC only started doing so in GCC 9. On GCC 8
// and earlier every value renders as "(amdsmi_status_t)N", so reflection finds
// no enumerators. On GCC < 9 or any other compiler the helper is unavailable and
// the positive test GTEST_SKIPs rather than failing the build; the driverless
// negative test below needs no reflection and always runs.
#if defined(__clang__)
#define AMDSMI_ENUM_REFLECTION_AVAILABLE 1
#elif defined(__GNUC__) && (__GNUC__ >= 9)
#define AMDSMI_ENUM_REFLECTION_AVAILABLE 1
#else
#define AMDSMI_ENUM_REFLECTION_AVAILABLE 0
#endif

namespace enum_reflect {

constexpr bool kAvailable = AMDSMI_ENUM_REFLECTION_AVAILABLE;

// Human-readable compiler identification for diagnostics, so a failure or skip
// on an unexpected toolchain names the compiler and version in its output.
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
  // Return type is `const char*` (not a library typedef) so the signature
  // carries no extra "= " clause from typedef expansion to confuse parsing.
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

// True when V names a real enumerator rather than an unused slot. A real
// enumerator name is a C identifier (starts with a letter or '_'); an unused
// slot starts with '(' on GCC ("(amdsmi_status_t)28") or a digit on Clang
// ("28"). Requiring an identifier-start character rejects both, so this works
// on either compiler.
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
  // Sentinels live at 0xFFFFFFFE/0xFFFFFFFF, far outside any practical scan.
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
  // bumping every time a status code is added). Reflection must have discovered
  // the specific normal codes whose missing cases prompted this test, plus the
  // two out-of-range sentinels. TIMEOUT (8) and MORE_DATA (39) are ordinary
  // scanned codes: if the __PRETTY_FUNCTION__ parser broke (e.g. a compiler
  // change) the scan finds no enumerators and they drop out, failing the check
  // below -- a test-harness problem, not an amdsmi library bug.
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

// Simpler alternative to the reflection-based test above. It does not aim for
// exhaustive coverage; it just spot-checks the codes that actually regressed
// (AMDSMI_STATUS_TIMEOUT, AMDSMI_STATUS_MORE_DATA) plus a couple of controls.
// No enum reflection, no scan, no compiler dependency -- it reads top to bottom
// and pins the exact bug this change fixes. The trade-off is that a future
// missing case for some *other* code would slip past it (the reflection test
// and the Python test cover that breadth).
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
