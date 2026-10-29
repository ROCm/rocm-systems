// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Null out-pointer contract for the error-string helpers: a NULL status_string
// must return AMDSMI_STATUS_INVAL instead of being dereferenced. No GPU required.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_common.h"
#include "test_common.h"

namespace {

// These are plain TEST()s with no TestBase, so the VERB() macro (which reads
// TestBase::verbosity()) is unavailable; gate on the global verbosity instead.
// VERBOSE_STANDARD == 1.
bool isGlobalTestBaseVerbose() {
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;
  return verbose;
}

// The helpers below call the API, print it the way the other amdsmi tests do
// ("### func(desc)" header, then the returned status against `expected`), and
// return the status so the caller can still assert on it. The caller's
// file/line are captured by the default arguments.
amdsmi_status_t StatusCodeToString(amdsmi_status_t code, const char** status_string,
                                   amdsmi_status_t expected, std::string_view desc,
                                   const char* file = __builtin_FILE(),
                                   uint32_t line = __builtin_LINE()) {
  DISPLAY_AMDSMI_API("amdsmi_status_code_to_string", desc, isGlobalTestBaseVerbose());
  const amdsmi_status_t ret = amdsmi_status_code_to_string(code, status_string);
  DISPLAY_AMDSMI_STATUS(isGlobalTestBaseVerbose(), file, line, ret, expected);
  return ret;
}

amdsmi_status_t RsmiToAmdsmiStatus(rsmi_status_t rsmi_status, amdsmi_status_t expected,
                                   std::string_view desc, const char* file = __builtin_FILE(),
                                   uint32_t line = __builtin_LINE()) {
  DISPLAY_AMDSMI_API("rsmi_to_amdsmi_status", desc, isGlobalTestBaseVerbose());
  const amdsmi_status_t ret = amd::smi::rsmi_to_amdsmi_status(rsmi_status);
  DISPLAY_AMDSMI_STATUS(isGlobalTestBaseVerbose(), file, line, ret, expected);
  return ret;
}

TEST(SystemUnit, status_code_to_string__StatusCodeToStringRejectsNullOutPtr) {
  const amdsmi_status_t ret = StatusCodeToString(AMDSMI_STATUS_SUCCESS, nullptr,
                                                 AMDSMI_STATUS_INVAL, "status_string=nullptr");
  EXPECT_EQ(ret, AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, status_code_to_string__StatusCodeToStringValidOutPtr) {
  const char* msg = nullptr;
  const amdsmi_status_t ret =
      StatusCodeToString(AMDSMI_STATUS_SUCCESS, &msg, AMDSMI_STATUS_SUCCESS, "status_string=&msg");
  EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(msg, nullptr);
}

#if ENABLE_ESMI_LIB
TEST(SystemUnit, status_code_to_string__EsmiErrMsgRejectsNullOutPtr) {
  DISPLAY_AMDSMI_API("amdsmi_get_esmi_err_msg", "status_string=nullptr", isGlobalTestBaseVerbose());
  const amdsmi_status_t ret = amdsmi_get_esmi_err_msg(AMDSMI_STATUS_SUCCESS, nullptr);
  DISPLAY_AMDSMI_STATUS(isGlobalTestBaseVerbose(), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(ret, AMDSMI_STATUS_INVAL);
}
#endif

// ---------------------------------------------------------------------------
// Compile-time enum reflection (stopgap until C++26 static reflection / P2996).
// C++17 cannot enumerate enum members, so this uses the magic_enum trick:
// instantiate a template on a value and parse its name from __PRETTY_FUNCTION__.
// Real enumerators render as names, whereas unused values render as numbers or casts.
//
// Normal status codes are discovered by scanning a low numeric range. The two
// high sentinels, AMDSMI_STATUS_MAP_ERROR and AMDSMI_STATUS_UNKNOWN_ERROR, are
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

// Scan beyond the current normal status-code range for future additions.
constexpr uint32_t kScanLimit = 128;

// {value, "AMDSMI_STATUS_..."} for every amdsmi_status_t enumerator.
std::vector<std::pair<amdsmi_status_t, std::string>> AllStatusCodes() {
  std::vector<std::pair<amdsmi_status_t, std::string>> out;
  Collect(out, std::make_integer_sequence<uint32_t, kScanLimit>{});
  // Append the two high-value sentinels the scan cannot reach.
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

// Every status code should resolve to a string that starts with its enum name.
// Missing cases fall through to AMDSMI_STATUS_UNKNOWN_ERROR and fail below.
TEST(SystemUnit, status_code_to_string__EveryStatusCodeResolvesToItsOwnName) {
  if (!enum_reflect::kAvailable) {
    GTEST_SKIP() << "enum reflection unavailable on this compiler (" << enum_reflect::CompilerId()
                 << "). This test needs Clang or GCC >= 9 "
                 << "(older GCC renders enum template args as numeric casts). "
                 << "The matching Python test covers this case elsewhere.";
  }

  const auto all_status_codes = enum_reflect::AllStatusCodes();

  // Keep this from passing with an empty or incomplete reflection result. We
  // check a small set of required codes instead of asserting the full enum
  // count, so adding a new status code does not require updating this test.
  // The required set covers the regression cases (TIMEOUT, MORE_DATA) and both
  // sentinels. If any are absent, the reflection parser or scan limit is broken;
  // the failure is in the test harness, not the amdsmi status-string code.
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
    const amdsmi_status_t ret = StatusCodeToString(code, &status_string, AMDSMI_STATUS_SUCCESS,
                                                   name + "=" + std::to_string(code));

    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS)
        << name << " (" << code << ") did not resolve, this status is missing from "
        << "amdsmi_status_code_to_string().";
    ASSERT_NE(status_string, nullptr) << name;

    const std::string description(status_string);
    if (isGlobalTestBaseVerbose()) {
      std::cout << "\t>>> " << name << " (" << code << ") -> \"" << description << "\""
                << std::endl;
    }
    EXPECT_TRUE(StartsWith(description, name))
        << name << " (" << code << ") resolved to \"" << description
        << "\". Expected the description to start with its own enum name. A "
        << "missing case falls through to the UNKNOWN_ERROR default.";
  }
}

// Reflection-free spot check for the regressed codes + a few other common returns.
TEST(SystemUnit, status_code_to_string__KnownStatusCodesResolveToTheirOwnName) {
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
    const amdsmi_status_t ret = StatusCodeToString(code, &status_string, AMDSMI_STATUS_SUCCESS,
                                                   std::string(name) + "=" + std::to_string(code));

    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS)
        << name << " (" << code << ") did not resolve, this case is missing from "
        << "amdsmi_status_code_to_string(). Must return AMDSMI_STATUS_SUCCESS "
        << "for a valid status code.";
    ASSERT_NE(status_string, nullptr) << name;
    if (isGlobalTestBaseVerbose()) {
      std::cout << "\t### " << name << " (" << code << ") -> \"" << status_string << "\""
                << std::endl;
    }
    EXPECT_TRUE(StartsWith(status_string, name))
        << name << " (" << code << ") resolved to \"" << status_string
        << "\". Expected the description to start with its own enum name.";
  }
}

// Negative: a status value in an enum gap (no case, no rsmi mapping) must fall
// through the default branch to AMDSMI_STATUS_UNKNOWN_ERROR with a non-empty
// description, rather than leaving status_string stale or empty.
TEST(SystemUnit, status_code_to_string__UndefinedStatusCodeYieldsUnknownError) {
  const char* status_string = nullptr;
  // 100 sits in an enum gap (above the highest normal code, below the two
  // high-value error codes).
  const auto undefined_code = static_cast<amdsmi_status_t>(100);
  const amdsmi_status_t ret =
      StatusCodeToString(undefined_code, &status_string, AMDSMI_STATUS_UNKNOWN_ERROR, "status=100");
  EXPECT_EQ(ret, AMDSMI_STATUS_UNKNOWN_ERROR);
  ASSERT_NE(status_string, nullptr);
  EXPECT_STRNE(status_string, "")
      << "an undefined status code returned UNKNOWN_ERROR but left an empty description.";
}

// Negative: a lower-level (rocm_smi) status that amdsmi has no mapping for must
// surface AMDSMI_STATUS_MAP_ERROR. This guards against rocm_smi adding a new
// status that silently goes untranslated through rsmi_to_amdsmi_status().
TEST(SystemUnit, status_code_to_string__UnmappedLowerLevelStatusYieldsMapError) {
  // A value that is intentionally not present in amd::smi::rsmi_status_map.
  const auto unmapped_rsmi_status = static_cast<rsmi_status_t>(0x0BADF00D);
  amdsmi_status_t ret =
      RsmiToAmdsmiStatus(unmapped_rsmi_status, AMDSMI_STATUS_MAP_ERROR, "rsmi_status=0x0BADF00D");
  EXPECT_EQ(ret, AMDSMI_STATUS_MAP_ERROR)
      << "An unmapped rocm_smi status must translate to AMDSMI_STATUS_MAP_ERROR; "
      << "a new rsmi status is missing from amd::smi::rsmi_status_map.";

  // Sanity: known mappings must NOT produce MAP_ERROR.
  ret = RsmiToAmdsmiStatus(RSMI_STATUS_SUCCESS, AMDSMI_STATUS_SUCCESS, "RSMI_STATUS_SUCCESS");
  EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);

  ret =
      RsmiToAmdsmiStatus(RSMI_STATUS_INVALID_ARGS, AMDSMI_STATUS_INVAL, "RSMI_STATUS_INVALID_ARGS");
  EXPECT_EQ(ret, AMDSMI_STATUS_INVAL);

  // No single expected code here; the only requirement is that it is not MAP_ERROR.
  ret = RsmiToAmdsmiStatus(RSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_SUPPORTED,
                           "RSMI_STATUS_NOT_SUPPORTED");
  EXPECT_NE(ret, AMDSMI_STATUS_MAP_ERROR);
}
