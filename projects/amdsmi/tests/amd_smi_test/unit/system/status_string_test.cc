// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Status-code tests: amdsmi_status_code_to_string() descriptions, fallback and null out-pointer
// contract (NULL status_string returns AMDSMI_STATUS_INVAL), and rsmi_to_amdsmi_status() mapping.
// No GPU required.

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

// The helpers below call the API, print it the way the other amdsmi tests do
// ("### func(desc)" header, then the returned status against `expected`), and
// return the status so the caller can still assert on it. The caller's
// file/line are captured by the default arguments.
amdsmi_status_t StatusCodeToString(amdsmi_status_t code, const char** status_string,
                                   amdsmi_status_t expected, std::string_view desc,
                                   const char* file = __builtin_FILE(),
                                   uint32_t line = __builtin_LINE()) {
  DISPLAY_AMDSMI_API("amdsmi_status_code_to_string", desc, IsVerbose());
  const amdsmi_status_t ret = amdsmi_status_code_to_string(code, status_string);
  DISPLAY_AMDSMI_STATUS(IsVerbose(), file, line, ret, expected);
  return ret;
}

amdsmi_status_t RsmiToAmdsmiStatus(rsmi_status_t rsmi_status, amdsmi_status_t expected,
                                   std::string_view desc, const char* file = __builtin_FILE(),
                                   uint32_t line = __builtin_LINE()) {
  DISPLAY_AMDSMI_API("rsmi_to_amdsmi_status", desc, IsVerbose());
  const amdsmi_status_t ret = amd::smi::rsmi_to_amdsmi_status(rsmi_status);
  DISPLAY_AMDSMI_STATUS(IsVerbose(), file, line, ret, expected);
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
  DISPLAY_AMDSMI_API("amdsmi_get_esmi_err_msg", "status_string=nullptr", IsVerbose());
  const amdsmi_status_t ret = amdsmi_get_esmi_err_msg(AMDSMI_STATUS_SUCCESS, nullptr);
  DISPLAY_AMDSMI_STATUS(IsVerbose(), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
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

// Scan beyond the current normal status-code range for future additions. Also bounds the
// reflection-free rsmi scan.
constexpr uint32_t kScanLimit = 128;

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

template <auto V>
constexpr const char* Signature() {
  // Return type is a plain `const char*` so the signature has no stray "= "
  // from typedef expansion to confuse the parse below.
  return __PRETTY_FUNCTION__;
}

// Enumerator name for V, e.g. "AMDSMI_STATUS_TIMEOUT".
template <auto V>
constexpr std::string_view EnumName() {
  std::string_view s = Signature<V>();
  const auto start = s.find("V = ") + 4;  // anchor on the value parameter
  const auto end = s.find_first_of(";]", start);
  return s.substr(start, end - start);
}

// Real enumerators start like C identifiers. Unused values start with '(' on
// GCC or a digit on Clang, so this rejects both renderings.
template <auto V>
constexpr bool IsEnumerator() {
  const auto name = EnumName<V>();
  if (name.empty()) return false;
  const char c = name.front();
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

template <typename E, uint32_t... Is>
void Collect(std::vector<std::pair<E, std::string>>& out, std::integer_sequence<uint32_t, Is...>) {
  ((IsEnumerator<static_cast<E>(Is)>()
        ? out.push_back({static_cast<E>(Is), std::string(EnumName<static_cast<E>(Is)>())})
        : void()),
   ...);
}

// {value, name} for every enumerator of E in the low range [0, kScanLimit).
template <typename E>
std::vector<std::pair<E, std::string>> ScanLowRange() {
  std::vector<std::pair<E, std::string>> out;
  Collect(out, std::make_integer_sequence<uint32_t, kScanLimit>{});
  return out;
}

// {value, "AMDSMI_STATUS_..."} for every amdsmi_status_t enumerator.
std::vector<std::pair<amdsmi_status_t, std::string>> AllStatusCodes() {
  auto out = ScanLowRange<amdsmi_status_t>();
  // Append the two high-value sentinels the scan cannot reach.
  out.push_back({AMDSMI_STATUS_MAP_ERROR, std::string(EnumName<AMDSMI_STATUS_MAP_ERROR>())});
  out.push_back(
      {AMDSMI_STATUS_UNKNOWN_ERROR, std::string(EnumName<AMDSMI_STATUS_UNKNOWN_ERROR>())});
  return out;
}

// Include a raw signature in failures so compiler-format changes are visible.
inline const char* SampleSignature() { return Signature<AMDSMI_STATUS_TIMEOUT>(); }

#else  // !AMDSMI_ENUM_REFLECTION_AVAILABLE

// Unsupported compiler: no codes discovered; the positive tests skip themselves.
template <typename E>
std::vector<std::pair<E, std::string>> ScanLowRange() {
  return {};
}
std::vector<std::pair<amdsmi_status_t, std::string>> AllStatusCodes() { return {}; }
inline const char* SampleSignature() { return "<enum reflection unavailable>"; }

#endif  // AMDSMI_ENUM_REFLECTION_AVAILABLE

}  // namespace enum_reflect

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

// Every status code should resolve to a string that starts with its enum name.
// A missing case is a build error (-Wswitch); this catches a wrong or copy-pasted description.
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
    if (IsVerbose()) {
      std::cout << "\t>>> " << name << " (" << code << ") -> \"" << description << "\""
                << std::endl;
    }
    EXPECT_TRUE(StartsWith(description, name))
        << name << " (" << code << ") resolved to \"" << description
        << "\". Expected the description to start with its own enum name.";
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
    if (IsVerbose()) {
      std::cout << "\t>>> " << name << " (" << code << ") -> \"" << status_string << "\""
                << std::endl;
    }
    EXPECT_TRUE(StartsWith(status_string, name))
        << name << " (" << code << ") resolved to \"" << status_string
        << "\". Expected the description to start with its own enum name.";
  }
}

// Negative: a status value in an enum gap (matches no case) must return
// AMDSMI_STATUS_UNKNOWN_ERROR and overwrite status_string with the fallback text,
// rather than leaving the caller's previous value in place.
TEST(SystemUnit, status_code_to_string__UndefinedStatusCodeYieldsUnknownError) {
  const char* const stale = "stale";
  const char* status_string = stale;
  // In an enum gap: above the highest normal code, below the two high-value error codes.
  constexpr uint32_t kUndefinedStatus = 100;
  const auto undefined_code = static_cast<amdsmi_status_t>(kUndefinedStatus);
  const amdsmi_status_t ret =
      StatusCodeToString(undefined_code, &status_string, AMDSMI_STATUS_UNKNOWN_ERROR,
                         "status=" + std::to_string(kUndefinedStatus));
  EXPECT_EQ(ret, AMDSMI_STATUS_UNKNOWN_ERROR);
  ASSERT_NE(status_string, nullptr);
  EXPECT_NE(status_string, stale) << "status_string was left pointing at the caller's old value.";
  // The fallback has no "AMDSMI_STATUS_UNKNOWN_ERROR:" prefix, unlike the real status.
  EXPECT_STREQ(status_string, "An unknown error occurred");
}

// Negative: a lower-level (rocm_smi) status that amdsmi has no mapping for must
// surface AMDSMI_STATUS_MAP_ERROR.
TEST(SystemUnit, status_code_to_string__UnmappedLowerLevelStatusYieldsMapError) {
  // A value outside rsmi_status_t, so rsmi_to_amdsmi_status() has no case for it.
  constexpr uint32_t kUnmappedRsmiStatus = 0x0BADF00D;
  const auto unmapped_rsmi_status = static_cast<rsmi_status_t>(kUnmappedRsmiStatus);
  amdsmi_status_t ret = RsmiToAmdsmiStatus(unmapped_rsmi_status, AMDSMI_STATUS_MAP_ERROR,
                                           "rsmi_status=" + std::to_string(kUnmappedRsmiStatus));
  EXPECT_EQ(ret, AMDSMI_STATUS_MAP_ERROR)
      << "An unmapped rocm_smi status must translate to AMDSMI_STATUS_MAP_ERROR.";

  // Sanity: known mappings must NOT produce MAP_ERROR.
  ret = RsmiToAmdsmiStatus(RSMI_STATUS_SUCCESS, AMDSMI_STATUS_SUCCESS, "RSMI_STATUS_SUCCESS");
  EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);

  ret =
      RsmiToAmdsmiStatus(RSMI_STATUS_INVALID_ARGS, AMDSMI_STATUS_INVAL, "RSMI_STATUS_INVALID_ARGS");
  EXPECT_EQ(ret, AMDSMI_STATUS_INVAL);

  ret = RsmiToAmdsmiStatus(RSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_SUPPORTED,
                           "RSMI_STATUS_NOT_SUPPORTED");
  EXPECT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);
}

// rsmi_to_amdsmi_status() and rsmi_status_string() must agree on which values are real statuses.
// No reflection, so it runs on every compiler. A status added to one but not the other fails here.
TEST(SystemUnit, status_code_to_string__RsmiMappingAndDescriptionsAgree) {
  std::vector<rsmi_status_t> candidates;
  for (uint32_t v = 0; v < enum_reflect::kScanLimit; ++v) {
    candidates.push_back(static_cast<rsmi_status_t>(v));
  }
  candidates.push_back(RSMI_STATUS_UNKNOWN_ERROR);  // High-value sentinel outside the scan range.

  for (rsmi_status_t code : candidates) {
    const char* description = nullptr;
    // For a value it does not know, rsmi_status_string() returns RSMI_STATUS_UNKNOWN_ERROR.
    const bool described = rsmi_status_string(code, &description) == RSMI_STATUS_SUCCESS;
    const bool mapped = amd::smi::rsmi_to_amdsmi_status(code) != AMDSMI_STATUS_MAP_ERROR;
    if (IsVerbose() && described) {
      std::cout << "\t>>> rsmi status " << code << " (" << description << ") mapped=" << mapped
                << std::endl;
    }
    EXPECT_EQ(mapped, described)
        << "rsmi status " << code
        << (mapped ? " is mapped in rsmi_to_amdsmi_status() but has no description in "
                     "rsmi_status_string()."
                   : " has a description in rsmi_status_string() but is not mapped in "
                     "rsmi_to_amdsmi_status().");
  }
}

// The compiler forces a case for every rsmi_status_t, but not what it returns. A real rsmi status
// must map to a specific amdsmi status: never MAP_ERROR, and UNKNOWN_ERROR only for the rsmi
// UNKNOWN_ERROR itself. Callers only see the amdsmi status, so a case that returns a generic error
// (e.g. a copy-paste slip) hides the real cause, such as BUSY or PERMISSION. It must also have a
// description in rsmi_status_string().
TEST(SystemUnit, status_code_to_string__EveryRsmiStatusMapsToASpecificStatus) {
  if (!enum_reflect::kAvailable) {
    GTEST_SKIP() << "enum reflection unavailable on this compiler (" << enum_reflect::CompilerId()
                 << "). This test needs Clang or GCC >= 9.";
  }

  auto rsmi_codes = enum_reflect::ScanLowRange<rsmi_status_t>();
  // The scan cannot reach this high-value sentinel.
  rsmi_codes.push_back({RSMI_STATUS_UNKNOWN_ERROR, "RSMI_STATUS_UNKNOWN_ERROR"});

  // Keep this from passing on an empty or broken reflection result.
  auto contains = [&](rsmi_status_t code) {
    for (const auto& [value, name] : rsmi_codes) {
      if (value == code) return true;
    }
    return false;
  };
  for (rsmi_status_t code :
       {RSMI_STATUS_SUCCESS, RSMI_STATUS_IPC_ERROR, RSMI_STATUS_UNKNOWN_ERROR}) {
    ASSERT_TRUE(contains(code)) << "enum reflection did not discover known rsmi status " << code
                                << ". This is a test-harness problem. Compiler: "
                                << enum_reflect::CompilerId();
  }

  for (const auto& [code, name] : rsmi_codes) {
    const amdsmi_status_t ret = amd::smi::rsmi_to_amdsmi_status(code);
    if (IsVerbose()) {
      std::cout << "\t>>> rsmi_to_amdsmi_status(" << name << "=" << code << ") -> " << ret
                << std::endl;
    }
    EXPECT_NE(ret, AMDSMI_STATUS_MAP_ERROR)
        << name << " (" << code << ") maps to AMDSMI_STATUS_MAP_ERROR in rsmi_to_amdsmi_status().";
    if (code != RSMI_STATUS_UNKNOWN_ERROR) {
      EXPECT_NE(ret, AMDSMI_STATUS_UNKNOWN_ERROR)
          << name << " (" << code
          << ") maps to AMDSMI_STATUS_UNKNOWN_ERROR in rsmi_to_amdsmi_status().";
    }

    const char* description = nullptr;
    EXPECT_EQ(rsmi_status_string(code, &description), RSMI_STATUS_SUCCESS)
        << name << " (" << code << ") has no description in rsmi_status_string().";
  }
}
