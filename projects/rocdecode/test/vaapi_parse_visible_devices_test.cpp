/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Unit tests for ParseVisibleDevicesCsv (rocdecode).
//
// Covers the strtok-on-getenv regression (C11 §7.22.4.6 UB) and input
// sanitisation of ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES.
//
// Tests are ordered to mirror the parser's validation stages:
//   1. Regression / env-var interaction
//   2. nullptr / empty              (parser stage 1)
//   3. Length cap                    (parser stage 2)
//   4. Charset                      (parser stage 3)
//   5. Token digit-count / range    (parser stage 4)
//   6. Valid topologies              (happy path)
//   7. Lenient / tolerant parsing
//   8. NUL truncation               (security-specific)
//
// No GPU or ROCm runtime required. Build and run:
//   c++ -std=c++17 -I projects/rocdecode/src
//       projects/rocdecode/test/vaapi_parse_visible_devices_test.cpp
//       -o vaapi_parse_visible_devices_test
//   ./vaapi_parse_visible_devices_test

#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rocdecode/vaapi/vaapi_parse_helpers.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a "0,1,2,...,n-1" CSV string.
static std::string build_csv(int count) {
    static constexpr std::size_t kReservePerEntry = 5;
    std::string csv;
    csv.reserve(static_cast<std::size_t>(count) * kReservePerEntry);
    for (int idx = 0; idx < count; ++idx) {
        if (idx != 0) {
            csv.push_back(',');
        }
        csv.append(std::to_string(idx));
    }
    return csv;
}

// Verify that the parsed result is exactly {0, 1, ..., count-1}.
static void verify_sequential(const std::vector<int>& result, int count) {
    assert(static_cast<int>(result.size()) == count);
    for (int idx = 0; idx < count; ++idx) {
        assert(result[idx] == idx);
    }
}

// Verify that every string in the table produces an empty (rejected) result.
template <std::size_t N>
static void assert_all_rejected(const std::array<const char*, N>& inputs) {
    for (const char* input : inputs) {
        assert(ParseVisibleDevicesCsv(input).empty());
    }
}

// ===========================================================================
// 1. Regression — getenv() return must not be mutated by the parser.
//    The strtok-on-getenv bug was introduced in a633867d2d ("Add support
//    for detecting visible devices before initializing va-api (#292)").
// ===========================================================================
static void test_does_not_mutate_getenv_return() {
    unsetenv("HIP_VISIBLE_DEVICES");
    setenv("ROCR_VISIBLE_DEVICES", "0,1,2", 1);

    const char* before = std::getenv("ROCR_VISIBLE_DEVICES");
    assert(before != nullptr);
    assert(std::string(before) == "0,1,2");

    auto parsed = ParseVisibleDevicesCsv(std::getenv("ROCR_VISIBLE_DEVICES"));

    const char* after = std::getenv("ROCR_VISIBLE_DEVICES");
    assert(after != nullptr);
    assert(std::string(after) == "0,1,2");  // fails pre-fix (strtok wrote '\0')

    assert(parsed.size() == 3);
    assert(parsed[0] == 0 && parsed[1] == 1 && parsed[2] == 2);
}

// ===========================================================================
// 2. HIP_VISIBLE_DEVICES fallback — exercises the env-var lookup pattern
//    that callers use around the parser.
// ===========================================================================
static void test_fallback_to_hip_visible_devices() {
    unsetenv("ROCR_VISIBLE_DEVICES");
    setenv("HIP_VISIBLE_DEVICES", "3,4", 1);

    const char* env = std::getenv("ROCR_VISIBLE_DEVICES");
    if (env == nullptr) {
        env = std::getenv("HIP_VISIBLE_DEVICES");
    }

    auto parsed = ParseVisibleDevicesCsv(env);
    assert(parsed.size() == 2 && parsed[0] == 3 && parsed[1] == 4);

    const char* hip_after = std::getenv("HIP_VISIBLE_DEVICES");
    assert(hip_after != nullptr);
    assert(std::string(hip_after) == "3,4");
}

// ===========================================================================
// 3. Parser stage 1: nullptr and empty string.
// ===========================================================================
static void test_rejects_null_and_empty() {
    assert(ParseVisibleDevicesCsv(nullptr).empty());
    assert(ParseVisibleDevicesCsv("").empty());
}

// ===========================================================================
// 4. Parser stage 2: total input length cap (kMaxEnvLength = 8192).
// ===========================================================================
static void test_rejects_oversize_input() {
    // 5000 single-digit tokens + 4999 commas = 9999 chars > 8192.
    static constexpr int kTokenCount = 5000;
    static constexpr int kDigitMod   = 10;
    std::string big;
    for (int idx = 0; idx < kTokenCount; ++idx) {
        if (idx != 0) {
            big.push_back(',');
        }
        big.push_back(static_cast<char>('0' + (idx % kDigitMod)));
    }
    assert(big.size() > rocm_vaapi_parse_detail::kMaxEnvLength);
    assert(ParseVisibleDevicesCsv(big.c_str()).empty());

    // Long run of digits with no commas — length cap and token cap both fire.
    static constexpr std::size_t kDigitRunLength = 10000;
    const std::string all_digits(kDigitRunLength, '9');
    assert(ParseVisibleDevicesCsv(all_digits.c_str()).empty());
}

// ===========================================================================
// 5. Parser stage 3: strict [0-9,] charset.
//    Data-driven: every string in the table must produce an empty result.
// ===========================================================================
static void test_rejects_invalid_characters() {
    static const std::array<const char*, 34> bad_inputs = {{
        // Whitespace
        "0, 1, 2", " 0,1,2", "0,1,2 ", "0\t1\t2", "0\n1",
        // Letters, UUID-style, alternate delimiters
        "abc", "0,foo,2", "GPU-0", "0;1;2", "0.1",
        // Sign characters
        "-1", "-1,0,1", "+0",
        // C0 control characters (SOH, BEL, BS, VT, FF, CR, ESC, DEL)
        "\x01,0",
        "0\x07""1", "0\x08""1", "0\x0b""1", "0\x0c""1",
        "0\r1", "0\x1b""1", "0\x7f""1",
        // High-bit bytes / UTF-8 payloads
        "0\xef\xbc\x8c""1",     // U+FF0C fullwidth comma
        "0\xd8\x8c""1",         // U+060C Arabic comma
        "\xef\xbb\xbf""0,1,2",  // UTF-8 BOM
        "0\xa0""1",              // Latin-1 NBSP
        "\xff",                  // 0xFF byte
        "\xd9\xa0",              // Arabic-Indic digit U+0660
        // Alternate numeric base prefixes
        "0x1", "0X1", "0o7", "0b1",
        // Hex digits / scientific notation
        "0,a,1", "ff", "1e2",
    }};
    assert_all_rejected(bad_inputs);
}

// ===========================================================================
// 6. Parser stage 4: per-token digit count and numeric range.
// ===========================================================================
static void test_rejects_invalid_tokens() {
    static const std::array<const char*, 8> bad_tokens = {{
        // Token too long (> kMaxTokenDigits = 4 digits)
        "00000", "99999", "0,99999,1",
        "2147483648",           // INT_MAX + 1
        "999999999999999999",   // way beyond int range
        // Value out of range (> kMaxDeviceIndex = 1023)
        "1024", "0,1,1024",
        "9999",  // fits 4 digits, exceeds numeric cap
    }};
    assert_all_rejected(bad_tokens);

    // Boundary: 1023 must still work.
    auto max_idx = ParseVisibleDevicesCsv("1023");
    assert(max_idx.size() == 1 && max_idx[0] == 1023);
}

// ===========================================================================
// 7. Happy path — realistic and large-scale GPU topologies.
// ===========================================================================
static void test_valid_topologies() {
    // Single GPU (local dev box: AMD Radeon Pro W7500).
    auto gpu1 = ParseVisibleDevicesCsv("0");
    assert(gpu1.size() == 1 && gpu1[0] == 0);

    // 3-GPU host (l2: Lexa XT + W5700 + MI50).
    auto gpu3 = ParseVisibleDevicesCsv("0,1,2");
    assert(gpu3.size() == 3 && gpu3[0] == 0 && gpu3[1] == 1 && gpu3[2] == 2);

    // Reordered — user prefers MI50 first.
    auto gpu3_rev = ParseVisibleDevicesCsv("2,0,1");
    assert(gpu3_rev.size() == 3 && gpu3_rev[0] == 2 && gpu3_rev[1] == 0 && gpu3_rev[2] == 1);

    // Non-contiguous subset.
    auto gpu_sub = ParseVisibleDevicesCsv("0,2");
    assert(gpu_sub.size() == 2 && gpu_sub[0] == 0 && gpu_sub[1] == 2);

    // 8-GPU MI250/MI300 node — sequential and reversed.
    static constexpr int kGpu8 = 8;
    verify_sequential(ParseVisibleDevicesCsv("0,1,2,3,4,5,6,7"), kGpu8);
    auto gpu8_rev = ParseVisibleDevicesCsv("7,6,5,4,3,2,1,0");
    assert(static_cast<int>(gpu8_rev.size()) == kGpu8);
    for (int idx = 0; idx < kGpu8; ++idx) {
        assert(gpu8_rev[idx] == kGpu8 - 1 - idx);
    }

    // Scale tests: increasing GPU counts up to the kMaxDeviceIndex ceiling.
    //   19  = PCIe-riser mining / multi-GPU render rig
    //   64  = bifurcated PCIe on high-lane-count EPYC board
    //   128 = PCIe switch fabric
    //   256 = well beyond any current deployment
    //   1024 = exactly at kMaxDeviceIndex ceiling (indices 0..1023)
    static constexpr std::array<int, 5> scale_counts = {19, 64, 128, 256, 1024};
    for (const int count : scale_counts) {
        verify_sequential(ParseVisibleDevicesCsv(build_csv(count).c_str()), count);
    }
}

// ===========================================================================
// 8. Lenient parsing — tolerated input shapes.
// ===========================================================================
static void test_lenient_parsing() {
    // Extra commas: leading, trailing, consecutive.
    const std::vector<int> expected_012 = {0, 1, 2};
    assert(ParseVisibleDevicesCsv(",0,1,2")  == expected_012);
    assert(ParseVisibleDevicesCsv("0,1,2,")  == expected_012);
    assert(ParseVisibleDevicesCsv("0,,1,,2") == expected_012);

    // Commas only — no tokens to extract.
    assert(ParseVisibleDevicesCsv(",,,,,,").empty());
    assert(ParseVisibleDevicesCsv(",").empty());

    // Duplicate indices: deduped, first-seen order preserved.
    auto deduped = ParseVisibleDevicesCsv("1,0,1,0,1,0");
    assert(deduped.size() == 2 && deduped[0] == 1 && deduped[1] == 0);

    auto all_same = ParseVisibleDevicesCsv("3,3,3,3");
    assert(all_same.size() == 1 && all_same[0] == 3);

    // Leading zeros: "01" means device 1, not an error.
    auto leading_zeros = ParseVisibleDevicesCsv("01,02,03");
    const std::vector<int> expected_123 = {1, 2, 3};
    assert(leading_zeros == expected_123);
}

// ===========================================================================
// 9. NUL truncation — attacker cannot smuggle a payload past an embedded NUL.
// ===========================================================================
static void test_embedded_nul_truncation() {
    std::string with_nul = "0,1";
    with_nul.push_back('\0');
    with_nul.append("malicious,9999");
    auto result = ParseVisibleDevicesCsv(with_nul.c_str());
    assert(result.size() == 2 && result[0] == 0 && result[1] == 1);
}

// ===========================================================================
int main() {
    test_does_not_mutate_getenv_return();
    test_fallback_to_hip_visible_devices();
    test_rejects_null_and_empty();
    test_rejects_oversize_input();
    test_rejects_invalid_characters();
    test_rejects_invalid_tokens();
    test_valid_topologies();
    test_lenient_parsing();
    test_embedded_nul_truncation();
    return 0;
}
