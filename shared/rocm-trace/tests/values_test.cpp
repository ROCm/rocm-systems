// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "values_c.h"

#include <rocm/trace/values.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {
// SHA-1 (FIPS 180-4), only to derive ETW provider GUIDs in this test.
std::array<uint8_t, 20> sha1(const std::vector<uint8_t> &message) {
  auto rotl = [](uint32_t x, int n) { return (x << n) | (x >> (32 - n)); };
  auto h = std::array<uint32_t, 5>{0x67452301u, 0xEFCDAB89u, 0x98BADCFEu,
                                   0x10325476u, 0xC3D2E1F0u};
  auto data = message;
  const auto bit_length = static_cast<uint64_t>(message.size()) * 8;
  data.push_back(0x80);
  while (data.size() % 64 != 56)
    data.push_back(0);
  for (int i = 7; i >= 0; --i)
    data.push_back(static_cast<uint8_t>(bit_length >> (8 * i)));
  for (size_t off = 0; off < data.size(); off += 64) {
    auto w = std::array<uint32_t, 80>{};
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t{data[off + 4 * i]} << 24) |
             (uint32_t{data[off + 4 * i + 1]} << 16) |
             (uint32_t{data[off + 4 * i + 2]} << 8) | data[off + 4 * i + 3];
    for (int i = 16; i < 80; ++i)
      w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    auto a = h[0];
    auto b = h[1];
    auto c = h[2];
    auto d = h[3];
    auto e = h[4];
    for (int i = 0; i < 80; ++i) {
      auto f = uint32_t{};
      auto k = uint32_t{};
      if (i < 20) {
        f = (b & c) | (~b & d);
        k = 0x5A827999u;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1u;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDCu;
      } else {
        f = b ^ c ^ d;
        k = 0xCA62C1D6u;
      }
      const auto t = rotl(a, 5) + f + e + k + w[i];
      e = d;
      d = c;
      c = rotl(b, 30);
      b = a;
      a = t;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
  }
  auto out = std::array<uint8_t, 20>{};
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 4; ++j)
      out[4 * i + j] = static_cast<uint8_t>(h[i] >> (24 - 8 * j));
  return out;
}

// ETW's provider GUID for a name (as .NET EventSource): SHA-1 over a fixed
// namespace and the upper-cased UTF-16BE name, truncated to 16 bytes,
// version 5.
std::string etw_guid(const std::string &name) {
  auto input =
      std::vector<uint8_t>{0x48, 0x2C, 0x2D, 0xB2, 0xC3, 0x90, 0x47, 0xC8,
                           0x87, 0xF8, 0x1A, 0x15, 0xBF, 0xC1, 0x30, 0xFB};
  for (char ch : name) {
    const auto upper =
        (ch >= 'a' && ch <= 'z') ? static_cast<char>(ch - 32) : ch;
    input.push_back(0);
    input.push_back(static_cast<uint8_t>(upper));
  }
  auto b = sha1(input);
  b[7] = static_cast<uint8_t>((b[7] & 0x0F) | 0x50);
  // The first three fields are little-endian.
  auto text = std::array<char, 37>{};
  std::snprintf(text.data(), text.size(),
                "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
                "%02x%02x%02x%02x%02x%02x",
                b[3], b[2], b[1], b[0], b[5], b[4], b[7], b[6], b[8], b[9],
                b[10], b[11], b[12], b[13], b[14], b[15]);
  return std::string{text.data()};
}

#define ROCM_TRACE_TEST_EXPAND(x) x
#define ROCM_TRACE_TEST_AS_ARRAY(...)                                          \
  std::array<unsigned long, 11> { __VA_ARGS__ }
#define ROCM_TRACE_TEST_INIT(ID)                                               \
  ROCM_TRACE_TEST_EXPAND(                                                      \
      ROCM_TRACE_TEST_AS_ARRAY ROCM_TRACE_PROVIDER_##ID##_GUID_INIT)

struct provider {
  std::string name;
  std::string guid;
  std::array<unsigned long, 11> init;
};

std::vector<provider> providers() {
  return {
      {ROCM_TRACE_PROVIDER_HIP_NAME, ROCM_TRACE_PROVIDER_HIP_GUID,
       ROCM_TRACE_TEST_INIT(HIP)},
      {ROCM_TRACE_PROVIDER_HSA_NAME, ROCM_TRACE_PROVIDER_HSA_GUID,
       ROCM_TRACE_TEST_INIT(HSA)},
      {ROCM_TRACE_PROVIDER_RCCL_NAME, ROCM_TRACE_PROVIDER_RCCL_GUID,
       ROCM_TRACE_TEST_INIT(RCCL)},
      {ROCM_TRACE_PROVIDER_ROCDECODE_NAME, ROCM_TRACE_PROVIDER_ROCDECODE_GUID,
       ROCM_TRACE_TEST_INIT(ROCDECODE)},
      {ROCM_TRACE_PROVIDER_ROCJPEG_NAME, ROCM_TRACE_PROVIDER_ROCJPEG_GUID,
       ROCM_TRACE_TEST_INIT(ROCJPEG)},
      {ROCM_TRACE_PROVIDER_ROCTX_NAME, ROCM_TRACE_PROVIDER_ROCTX_GUID,
       ROCM_TRACE_TEST_INIT(ROCTX)},
      {ROCM_TRACE_PROVIDER_ROCSHMEM_NAME, ROCM_TRACE_PROVIDER_ROCSHMEM_GUID,
       ROCM_TRACE_TEST_INIT(ROCSHMEM)},
      {ROCM_TRACE_PROVIDER_HIPFILE_NAME, ROCM_TRACE_PROVIDER_HIPFILE_GUID,
       ROCM_TRACE_TEST_INIT(HIPFILE)},
  };
}

std::string init_to_string(const std::array<unsigned long, 11> &v) {
  auto text = std::array<char, 37>{};
  std::snprintf(text.data(), text.size(),
                "%08lx-%04lx-%04lx-%02lx%02lx-%02lx%02lx%02lx%02lx%02lx%02lx",
                v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9],
                v[10]);
  return std::string{text.data()};
}
} // namespace

TEST(rocm_trace_tests, etw_guid_matches_documented_example) {
  // The example from Microsoft's TraceLoggingProvider.h documentation.
  EXPECT_EQ(etw_guid("MyCompany.MyComponent"),
            "ce5fa4ea-ab00-5402-8b76-9f76ac858fb5");
}

TEST(rocm_trace_tests, provider_guids_are_etw_name_hashes) {
  for (const auto &p : providers()) {
    EXPECT_EQ(p.guid, etw_guid(p.name)) << p.name;
    EXPECT_EQ(init_to_string(p.init), p.guid) << p.name;
  }
}

TEST(rocm_trace_tests, provider_names_are_rocm_library) {
  for (const auto &p : providers()) {
    EXPECT_EQ(p.name.rfind("rocm_", 0), 0u) << p.name;
    EXPECT_LT(p.name.size(), 64u) << p.name;
  }
}

TEST(rocm_trace_tests, keywords_are_distinct_single_bits) {
  const auto keywords =
      std::array<uint64_t, 7>{ROCM_TRACE_KW_API,       ROCM_TRACE_KW_DISPATCH,
                              ROCM_TRACE_KW_STATE,     ROCM_TRACE_KW_SERIALIZE,
                              ROCM_TRACE_KW_COPY_TIME, ROCM_TRACE_KW_CODE_BYTES,
                              ROCM_TRACE_KW_ALLOC};
  auto seen = uint64_t{0};
  for (auto kw : keywords) {
    EXPECT_NE(kw, 0u);
    EXPECT_EQ(kw & (kw - 1), 0u) << "not a single bit: " << kw;
    EXPECT_EQ(seen & kw, 0u) << "duplicate keyword: " << kw;
    seen |= kw;
  }
}

TEST(rocm_trace_tests, levels_order_information_before_verbose) {
  EXPECT_LT(ROCM_TRACE_ETW_LEVEL_INFORMATION, ROCM_TRACE_ETW_LEVEL_VERBOSE);
  EXPECT_LT(ROCM_TRACE_LTTNG_LEVEL_INFORMATION, ROCM_TRACE_LTTNG_LEVEL_VERBOSE);
}

TEST(rocm_trace_tests, capture_levels_order_none_scalars_full) {
  EXPECT_EQ(ROCM_TRACE_CAPTURE_NONE, 0);
  EXPECT_LT(ROCM_TRACE_CAPTURE_NONE, ROCM_TRACE_CAPTURE_SCALARS);
  EXPECT_LT(ROCM_TRACE_CAPTURE_SCALARS, ROCM_TRACE_CAPTURE_FULL);
}

TEST(rocm_trace_tests, string_cap_matches_the_contract) {
  EXPECT_EQ(ROCM_TRACE_STRING_CAP, 256); // RC-GEN-8
}

TEST(rocm_trace_tests, c_and_cpp_see_the_same_values) {
  EXPECT_EQ(rocm_trace_test_c_keywords(),
            ROCM_TRACE_KW_API | ROCM_TRACE_KW_DISPATCH | ROCM_TRACE_KW_STATE |
                ROCM_TRACE_KW_SERIALIZE | ROCM_TRACE_KW_COPY_TIME |
                ROCM_TRACE_KW_CODE_BYTES | ROCM_TRACE_KW_ALLOC);
  EXPECT_EQ(std::string{rocm_trace_test_c_hip_name()},
            ROCM_TRACE_PROVIDER_HIP_NAME);
  EXPECT_EQ(std::string{rocm_trace_test_c_hsa_guid()},
            ROCM_TRACE_PROVIDER_HSA_GUID);
  EXPECT_EQ(rocm_trace_test_c_values_version(), ROCM_TRACE_VALUES_VERSION);
  EXPECT_EQ(rocm_trace_test_c_capture_full(), ROCM_TRACE_CAPTURE_FULL);
  EXPECT_EQ(rocm_trace_test_c_string_cap(), ROCM_TRACE_STRING_CAP);
}
