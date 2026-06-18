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

// Mock-value tests for the CPER read path via
// amdsmi_get_gpu_cper_entries_by_path(); no GPU required.
// Fixtures live in functional/mock_values/ (real GFX950 CPER captures,
// sanitized: timestamps normalized and the injected error-payload body zeroed).
// See that folder's README for provenance. The fixtures are installed next to
// the test binary and located at runtime (see MockDir); AMDSMI_TEST_MOCK_DIR is
// the build-tree fallback for in-tree runs.
// Single-record parsing is also covered synthetically in cper_read.cc; these
// fixtures add real-capture coverage: header byte alignment, multi-record
// rings, and severity_mask filtering.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <climits>
#include <cstdint>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_cper.h"

#ifndef AMDSMI_TEST_MOCK_DIR
#error "AMDSMI_TEST_MOCK_DIR must be defined by the build"
#endif

namespace {

// Severity bit in the severity_mask for a given amdsmi_cper_sev_t.
constexpr uint32_t SevBit(amdsmi_cper_sev_t sev) { return 1u << static_cast<uint32_t>(sev); }

// Directory holding the mock fixtures. Prefer a mock_values/ folder installed
// next to the test binary (the packaged layout, portable across machines); fall
// back to the build-tree path baked in at compile time for in-tree runs.
std::string MockDir() {
  char exe[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (n > 0) {
    exe[n] = '\0';
    std::string dir(exe);
    std::string::size_type slash = dir.find_last_of('/');
    if (slash != std::string::npos) {
      std::string candidate = dir.substr(0, slash) + "/mock_values";
      struct stat st;
      if (stat(candidate.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        return candidate;
      }
    }
  }
  return AMDSMI_TEST_MOCK_DIR;
}

std::string MockPath(const char* name) { return MockDir() + "/" + name; }

// Reads a mock fixture and returns the parsed severities of the accepted
// records (those passing severity_mask). entry_count and buf_size are reported
// via out-params.
std::vector<amdsmi_cper_sev_t> ReadMock(const char* name, uint32_t severity_mask,
                                        amdsmi_status_t* out_status,
                                        uint64_t* out_entry_count = nullptr,
                                        uint64_t* out_buf_size = nullptr) {
  std::vector<char> cper_data(8192, 0);
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(16, nullptr);
  uint64_t buf_size = cper_data.size();
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  amdsmi_status_t status = amdsmi_get_gpu_cper_entries_by_path(
      MockPath(name).c_str(), severity_mask, cper_data.data(), &buf_size, cper_hdrs.data(),
      &entry_count, &cursor, /*product_serial=*/0);

  *out_status = status;
  if (out_entry_count) *out_entry_count = entry_count;
  if (out_buf_size) *out_buf_size = buf_size;

  std::vector<amdsmi_cper_sev_t> severities;
  for (uint64_t i = 0; i < entry_count; ++i) {
    severities.push_back(cper_hdrs[i]->error_severity);
  }
  return severities;
}

}  // namespace

// Five non-fatal corrected records parse to five entries, all severity 2.
TEST(amdsmitstReadOnly, CperMockCorrectedRecords) {
  amdsmi_status_t status = AMDSMI_STATUS_UNKNOWN_ERROR;
  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  auto sevs = ReadMock("cper_corrected.cper", 0xFFFFFFFF, &status, &entry_count, &buf_size);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 5u);
  EXPECT_GT(buf_size, 0u);
  ASSERT_EQ(sevs.size(), 5u);
  for (amdsmi_cper_sev_t sev : sevs) {
    EXPECT_EQ(sev, AMDSMI_CPER_SEV_NON_FATAL_CORRECTED);
  }
}

// A single non-fatal uncorrected record parses to one entry, severity 0.
TEST(amdsmitstReadOnly, CperMockUncorrectedRecord) {
  amdsmi_status_t status = AMDSMI_STATUS_UNKNOWN_ERROR;
  uint64_t entry_count = 0;
  auto sevs = ReadMock("cper_uncorrected.cper", 0xFFFFFFFF, &status, &entry_count);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 1u);
  ASSERT_EQ(sevs.size(), 1u);
  EXPECT_EQ(sevs[0], AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED);
}

// A single fatal record parses to one entry, severity 1.
TEST(amdsmitstReadOnly, CperMockFatalRecord) {
  amdsmi_status_t status = AMDSMI_STATUS_UNKNOWN_ERROR;
  uint64_t entry_count = 0;
  auto sevs = ReadMock("cper_fatal.cper", 0xFFFFFFFF, &status, &entry_count);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 1u);
  ASSERT_EQ(sevs.size(), 1u);
  EXPECT_EQ(sevs[0], AMDSMI_CPER_SEV_FATAL);
}

// The mixed ring (5 corrected + 1 uncorrected + 1 fatal) parses to seven
// entries under a full severity mask.
TEST(amdsmitstReadOnly, CperMockMixedFullMask) {
  amdsmi_status_t status = AMDSMI_STATUS_UNKNOWN_ERROR;
  uint64_t entry_count = 0;
  auto sevs = ReadMock("cper_mixed.cper", 0xFFFFFFFF, &status, &entry_count);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 7u);
  ASSERT_EQ(sevs.size(), 7u);

  int corrected = 0, uncorrected = 0, fatal = 0;
  for (amdsmi_cper_sev_t sev : sevs) {
    if (sev == AMDSMI_CPER_SEV_NON_FATAL_CORRECTED)
      ++corrected;
    else if (sev == AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED)
      ++uncorrected;
    else if (sev == AMDSMI_CPER_SEV_FATAL)
      ++fatal;
  }
  EXPECT_EQ(corrected, 5);
  EXPECT_EQ(uncorrected, 1);
  EXPECT_EQ(fatal, 1);
}

// The severity_mask filters the mixed ring: each single-severity mask yields
// only the matching records.
TEST(amdsmitstReadOnly, CperMockSeverityMaskFilter) {
  amdsmi_status_t status = AMDSMI_STATUS_UNKNOWN_ERROR;
  uint64_t entry_count = 0;

  auto fatal_only =
      ReadMock("cper_mixed.cper", SevBit(AMDSMI_CPER_SEV_FATAL), &status, &entry_count);
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 1u);
  ASSERT_EQ(fatal_only.size(), 1u);
  EXPECT_EQ(fatal_only[0], AMDSMI_CPER_SEV_FATAL);

  auto corrected_only = ReadMock("cper_mixed.cper", SevBit(AMDSMI_CPER_SEV_NON_FATAL_CORRECTED),
                                 &status, &entry_count);
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 5u);
  ASSERT_EQ(corrected_only.size(), 5u);
  for (amdsmi_cper_sev_t sev : corrected_only) {
    EXPECT_EQ(sev, AMDSMI_CPER_SEV_NON_FATAL_CORRECTED);
  }

  auto uncorrected_only = ReadMock("cper_mixed.cper", SevBit(AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED),
                                   &status, &entry_count);
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 1u);
  ASSERT_EQ(uncorrected_only.size(), 1u);
  EXPECT_EQ(uncorrected_only[0], AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED);
}

// A zero severity_mask rejects every record: SUCCESS with no entries.
TEST(amdsmitstReadOnly, CperMockSeverityMaskRejectAll) {
  amdsmi_status_t status = AMDSMI_STATUS_UNKNOWN_ERROR;
  uint64_t entry_count = 99;
  uint64_t buf_size = 99;
  auto sevs = ReadMock("cper_mixed.cper", 0u, &status, &entry_count, &buf_size);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
  EXPECT_TRUE(sevs.empty());
}
