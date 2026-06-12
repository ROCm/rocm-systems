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

// Regression tests for the CPER read path via
// amdsmi_get_gpu_cper_entries_by_path(); no GPU required.
// ROCM-25398: a zero-byte CPER node must not abort the process.
// ROCM-25954: an empty ring returns SUCCESS with zero entries, not an error.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_cper.h"

namespace {

// Runs the CPER read path against a file. Out-params report the final
// entry_count and buf_size.
static amdsmi_status_t CallCperByPath(const char* path, uint64_t* out_entry_count = nullptr,
                                      uint64_t* out_buf_size = nullptr) {
  std::vector<char> cper_data(4096, 0);
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(8, nullptr);
  uint64_t buf_size = cper_data.size();
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  amdsmi_status_t status = amdsmi_get_gpu_cper_entries_by_path(
      path, 0xFFFFFFFF, cper_data.data(), &buf_size, cper_hdrs.data(), &entry_count, &cursor,
      /*product_serial=*/0);

  if (out_entry_count) *out_entry_count = entry_count;
  if (out_buf_size) *out_buf_size = buf_size;
  return status;
}

}  // namespace

// Zero-size regular file mimics an empty amdgpu_ring_cper node: read() returns
// 0 bytes. Empty ring -> SUCCESS with zero entries, and no crash.
TEST(amdsmitstReadOnly, CperReadZeroSizeFile) {
  std::string tmpl = "/tmp/amdsmi_cper_zero_XXXXXX";
  int fd = mkstemp(tmpl.data());
  ASSERT_NE(fd, -1) << "failed to create temp file";
  close(fd);  // leave it empty -> st_size == 0

  uint64_t entry_count = 0;
  uint64_t buf_size = 0;
  amdsmi_status_t status = CallCperByPath(tmpl.c_str(), &entry_count, &buf_size);
  unlink(tmpl.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(entry_count, 0u);
  EXPECT_EQ(buf_size, 0u);
}

// Missing path -> NOT_SUPPORTED (stat() fails), no crash.
TEST(amdsmitstReadOnly, CperReadMissingFile) {
  amdsmi_status_t status = CallCperByPath("/tmp/amdsmi_cper_does_not_exist_12345");
  EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
}
