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

// Regression tests for the CPER read path exercised via
// amdsmi_get_gpu_cper_entries_by_path(). They require no GPU.
//
// ROCM-25398: amdsmi_get_gpu_cper_entries crashed with "free(): invalid
// pointer" / SIGABRT when the CPER node reported a zero byte size (the case for
// debugfs amdgpu_ring_cper nodes, whose content is generated on read). The read
// path must never abort the process.
//
// ROCM-25954: an empty CPER ring (zero bytes read) is a valid "no records"
// state and must return AMDSMI_STATUS_SUCCESS with zero entries, not an error.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_cper.h"

namespace {

// Drives the CPER read path against a caller supplied file and returns the
// status. All output parameters are valid so that validation succeeds and the
// file read is actually attempted. The final entry_count and buf_size are
// reported back through the optional out-parameters.
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

// A zero-size regular file reproduces the debugfs amdgpu_ring_cper case exactly:
// stat() reports S_ISREG and read() returns 0 bytes. This is an empty ring (no
// records), which must return AMDSMI_STATUS_SUCCESS with zero entries. It must
// also never abort the process (ROCM-25398).
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

// A non-existent path must report a clean status, never crash. stat() fails for
// a missing path, so the read helper reports AMDSMI_STATUS_NOT_SUPPORTED.
TEST(amdsmitstReadOnly, CperReadMissingFile) {
  amdsmi_status_t status = CallCperByPath("/tmp/amdsmi_cper_does_not_exist_12345");
  EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
}
