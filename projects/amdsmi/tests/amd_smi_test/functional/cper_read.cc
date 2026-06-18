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

// Regression tests for ROCM-25398: amdsmi_get_gpu_cper_entries crashed with
// "free(): invalid pointer" / SIGABRT when the CPER node reported a zero byte
// size (the case for debugfs amdgpu_ring_cper nodes, whose content is generated
// on read). These tests exercise the read error path directly via
// amdsmi_get_gpu_cper_entries_by_path() and require no GPU. They must return a
// clean error status without aborting the process.

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
// file read is actually attempted.
static amdsmi_status_t CallCperByPath(const char* path) {
  std::vector<char> cper_data(4096, 0);
  std::vector<amdsmi_cper_hdr_t*> cper_hdrs(8, nullptr);
  uint64_t buf_size = cper_data.size();
  uint64_t entry_count = cper_hdrs.size();
  uint64_t cursor = 0;

  return amdsmi_get_gpu_cper_entries_by_path(path, 0xFFFFFFFF, cper_data.data(), &buf_size,
                                             cper_hdrs.data(), &entry_count, &cursor,
                                             /*product_serial=*/0);
}

}  // namespace

// A zero-size regular file reproduces the debugfs amdgpu_ring_cper case exactly:
// stat() reports S_ISREG with st_size == 0. Before the fix this aborted the
// process inside the std::ifstream filebuf destructor.
TEST(amdsmitstReadOnly, CperReadZeroSizeFile) {
  std::string tmpl = "/tmp/amdsmi_cper_zero_XXXXXX";
  int fd = mkstemp(tmpl.data());
  ASSERT_NE(fd, -1) << "failed to create temp file";
  close(fd);  // leave it empty -> st_size == 0

  amdsmi_status_t status = CallCperByPath(tmpl.c_str());
  unlink(tmpl.c_str());

  EXPECT_EQ(status, AMDSMI_STATUS_FILE_ERROR);
}

// A non-existent path must report a clean status, never crash. stat() fails for
// a missing path, so the read helper reports AMDSMI_STATUS_NOT_SUPPORTED.
TEST(amdsmitstReadOnly, CperReadMissingFile) {
  amdsmi_status_t status = CallCperByPath("/tmp/amdsmi_cper_does_not_exist_12345");
  EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
}
