/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Checks in the RDC library that need no GPU: the C API rejects job IDs and
// names that do not fit it before it uses its handle, and the cache gives each
// value it returns a status.

#include <gtest/gtest.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#include "rdc/rdc.h"
#include "rdc_lib/impl/RdcCacheManagerImpl.h"

namespace {

// A handle the checks below must reject the arguments before they use: using
// it crashes.
const rdc_handle_t kBadHandle = reinterpret_cast<rdc_handle_t>(uintptr_t{8});

// Runs a check in a child process, which a call that uses kBadHandle kills.
// The child fails on its own failures only, not on those the test had before.
void in_child(const std::function<void()>& check) {
  const ::testing::TestResult& result =
      *::testing::UnitTest::GetInstance()->current_test_info()->result();
  const int parts_before = result.total_part_count();
  fflush(nullptr);
  const pid_t pid = fork();
  ASSERT_NE(pid, -1);
  if (pid == 0) {
    check();
    bool failed = false;
    for (int i = parts_before; i < result.total_part_count(); ++i) {
      failed = failed || result.GetTestPartResult(i).failed();
    }
    fflush(nullptr);
    _exit(failed ? 1 : 0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
      << (WIFSIGNALED(status) ? "killed by signal " + std::to_string(WTERMSIG(status))
                              : "exit status " + std::to_string(WEXITSTATUS(status)));
}

TEST(ApiChecks, JobCallsRejectJobIdsThatDoNotFit) {
  char unterminated[64];
  std::memset(unterminated, 'j', sizeof(unterminated));  // no NUL in its 64 bytes
  const char* rejected[] = {unterminated, "a\nb", "tab\there", "del\x7f", nullptr};
  for (const char* job_id : rejected) {
    in_child([&] {
      rdc_job_info_t info = {};
      EXPECT_EQ(rdc_job_start_stats(kBadHandle, 1, job_id, 1000000), RDC_ST_BAD_PARAMETER);
      EXPECT_EQ(rdc_job_get_stats(kBadHandle, job_id, &info), RDC_ST_BAD_PARAMETER);
      EXPECT_EQ(rdc_job_stop_stats(kBadHandle, job_id), RDC_ST_BAD_PARAMETER);
      EXPECT_EQ(rdc_job_remove(kBadHandle, job_id), RDC_ST_BAD_PARAMETER);
    });
  }
}

TEST(ApiChecks, GroupsRejectNamesThatDoNotFit) {
  const std::string too_long(RDC_MAX_STR_LENGTH, 'g');
  in_child([&] {
    rdc_gpu_group_t group_id = 0;
    EXPECT_EQ(rdc_group_gpu_create(kBadHandle, RDC_GROUP_EMPTY, too_long.c_str(), &group_id),
              RDC_ST_BAD_PARAMETER);
    EXPECT_EQ(rdc_group_gpu_create(kBadHandle, RDC_GROUP_EMPTY, nullptr, &group_id),
              RDC_ST_BAD_PARAMETER);
    rdc_field_t fields[] = {RDC_FI_GPU_CLOCK};
    rdc_field_grp_t field_group_id = 0;
    EXPECT_EQ(rdc_group_field_create(kBadHandle, 1, fields, too_long.c_str(), &field_group_id),
              RDC_ST_BAD_PARAMETER);
  });
}

// Values the cache returns carry RDC_ST_OK and all of their data, also into a
// struct the caller did not clear.
TEST(CacheValues, ValuesCarryTheirStatusAndData) {
  amd::rdc::RdcCacheManagerImpl cache;
  rdc_field_value clock = {};
  clock.field_id = RDC_FI_GPU_CLOCK;
  clock.type = INTEGER;
  clock.ts = 1000;
  clock.value.l_int = 5;
  ASSERT_EQ(cache.rdc_update_cache(0, clock), RDC_ST_OK);
  rdc_field_value blob = {};
  blob.field_id = RDC_FI_GPU_MEMORY_USAGE;
  blob.type = BLOB;
  blob.ts = 1000;
  std::strcpy(blob.value.str, "blob data");  // NOLINT
  ASSERT_EQ(cache.rdc_update_cache(0, blob), RDC_ST_OK);

  rdc_field_value value;
  std::memset(&value, 0xA5, sizeof(value));
  ASSERT_EQ(cache.rdc_field_get_latest_value(0, RDC_FI_GPU_CLOCK, &value), RDC_ST_OK);
  EXPECT_EQ(value.status, RDC_ST_OK);
  EXPECT_EQ(value.value.l_int, 5);

  uint64_t next = 0;
  std::memset(&value, 0xA5, sizeof(value));
  ASSERT_EQ(cache.rdc_field_get_value_since(0, RDC_FI_GPU_CLOCK, 0, &next, &value), RDC_ST_OK);
  EXPECT_EQ(value.status, RDC_ST_OK);
  EXPECT_EQ(value.value.l_int, 5);

  std::memset(&value, 0xA5, sizeof(value));
  ASSERT_EQ(cache.rdc_field_get_value_since(0, RDC_FI_GPU_MEMORY_USAGE, 0, &next, &value),
            RDC_ST_OK);
  EXPECT_EQ(value.status, RDC_ST_OK);
  EXPECT_EQ(value.type, BLOB);
  EXPECT_STREQ(value.value.str, "blob data");
}

}  // namespace
