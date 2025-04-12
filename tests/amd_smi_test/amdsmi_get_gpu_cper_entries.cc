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

#include <cstdint>
#include <gtest/gtest.h>

#include "amd_smi/amdsmi.h"
#include "rocm_smi/rocm_smi_logger.h"

extern amdsmi_status_t
amdsmi_get_gpu_cper_entries_by_path(
    const std::string &amdgpu_ring_cper_file,
    uint32_t severity_mask,
    char *cper_data,
    uint64_t *buf_size,
    amdsmi_cper_hdr_t **cper_hdrs,
    uint64_t *entry_count,
    uint64_t *cursor);

class CperEntriesTest : public testing::Test{
    //class object public so that it is accessible
    //within the tests that are written
public:
  CperEntriesTest() {
      setenv("CPER_SYS_ROOT", CPER_SYS_ROOT, 1);
      ROCmLogging::Logger::getInstance()->
        updateLogLevel(ROCmLogging::LogLevel::LOG_LEVEL_DEBUG);
  }
};

TEST_F(CperEntriesTest, TestNullCperData){
  uint32_t gpu_num = 9;
  uint32_t severity_mask = amdsmi_cper_sev_t::AMDSMI_CPER_SEV_FATAL;
  char *cper_data = nullptr;
  uint64_t buf_size = 0;
  amdsmi_cper_hdr_t *cper_hdrs = nullptr;
  uint64_t entry_count = 0;
  uint64_t cursor = 0;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data,
    nullptr,
    &cper_hdrs,
    &entry_count,
    &cursor);
  ASSERT_EQ(err, AMDSMI_STATUS_OUT_OF_RESOURCES);
}

TEST_F(CperEntriesTest, TestNullBufferSize){
  uint32_t gpu_num = 9;
  uint32_t severity_mask = amdsmi_cper_sev_t::AMDSMI_CPER_SEV_FATAL;
  uint64_t buf_size = 0;
  auto cper_data = std::make_unique<char[]>(buf_size);
  amdsmi_cper_hdr_t *cper_hdrs = nullptr;
  uint64_t entry_count = 0;
  uint64_t cursor = 0;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data.get(),
    nullptr,
    &cper_hdrs,
    &entry_count,
    &cursor);
  ASSERT_EQ(err, AMDSMI_STATUS_OUT_OF_RESOURCES);
}

TEST_F(CperEntriesTest, TestNullCperHeaders){
  uint32_t gpu_num = 9;
  uint32_t severity_mask = amdsmi_cper_sev_t::AMDSMI_CPER_SEV_FATAL;
  uint64_t buf_size = 4 * (1<<20); //4 MB;
  auto cper_data = std::make_unique<char[]>(buf_size);
  amdsmi_cper_hdr_t *cper_hdrs = nullptr;
  uint64_t entry_count = 0;
  uint64_t cursor = 0;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data.get(),
    &buf_size,
    &cper_hdrs,
    &entry_count,
    &cursor);
  ASSERT_EQ(err, AMDSMI_STATUS_OUT_OF_RESOURCES);
}

TEST_F(CperEntriesTest, TestNullCperHeaderEntryCount){
  uint32_t gpu_num = 9;
  uint32_t severity_mask = amdsmi_cper_sev_t::AMDSMI_CPER_SEV_FATAL;
  uint64_t buf_size = 4 * (1<<20); //4 MB;
  auto cper_data = std::make_unique<char[]>(buf_size);
  uint64_t entry_count = 0;
  auto cper_hdrs = std::make_unique<amdsmi_cper_hdr_t*[]>(entry_count);
  uint64_t cursor = 0;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data.get(),
    &buf_size,
    cper_hdrs.get(),
    nullptr,
    &cursor);
  ASSERT_EQ(err, AMDSMI_STATUS_OUT_OF_RESOURCES);
}

TEST_F(CperEntriesTest, TestNotEnoughBufferSize){
  uint32_t gpu_num = 9;
  uint32_t severity_mask =
    AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED|
    AMDSMI_CPER_SEV_NON_FATAL_CORRECTED|
    AMDSMI_CPER_SEV_FATAL;
  uint64_t buf_size = 1024;
  auto cper_data = std::make_unique<char[]>(buf_size);
  uint64_t entry_count = 10;
  auto cper_hdrs = std::make_unique<amdsmi_cper_hdr_t*[]>(entry_count);
  uint64_t cursor = 0;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data.get(),
    &buf_size,
    cper_hdrs.get(),
    &entry_count,
    &cursor);
  ASSERT_EQ(err, AMDSMI_STATUS_MORE_DATA);
  ASSERT_EQ(entry_count, 2);
}

TEST_F(CperEntriesTest, TestNotEnoughHeaderPtrs){
  uint32_t gpu_num = 9;
  uint32_t severity_mask =
    AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED|
    AMDSMI_CPER_SEV_NON_FATAL_CORRECTED|
    AMDSMI_CPER_SEV_FATAL;
  uint64_t buf_size = 4 * (1<<20); //4 MB;
  auto cper_data = std::make_unique<char[]>(buf_size);
  uint64_t entry_count = 4;
  auto cper_hdrs = std::make_unique<amdsmi_cper_hdr_t*[]>(entry_count);
  uint64_t cursor = 0;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data.get(),
    &buf_size,
    cper_hdrs.get(),
    &entry_count,
    &cursor);
  ASSERT_EQ(entry_count, 4);
  ASSERT_EQ(err, AMDSMI_STATUS_MORE_DATA);
}

TEST_F(CperEntriesTest, TestGetsAllSeverityErrors){
  uint32_t gpu_num = 9;
  uint32_t severity_mask =
    (1 << AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED)|
    (1 << AMDSMI_CPER_SEV_NON_FATAL_CORRECTED)|
    (1 << AMDSMI_CPER_SEV_FATAL);
  uint64_t buf_size = 4 * (1<<20); //4 MB;
  auto cper_data = std::make_unique<char[]>(buf_size);
  uint64_t entry_count = 10;
  auto cper_hdrs = std::make_unique<amdsmi_cper_hdr_t*[]>(entry_count);
  uint64_t cursor = 0;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data.get(),
    &buf_size,
    cper_hdrs.get(),
    &entry_count,
    &cursor);
  ASSERT_EQ(entry_count, 8);
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CperEntriesTest, TestGetsCorrectableSeverityErrors){
  uint32_t gpu_num = 9;
  uint32_t severity_mask =
    (1 << AMDSMI_CPER_SEV_NON_FATAL_CORRECTED);
  uint64_t buf_size = 4 * (1<<20); //4 MB;
  auto cper_data = std::make_unique<char[]>(buf_size);
  uint64_t entry_count = 10;
  auto cper_hdrs = std::make_unique<amdsmi_cper_hdr_t*[]>(entry_count);
  uint64_t cursor = 0;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data.get(),
    &buf_size,
    cper_hdrs.get(),
    &entry_count,
    &cursor);
  ASSERT_EQ(entry_count, 1);
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CperEntriesTest, TestGetsFatalSeverityErrors){
  uint32_t gpu_num = 9;
  uint32_t severity_mask =
    (1 << AMDSMI_CPER_SEV_FATAL);
  uint64_t buf_size = 4 * (1<<20); //4 MB;
  auto cper_data = std::make_unique<char[]>(buf_size);
  uint64_t entry_count = 10;
  auto cper_hdrs = std::make_unique<amdsmi_cper_hdr_t*[]>(entry_count);
  uint64_t cursor = 0;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data.get(),
    &buf_size,
    cper_hdrs.get(),
    &entry_count,
    &cursor);
  ASSERT_EQ(entry_count, 1);
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CperEntriesTest, TestGetsUncorrectableSeverityErrors){
  uint32_t gpu_num = 9;
  uint32_t severity_mask =
    (1 << AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED);
  uint64_t buf_size = 4 * (1<<20); //4 MB;
  auto cper_data = std::make_unique<char[]>(buf_size);
  uint64_t entry_count = 10;
  auto cper_hdrs = std::make_unique<amdsmi_cper_hdr_t*[]>(entry_count);
  uint64_t cursor = 0;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data.get(),
    &buf_size,
    cper_hdrs.get(),
    &entry_count,
    &cursor);
  ASSERT_EQ(entry_count, 6);
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CperEntriesTest, TestCursor5GetsLast3HeadersGivenTotal8Headers){
  uint32_t gpu_num = 9;
  uint32_t severity_mask =
    (1 << AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED)|
    (1 << AMDSMI_CPER_SEV_NON_FATAL_CORRECTED)|
    (1 << AMDSMI_CPER_SEV_FATAL);
  uint64_t buf_size = 4 * (1<<20); //4 MB;
  auto cper_data = std::make_unique<char[]>(buf_size);
  uint64_t entry_count = 10;
  auto cper_hdrs = std::make_unique<amdsmi_cper_hdr_t*[]>(entry_count);
  uint64_t cursor = 5;
  std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
  amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
    gpu,
    severity_mask,
    cper_data.get(),
    &buf_size,
    cper_hdrs.get(),
    &entry_count,
    &cursor);
  ASSERT_EQ(entry_count, 3);
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CperEntriesTest, TestCursorAdvances){
  uint32_t gpu_num = 9;
  uint32_t severity_mask =
    (1 << AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED)|
    (1 << AMDSMI_CPER_SEV_NON_FATAL_CORRECTED)|
    (1 << AMDSMI_CPER_SEV_FATAL);
  uint64_t buf_size = 512;//4 * (1<<20); //4 MB;
  auto cper_data = std::make_unique<char[]>(buf_size);
  uint64_t entry_count = 10;
  auto cper_hdrs = std::make_unique<amdsmi_cper_hdr_t*[]>(entry_count);

  uint64_t buf_size_original = buf_size;
  uint64_t entry_count_original = entry_count;
  uint64_t cursor_idx = 0;
  uint64_t cursor = 0;
  while(true) {
    std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
    amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
      gpu,
      severity_mask,
      cper_data.get(),
      &buf_size,
      cper_hdrs.get(),
      &entry_count,
      &cursor);
    ASSERT_EQ(entry_count, 1);
    ASSERT_EQ(cursor, ++cursor_idx);
    ASSERT_TRUE(err == AMDSMI_STATUS_MORE_DATA || err == AMDSMI_STATUS_SUCCESS);
    if(err == AMDSMI_STATUS_SUCCESS) {
      break;
    }
    buf_size = buf_size_original;
    entry_count = entry_count_original;
  }
}

TEST_F(CperEntriesTest, TestGetsCorrectHeaderCountFromAllDevices) {
  //we can get these deviceids by calling:
  // ls -alh tests/amd_smi_test/cper/sys/kernel/debug/dri/
  static constexpr int deviceids[] = { 1, 9, 17, 25, 33};
  //we can get the numbers in the expected_num_headers array below by calling:
  // hexdump -C tests/amd_smi_test/cper/sys/kernel/debug/dri/<deviceid>/amdgpu_ring_cper | grep CPER|wc -l
  // where <deviceid> is one of the entries in the deviceids array above.
  static constexpr int expected_num_headers[] = { 19, 8, 7, 4, 7};

  for(int device_idx = 0;
    device_idx < sizeof(deviceids)/sizeof(deviceids[0]);
    ++device_idx) {

    uint32_t gpu_num = deviceids[device_idx];
    uint32_t severity_mask =
      (1 << AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED)|
      (1 << AMDSMI_CPER_SEV_NON_FATAL_CORRECTED)|
      (1 << AMDSMI_CPER_SEV_FATAL);
    uint64_t buf_size = 4 * (1<<20); //4 MB;
    auto cper_data = std::make_unique<char[]>(buf_size);
    uint64_t entry_count = 20;
    auto cper_hdrs = std::make_unique<amdsmi_cper_hdr_t*[]>(entry_count);
    uint64_t cursor = 0;

    std::string gpu = std::string(CPER_SYS_ROOT) + "/sys/kernel/debug/dri/" + std::to_string(gpu_num) + "/amdgpu_ring_cper";
    amdsmi_status_t err = amdsmi_get_gpu_cper_entries_by_path(
      gpu,
      severity_mask,
      cper_data.get(),
      &buf_size,
      cper_hdrs.get(),
      &entry_count,
      &cursor);
    ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(entry_count, expected_num_headers[device_idx]);
    ASSERT_EQ(cursor, expected_num_headers[device_idx]);
  }
}
