// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "memorypartition_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "test_base.h"
#include "test_common.h"

TestMemoryPartitionRead::TestMemoryPartitionRead() : TestBase() {
  set_title("AMDSMI Memory Partition Read Test");
  set_description(
      "Reads the current memory partition into buffers of several sizes and "
      "verifies that the result always stays inside the caller's buffer.");
}

TestMemoryPartitionRead::~TestMemoryPartitionRead(void) {}

void TestMemoryPartitionRead::SetUp(void) { TestBase::SetUp(); }

void TestMemoryPartitionRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestMemoryPartitionRead::DisplayResults(void) const { TestBase::DisplayResults(); }

void TestMemoryPartitionRead::Close() { TestBase::Close(); }

void TestMemoryPartitionRead::Run(void) {
  constexpr uint32_t k255Len = 255;
  constexpr char kGuard = '#';

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    char full[k255Len] = {};
    amdsmi_status_t err =
        amdsmi_get_gpu_memory_partition(processor_handles_[dv_ind], full, k255Len);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Memory partition is not supported on this device" << std::endl;
      }
      continue;
    }
    ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
    const std::string name(full);
    if (name.empty()) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Memory partition is empty on this device" << std::endl;
      }
      continue;
    }
    IF_VERB(STANDARD) { std::cout << "\t**Current memory partition: " << name << std::endl; }

    // Each buffer has `len` usable bytes followed by one guard byte. A buffer
    // too small for the name must hold a terminated prefix of it, and a buffer
    // that fits must hold all of it. The guard byte must stay untouched.
    const auto name_len = static_cast<uint32_t>(name.size());
    for (uint32_t len : {1u, 2u, name_len, name_len + 1}) {
      std::vector<char> buf(len + 1, kGuard);
      err = amdsmi_get_gpu_memory_partition(processor_handles_[dv_ind], buf.data(), len);
      const bool fits = len > name_len;
      EXPECT_EQ(err, fits ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_INSUFFICIENT_SIZE)
          << "len=" << len;
      EXPECT_EQ(buf[len], kGuard) << "wrote past the end of a " << len << "-byte buffer";
      const size_t used = strnlen(buf.data(), len);
      EXPECT_LT(used, len) << "no terminator inside a " << len << "-byte buffer";
      EXPECT_EQ(std::string(buf.data(), used), fits ? name : name.substr(0, len - 1))
          << "len=" << len;
    }
  }
}
