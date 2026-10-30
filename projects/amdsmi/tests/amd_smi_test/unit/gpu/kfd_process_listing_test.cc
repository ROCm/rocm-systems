// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "amd_smi/impl/amd_smi_utils.h"

namespace {

// A KFD process root that holds counts[i] processes during listing i (the
// last count repeats), listed the way GetProcessInfo() lists them.
class FakeKfdProcessRoot {
 public:
  explicit FakeKfdProcessRoot(std::vector<uint32_t> counts) : counts_(std::move(counts)) {}

  SmiKfdProcessLister Lister() {
    return [this](rsmi_process_info_t* procs, uint32_t num_allocated, uint32_t* num_found) {
      buffer_sizes_.push_back(num_allocated);
      const uint32_t count = counts_[std::min(buffer_sizes_.size(), counts_.size()) - 1];
      for (uint32_t i = 0; i < count && i < num_allocated; ++i) {
        procs[i].process_id = kFirstPid + i;
      }
      *num_found = count;
      return 0;
    };
  }

  // The buffer size each listing was given.
  const std::vector<uint32_t>& buffer_sizes() const { return buffer_sizes_; }

  static constexpr uint32_t kFirstPid = 1000;

 private:
  std::vector<uint32_t> counts_;
  std::vector<uint32_t> buffer_sizes_;
};

TEST(GpuUnit, KfdProcessListingListsAgainWhenProcessesStartDuringIt) {
  // 100 processes overflow the first buffer: none expected plus 64 of room.
  FakeKfdProcessRoot root({100});
  std::vector<rsmi_process_info_t> procs;
  ASSERT_EQ(smi_amdgpu_list_kfd_processes(root.Lister(), 0, &procs), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(root.buffer_sizes(), (std::vector<uint32_t>{64, 164}));
  ASSERT_EQ(procs.size(), 100U);
  for (size_t i = 0; i < procs.size(); ++i) {
    EXPECT_EQ(procs[i].process_id, FakeKfdProcessRoot::kFirstPid + i);
  }
}

TEST(GpuUnit, KfdProcessListingAsksForARetryWhileProcessesKeepStarting) {
  // Each listing finds more processes than the room the one before left.
  FakeKfdProcessRoot root({100, 200, 300});
  std::vector<rsmi_process_info_t> procs;
  EXPECT_EQ(smi_amdgpu_list_kfd_processes(root.Lister(), 0, &procs), AMDSMI_STATUS_RETRY);
  EXPECT_EQ(root.buffer_sizes(), (std::vector<uint32_t>{64, 164, 264}));
}

TEST(GpuUnit, KfdProcessListingSizesTheFirstBufferFromTheLastCount) {
  FakeKfdProcessRoot root({500});
  std::vector<rsmi_process_info_t> procs;
  ASSERT_EQ(smi_amdgpu_list_kfd_processes(root.Lister(), 500, &procs), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(root.buffer_sizes(), (std::vector<uint32_t>{564}));
  EXPECT_EQ(procs.size(), 500U);
}

TEST(GpuUnit, KfdProcessListingWithoutKfdIsEmpty) {
  std::vector<rsmi_process_info_t> procs(3);
  const SmiKfdProcessLister no_kfd = [](rsmi_process_info_t*, uint32_t, uint32_t* num_found) {
    *num_found = 0;
    return ENOENT;
  };
  EXPECT_EQ(smi_amdgpu_list_kfd_processes(no_kfd, 0, &procs), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(procs.empty());
}

TEST(GpuUnit, KfdProcessListingReportsOtherErrors) {
  std::vector<rsmi_process_info_t> procs;
  const SmiKfdProcessLister denied = [](rsmi_process_info_t*, uint32_t, uint32_t*) {
    return EACCES;
  };
  EXPECT_EQ(smi_amdgpu_list_kfd_processes(denied, 0, &procs), AMDSMI_STATUS_NO_PERM);
}

}  // namespace
