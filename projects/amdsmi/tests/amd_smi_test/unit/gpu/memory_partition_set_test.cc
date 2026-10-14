// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free tests for rsmi_dev_memory_partition_set() on a fake card
// directory appended to the rocm_smi device list.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <linux/capability.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"

namespace {

namespace fs = std::filesystem;

class FakeCard {
 public:
  explicit FakeCard(const std::string& current_mode) {
    std::string root = (fs::temp_directory_path() / "amdsmitst_nps_XXXXXX").string();
    if (mkdtemp(root.data()) == nullptr) {
      return;
    }
    root_ = root;
    fs::create_directories(card() / "device");
    std::ofstream(current_file()) << current_mode << "\n";
    std::ofstream(card() / "device" / "available_memory_partition") << "NPS1, NPS4\n";
    amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
    index_ = static_cast<uint32_t>(smi.devices().size());
    smi.devices().push_back(std::make_shared<amd::smi::Device>(card().string(), &smi.getEnv()));
  }
  ~FakeCard() {
    if (root_.empty()) {
      return;
    }
    amd::smi::RocmSMI::getInstance().devices().resize(index_);
    std::error_code ec;
    fs::remove_all(root_, ec);
  }
  FakeCard(const FakeCard&) = delete;
  FakeCard& operator=(const FakeCard&) = delete;

  bool ok() const { return !root_.empty(); }
  uint32_t index() const { return index_; }
  fs::path current_file() const { return card() / "device" / "current_memory_partition"; }
  std::string ReadCurrent() const {
    std::string mode;
    std::ifstream(current_file()) >> mode;
    return mode;
  }

 private:
  // The Device mutex is named after this directory; keep it apart from real cards.
  fs::path card() const { return root_ / "amdsmitst_nps_card"; }

  fs::path root_;
  uint32_t index_ = 0;
};

// Root bypasses file modes through CAP_DAC_OVERRIDE; without it a 0444 file
// refuses root's writes the way a read-only sysfs attribute does.
class ScopedNoDacOverride {
 public:
  ScopedNoDacOverride() {
    if (syscall(SYS_capget, &header_, saved_) != 0) {
      error_ = errno;
      return;
    }
    __user_cap_data_struct reduced[2] = {saved_[0], saved_[1]};
    reduced[CAP_TO_INDEX(CAP_DAC_OVERRIDE)].effective &= ~CAP_TO_MASK(CAP_DAC_OVERRIDE);
    dropped_ = syscall(SYS_capset, &header_, reduced) == 0;
    error_ = dropped_ ? 0 : errno;
  }
  ~ScopedNoDacOverride() {
    if (dropped_ && syscall(SYS_capset, &header_, saved_) != 0) {
      ADD_FAILURE() << "could not restore CAP_DAC_OVERRIDE: " << std::strerror(errno);
    }
  }
  int error() const { return error_; }
  ScopedNoDacOverride(const ScopedNoDacOverride&) = delete;
  ScopedNoDacOverride& operator=(const ScopedNoDacOverride&) = delete;

 private:
  __user_cap_header_struct header_{_LINUX_CAPABILITY_VERSION_3, 0};
  __user_cap_data_struct saved_[2] = {};
  bool dropped_ = false;
  int error_ = 0;
};

}  // namespace

TEST(GpuUnit, MemoryPartitionSetCurrentModeWritesRequest) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "rsmi_dev_memory_partition_set requires root";
  }
  FakeCard card("NPS1");
  ASSERT_TRUE(card.ok());
  // Only a write moves the timestamp forward again.
  const auto before = fs::last_write_time(card.current_file()) - std::chrono::hours(1);
  fs::last_write_time(card.current_file(), before);

  EXPECT_EQ(rsmi_dev_memory_partition_set(card.index(), RSMI_MEMORY_PARTITION_NPS1),
            RSMI_STATUS_SUCCESS);
  EXPECT_TRUE(fs::last_write_time(card.current_file()) > before)
      << "the current mode was not written to " << card.current_file();
  EXPECT_EQ(card.ReadCurrent(), "NPS1");
}

TEST(GpuUnit, MemoryPartitionSetCurrentModeReadOnlyFileIsSuccess) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "rsmi_dev_memory_partition_set requires root";
  }
  FakeCard card("NPS1");
  ASSERT_TRUE(card.ok());
  ASSERT_EQ(chmod(card.current_file().c_str(), 0444), 0);
  ScopedNoDacOverride no_override;
  const int probe = open(card.current_file().c_str(), O_WRONLY);
  if (probe >= 0) {
    close(probe);
    GTEST_SKIP() << (no_override.error() != 0 ? std::strerror(no_override.error())
                                              : "the read-only file is still writable");
  }

  EXPECT_EQ(rsmi_dev_memory_partition_set(card.index(), RSMI_MEMORY_PARTITION_NPS1),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(rsmi_dev_memory_partition_set(card.index(), RSMI_MEMORY_PARTITION_NPS4),
            RSMI_STATUS_NOT_SUPPORTED);
}

// A refusal of a file that keeps its write bits (read-only mount, missing privileges)
// means the request never reached the driver, so it is reported.
TEST(GpuUnit, MemoryPartitionSetCurrentModeRefusedWritableFileIsError) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "rsmi_dev_memory_partition_set requires root";
  }
  FakeCard card("NPS1");
  ASSERT_TRUE(card.ok());
  // Owner-only write access refuses root once CAP_DAC_OVERRIDE is dropped.
  if (chown(card.current_file().c_str(), 65534, 65534) != 0) {
    GTEST_SKIP() << "cannot chown the fake file: " << std::strerror(errno);
  }
  const auto before = fs::last_write_time(card.current_file());
  ScopedNoDacOverride no_override;
  const int probe = open(card.current_file().c_str(), O_WRONLY);
  const int probe_errno = errno;
  if (probe >= 0) {
    close(probe);
    GTEST_SKIP() << (no_override.error() != 0 ? std::strerror(no_override.error())
                                              : "the file is still writable");
  }
  ASSERT_EQ(probe_errno, EACCES);

  EXPECT_EQ(rsmi_dev_memory_partition_set(card.index(), RSMI_MEMORY_PARTITION_NPS1),
            RSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(fs::last_write_time(card.current_file()) == before) << "the file was written";
  EXPECT_EQ(card.ReadCurrent(), "NPS1");
}
