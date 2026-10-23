// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "memorypartition_read_write.h"

#include <gtest/gtest.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "test_base.h"
#include "test_common.h"

const uint32_t MAX_UNSUPPORTED_PARTITIONS = 0;
const uint32_t MAX_SPX_PARTITIONS = 1;  // Single GPU node
const uint32_t MAX_DPX_PARTITIONS = 2;
const uint32_t MAX_TPX_PARTITIONS = 3;
const uint32_t MAX_QPX_PARTITIONS = 4;

TestMemoryPartitionReadWrite::TestMemoryPartitionReadWrite() : TestBase() {
  set_title("AMDSMI Memory Partition Read Test");
  set_description(
      "Verify memory partition reads, capabilities, invalid inputs, and current-mode sets. "
      "With AMDSMI_ALLOW_DESTRUCTIVE_TESTS=1, also apply each supported NPS mode with a "
      "driver reload, then restore the starting NPS mode and accelerator profiles.");
  const char* destructive = std::getenv("AMDSMI_ALLOW_DESTRUCTIVE_TESTS");
  allow_destructive_ = destructive && std::strcmp(destructive, "1") == 0;
  api_.sleep_ms = [](uint32_t ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); };
}

TestMemoryPartitionReadWrite::~TestMemoryPartitionReadWrite(void) {}

void TestMemoryPartitionReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestMemoryPartitionReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestMemoryPartitionReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestMemoryPartitionReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const std::string memoryPartitionString(amdsmi_memory_partition_type_t memoryPartitionType) {
  switch (memoryPartitionType) {
    case AMDSMI_MEMORY_PARTITION_NPS1:
      return "NPS1";
    case AMDSMI_MEMORY_PARTITION_NPS2:
      return "NPS2";
    case AMDSMI_MEMORY_PARTITION_NPS4:
      return "NPS4";
    case AMDSMI_MEMORY_PARTITION_NPS8:
      return "NPS8";
    default:
      return "UNKNOWN";
  }
}

namespace {

constexpr amdsmi_memory_partition_type_t kNpsModes[] = {
    AMDSMI_MEMORY_PARTITION_NPS1, AMDSMI_MEMORY_PARTITION_NPS2, AMDSMI_MEMORY_PARTITION_NPS4,
    AMDSMI_MEMORY_PARTITION_NPS8};
constexpr int kModprobeTimeoutSec = 300;
constexpr int kDriverLiveTimeoutSec = 120;
constexpr int kEnumerateAttempts = 30;
constexpr int kProfileAttempts = 10;
constexpr uint32_t kRetryDelayMs = 2000;

amdsmi_memory_partition_type_t ModeFromName(const std::string& name) {
  for (auto mode : kNpsModes) {
    if (name == memoryPartitionString(mode)) {
      return mode;
    }
  }
  return AMDSMI_MEMORY_PARTITION_UNKNOWN;
}

const char* ProfileName(amdsmi_accelerator_partition_type_t type) {
  switch (type) {
    case AMDSMI_ACCELERATOR_PARTITION_SPX:
      return "SPX";
    case AMDSMI_ACCELERATOR_PARTITION_DPX:
      return "DPX";
    case AMDSMI_ACCELERATOR_PARTITION_TPX:
      return "TPX";
    case AMDSMI_ACCELERATOR_PARTITION_QPX:
      return "QPX";
    case AMDSMI_ACCELERATOR_PARTITION_CPX:
      return "CPX";
    default:
      return "N/A";
  }
}

bool ModeInCaps(const amdsmi_nps_caps_t& caps, amdsmi_memory_partition_type_t mode) {
  switch (mode) {
    case AMDSMI_MEMORY_PARTITION_NPS1:
      return caps.nps_flags.nps1_cap;
    case AMDSMI_MEMORY_PARTITION_NPS2:
      return caps.nps_flags.nps2_cap;
    case AMDSMI_MEMORY_PARTITION_NPS4:
      return caps.nps_flags.nps4_cap;
    case AMDSMI_MEMORY_PARTITION_NPS8:
      return caps.nps_flags.nps8_cap;
    default:
      return false;
  }
}

std::string ReadSysfsLine(const char* path) {
  std::ifstream file(path);
  std::string line;
  std::getline(file, line);
  return line;
}

// -1 when amdgpu is not loaded.
int AmdgpuRefcount() {
  const std::string value = ReadSysfsLine("/sys/module/amdgpu/refcnt");
  return value.empty() ? -1 : std::atoi(value.c_str());
}

// Runs modprobe from a fixed path with fixed arguments: no shell and no PATH lookup.
bool RunModprobe(const std::vector<std::string>& args, std::string* error) {
  std::string command = "modprobe";
  for (const auto& arg : args) {
    command += " " + arg;
  }
  const char* path = nullptr;
  for (const char* candidate : {"/usr/sbin/modprobe", "/sbin/modprobe"}) {
    if (access(candidate, X_OK) == 0) {
      path = candidate;
      break;
    }
  }
  if (!path) {
    *error = "modprobe not found in /usr/sbin or /sbin (install kmod)";
    return false;
  }
  std::vector<char*> argv;
  argv.push_back(const_cast<char*>("modprobe"));
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);

  std::cout << "\t** " << command << std::endl;
  const pid_t pid = fork();
  if (pid < 0) {
    *error = command + ": fork failed: " + std::strerror(errno);
    return false;
  }
  if (pid == 0) {
    execv(path, argv.data());
    _exit(127);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(kModprobeTimeoutSec);
  int status = 0;
  for (;;) {
    const pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      break;
    }
    if (done < 0 && errno != EINTR) {
      *error = command + ": waitpid failed: " + std::strerror(errno);
      return false;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      *error = command + " timed out after " + std::to_string(kModprobeTimeoutSec) + "s";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    *error = command + " failed (exit status " +
             std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) + ")";
    return false;
  }
  return true;
}

}  // namespace

void TestMemoryPartitionReadWrite::Run(void) {
  amdsmi_status_t ret, err, ret_set;
  constexpr uint32_t k255Len = 255;
  constexpr uint32_t k0Len = 0;
  char orig_memory_partition[k255Len] = {};
  amdsmi_memory_partition_config_t current_memory_config;
  const uint32_t kMAX_UINT32 = std::numeric_limits<uint32_t>::max();
  std::map<uint32_t, amdsmi_accelerator_partition_profile_t> original_profiles;
  std::map<uint32_t, std::string> original_memory_modes;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  bool isVerbose =
      (this->verbosity() && this->verbosity() >= (this->TestBase::VERBOSE_STANDARD)) ? true : false;

  // Save each device independently; mixed compute configurations are valid.
  IF_VERB(STANDARD) {
    std::cout << "\t**=========================================================\n";
    std::cout << "\t**Save Original Compute Partition Settings ================\n";
    std::cout << "\t**=========================================================\n";
  }
  auto initial_num_devices = num_monitor_devs();
  std::vector<bool> safe_to_test(initial_num_devices, true);
  amdsmi_accelerator_partition_type_t primary_partition_type = AMDSMI_ACCELERATOR_PARTITION_INVALID;
  uint32_t primary_index = 0;
  for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind) {
    if (dv_ind != 0) {
      std::cout << "\n";
    }
    PrintDeviceHeader(processor_handles_[dv_ind]);
    amdsmi_accelerator_partition_profile_t profile = {};
    uint32_t partition_id[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile", "", VERB(STANDARD));
    ret = api_.get_profile(processor_handles_[dv_ind], &profile, &partition_id[0]);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS) << "Compute partition snapshot failed";
    if (ret != AMDSMI_STATUS_SUCCESS) {
      safe_to_test[dv_ind] = false;
      continue;
    }
    original_profiles[dv_ind] = profile;
    std::string nps_caps_str = "";
    if ((profile.memory_caps.nps_flags.nps1_cap == 0 &&
         profile.memory_caps.nps_flags.nps2_cap == 0 &&
         profile.memory_caps.nps_flags.nps4_cap == 0 &&
         profile.memory_caps.nps_flags.nps8_cap == 0)) {
      nps_caps_str = "N/A";
    } else {
      nps_caps_str.clear();
      if (profile.memory_caps.nps_flags.nps1_cap) {
        (nps_caps_str.empty()) ? nps_caps_str += "NPS1" : nps_caps_str += ", NPS1";
      }
      if (profile.memory_caps.nps_flags.nps2_cap) {
        (nps_caps_str.empty()) ? nps_caps_str += "NPS2" : nps_caps_str += ", NPS2";
      }
      if (profile.memory_caps.nps_flags.nps4_cap) {
        (nps_caps_str.empty()) ? nps_caps_str += "NPS4" : nps_caps_str += ", NPS4";
      }
      if (profile.memory_caps.nps_flags.nps8_cap) {
        (nps_caps_str.empty()) ? nps_caps_str += "NPS8" : nps_caps_str += ", NPS8";
      }
    }

    std::string profile_type_str = "N/A";
    if (profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_SPX) {
      profile_type_str = "SPX";
    } else if (profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_DPX) {
      profile_type_str = "DPX";
    } else if (profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_TPX) {
      profile_type_str = "TPX";
    } else if (profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_QPX) {
      profile_type_str = "QPX";
    } else if (profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_CPX) {
      profile_type_str = "CPX";
    }

    // save the primary partition type
    if (profile.profile_type != AMDSMI_ACCELERATOR_PARTITION_INVALID) {
      primary_partition_type = profile.profile_type;
      primary_index = dv_ind;
    }

    std::string partition_id_str = "";
    for (int i = 0; i < 8; i++) {
      partition_id_str += std::to_string(partition_id[i]);
      if (i < 7) {
        partition_id_str += ", ";
      }

      switch (primary_partition_type) {
        case AMDSMI_ACCELERATOR_PARTITION_SPX:
          EXPECT_LT(partition_id[i], MAX_SPX_PARTITIONS);
          break;
        case AMDSMI_ACCELERATOR_PARTITION_DPX:
          EXPECT_LT(partition_id[i], MAX_DPX_PARTITIONS);
          break;
        case AMDSMI_ACCELERATOR_PARTITION_TPX:
          EXPECT_LT(partition_id[i], MAX_TPX_PARTITIONS);
          break;
        case AMDSMI_ACCELERATOR_PARTITION_QPX:
          EXPECT_LT(partition_id[i], MAX_QPX_PARTITIONS);
          break;
        case AMDSMI_ACCELERATOR_PARTITION_CPX: {
          uint16_t num_xcd;
          uint32_t max_xcps = 0;
          DISPLAY_AMDSMI_API("amdsmi_get_gpu_xcd_counter", "gpu=" + std::to_string(primary_index),
                             VERB(STANDARD));
          ret = api_.get_xcd_counter(processor_handles_[primary_index], &num_xcd);
          DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
          if (ret == AMDSMI_STATUS_SUCCESS) {
            max_xcps = static_cast<uint32_t>(num_xcd);
          }
          if (!amd::smi::is_vm_guest()) {
            // In BM, we can get the number of XCDs (calculated by getting # of gfx_clocks)
            EXPECT_LT(partition_id[i], max_xcps);
          } else {
            // In guest, we may not be able to get the number of XCDs
            // (calculated by getting # of gfx_clocks)
            EXPECT_LE(partition_id[i], max_xcps);
          }
          break;
        }
        case AMDSMI_ACCELERATOR_PARTITION_INVALID:
          EXPECT_EQ(partition_id[i], MAX_UNSUPPORTED_PARTITIONS);
          break;
        default:
          EXPECT_EQ(partition_id[i], MAX_UNSUPPORTED_PARTITIONS);
          break;
      }
    }

    IF_VERB(STANDARD) {
      std::cout << "\t**amdsmi_get_gpu_accelerator_partition_profile(processor_handles_[" << dv_ind
                << "], &profile, &partition_id[0]):\n"
                << "\t\t" << smi_amdgpu_get_status_string(ret, false)
                << "\n\t**Current profile.profile_type: " << profile_type_str
                << "\n\t**profile.num_partitions: "
                << (profile.num_partitions == kMAX_UINT32 ? "N/A"
                                                          : std::to_string(profile.num_partitions))
                << "\n\t**profile.memory_caps: " << nps_caps_str << "\n\t**profile.profile_index: "
                << (profile.profile_index == kMAX_UINT32 ? "N/A"
                                                         : std::to_string(profile.profile_index))
                << "\n\t**profile.num_resources: " << profile.num_resources
                << "\n\t**partition_id: " << partition_id_str << std::endl;
    }
    EXPECT_TRUE(ret == AMDSMI_STATUS_SUCCESS || ret == AMDSMI_STATUS_NOT_SUPPORTED);
    amdsmi_accelerator_partition_profile_config_t profile_config = {};
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile_config",
                       "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
    ret = api_.get_profile_config(processor_handles_[dv_ind], &profile_config);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    IF_VERB(STANDARD) {
      std::cout << "\t**amdsmi_get_gpu_accelerator_partition_profile_config(processor_handles_["
                << dv_ind << "], &profile_config):\n"
                << "\t\t" << smi_amdgpu_get_status_string(ret, false)
                << "\n\t**profile_config.num_profiles: " << profile_config.num_profiles
                << "\n\t**profile_config.num_resource_profiles: "
                << profile_config.num_resource_profiles << std::endl;
    }
    getAvailableProfileConfigs(dv_ind, profile, profile_config, isVerbose);

    IF_VERB(STANDARD) {
      std::cout << "\t**=========================================================\n";
      std::cout << "\t**Checking valid profile Sets =============================\n";
      std::cout << "\t**=========================================================\n";
    }
    int resource_index = 0;
    for (uint32_t i = 0; i < profile_config.num_profiles; i++) {
      auto current_profile = profile_config.profiles[i];
      std::string profile_type_str = "N/A";
      if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_SPX) {
        profile_type_str = "SPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_DPX) {
        profile_type_str = "DPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_TPX) {
        profile_type_str = "TPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_QPX) {
        profile_type_str = "QPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_CPX) {
        profile_type_str = "CPX";
      }

      std::string nps_caps_str = "";
      if ((current_profile.memory_caps.nps_flags.nps1_cap == 0 &&
           current_profile.memory_caps.nps_flags.nps2_cap == 0 &&
           current_profile.memory_caps.nps_flags.nps4_cap == 0 &&
           current_profile.memory_caps.nps_flags.nps8_cap == 0)) {
        nps_caps_str = "N/A";
      } else {
        nps_caps_str.clear();
        if (current_profile.memory_caps.nps_flags.nps1_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS1" : nps_caps_str += ", NPS1";
        }
        if (current_profile.memory_caps.nps_flags.nps2_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS2" : nps_caps_str += ", NPS2";
        }
        if (current_profile.memory_caps.nps_flags.nps4_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS4" : nps_caps_str += ", NPS4";
        }
        if (current_profile.memory_caps.nps_flags.nps8_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS8" : nps_caps_str += ", NPS8";
        }
      }
      IF_VERB(STANDARD) {
        std::cout << "\t**profile_config.profiles[" << i << "]:\n"
                  << "\t\tprofile_type: " << profile_type_str
                  << "\n\t\tnum_partitions: " << current_profile.num_partitions
                  << "\n\t\tmemory_caps: " << nps_caps_str
                  << "\n\t\tcurrent_profile.num_resources: " << current_profile.num_resources
                  << std::endl;
      }
      for (uint32_t j = 0; j < current_profile.num_resources; j++) {
        auto rp = profile_config.resource_profiles[resource_index];

        IF_VERB(STANDARD) {
          std::cout << "\n\t\t\tprofile_index: " << current_profile.profile_index
                    << "\n\t\t\tresource_index: " << resource_index
                    << "\n\t\t\tprofile_config.resource_profiles[" << resource_index
                    << "].resource_type: " << getResourceType(rp.resource_type)
                    << "\n\t\t\tprofile_config.resource_profiles[" << resource_index
                    << "].partition_resource: " << rp.partition_resource
                    << "\n\t\t\tprofile_config.resource_profiles[" << resource_index
                    << "].num_partitions_share_resource: " << rp.num_partitions_share_resource
                    << std::endl;
        }
        resource_index++;
      }
    }
    EXPECT_TRUE(ret == AMDSMI_STATUS_SUCCESS || ret == AMDSMI_STATUS_NOT_SUPPORTED);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "amdsmi_get_gpu_accelerator_partition_profile_config(): "
                  << "Not supported on this machine" << std::endl;
      }
      continue;
    }
  }

  // Run memory partition tests
  IF_VERB(STANDARD) {
    std::cout << "\t**=========================================================\n";
    std::cout << "\t**Test: Memory Partition Sets =============================\n";
    std::cout << "\t**=========================================================\n";
  }
  for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind) {
    if (!safe_to_test[dv_ind]) {
      continue;
    }
    if (dv_ind != 0) {
      IF_VERB(STANDARD) { std::cout << std::endl; }
    }
    PrintDeviceHeader(processor_handles_[dv_ind]);

    // Standard checks to see if API is supported, before running full tests
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    std::memset(orig_memory_partition, 0, sizeof(orig_memory_partition));
    ret = api_.get_memory(processor_handles_[dv_ind], orig_memory_partition, k255Len);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS) << "Memory partition snapshot failed";
    if (ret != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    const std::map<std::string, amdsmi_memory_partition_type_t> memory_modes = {
        {"NPS1", AMDSMI_MEMORY_PARTITION_NPS1},
        {"NPS2", AMDSMI_MEMORY_PARTITION_NPS2},
        {"NPS4", AMDSMI_MEMORY_PARTITION_NPS4},
        {"NPS8", AMDSMI_MEMORY_PARTITION_NPS8}};
    const std::string original_memory(
        orig_memory_partition, strnlen(orig_memory_partition, sizeof(orig_memory_partition)));
    const auto original_mode = memory_modes.find(original_memory);
    EXPECT_NE(original_mode, memory_modes.end()) << "Invalid memory partition snapshot";
    if (original_mode == memory_modes.end()) {
      continue;
    }
    original_memory_modes[dv_ind] = original_memory;
    IF_VERB(STANDARD) {
      std::cout << "\t**Current Memory Partition: " << original_memory << std::endl;
    }

    // Verify api support checking functionality is working
    constexpr uint32_t k2Len = 2;
    char smallBuffer[k2Len];
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.get_memory(processor_handles_[dv_ind], smallBuffer, k2Len);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INSUFFICIENT_SIZE);
    uint32_t size = static_cast<uint32_t>(sizeof(smallBuffer) / sizeof(*smallBuffer));
    ASSERT_EQ(err, AMDSMI_STATUS_INSUFFICIENT_SIZE);
    ASSERT_EQ(k2Len, size);
    if (err == AMDSMI_STATUS_INSUFFICIENT_SIZE) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed AMDSMI_STATUS_INSUFFICIENT_SIZE was returned "
                  << "and size is 2, as requested." << std::endl;
      }
    }

    // Verify api support checking functionality is working
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.get_memory(processor_handles_[dv_ind], nullptr, k255Len);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_memory_partition(processor_handles_[" << dv_ind << "], "
                  << "nullptr, 255): "
                  << "Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
    }

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.get_memory_config(processor_handles_[dv_ind], nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_memory_partition(processor_handles_[" << dv_ind
                  << "], nullptr): Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
    }

    // Verify api support checking functionality is working
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.get_memory(processor_handles_[dv_ind], orig_memory_partition, k0Len);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_TRUE(err == AMDSMI_STATUS_INVAL);
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_memory_partition(processor_handles_[" << dv_ind << "], "
                  << "orig_memory_partition, 0): "
                  << "Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
    }

    amdsmi_memory_partition_config_t* null_memory_partition_config = nullptr;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.get_memory_config(processor_handles_[dv_ind], null_memory_partition_config);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                          AMDSMI_STATUS_NOT_SUPPORTED);
    ASSERT_TRUE((err == AMDSMI_STATUS_INVAL) || (err == AMDSMI_STATUS_NOT_SUPPORTED));
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "amdsmi_get_gpu_memory_partition_config(processor_handles_[" << dv_ind << "], "
                  << "nullptr): "
                  << "Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
    }

    /****************************************/
    /* amdsmi_set_gpu_memory_partition_mode(...) */
    /****************************************/
    // Verify api support checking functionality is working
    amdsmi_memory_partition_type_t null_memory_partition = {};
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition_mode", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.set_memory(processor_handles_[dv_ind], null_memory_partition);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    std::cout << "\t**amdsmi_set_gpu_memory_partition_mode"
              << "(processor_handles_[" << dv_ind
              << "], null_memory_partition): " << smi_amdgpu_get_status_string(err, false) << "\n";
    // Note: new_memory_partition is not set
    ASSERT_TRUE(err == AMDSMI_STATUS_INVAL);
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
    } else if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << ": "
                  << "amdsmi_set_gpu_memory_partition_mode not supported on this "
                  << "device\n\t    (if amdsmi_get_gpu_memory_partition works, "
                  << "then likely need to set in bios)" << std::endl;
      }
      continue;
    }
    ASSERT_FALSE(err == AMDSMI_STATUS_NO_PERM);

    // Verify api support checking functionality is working
    amdsmi_memory_partition_type_t new_memory_partition = AMDSMI_MEMORY_PARTITION_UNKNOWN;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition_mode", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.set_memory(processor_handles_[dv_ind], new_memory_partition);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_TRUE((err == AMDSMI_STATUS_INVAL) || (err == AMDSMI_STATUS_NOT_SUPPORTED) ||
                (err == AMDSMI_STATUS_NO_PERM));
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
      else if (err == AMDSMI_STATUS_NO_PERM) {
        // tests should not continue if err is a permission issue
        ASSERT_FALSE(err == AMDSMI_STATUS_NO_PERM);
      }
    }

    // Do not turn a same-mode check into a write after an observed external change.
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = api_.get_memory(processor_handles_[dv_ind], orig_memory_partition, k255Len);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS) << "Memory partition pre-set read failed";
    if (ret != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    const std::string before_set(orig_memory_partition,
                                 strnlen(orig_memory_partition, sizeof(orig_memory_partition)));
    EXPECT_EQ(before_set, original_memory) << "Memory partition changed before same-mode set";
    if (before_set != original_memory) {
      continue;
    }

    // Active-mode requests cannot cancel a deferred alternate-NPS write.
    new_memory_partition = original_mode->second;
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "======== TEST AMDSMI_MEMORY_PARTITION_"
                << memoryPartitionString(new_memory_partition) << " ===============" << std::endl;
    }

    // Read capabilities before attempting set; used to validate set return code.
    current_memory_config = {};
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    auto ret_caps = api_.get_memory_config(processor_handles_[dv_ind], &current_memory_config);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret_caps, AMDSMI_STATUS_SUCCESS);
    ASSERT_TRUE((ret_caps == AMDSMI_STATUS_NOT_SUPPORTED) || (ret_caps == AMDSMI_STATUS_SUCCESS));

    std::string memory_caps_str = "N/A";
    bool partition_is_supported = false;
    if (ret_caps == AMDSMI_STATUS_SUCCESS) {
      memory_caps_str.clear();
      if (current_memory_config.partition_caps.nps_flags.nps1_cap) {
        memory_caps_str += (memory_caps_str.empty() ? "NPS1" : ", NPS1");
        if (new_memory_partition == AMDSMI_MEMORY_PARTITION_NPS1) partition_is_supported = true;
      }
      if (current_memory_config.partition_caps.nps_flags.nps2_cap) {
        memory_caps_str += (memory_caps_str.empty() ? "NPS2" : ", NPS2");
        if (new_memory_partition == AMDSMI_MEMORY_PARTITION_NPS2) partition_is_supported = true;
      }
      if (current_memory_config.partition_caps.nps_flags.nps4_cap) {
        memory_caps_str += (memory_caps_str.empty() ? "NPS4" : ", NPS4");
        if (new_memory_partition == AMDSMI_MEMORY_PARTITION_NPS4) partition_is_supported = true;
      }
      if (current_memory_config.partition_caps.nps_flags.nps8_cap) {
        memory_caps_str += (memory_caps_str.empty() ? "NPS8" : ", NPS8");
        if (new_memory_partition == AMDSMI_MEMORY_PARTITION_NPS8) partition_is_supported = true;
      }
    }

    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Available Memory Partition Capabilities: " << memory_caps_str << "\n"
                << "\t**"
                << "current_memory_partition_mode: "
                << (ret_caps == AMDSMI_STATUS_SUCCESS
                        ? memoryPartitionString(current_memory_config.mp_mode)
                        : "N/A")
                << "\n"
                << "\t**"
                << "Requested partition supported by hardware: "
                << (partition_is_supported ? "YES" : "NO") << "\n"
                << "\t**"
                << "Attempting to set memory partition to: "
                << memoryPartitionString(new_memory_partition) << std::endl;
    }

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition_mode", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret_set = api_.set_memory(processor_handles_[dv_ind], new_memory_partition);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret_set, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED);
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "amdsmi_set_gpu_memory_partition_mode(processor_handles_[" << dv_ind << "], "
                << memoryPartitionString(new_memory_partition)
                << "): " << smi_amdgpu_get_status_string(ret_set, false) << "\n";
    }
    if (ret_set == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    ASSERT_TRUE((ret_set == AMDSMI_STATUS_SUCCESS) || (ret_set == AMDSMI_STATUS_BUSY) ||
                (ret_set == AMDSMI_STATUS_AMDGPU_RESTART_ERR) || (ret_set == AMDSMI_STATUS_INVAL) ||
                (ret_set == AMDSMI_STATUS_NOT_SUPPORTED));

    // If the hardware advertises support for this mode, the set should succeed unless the
    // device is busy (e.g., workloads running).
    if (partition_is_supported) {
      EXPECT_TRUE(ret_set == AMDSMI_STATUS_SUCCESS || ret_set == AMDSMI_STATUS_BUSY)
          << "Set did not succeed for a partition mode advertised as supported by hardware: "
          << memoryPartitionString(new_memory_partition);
    }

    if (ret_set == AMDSMI_STATUS_SUCCESS) {
      std::cout << "\t** Current-mode memory partition request accepted; "
                   "alternate NPS transitions were not tested.\n";
    }

    // Verify capabilities are unchanged after the set attempt.
    amdsmi_memory_partition_config_t post_set_config = {};
    auto ret_caps_post = api_.get_memory_config(processor_handles_[dv_ind], &post_set_config);
    if (ret_caps_post == AMDSMI_STATUS_SUCCESS && ret_caps == AMDSMI_STATUS_SUCCESS) {
      EXPECT_EQ(post_set_config.partition_caps.nps_flags.nps1_cap,
                current_memory_config.partition_caps.nps_flags.nps1_cap);
      EXPECT_EQ(post_set_config.partition_caps.nps_flags.nps2_cap,
                current_memory_config.partition_caps.nps_flags.nps2_cap);
      EXPECT_EQ(post_set_config.partition_caps.nps_flags.nps4_cap,
                current_memory_config.partition_caps.nps_flags.nps4_cap);
      EXPECT_EQ(post_set_config.partition_caps.nps_flags.nps8_cap,
                current_memory_config.partition_caps.nps_flags.nps8_cap);
    }
  }  // END DEVICE FOR LOOP

  for (const auto& entry : original_memory_modes) {
    SCOPED_TRACE(::testing::Message() << "gpu=" << entry.first);
    char current[k255Len] = {};
    const auto status = api_.get_memory(processor_handles_[entry.first], current, sizeof(current));
    EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS) << "Memory partition readback failed";
    if (status != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    EXPECT_EQ(std::string(current, strnlen(current, sizeof(current))), entry.second)
        << "Memory partition drift";
  }

  // Another actor's changes must fail verification, not be overwritten.
  for (const auto& entry : original_profiles) {
    SCOPED_TRACE(::testing::Message() << "gpu=" << entry.first);
    amdsmi_accelerator_partition_profile_t profile = {};
    uint32_t partition_id[8] = {};
    const auto status = api_.get_profile(processor_handles_[entry.first], &profile, partition_id);
    EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS) << "Compute partition readback failed";
    if (status != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    EXPECT_EQ(profile.profile_type, entry.second.profile_type) << "Compute partition drift";
    EXPECT_EQ(profile.profile_index, entry.second.profile_index) << "Compute profile index drift";
  }

  if (allow_destructive_) {
    RunMemoryPartitionCycle();
  } else {
    std::cout << "\t** NPS cycle not run: set AMDSMI_ALLOW_DESTRUCTIVE_TESTS=1 to apply each "
                 "supported NPS mode with a driver reload.\n";
  }
}

bool TestMemoryPartitionReadWrite::RefreshHandles(std::string* error) {
  // Not initialized after a failed reload; nothing to shut down then.
  amdsmi_shut_down();
  num_monitor_devs_ = 0;
  amdsmi_status_t err = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (err != AMDSMI_STATUS_SUCCESS) {
    if (error) {
      *error = std::string("amdsmi_init failed: ") + smi_amdgpu_get_status_string(err, false);
    }
    return false;
  }
  uint32_t socket_count = 0;
  err = amdsmi_get_socket_handles(&socket_count, nullptr);
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (err == AMDSMI_STATUS_SUCCESS) {
    err = amdsmi_get_socket_handles(&socket_count, sockets.data());
  }
  for (uint32_t i = 0; err == AMDSMI_STATUS_SUCCESS && i < socket_count; ++i) {
    uint32_t count = 0;
    err = amdsmi_get_processor_handles(sockets[i], &count, nullptr);
    std::vector<amdsmi_processor_handle> handles(count);
    if (err == AMDSMI_STATUS_SUCCESS) {
      err = amdsmi_get_processor_handles(sockets[i], &count, handles.data());
    }
    for (uint32_t j = 0; err == AMDSMI_STATUS_SUCCESS && j < count; ++j) {
      if (num_monitor_devs_ >= MAX_MONITOR_DEVICES) {
        err = AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
        break;
      }
      processor_handles_[num_monitor_devs_++] = handles[j];
    }
  }
  if (err != AMDSMI_STATUS_SUCCESS && error) {
    *error = std::string("enumerating devices failed: ") + smi_amdgpu_get_status_string(err, false);
  }
  return err == AMDSMI_STATUS_SUCCESS;
}

bool TestMemoryPartitionReadWrite::DriverIdle(std::string* error) {
  amdsmi_shut_down();
  const int refcount = AmdgpuRefcount();
  RefreshHandles(nullptr);
  if (refcount != 0) {
    *error = refcount < 0 ? "amdgpu is not loaded"
                          : "amdgpu is in use by another process (refcnt " +
                                std::to_string(refcount) + "); the NPS cycle needs the GPUs idle";
    return false;
  }
  return true;
}

bool TestMemoryPartitionReadWrite::ReloadDriver(std::string* error) {
  // Our own descriptors would keep amdgpu loaded.
  amdsmi_shut_down();
  num_monitor_devs_ = 0;
  const int refcount = AmdgpuRefcount();
  if (refcount != 0) {
    *error = "amdgpu is in use (refcnt " + std::to_string(refcount) + "); not reloading";
    RefreshHandles(nullptr);
    return false;
  }
  std::cout << "\t** Reloading amdgpu, this may take a minute..." << std::endl;
  const auto started = std::chrono::steady_clock::now();
  if (!RunModprobe({"-r", "amdgpu"}, error)) {
    RefreshHandles(nullptr);
    return false;
  }
  // Never leave the driver unloaded: try the load twice.
  std::string load_error;
  if (!RunModprobe({"amdgpu"}, &load_error) && !RunModprobe({"amdgpu"}, &load_error)) {
    *error = "amdgpu did not load again, GPUs are unavailable: " + load_error;
    return false;
  }
  for (int waited = 0; ReadSysfsLine("/sys/module/amdgpu/initstate") != "live"; ++waited) {
    if (waited >= kDriverLiveTimeoutSec) {
      *error = "amdgpu loaded but did not report live";
      return false;
    }
    api_.sleep_ms(1000);
  }
  std::cout << "\t** Reload took "
            << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
                                                                started)
                   .count()
            << "s" << std::endl;
  return RefreshHandles(error);
}

std::string TestMemoryPartitionReadWrite::PhysicalKey(amdsmi_processor_handle handle) {
  amdsmi_kfd_info_t kfd = {};
  if (amdsmi_get_gpu_kfd_info(handle, &kfd) == AMDSMI_STATUS_SUCCESS &&
      kfd.current_partition_id != 0 &&
      kfd.current_partition_id != std::numeric_limits<uint32_t>::max()) {
    return "";
  }
  amdsmi_bdf_t bdf = {};
  if (amdsmi_get_gpu_device_bdf(handle, &bdf) != AMDSMI_STATUS_SUCCESS) {
    return "";
  }
  char key[32];
  std::snprintf(key, sizeof(key), "%04llx:%02x:%02x",
                static_cast<unsigned long long>(bdf.domain_number),
                static_cast<unsigned>(bdf.bus_number), static_cast<unsigned>(bdf.device_number));
  return key;
}

std::map<std::string, TestMemoryPartitionReadWrite::PhysicalGpu>
TestMemoryPartitionReadWrite::ReadPhysicalGpus(bool* complete) {
  std::map<std::string, PhysicalGpu> gpus;
  *complete = true;
  for (uint32_t i = 0; i < num_monitor_devs_; ++i) {
    const std::string key = PhysicalKey(processor_handles_[i]);
    if (key.empty() || gpus.count(key)) {
      continue;
    }
    PhysicalGpu gpu;
    gpu.handle = processor_handles_[i];
    char memory[255] = {};
    const auto status = api_.get_memory(gpu.handle, memory, sizeof(memory));
    if (status == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    if (status != AMDSMI_STATUS_SUCCESS) {
      *complete = false;
      continue;
    }
    gpu.memory.assign(memory, strnlen(memory, sizeof(memory)));
    uint32_t partition_id[8] = {};
    gpu.profile_valid =
        api_.get_profile(gpu.handle, &gpu.profile, partition_id) == AMDSMI_STATUS_SUCCESS;
    amdsmi_memory_partition_config_t config = {};
    if (api_.get_memory_config(gpu.handle, &config) == AMDSMI_STATUS_SUCCESS) {
      gpu.caps = config.partition_caps;
    }
    gpus[key] = gpu;
  }
  return gpus;
}

bool TestMemoryPartitionReadWrite::StageOnAll(amdsmi_memory_partition_type_t mode,
                                              bool* any_staged) {
  bool complete = true;
  const auto gpus = ReadPhysicalGpus(&complete);
  size_t accepted = 0;
  for (const auto& entry : gpus) {
    const auto status = api_.set_memory(entry.second.handle, mode);
    std::cout << "\t** " << entry.first << ": set " << memoryPartitionString(mode) << " -> "
              << smi_amdgpu_get_status_string(status, false) << std::endl;
    if (status == AMDSMI_STATUS_SUCCESS) {
      ++accepted;
    } else {
      EXPECT_TRUE(status == AMDSMI_STATUS_INVAL || status == AMDSMI_STATUS_NOT_SUPPORTED)
          << entry.first << ": unexpected status staging " << memoryPartitionString(mode);
    }
  }
  *any_staged = accepted > 0;
  if (accepted > 0 && accepted < gpus.size()) {
    ADD_FAILURE() << memoryPartitionString(mode) << " accepted by " << accepted << " of "
                  << gpus.size() << " GPUs";
  }
  return complete && accepted == gpus.size();
}

std::string TestMemoryPartitionReadWrite::VerifyMode(const std::string& expected,
                                                     size_t physical_count) {
  // Partition devices can take a moment to appear after the load.
  std::map<std::string, PhysicalGpu> gpus;
  bool complete = false;
  for (int attempt = 0; attempt < kEnumerateAttempts; ++attempt) {
    gpus = ReadPhysicalGpus(&complete);
    if (complete && gpus.size() == physical_count) {
      break;
    }
    api_.sleep_ms(kRetryDelayMs);
    RefreshHandles(nullptr);
  }
  EXPECT_TRUE(complete) << "Memory partition unreadable after reload";
  EXPECT_EQ(gpus.size(), physical_count) << "GPU count changed after reload";
  std::string agreed = gpus.empty() ? "" : gpus.begin()->second.memory;
  for (const auto& entry : gpus) {
    EXPECT_EQ(entry.second.memory, expected) << entry.first << ": reload applied the wrong mode";
    if (entry.second.memory != agreed) {
      agreed.clear();
    }
    std::cout << "\t** " << entry.first << ": " << entry.second.memory << " / "
              << ProfileName(entry.second.profile.profile_type) << std::endl;
  }
  return (complete && gpus.size() == physical_count) ? agreed : "";
}

void TestMemoryPartitionReadWrite::RestoreAcceleratorProfiles(
    const std::map<std::string, PhysicalGpu>& original) {
  for (const auto& entry : original) {
    if (!entry.second.profile_valid) {
      continue;
    }
    const auto& want = entry.second.profile;
    SCOPED_TRACE(::testing::Message() << entry.first);
    bool restored = false;
    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    for (int attempt = 0; attempt < kProfileAttempts && !restored; ++attempt) {
      bool complete = true;
      const auto gpus = ReadPhysicalGpus(&complete);
      const auto current = gpus.find(entry.first);
      if (current == gpus.end() || !current->second.profile_valid) {
        api_.sleep_ms(kRetryDelayMs);
        RefreshHandles(nullptr);
        continue;
      }
      if (current->second.profile.profile_index == want.profile_index) {
        restored = true;
        break;
      }
      status = api_.set_profile(current->second.handle, want.profile_index);
      std::cout << "\t** " << entry.first << ": set accelerator profile "
                << ProfileName(want.profile_type) << " (index " << want.profile_index << ") -> "
                << smi_amdgpu_get_status_string(status, false) << std::endl;
      if (status != AMDSMI_STATUS_SUCCESS && status != AMDSMI_STATUS_BUSY) {
        break;
      }
      // A profile change replaces the partition devices; BUSY is retried.
      api_.sleep_ms(kRetryDelayMs);
      RefreshHandles(nullptr);
    }
    EXPECT_TRUE(restored) << "Accelerator profile not restored to "
                          << ProfileName(want.profile_type) << " (index " << want.profile_index
                          << "), last status " << smi_amdgpu_get_status_string(status, false);
  }
}

void TestMemoryPartitionReadWrite::RunMemoryPartitionCycle() {
  std::cout << "\t**=========================================================\n"
            << "\t**NPS cycle (AMDSMI_ALLOW_DESTRUCTIVE_TESTS=1) =============\n"
            << "\t**=========================================================\n";
  bool complete = true;
  const auto original = ReadPhysicalGpus(&complete);
  if (!complete) {
    ADD_FAILURE() << "Memory partition unreadable on some GPUs; NPS cycle not run";
    return;
  }
  if (original.empty()) {
    std::cout << "\t** No GPU reports a memory partition; nothing to cycle.\n";
    return;
  }
  const std::string start = original.begin()->second.memory;
  amdsmi_nps_caps_t caps = original.begin()->second.caps;
  for (const auto& entry : original) {
    if (entry.second.memory != start) {
      std::cout << "\t** GPUs start in different memory partition modes; NPS cycle not run.\n";
      return;
    }
    caps.nps_cap_mask &= entry.second.caps.nps_cap_mask;
  }
  if (ModeFromName(start) == AMDSMI_MEMORY_PARTITION_UNKNOWN) {
    ADD_FAILURE() << "Starting memory partition " << start << " cannot be restored";
    return;
  }
  std::string error;
  if (!DriverIdle(&error)) {
    ADD_FAILURE() << error << "; NPS cycle not run, nothing was staged";
    return;
  }

  std::cout << "\t** Starting mode: " << start << " on " << original.size() << " GPUs\n";
  // ReloadDriver leaves no devices only when amdgpu failed to load again.
  auto reload = [&](std::string* reload_error) {
    if (ReloadDriver(reload_error)) {
      return true;
    }
    ADD_FAILURE() << *reload_error;
    return false;
  };
  std::string current = start;
  bool staged_pending = false;
  for (auto mode : kNpsModes) {
    const std::string name = memoryPartitionString(mode);
    if (name == start || !ModeInCaps(caps, mode)) {
      continue;
    }
    std::cout << "\n\t** ---- " << current << " -> " << name << " ----" << std::endl;
    bool any_staged = false;
    StageOnAll(mode, &any_staged);
    if (!any_staged) {
      std::cout << "\t** " << name << " rejected by every GPU; no reload needed.\n";
      continue;
    }
    staged_pending = true;
    if (!reload(&error)) {
      break;
    }
    staged_pending = false;
    current = VerifyMode(name, original.size());
    if (current != name) {
      break;
    }
  }

  std::cout << "\n\t** ---- Restore " << start << " ----" << std::endl;
  if (num_monitor_devs_ == 0) {
    ADD_FAILURE() << "amdgpu is not loaded; " << start
                  << " and the accelerator profiles were not restored";
    return;
  }
  if (staged_pending) {
    // Requesting the active mode does not cancel a staged one, so apply it first.
    if (!reload(&error)) {
      ADD_FAILURE() << "A memory partition mode is still staged; the next driver reload "
                    << "applies it instead of " << start;
      return;
    }
    bool read = true;
    const auto gpus = ReadPhysicalGpus(&read);
    current = gpus.empty() ? "" : gpus.begin()->second.memory;
  }
  if (current != start) {
    bool any_staged = false;
    StageOnAll(ModeFromName(start), &any_staged);
    EXPECT_TRUE(any_staged) << "Could not stage the restore to " << start;
    if (any_staged && reload(&error)) {
      VerifyMode(start, original.size());
    }
    if (num_monitor_devs_ == 0) {
      ADD_FAILURE() << "amdgpu is not loaded; accelerator profiles were not restored";
      return;
    }
  }
  RestoreAcceleratorProfiles(original);

  bool read = true;
  const auto final_state = ReadPhysicalGpus(&read);
  EXPECT_TRUE(read) << "Memory partition unreadable after restore";
  EXPECT_EQ(final_state.size(), original.size()) << "GPU count changed after restore";
  for (const auto& entry : original) {
    const auto now = final_state.find(entry.first);
    if (now == final_state.end()) {
      ADD_FAILURE() << entry.first << ": missing after restore";
      continue;
    }
    EXPECT_EQ(now->second.memory, start) << entry.first << ": memory partition not restored";
    if (entry.second.profile_valid) {
      EXPECT_EQ(now->second.profile.profile_type, entry.second.profile.profile_type)
          << entry.first << ": compute partition not restored";
      EXPECT_EQ(now->second.profile.profile_index, entry.second.profile.profile_index)
          << entry.first << ": accelerator profile index not restored";
    }
  }
}
