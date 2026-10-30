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
#include <set>
#include <sstream>
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
// Same bounds as test_memory_partition_cycle.py.
constexpr int kModprobeTimeoutSec = 120;
constexpr int kReapTimeoutSec = 10;
constexpr int kDriverLiveTimeoutSec = 60;
constexpr int kEnumerateAttempts = 30;
constexpr uint32_t kRetryDelayMs = 2000;
constexpr uint32_t kNameLen = 255;
constexpr int kCapSysModule = 16;
const char kRefcntPath[] = "/sys/module/amdgpu/refcnt";
const char kInitstatePath[] = "/sys/module/amdgpu/initstate";
const char kProcStatusPath[] = "/proc/self/status";
const char* const kModprobePaths[] = {"/usr/sbin/modprobe", "/sbin/modprobe"};

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

std::string CapsString(const amdsmi_nps_caps_t& caps) {
  std::string names;
  for (auto mode : kNpsModes) {
    if (ModeInCaps(caps, mode)) {
      names += (names.empty() ? "" : ", ") + memoryPartitionString(mode);
    }
  }
  return names.empty() ? "N/A" : names;
}

std::string ReadFile(const std::string& path) {
  std::ifstream file(path);
  std::stringstream text;
  text << file.rdbuf();
  return text.str();
}

std::string FirstLine(const std::string& text) { return text.substr(0, text.find('\n')); }

// -1 unless the first line is a non-negative decimal number.
int ParseCount(const std::string& text) {
  const std::string line = FirstLine(text);
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(line.c_str(), &end, 10);
  if (line.empty() || errno != 0 || *end != '\0' || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    return -1;
  }
  return static_cast<int>(value);
}

// CAP_SYS_MODULE in the CapEff line of /proc/self/status.
bool HasCapSysModule(const std::string& status) {
  std::istringstream lines(status);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.rfind("CapEff:", 0) != 0) {
      continue;
    }
    const char* digits = line.c_str() + std::strlen("CapEff:");
    char* end = nullptr;
    const unsigned long long caps = std::strtoull(digits, &end, 16);
    return end != digits && ((caps >> kCapSysModule) & 1ULL);
  }
  return false;
}

// 1 once `pid` is reaped, 0 at the deadline, -1 on a waitpid error.
int Reap(pid_t pid, std::chrono::steady_clock::time_point deadline, int* status) {
  for (;;) {
    const pid_t done = waitpid(pid, status, WNOHANG);
    if (done == pid) {
      return 1;
    }
    if (done < 0 && errno != EINTR) {
      return -1;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return 0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

}  // namespace

TestMemoryPartitionReadWrite::TestMemoryPartitionReadWrite() : TestBase() {
  set_title("AMDSMI Memory Partition Read Test");
  set_description(
      "Verify memory partition reads, capabilities, invalid inputs, and current-mode sets. "
      "With AMDSMI_ALLOW_DESTRUCTIVE_TESTS=1, also apply each supported NPS mode with a "
      "driver reload, then restore the starting NPS mode and accelerator profiles.");
  const char* destructive = std::getenv("AMDSMI_ALLOW_DESTRUCTIVE_TESTS");
  allow_destructive_ = destructive && std::strcmp(destructive, "1") == 0;
  api_.read_file = ReadFile;
  api_.is_executable = [](const std::string& path) { return access(path.c_str(), X_OK) == 0; };
  api_.run = [](const std::string& path, const std::vector<std::string>& args, std::string* error) {
    return RunProgram(path, args, kModprobeTimeoutSec, kReapTimeoutSec, error);
  };
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

TestMemoryPartitionReadWrite::RunResult TestMemoryPartitionReadWrite::RunProgram(
    const std::string& path, const std::vector<std::string>& args, int timeout_sec, int reap_sec,
    std::string* error) {
  std::string command = path;
  std::vector<char*> argv{const_cast<char*>(path.c_str())};
  for (const auto& arg : args) {
    command += " " + arg;
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);
  // A fixed environment, so variables such as MODPROBE_OPTIONS do not pass through sudo -E.
  char env_path[] = "PATH=/usr/sbin:/usr/bin:/sbin:/bin";
  char* envp[] = {env_path, nullptr};

  std::cout << "\t** " << command << std::endl;
  const pid_t pid = fork();
  if (pid < 0) {
    *error = command + ": fork failed: " + std::strerror(errno);
    return RunResult::kFailed;
  }
  if (pid == 0) {
    execve(path.c_str(), argv.data(), envp);
    _exit(127);
  }
  int status = 0;
  const auto now = std::chrono::steady_clock::now;
  const int reaped = Reap(pid, now() + std::chrono::seconds(timeout_sec), &status);
  if (reaped < 0) {
    *error = command + ": waitpid failed: " + std::strerror(errno);
    return RunResult::kFailed;
  }
  if (reaped == 0) {
    kill(pid, SIGKILL);
    if (Reap(pid, now() + std::chrono::seconds(reap_sec), &status) != 1) {
      *error = command + " timed out after " + std::to_string(timeout_sec) +
               "s and ignored SIGKILL; it is stuck in the kernel, so the node needs a reboot";
      return RunResult::kStuck;
    }
    *error = command + " timed out after " + std::to_string(timeout_sec) + "s";
    return RunResult::kTimedOut;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    *error = command + " failed (" +
             (WIFEXITED(status) ? "exit status " + std::to_string(WEXITSTATUS(status))
                                : "signal " + std::to_string(WTERMSIG(status))) +
             ")";
    return RunResult::kFailed;
  }
  return RunResult::kOk;
}

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
    const std::string nps_caps_str = CapsString(profile.memory_caps);
    const std::string profile_type_str = ProfileName(profile.profile_type);

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
      const std::string profile_type_str = ProfileName(current_profile.profile_type);
      const std::string nps_caps_str = CapsString(current_profile.memory_caps);
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
    const std::string original_memory(
        orig_memory_partition, strnlen(orig_memory_partition, sizeof(orig_memory_partition)));
    const amdsmi_memory_partition_type_t original_mode = ModeFromName(original_memory);
    EXPECT_NE(original_mode, AMDSMI_MEMORY_PARTITION_UNKNOWN)
        << "Invalid memory partition snapshot";
    if (original_mode == AMDSMI_MEMORY_PARTITION_UNKNOWN) {
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
    new_memory_partition = original_mode;
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
  // amdsmi_init is reference counted, so this re-enumerates only while the fixture
  // holds the only reference, as it does in amdsmitst.
  api_.shut_down();
  num_monitor_devs_ = 0;
  amdsmi_status_t err = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (err == AMDSMI_STATUS_SUCCESS) {
    err = EnumerateHandles();
  }
  if (err != AMDSMI_STATUS_SUCCESS) {
    num_monitor_devs_ = 0;
    *error = std::string("amdsmi_init or device enumeration failed: ") +
             smi_amdgpu_get_status_string(err, false);
    return false;
  }
  return true;
}

std::string TestMemoryPartitionReadWrite::ModprobePath() {
  for (const char* path : kModprobePaths) {
    if (api_.is_executable(path)) {
      return path;
    }
  }
  return "";
}

int TestMemoryPartitionReadWrite::AmdgpuRefcount() {
  return ParseCount(api_.read_file(kRefcntPath));
}

std::string TestMemoryPartitionReadWrite::ReloadBlocker() {
  std::string blocker;
  if (ModprobePath().empty()) {
    blocker = "modprobe not found in /usr/sbin or /sbin (install kmod)";
  } else if (!HasCapSysModule(api_.read_file(kProcStatusPath))) {
    blocker = "this process cannot load kernel modules (needs root with CAP_SYS_MODULE)";
  } else if (FirstLine(api_.read_file(kInitstatePath)) != "live") {
    blocker =
        "amdgpu is not a loaded module (it may be built into the kernel), so it cannot be "
        "reloaded";
  } else {
    // Our own descriptors would count as a user of amdgpu.
    api_.shut_down();
    const int refcount = AmdgpuRefcount();
    if (refcount != 0) {
      blocker = refcount < 0
                    ? "amdgpu refcnt is unreadable"
                    : "amdgpu is in use by another process (refcnt " + std::to_string(refcount) +
                          "); the NPS cycle needs the GPUs idle";
    }
  }
  std::string error;
  if (!RefreshHandles(&error) && blocker.empty()) {
    blocker = error;
  }
  return blocker;
}

TestMemoryPartitionReadWrite::Reload TestMemoryPartitionReadWrite::ReloadDriver(
    std::string* error) {
  std::string ignored;
  const std::string modprobe = ModprobePath();
  // Our own descriptors would keep amdgpu loaded.
  api_.shut_down();
  num_monitor_devs_ = 0;
  const int refcount = AmdgpuRefcount();
  if (modprobe.empty() || refcount != 0) {
    *error = modprobe.empty() ? "modprobe not found in /usr/sbin or /sbin"
                              : "amdgpu is in use (refcnt " + std::to_string(refcount) + ")";
    RefreshHandles(&ignored);
    return Reload::kNotUnloaded;
  }
  std::cout << "\t** Reloading amdgpu, this may take a minute..." << std::endl;
  const auto started = std::chrono::steady_clock::now();
  const RunResult unload = api_.run(modprobe, {"-r", "amdgpu"}, error);
  if (unload != RunResult::kOk) {
    if (unload != RunResult::kStuck && FirstLine(api_.read_file(kInitstatePath)) == "live") {
      RefreshHandles(&ignored);
      return Reload::kNotUnloaded;
    }
    return Reload::kNotLoaded;
  }
  // Never leave the driver unloaded: try the load twice, unless modprobe is stuck.
  RunResult load = api_.run(modprobe, {"amdgpu"}, error);
  if (load == RunResult::kFailed || load == RunResult::kTimedOut) {
    load = api_.run(modprobe, {"amdgpu"}, error);
  }
  if (load != RunResult::kOk) {
    return Reload::kNotLoaded;
  }
  for (int waited = 0; FirstLine(api_.read_file(kInitstatePath)) != "live"; ++waited) {
    if (waited >= kDriverLiveTimeoutSec) {
      *error = "amdgpu loaded but did not report live within " +
               std::to_string(kDriverLiveTimeoutSec) + "s";
      return Reload::kNotLoaded;
    }
    api_.sleep_ms(1000);
  }
  std::cout << "\t** Reload took "
            << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() -
                                                                started)
                   .count()
            << "s" << std::endl;
  // The devices probe asynchronously after the load.
  for (int attempt = 1; !RefreshHandles(error); ++attempt) {
    if (attempt >= kEnumerateAttempts) {
      return Reload::kNoDevices;
    }
    api_.sleep_ms(kRetryDelayMs);
  }
  return Reload::kApplied;
}

std::string TestMemoryPartitionReadWrite::PhysicalKey(amdsmi_processor_handle handle) {
  amdsmi_kfd_info_t kfd = {};
  if (api_.get_kfd_info(handle, &kfd) == AMDSMI_STATUS_SUCCESS && kfd.current_partition_id != 0 &&
      kfd.current_partition_id != std::numeric_limits<uint32_t>::max()) {
    return "";
  }
  amdsmi_bdf_t bdf = {};
  if (api_.get_bdf(handle, &bdf) != AMDSMI_STATUS_SUCCESS) {
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
    char memory[kNameLen] = {};
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
    gpu.caps_valid = api_.get_memory_config(gpu.handle, &config) == AMDSMI_STATUS_SUCCESS;
    gpu.caps = config.partition_caps;
    gpus[key] = gpu;
  }
  return gpus;
}

std::string TestMemoryPartitionReadWrite::AgreedMode(const std::map<std::string, PhysicalGpu>& gpus,
                                                     bool complete, size_t physical_count) {
  if (!complete || gpus.empty() || gpus.size() != physical_count) {
    return "";
  }
  const std::string& mode = gpus.begin()->second.memory;
  for (const auto& entry : gpus) {
    if (entry.second.memory != mode) {
      return "";
    }
  }
  return mode;
}

TestMemoryPartitionReadWrite::StageResult TestMemoryPartitionReadWrite::StageOnAll(
    amdsmi_memory_partition_type_t mode) {
  StageResult result;
  const auto gpus = ReadPhysicalGpus(&result.complete);
  result.total = gpus.size();
  for (const auto& entry : gpus) {
    const auto status = api_.set_memory(entry.second.handle, mode);
    std::cout << "\t** " << entry.first << ": set " << memoryPartitionString(mode) << " -> "
              << smi_amdgpu_get_status_string(status, false) << std::endl;
    if (status == AMDSMI_STATUS_SUCCESS) {
      ++result.accepted;
    } else {
      EXPECT_TRUE(status == AMDSMI_STATUS_INVAL || status == AMDSMI_STATUS_NOT_SUPPORTED)
          << entry.first << ": unexpected status staging " << memoryPartitionString(mode);
    }
  }
  if (result.accepted > 0 && result.accepted < result.total) {
    ADD_FAILURE() << memoryPartitionString(mode) << " accepted by " << result.accepted << " of "
                  << result.total << " GPUs";
  }
  return result;
}

bool TestMemoryPartitionReadWrite::ApplyStaged(const std::string& staged,
                                               const std::string& start) {
  std::string error;
  Reload result = ReloadDriver(&error);
  if (result == Reload::kNotUnloaded) {
    std::cout << "\t** " << error << "; retrying the reload once" << std::endl;
    api_.sleep_ms(kRetryDelayMs);
    result = ReloadDriver(&error);
  }
  const std::string not_restored =
      ", so " + start + " and the accelerator profiles were not restored";
  switch (result) {
    case Reload::kApplied:
      return true;
    case Reload::kNotUnloaded:
      ADD_FAILURE() << error << "; " << staged
                    << " is still staged, and the next amdgpu reload applies it" << not_restored;
      break;
    case Reload::kNotLoaded:
      ADD_FAILURE() << error << "; amdgpu is not loaded, and its next load may bring the GPUs "
                    << "up in " << staged << not_restored;
      break;
    case Reload::kNoDevices:
      ADD_FAILURE() << error << "; amdgpu reloaded but the library found no devices"
                    << not_restored;
      break;
  }
  return false;
}

std::string TestMemoryPartitionReadWrite::VerifyMode(const std::string& expected,
                                                     size_t physical_count) {
  // Partition devices can take a moment to appear after the load.
  std::map<std::string, PhysicalGpu> gpus;
  bool complete = false;
  std::string ignored;
  for (int attempt = 1; attempt <= kEnumerateAttempts; ++attempt) {
    gpus = ReadPhysicalGpus(&complete);
    if ((complete && gpus.size() == physical_count) || attempt == kEnumerateAttempts) {
      break;
    }
    api_.sleep_ms(kRetryDelayMs);
    RefreshHandles(&ignored);
  }
  EXPECT_TRUE(complete) << "Memory partition unreadable after reload";
  EXPECT_EQ(gpus.size(), physical_count) << "GPU count changed after reload";
  for (const auto& entry : gpus) {
    EXPECT_EQ(entry.second.memory, expected) << entry.first << ": reload applied the wrong mode";
    std::cout << "\t** " << entry.first << ": " << entry.second.memory << " / "
              << ProfileName(entry.second.profile.profile_type) << std::endl;
  }
  return AgreedMode(gpus, complete, physical_count);
}

void TestMemoryPartitionReadWrite::RestoreAcceleratorProfiles(
    const std::map<std::string, PhysicalGpu>& original) {
  // Shared rounds: read every GPU, set each one that differs, then wait once.
  std::set<std::string> restored, rejected;
  std::map<std::string, amdsmi_status_t> last_status;
  std::string ignored;
  for (int round = 1; round <= kProfileAttempts; ++round) {
    bool complete = true;
    const auto gpus = ReadPhysicalGpus(&complete);
    bool pending = false;
    for (const auto& entry : original) {
      const std::string& key = entry.first;
      const auto& want = entry.second.profile;
      if (!entry.second.profile_valid || restored.count(key) || rejected.count(key)) {
        continue;
      }
      const auto current = gpus.find(key);
      if (current == gpus.end() || !current->second.profile_valid) {
        pending = true;
        continue;
      }
      if (current->second.profile.profile_index == want.profile_index) {
        restored.insert(key);
        continue;
      }
      pending = true;
      if (round == kProfileAttempts) {
        continue;  // the last round only reads back
      }
      const auto status = api_.set_profile(current->second.handle, want.profile_index);
      last_status[key] = status;
      std::cout << "\t** " << key << ": set accelerator profile " << ProfileName(want.profile_type)
                << " (index " << want.profile_index << ") -> "
                << smi_amdgpu_get_status_string(status, false) << std::endl;
      if (status != AMDSMI_STATUS_SUCCESS && status != AMDSMI_STATUS_BUSY) {
        rejected.insert(key);
      }
    }
    if (!pending || round == kProfileAttempts) {
      break;
    }
    // A profile change replaces the partition devices, and BUSY is retried.
    api_.sleep_ms(kRetryDelayMs);
    RefreshHandles(&ignored);
  }
  for (const auto& entry : original) {
    if (!entry.second.profile_valid || restored.count(entry.first)) {
      continue;
    }
    const auto last = last_status.find(entry.first);
    ADD_FAILURE() << entry.first << ": accelerator profile not restored to "
                  << ProfileName(entry.second.profile.profile_type) << " (index "
                  << entry.second.profile.profile_index << "), last status "
                  << (last == last_status.end()
                          ? "none"
                          : smi_amdgpu_get_status_string(last->second, false));
  }
}

void TestMemoryPartitionReadWrite::RunMemoryPartitionCycle() {
  std::cout << "\t**=========================================================\n"
            << "\t**NPS cycle (AMDSMI_ALLOW_DESTRUCTIVE_TESTS=1) =============\n"
            << "\t**=========================================================\n";
  bool complete = true;
  const auto original = ReadPhysicalGpus(&complete);
  if (!complete) {
    ADD_FAILURE() << "Memory partition unreadable on some GPUs; NPS cycle not run, nothing was "
                     "staged";
    return;
  }
  if (original.empty()) {
    GTEST_SKIP() << "No GPU reports a memory partition; nothing to cycle";
  }
  const std::string start = original.begin()->second.memory;
  amdsmi_nps_caps_t caps = original.begin()->second.caps;
  std::string unreadable_caps;
  for (const auto& entry : original) {
    if (entry.second.memory != start) {
      GTEST_SKIP() << "GPUs start in different memory partition modes; NPS cycle not run";
    }
    if (!entry.second.caps_valid) {
      unreadable_caps += " " + entry.first;
    }
    caps.nps_cap_mask &= entry.second.caps.nps_cap_mask;
  }
  if (!unreadable_caps.empty()) {
    ADD_FAILURE() << "Memory partition capabilities unreadable on" << unreadable_caps
                  << "; NPS cycle not run, nothing was staged";
    return;
  }
  if (ModeFromName(start) == AMDSMI_MEMORY_PARTITION_UNKNOWN) {
    ADD_FAILURE() << "Starting memory partition " << start << " cannot be restored";
    return;
  }
  std::vector<amdsmi_memory_partition_type_t> candidates;
  for (auto mode : kNpsModes) {
    if (memoryPartitionString(mode) != start && ModeInCaps(caps, mode)) {
      candidates.push_back(mode);
    }
  }
  if (candidates.empty()) {
    GTEST_SKIP() << "Every GPU supports only " << start << " (" << CapsString(caps)
                 << "); no other NPS mode to cycle";
  }
  const std::string blocker = ReloadBlocker();
  if (!blocker.empty()) {
    ADD_FAILURE() << blocker << "; NPS cycle not run, nothing was staged";
    return;
  }

  std::cout << "\t** Starting mode: " << start << " on " << original.size()
            << " GPUs; supported: " << CapsString(caps) << "\n";
  std::string current = start;
  std::vector<std::string> applied;
  for (auto mode : candidates) {
    const std::string name = memoryPartitionString(mode);
    std::cout << "\n\t** ---- " << current << " -> " << name << " ----" << std::endl;
    const StageResult stage = StageOnAll(mode);
    if (stage.accepted == 0) {
      std::cout << "\t** " << name << " rejected by every GPU; no reload needed.\n";
      continue;
    }
    if (!ApplyStaged(name, start)) {
      return;
    }
    current = VerifyMode(name, original.size());
    if (current != name || stage.accepted != stage.total || !stage.complete) {
      break;
    }
    applied.push_back(name);
  }

  std::cout << "\n\t** ---- Restore " << start << " ----" << std::endl;
  if (current != start) {
    const StageResult stage = StageOnAll(ModeFromName(start));
    if (stage.accepted == 0) {
      ADD_FAILURE() << "Could not stage the restore to " << start;
      return;
    }
    if (!ApplyStaged(start, start)) {
      return;
    }
    VerifyMode(start, original.size());
  }
  RestoreAcceleratorProfiles(original);

  bool final_complete = true;
  const auto final_state = ReadPhysicalGpus(&final_complete);
  EXPECT_TRUE(final_complete) << "Memory partition unreadable after restore";
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
  std::string applied_names;
  for (const auto& name : applied) {
    applied_names += (applied_names.empty() ? "" : ", ") + name;
  }
  std::cout << "\t** Modes applied: " << (applied.empty() ? "none" : applied_names) << std::endl;
  if (applied.empty()) {
    GTEST_SKIP() << "No other NPS mode could be applied (modes applied: none)";
  }
}
