/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: NCSA
 */

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <elf.h>
#include <glob.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>

#include "suites/functional/gpu_coredump.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const uint32_t kNumBufferElements = rocrtst::isEmuModeEnabled() ? 4 : 256;

GpuCoreDumpTest::GpuCoreDumpTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("GPU Core Dump Configuration Tests");
  set_description("Tests for configurable GPU core dump functionality including "
                  "custom patterns, format specifiers, and disable flag.");

  // Save original ulimit
  getrlimit(RLIMIT_CORE, &original_rlimit_);
}

GpuCoreDumpTest::~GpuCoreDumpTest(void) {
  // Restore original ulimit
  setrlimit(RLIMIT_CORE, &original_rlimit_);
}

void GpuCoreDumpTest::SetUp(void) {
  // Don't call TestBase::SetUp() - we don't want hsa_init() in parent
  // Just do minimal setup

  // Create temporary directory for test dumps
  test_dir_ = "/tmp/rocr_coredump_test_" + std::to_string(getpid());
  mkdir(test_dir_.c_str(), 0755);

  // Set ulimit for core dumps
  struct rlimit rlim;
  rlim.rlim_cur = 100 * 1024 * 1024;  // 100MB
  rlim.rlim_max = 100 * 1024 * 1024;
  setrlimit(RLIMIT_CORE, &rlim);
}

void GpuCoreDumpTest::Run(void) {
  // Nothing to do here - each test method handles its own execution
}

void GpuCoreDumpTest::DisplayTestInfo(void) {
  std::cout << "Test: " << title() << std::endl;
  std::cout << description() << std::endl;
}

void GpuCoreDumpTest::DisplayResults(void) const {
  // Nothing to display
}

void GpuCoreDumpTest::Close() {
  // Don't call TestBase::Close() - we never called hsa_init() in parent

  // Clean up test directory
  if (!test_dir_.empty()) {
    CleanupCoreDumps(test_dir_ + "/*");
    rmdir(test_dir_.c_str());
  }
}

pid_t GpuCoreDumpTest::RunFaultingKernelInChild() {
  pid_t pid = fork();

  if (pid < 0) {
    return -1;  // Fork failed
  }

  if (pid == 0) {
    // Child process - do ALL HSA work here
    hsa_status_t err;

    // Initialize HSA
    err = hsa_init();
    if (err != HSA_STATUS_SUCCESS) {
      _exit(1);
    }

    // Find agents
    hsa_agent_t cpu_agent = {0};
    hsa_agent_t gpu_agent = {0};

    err = hsa_iterate_agents(rocrtst::FindCPUDevice, &cpu_agent);
    // ProcessIterateError: INFO_BREAK -> SUCCESS, SUCCESS -> ERROR
    if (err == HSA_STATUS_INFO_BREAK) {
      err = HSA_STATUS_SUCCESS;
    } else if (err == HSA_STATUS_SUCCESS) {
      err = HSA_STATUS_ERROR;
    }
    if (err != HSA_STATUS_SUCCESS || cpu_agent.handle == 0) {
      hsa_shut_down();
      _exit(1);
    }

    err = hsa_iterate_agents(rocrtst::FindGPUDevice, &gpu_agent);
    if (err == HSA_STATUS_INFO_BREAK) {
      err = HSA_STATUS_SUCCESS;
    } else if (err == HSA_STATUS_SUCCESS) {
      err = HSA_STATUS_ERROR;
    }
    if (err != HSA_STATUS_SUCCESS || gpu_agent.handle == 0) {
      hsa_shut_down();
      _exit(1);
    }

    // Get profile to determine which pool finder to use
    hsa_profile_t profile;
    err = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_PROFILE, &profile);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_shut_down();
      _exit(1);
    }

    // Find memory pools based on profile (matching SetPoolsTypical logic)
    hsa_amd_memory_pool_t cpu_pool;
    hsa_amd_memory_pool_t kernarg_pool;

    if (profile == HSA_PROFILE_FULL) {
      // APU - use FindAPUStandardPool
      err = hsa_amd_agent_iterate_memory_pools(cpu_agent,
                                               rocrtst::FindAPUStandardPool,
                                               &cpu_pool);
      if (err == HSA_STATUS_INFO_BREAK) {
        err = HSA_STATUS_SUCCESS;
      } else if (err == HSA_STATUS_SUCCESS) {
        err = HSA_STATUS_ERROR;
      }
      if (err != HSA_STATUS_SUCCESS) {
        hsa_shut_down();
        _exit(1);
      }

      err = hsa_amd_agent_iterate_memory_pools(cpu_agent,
                                               rocrtst::FindAPUStandardPool,
                                               &kernarg_pool);
      if (err == HSA_STATUS_INFO_BREAK) {
        err = HSA_STATUS_SUCCESS;
      } else if (err == HSA_STATUS_SUCCESS) {
        err = HSA_STATUS_ERROR;
      }
      if (err != HSA_STATUS_SUCCESS) {
        hsa_shut_down();
        _exit(1);
      }
    } else {
      // Discrete GPU - use FindStandardPool and FindKernArgPool
      err = hsa_amd_agent_iterate_memory_pools(cpu_agent,
                                               rocrtst::FindStandardPool,
                                               &cpu_pool);
      if (err == HSA_STATUS_INFO_BREAK) {
        err = HSA_STATUS_SUCCESS;
      } else if (err == HSA_STATUS_SUCCESS) {
        err = HSA_STATUS_ERROR;
      }
      if (err != HSA_STATUS_SUCCESS) {
        hsa_shut_down();
        _exit(1);
      }

      err = hsa_amd_agent_iterate_memory_pools(cpu_agent,
                                               rocrtst::FindKernArgPool,
                                               &kernarg_pool);
      if (err == HSA_STATUS_INFO_BREAK) {
        err = HSA_STATUS_SUCCESS;
      } else if (err == HSA_STATUS_SUCCESS) {
        err = HSA_STATUS_ERROR;
      }
      if (err != HSA_STATUS_SUCCESS) {
        hsa_shut_down();
        _exit(1);
      }
    }

    // Create queue
    uint32_t queue_size = 0;
    err = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_shut_down();
      _exit(1);
    }

    hsa_queue_t* queue = nullptr;
    err = hsa_queue_create(gpu_agent, queue_size, HSA_QUEUE_TYPE_MULTI,
                           nullptr, nullptr, 0, 0, &queue);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_shut_down();
      _exit(1);
    }

    // Allocate buffers
    void* src_buffer = nullptr;
    void* dst_buffer = nullptr;

    err = hsa_amd_memory_pool_allocate(cpu_pool,
                                       kNumBufferElements * sizeof(uint32_t),
                                       0, &src_buffer);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    err = hsa_amd_memory_pool_allocate(cpu_pool,
                                       kNumBufferElements * sizeof(uint32_t),
                                       0, &dst_buffer);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_memory_free(src_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    // Allow GPU access
    hsa_agent_t ag_list[2] = {gpu_agent, cpu_agent};
    err = hsa_amd_agents_allow_access(2, ag_list, nullptr, src_buffer);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    err = hsa_amd_agents_allow_access(2, ag_list, nullptr, dst_buffer);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    // Load kernel from file
    int file_fd = open("test_case_template_kernels.hsaco", O_RDONLY);
    if (file_fd == -1) {
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    hsa_code_object_reader_t code_obj_rdr = {0};
    err = hsa_code_object_reader_create_from_file(file_fd, &code_obj_rdr);
    if (err != HSA_STATUS_SUCCESS) {
      close(file_fd);
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    hsa_executable_t executable = {0};
    err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                    nullptr, &executable);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_code_object_reader_destroy(code_obj_rdr);
      close(file_fd);
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    err = hsa_executable_load_agent_code_object(executable, gpu_agent, code_obj_rdr,
                                                nullptr, nullptr);
    hsa_code_object_reader_destroy(code_obj_rdr);
    close(file_fd);

    if (err != HSA_STATUS_SUCCESS) {
      hsa_executable_destroy(executable);
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    err = hsa_executable_freeze(executable, nullptr);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_executable_destroy(executable);
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    // Get kernel symbol
    hsa_executable_symbol_t symbol;
    err = hsa_executable_get_symbol_by_name(executable, "square.kd", &gpu_agent, &symbol);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_executable_destroy(executable);
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    uint64_t kernel_object = 0;
    err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                         &kernel_object);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_executable_destroy(executable);
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    // Allocate kernel arguments with nullptr arrays to cause fault
    void* kernarg_buffer = nullptr;
    struct __attribute__((aligned(16))) kernel_args_t {
      uint32_t* dstArray;
      uint32_t* srcArray;
      uint32_t size;
      uint32_t pad;
      uint64_t global_offset_x;
      uint64_t global_offset_y;
      uint64_t global_offset_z;
      uint64_t printf_buffer;
      uint64_t default_queue;
      uint64_t completion_action;
    } kernel_args;

    // Intentionally set to nullptr to cause a fault
    kernel_args.dstArray = nullptr;
    kernel_args.srcArray = nullptr;
    kernel_args.size = kNumBufferElements;
    kernel_args.pad = 0;
    kernel_args.global_offset_x = 0;
    kernel_args.global_offset_y = 0;
    kernel_args.global_offset_z = 0;
    kernel_args.printf_buffer = 0;
    kernel_args.default_queue = 0;
    kernel_args.completion_action = 0;

    err = hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(kernel_args), 0, &kernarg_buffer);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_executable_destroy(executable);
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    memcpy(kernarg_buffer, &kernel_args, sizeof(kernel_args));

    // Create completion signal
    hsa_signal_t signal;
    err = hsa_signal_create(1, 0, nullptr, &signal);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_memory_free(kernarg_buffer);
      hsa_executable_destroy(executable);
      hsa_memory_free(src_buffer);
      hsa_memory_free(dst_buffer);
      hsa_queue_destroy(queue);
      hsa_shut_down();
      _exit(1);
    }

    // Create and dispatch AQL packet
    hsa_kernel_dispatch_packet_t aql;
    memset(&aql, 0, sizeof(aql));

    aql.header = 0;
    aql.setup = 1;
    aql.workgroup_size_x = kNumBufferElements;
    aql.workgroup_size_y = 1;
    aql.workgroup_size_z = 1;
    aql.grid_size_x = kNumBufferElements;
    aql.grid_size_y = 1;
    aql.grid_size_z = 1;
    aql.private_segment_size = 0;
    aql.group_segment_size = 0;
    aql.kernel_object = kernel_object;
    aql.kernarg_address = kernarg_buffer;
    aql.completion_signal = signal;

    const uint32_t queue_mask = queue->size - 1;
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    hsa_queue_store_write_index_relaxed(queue, index + 1);

    // Write packet to queue
    hsa_kernel_dispatch_packet_t* queue_packet =
        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address))[index & queue_mask];

    queue_packet->workgroup_size_x = aql.workgroup_size_x;
    queue_packet->workgroup_size_y = aql.workgroup_size_y;
    queue_packet->workgroup_size_z = aql.workgroup_size_z;
    queue_packet->grid_size_x = aql.grid_size_x;
    queue_packet->grid_size_y = aql.grid_size_y;
    queue_packet->grid_size_z = aql.grid_size_z;
    queue_packet->private_segment_size = aql.private_segment_size;
    queue_packet->group_segment_size = aql.group_segment_size;
    queue_packet->kernel_object = aql.kernel_object;
    queue_packet->kernarg_address = aql.kernarg_address;
    queue_packet->completion_signal = aql.completion_signal;

    uint32_t aql_header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    __atomic_store_n(reinterpret_cast<uint32_t*>(queue_packet),
                     aql_header | (aql.setup << 16), __ATOMIC_RELEASE);

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, index);

    // Wait for completion (or fault)
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                              UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Should not reach here if fault occurs
    hsa_signal_destroy(signal);
    hsa_memory_free(kernarg_buffer);
    hsa_executable_destroy(executable);
    hsa_memory_free(src_buffer);
    hsa_memory_free(dst_buffer);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    _exit(0);
  }

  // Parent process - wait for child with timeout
  int status;
  int timeout_ms = 1000;  // 1 second
  int elapsed = 0;

  while (elapsed < timeout_ms) {
    pid_t result = waitpid(pid, &status, WNOHANG);
    if (result == pid) {
      // Child finished
      usleep(100000);  // 100ms for core dump to finish writing
      return pid;
    } else if (result < 0) {
      // Error
      return -1;
    }
    // Still running, wait a bit
    usleep(10000);  // 10ms
    elapsed += 10;
  }

  // Timeout - kill the child
  if (verbosity() > 0) {
    std::cout << "    Child process timeout, killing..." << std::endl;
  }
  kill(pid, SIGKILL);
  waitpid(pid, &status, 0);
  usleep(100000);  // Give time for core dump
  return pid;
}

bool GpuCoreDumpTest::VerifyCoreDumpFile(const std::string& filename) {
  if (access(filename.c_str(), F_OK) != 0) {
    if (verbosity() > 0) {
      std::cout << "    Core dump file not found: " << filename << std::endl;
    }
    return false;
  }

  if (access(filename.c_str(), R_OK) != 0) {
    if (verbosity() > 0) {
      std::cout << "    Core dump file not readable: " << filename << std::endl;
    }
    return false;
  }

  if (!IsValidGPUCoreDump(filename)) {
    if (verbosity() > 0) {
      std::cout << "    File is not a valid GPU core dump: " << filename << std::endl;
    }
    return false;
  }

  if (verbosity() > 0) {
    struct stat st;
    stat(filename.c_str(), &st);
    std::cout << "    Core dump verified: " << filename
              << " (size: " << st.st_size << " bytes)" << std::endl;
  }

  return true;
}

bool GpuCoreDumpTest::IsValidGPUCoreDump(const std::string& filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    return false;
  }

  Elf64_Ehdr ehdr;
  if (read(fd, &ehdr, sizeof(ehdr)) != sizeof(ehdr)) {
    close(fd);
    return false;
  }

  close(fd);

  // Verify ELF magic number
  if (ehdr.e_ident[EI_MAG0] != ELFMAG0 ||
      ehdr.e_ident[EI_MAG1] != ELFMAG1 ||
      ehdr.e_ident[EI_MAG2] != ELFMAG2 ||
      ehdr.e_ident[EI_MAG3] != ELFMAG3) {
    return false;
  }

  if (ehdr.e_ident[EI_CLASS] != ELFCLASS64) {
    return false;
  }

  if (ehdr.e_type != ET_CORE) {
    return false;
  }

  // EM_AMDGPU = 224
  if (ehdr.e_machine != 224) {
    return false;
  }

  return true;
}

void GpuCoreDumpTest::CleanupCoreDumps(const std::string& pattern) {
  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  int ret = glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
  if (ret == 0) {
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
      unlink(glob_result.gl_pathv[i]);
    }
  }
  globfree(&glob_result);
}

void GpuCoreDumpTest::TestDefaultPattern(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing default pattern (kernel core_pattern + .gpu suffix)..." << std::endl;
  }

  // Read kernel core pattern
  std::ifstream pattern_file("/proc/sys/kernel/core_pattern");
  std::string kernel_pattern;
  if (pattern_file.is_open()) {
    std::getline(pattern_file, kernel_pattern);
  }

  // Unset HSA_COREDUMP_FILE to use default
  unsetenv("HSA_COREDUMP_FILE");
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  std::string expected;

  if (kernel_pattern.empty()) {
    expected = "gpucore.%p.gpu";
  } else if (kernel_pattern[0] == '|') {
    if (verbosity() > 0) {
      std::cout << "    Kernel uses pipe pattern - testing graceful fault handling" << std::endl;
    }

    pid_t child_pid = RunFaultingKernelInChild();
    if (child_pid < 0) {
      FAIL() << "Failed to run test in child process";
      return;
    }

    if (verbosity() > 0) {
      std::cout << "    GPU fault handled successfully (pipe pattern in use)" << std::endl;
    }
    return;
  } else {
    expected = kernel_pattern + ".gpu";
  }

  // Run test in child and get PID
  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    return;
  }

  // Substitute child PID in expected filename
  size_t pos = expected.find("%p");
  if (pos != std::string::npos) {
    expected.replace(pos, 2, std::to_string(child_pid));
  }

  bool success = VerifyCoreDumpFile(expected);
  EXPECT_TRUE(success);

  if (success) {
    unlink(expected.c_str());
  }
}

void GpuCoreDumpTest::TestCustomPattern(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing custom pattern (HSA_COREDUMP_FILE)..." << std::endl;
  }

  std::string pattern = test_dir_ + "/custom_gpu_core.%p";
  setenv("HSA_COREDUMP_FILE", pattern.c_str(), 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    unsetenv("HSA_COREDUMP_FILE");
    return;
  }

  std::string expected = test_dir_ + "/custom_gpu_core." + std::to_string(child_pid);
  bool success = VerifyCoreDumpFile(expected);
  EXPECT_TRUE(success);

  if (success) {
    unlink(expected.c_str());
  }

  unsetenv("HSA_COREDUMP_FILE");
}

void GpuCoreDumpTest::TestDisableFlag(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing disable flag (HSA_DISABLE_COREDUMP_ON_EXCEPTION=1)..." << std::endl;
  }

  std::string pattern = test_dir_ + "/should_not_exist.%p";
  setenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION", "1", 1);
  setenv("HSA_COREDUMP_FILE", pattern.c_str(), 1);

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");
    unsetenv("HSA_COREDUMP_FILE");
    return;
  }

  std::string test_file = test_dir_ + "/should_not_exist." + std::to_string(child_pid);
  bool file_exists = (access(test_file.c_str(), F_OK) == 0);

  EXPECT_FALSE(file_exists) << "Core dump should not have been created when disabled";

  if (verbosity() > 0) {
    if (!file_exists) {
      std::cout << "    Correctly prevented core dump creation" << std::endl;
    }
  }

  if (file_exists) {
    unlink(test_file.c_str());
  }

  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");
  unsetenv("HSA_COREDUMP_FILE");
}

void GpuCoreDumpTest::TestPatternSubstitution(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing pattern substitution (%p, %e, %t)..." << std::endl;
  }

  std::string pattern = test_dir_ + "/core.%p.dump";
  setenv("HSA_COREDUMP_FILE", pattern.c_str(), 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    unsetenv("HSA_COREDUMP_FILE");
    return;
  }

  std::string expected = test_dir_ + "/core." + std::to_string(child_pid) + ".dump";
  bool success = VerifyCoreDumpFile(expected);
  EXPECT_TRUE(success);

  if (success) {
    unlink(expected.c_str());
  }

  unsetenv("HSA_COREDUMP_FILE");
}

void GpuCoreDumpTest::TestInvalidPath(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing invalid path handling..." << std::endl;
  }

  std::string pattern = "/nonexistent_dir_12345/core.%p";
  setenv("HSA_COREDUMP_FILE", pattern.c_str(), 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    unsetenv("HSA_COREDUMP_FILE");
    return;
  }

  std::string invalid_file = "/nonexistent_dir_12345/core." + std::to_string(child_pid);
  bool file_exists = (access(invalid_file.c_str(), F_OK) == 0);

  EXPECT_FALSE(file_exists) << "Core dump should not be created with invalid path";

  if (verbosity() > 0) {
    if (!file_exists) {
      std::cout << "    Correctly handled invalid path" << std::endl;
    }
  }

  unsetenv("HSA_COREDUMP_FILE");
}
