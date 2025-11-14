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
#include <limits.h>
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

// Convert core pattern to glob pattern, substituting known values
// and using wildcards for unknowable values (timestamp, TID)
namespace {
std::string PatternToGlob(const std::string& pattern, pid_t child_pid) {
  std::string result;

  // Get values we can know
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  hostname[sizeof(hostname) - 1] = '\0';

  char exe_path[PATH_MAX];
  std::string exe_name = "unknown";
  ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  if (len > 0) {
    exe_path[len] = '\0';
    char* base = basename(exe_path);
    if (base) exe_name = base;
  }

  // Parse and substitute
  for (size_t i = 0; i < pattern.length(); i++) {
    if (pattern[i] == '%' && i + 1 < pattern.length()) {
      switch (pattern[i + 1]) {
        case '%': result += '%'; break;
        case 'p': result += std::to_string(child_pid); break;
        case 'i': result += '*'; break;  // TID - use wildcard
        case 'h': result += hostname; break;
        case 'e': result += exe_name; break;
        case 't': result += '*'; break;  // Timestamp - use wildcard
        default: break;  // Drop unsupported specifiers
      }
      i++;
    } else {
      result += pattern[i];
    }
  }

  return result;
}

// Find core dump file matching the glob pattern
std::string FindMatchingCoreDump(const std::string& glob_pattern) {
  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  int ret = glob(glob_pattern.c_str(), 0, nullptr, &glob_result);
  std::string found_file;

  if (ret == 0 && glob_result.gl_pathc > 0) {
    found_file = glob_result.gl_pathv[0];
  }

  globfree(&glob_result);
  return found_file;
}
}  // anonymous namespace
// RAII helper class for automatic HSA resource cleanup
class HSAResourceGuard {
public:
  hsa_queue_t* queue = nullptr;
  hsa_executable_t executable = {0};
  void* kernarg_buffer = nullptr;
  hsa_signal_t signal = {0};
  int file_fd = -1;
  hsa_code_object_reader_t code_obj_rdr = {0};

  HSAResourceGuard() = default;
  ~HSAResourceGuard() {
    // Cleanup in reverse order of typical acquisition
    if (signal.handle) hsa_signal_destroy(signal);
    if (kernarg_buffer) hsa_memory_free(kernarg_buffer);
    if (executable.handle) hsa_executable_destroy(executable);
    if (code_obj_rdr.handle) hsa_code_object_reader_destroy(code_obj_rdr);
    if (file_fd != -1) close(file_fd);
    if (queue) hsa_queue_destroy(queue);
    hsa_shut_down();
  }

  // Prevent copying
  HSAResourceGuard(const HSAResourceGuard&) = delete;
  HSAResourceGuard& operator=(const HSAResourceGuard&) = delete;
};

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
    // Child process - verify environment is inherited
    const char* coredump_file = getenv("HSA_COREDUMP_PATTERN");
    const char* disable_flag = getenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");
    const char* show_progress = getenv("HSA_COREDUMP_SHOW_PROGRESS");

    fprintf(stderr, "CHILD: HSA_COREDUMP_PATTERN=%s\n",
            coredump_file ? coredump_file : "(null)");
    fprintf(stderr, "CHILD: HSA_DISABLE_COREDUMP_ON_EXCEPTION=%s\n",
            disable_flag ? disable_flag : "(null)");
    fprintf(stderr, "CHILD: HSA_COREDUMP_SHOW_PROGRESS=%s\n",
            show_progress ? show_progress : "(null)");



    // Child process - do ALL HSA work here
    hsa_status_t err;

    // RAII guard will cleanup all resources on exit
    HSAResourceGuard resources;

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

    // Find kernarg pool only; we don't need cpu_pool since
    // we're not allocating buffers
    hsa_amd_memory_pool_t kernarg_pool;

    if (profile == HSA_PROFILE_FULL) {
      // APU - use FindAPUStandardPool
      err = hsa_amd_agent_iterate_memory_pools(cpu_agent,
                                               rocrtst::FindAPUStandardPool,
                                               &kernarg_pool);
    } else {
      // Discrete GPU - use FindKernArgPool
      err = hsa_amd_agent_iterate_memory_pools(cpu_agent,
                                               rocrtst::FindKernArgPool,
                                               &kernarg_pool);
    }

    if (err == HSA_STATUS_INFO_BREAK) {
      err = HSA_STATUS_SUCCESS;
    } else if (err == HSA_STATUS_SUCCESS) {
      err = HSA_STATUS_ERROR;
    }
    if (err != HSA_STATUS_SUCCESS) {
      hsa_shut_down();
      _exit(1);
    }

    // Create queue
    uint32_t queue_size = 0;
    err = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_shut_down();
      _exit(1);
    }

    err = hsa_queue_create(gpu_agent, queue_size, HSA_QUEUE_TYPE_MULTI,
                           nullptr, nullptr, 0, 0, &resources.queue);
    if (err != HSA_STATUS_SUCCESS) _exit(1);

    // Load kernel from file (no need to allocate src/dst buffers - we pass nullptr)
    resources.file_fd = open("test_case_template_kernels.hsaco", O_RDONLY);
    if (resources.file_fd == -1) _exit(1);

    err = hsa_code_object_reader_create_from_file(resources.file_fd, &resources.code_obj_rdr);
    if (err != HSA_STATUS_SUCCESS) _exit(1);

    err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                    nullptr, &resources.executable);
    if (err != HSA_STATUS_SUCCESS) _exit(1);

    err = hsa_executable_load_agent_code_object(resources.executable,
                        gpu_agent, resources.code_obj_rdr, nullptr, nullptr);
    // Can destroy reader now
    hsa_code_object_reader_destroy(resources.code_obj_rdr);
    resources.code_obj_rdr = {0};
    close(resources.file_fd);
    resources.file_fd = -1;

    if (err != HSA_STATUS_SUCCESS) _exit(1);

    err = hsa_executable_freeze(resources.executable, nullptr);
    if (err != HSA_STATUS_SUCCESS) _exit(1);

    // Get kernel symbol
    hsa_executable_symbol_t symbol;
    err = hsa_executable_get_symbol_by_name(resources.executable, "square.kd", &gpu_agent, &symbol);
    if (err != HSA_STATUS_SUCCESS) _exit(1);

    uint64_t kernel_object = 0;
    err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                         &kernel_object);
    if (err != HSA_STATUS_SUCCESS) _exit(1);

    // Allocate kernel arguments with nullptr arrays to cause fault
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

    err = hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(kernel_args), 0, &resources.kernarg_buffer);
    if (err != HSA_STATUS_SUCCESS) _exit(1);

    memcpy(resources.kernarg_buffer, &kernel_args, sizeof(kernel_args));

    // Create completion signal
    err = hsa_signal_create(1, 0, nullptr, &resources.signal);
    if (err != HSA_STATUS_SUCCESS) _exit(1);

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
    aql.kernarg_address = resources.kernarg_buffer;
    aql.completion_signal = resources.signal;

    const uint32_t queue_mask = resources.queue->size - 1;
    uint64_t index = hsa_queue_load_write_index_relaxed(resources.queue);
    hsa_queue_store_write_index_relaxed(resources.queue, index + 1);

    // Write packet to queue
    hsa_kernel_dispatch_packet_t* queue_packet =
        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(resources.queue->base_address))[index & queue_mask];

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
    hsa_signal_store_screlease(resources.queue->doorbell_signal, index);

    // Wait for completion (or fault)
    hsa_signal_wait_scacquire(resources.signal, HSA_SIGNAL_CONDITION_LT, 1,
                              UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Should not reach here if fault occurs - destructor will cleanup
    _exit(0);
  }

  // Parent process - wait for child with timeout
  int status;
  int timeout_ms = 10000;  // 10 second
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

  // Unset HSA_COREDUMP_PATTERN to use default
  unsetenv("HSA_COREDUMP_PATTERN");
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

  // Convert pattern to glob pattern (handles %t and %i with wildcards)
  std::string glob_pattern = PatternToGlob(expected, child_pid);
  std::string actual_file = FindMatchingCoreDump(glob_pattern);

  if (!actual_file.empty()) {
    bool success = VerifyCoreDumpFile(actual_file);
    EXPECT_TRUE(success);
    if (success) {
      unlink(actual_file.c_str());
    }
  } else {
    FAIL() << "No core dump found matching pattern: " << glob_pattern;
  }
}

void GpuCoreDumpTest::TestCustomPattern(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing custom pattern (HSA_COREDUMP_PATTERN)..." << std::endl;
  }

  std::string pattern = test_dir_ + "/custom_gpu_core.%p";
  setenv("HSA_COREDUMP_PATTERN", pattern.c_str(), 1);
  setenv("HSA_COREDUMP_SHOW_PROGRESS" , "1", 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  // Use glob to find file
  std::string glob_pattern = PatternToGlob(pattern, child_pid);
  std::string actual_file = FindMatchingCoreDump(glob_pattern);

  if (!actual_file.empty()) {
    bool success = VerifyCoreDumpFile(actual_file);
    EXPECT_TRUE(success);
    if (success) {
      unlink(actual_file.c_str());
    }
  } else {
    FAIL() << "No core dump found matching pattern: " << glob_pattern;
  }

  unsetenv("HSA_COREDUMP_PATTERN");
}

void GpuCoreDumpTest::TestDisableFlag(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing disable flag (HSA_DISABLE_COREDUMP_ON_EXCEPTION=1)..." << std::endl;
  }

  std::string pattern = test_dir_ + "/should_not_exist.%p";
  setenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION", "1", 1);
  setenv("HSA_COREDUMP_PATTERN", pattern.c_str(), 1);

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  // Use glob to check if any file was created
  std::string glob_pattern = PatternToGlob(pattern, child_pid);
  std::string actual_file = FindMatchingCoreDump(glob_pattern);

  EXPECT_TRUE(actual_file.empty()) << "Core dump should not have been created when disabled";

  if (verbosity() > 0) {
    if (actual_file.empty()) {
      std::cout << "    Correctly prevented core dump creation" << std::endl;
    }
  }

  if (!actual_file.empty()) {
    unlink(actual_file.c_str());
  }

  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");
  unsetenv("HSA_COREDUMP_PATTERN");
}

void GpuCoreDumpTest::TestPatternSubstitution(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing pattern substitution (%p, %e, %t)..." << std::endl;
  }

  // Test pattern with multiple specifiers including timestamp
  std::string pattern = test_dir_ + "/core.%p.%e.%t.dump";
  setenv("HSA_COREDUMP_PATTERN", pattern.c_str(), 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  // Use glob to find file (handles %t wildcard)
  std::string glob_pattern = PatternToGlob(pattern, child_pid);
  std::string actual_file = FindMatchingCoreDump(glob_pattern);

  if (!actual_file.empty()) {
    bool success = VerifyCoreDumpFile(actual_file);
    EXPECT_TRUE(success);
    if (success) {
      unlink(actual_file.c_str());
    }
  } else {
    FAIL() << "No core dump found matching pattern: " << glob_pattern;
  }

  unsetenv("HSA_COREDUMP_PATTERN");
}

void GpuCoreDumpTest::TestInvalidPath(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing invalid path handling..." << std::endl;
  }

  std::string pattern = "/nonexistent_dir_12345/core.%p";
  setenv("HSA_COREDUMP_PATTERN", pattern.c_str(), 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  // Use glob to check if any file was created
  std::string glob_pattern = PatternToGlob(pattern, child_pid);
  std::string actual_file = FindMatchingCoreDump(glob_pattern);

  EXPECT_TRUE(actual_file.empty()) << "Core dump should not be created with invalid path";

  if (verbosity() > 0) {
    if (actual_file.empty()) {
      std::cout << "    Correctly handled invalid path" << std::endl;
    }
  }

  unsetenv("HSA_COREDUMP_PATTERN");
}
