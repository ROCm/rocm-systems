/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <limits.h>
#include <elf.h>
#include <glob.h>
#include <sys/stat.h>
#include <errno.h>
#include <libgen.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <system_error>

#include "suites/functional/gpu_coredump.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

// NT_AMDGPU_CORE_STATE from runtime/hsa-runtime/inc/amd_hsa_elf.h
// AMDGPU snapshots of runtime, agent and queues state for use in core dump
#define NT_AMDGPU_CORE_STATE 33

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
  prerequisites_met_ = false;
}

GpuCoreDumpTest::~GpuCoreDumpTest(void) {
  // Restore original ulimit
  setrlimit(RLIMIT_CORE, &original_rlimit_);
}

bool GpuCoreDumpTest::CheckPrerequisites() {
  // Check if /tmp is writable
  if (access("/tmp", W_OK) != 0) {
    std::cout << "SKIP: /tmp is not writable, cannot run core dump tests" << std::endl;
    return false;
  }

  // Try to set ulimit to 200MB
  struct rlimit rlim;
  rlim.rlim_cur = 200 * 1024 * 1024;
  rlim.rlim_max = 200 * 1024 * 1024;
  if (setrlimit(RLIMIT_CORE, &rlim) != 0) {
    std::cout << "SKIP: Cannot set ulimit for core dumps: " << strerror(errno) << std::endl;
    return false;
  }

  // Verify ulimit was actually set
  struct rlimit check_rlim;
  if (getrlimit(RLIMIT_CORE, &check_rlim) == 0) {
    if (check_rlim.rlim_cur < 200 * 1024 * 1024) {
      std::cout << "SKIP: ulimit could not be set to required value (got "
                << check_rlim.rlim_cur << " bytes)" << std::endl;
      return false;
    }
  }

  return true;
}

void GpuCoreDumpTest::SetUp(void) {
  // Don't call TestBase::SetUp() - we don't want hsa_init() in parent

  // Check prerequisites first
  prerequisites_met_ = CheckPrerequisites();
  if (!prerequisites_met_) {
    return;  // Skip setup if prerequisites not met
  }

  // Create temporary directory for test dumps
  test_dir_ = "/tmp/rocr_coredump_test_" + std::to_string(getpid());

  int mkdir_result = mkdir(test_dir_.c_str(), 0755);
  if (mkdir_result != 0 && errno != EEXIST) {
    std::cerr << "Failed to create test directory " << test_dir_
              << ": " << strerror(errno) << std::endl;
    prerequisites_met_ = false;
    return;
  }

  if (verbosity() > 0) {
    std::cout << "Created test directory: " << test_dir_ << std::endl;
  }
}

void GpuCoreDumpTest::Run(void) {
  // Nothing to do here - each test method handles its own execution
}

void GpuCoreDumpTest::DisplayTestInfo(void) {
  std::cout << "Test: " << title() << '\n';
  std::cout << description() << '\n';
}

void GpuCoreDumpTest::DisplayResults(void) const {
  // Nothing to display
}

void GpuCoreDumpTest::Close() {
  // Don't call TestBase::Close() - we never called hsa_init() in parent

  // Clean up test directory (unless debugging)
  const char* preserve = getenv("ROCR_TEST_PRESERVE_COREDUMP");
  if (!test_dir_.empty() && !preserve) {
    CleanupCoreDumps(test_dir_ + "/*");
    rmdir(test_dir_.c_str());
  } else if (preserve) {
    std::cout << "  DEBUG: Test directory preserved: " << test_dir_ << '\n';
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
      fprintf(stderr, "CHILD: hsa_init failed with error %d\n", err);
      _exit(1);
    }
    fprintf(stderr, "CHILD: HSA initialized successfully\n");

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
      fprintf(stderr, "CHILD: Failed to find CPU agent\n");
      _exit(1);
    }

    err = hsa_iterate_agents(rocrtst::FindGPUDevice, &gpu_agent);
    if (err == HSA_STATUS_INFO_BREAK) {
      err = HSA_STATUS_SUCCESS;
    } else if (err == HSA_STATUS_SUCCESS) {
      err = HSA_STATUS_ERROR;
    }
    if (err != HSA_STATUS_SUCCESS || gpu_agent.handle == 0) {
      fprintf(stderr, "CHILD: Failed to find GPU agent\n");
      _exit(1);
    }

    // Get profile to determine which pool finder to use
    hsa_profile_t profile;
    err = hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_PROFILE, &profile);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to get GPU agent profile\n");
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
      fprintf(stderr, "CHILD: Failed to find kernarg memory pool\n");
      _exit(1);
    }

    // Create queue
    uint32_t queue_size = 0;
    err = hsa_agent_get_info(gpu_agent,
                                  HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to get GPU agent max queue size\n");
      _exit(1);
    }

    err = hsa_queue_create(gpu_agent, queue_size, HSA_QUEUE_TYPE_MULTI,
                           nullptr, nullptr, 0, 0, &resources.queue);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to create GPU queue\n");
      _exit(1);
    }

    // Load kernel from file (no need to allocate src/dst buffers
    //  - we pass nullptr)
    std::string kernel_path = rocrtst::LocateKernelFile(
        "test_case_template_kernels.hsaco", gpu_agent);
    if (kernel_path.empty()) {
      fprintf(stderr, "CHILD: Failed to locate kernel file\n");
      _exit(1);
    }
    resources.file_fd = open(kernel_path.c_str(), O_RDONLY);
    if (resources.file_fd == -1) {
      fprintf(stderr, "CHILD: Failed to open kernel file %s: %s\n",
              kernel_path.c_str(), strerror(errno));
      _exit(1);
    }
    fprintf(stderr, "CHILD: Kernel file opened: %s\n",
            kernel_path.c_str());

    err = hsa_code_object_reader_create_from_file(resources.file_fd,
                                                    &resources.code_obj_rdr);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to create code object reader\n");
      _exit(1);
    }

    err = hsa_executable_create_alt(HSA_PROFILE_FULL,
                                    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                              nullptr, &resources.executable);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to create executable\n");
      _exit(1);
    }

    err = hsa_executable_load_agent_code_object(resources.executable,
                        gpu_agent, resources.code_obj_rdr, nullptr, nullptr);
    // Can destroy reader now
    hsa_code_object_reader_destroy(resources.code_obj_rdr);
    resources.code_obj_rdr = {0};
    close(resources.file_fd);
    resources.file_fd = -1;

    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to load code object into executable\n");
      _exit(1);
    }

    err = hsa_executable_freeze(resources.executable, nullptr);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to freeze executable\n");
      _exit(1);
    }
    // Get kernel symbol
    hsa_executable_symbol_t symbol;
    err = hsa_executable_get_symbol_by_name(resources.executable,
                                            "square.kd", &gpu_agent, &symbol);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to get kernel symbol\n");
      _exit(1);
    }

    uint64_t kernel_object = 0;
    err = hsa_executable_symbol_get_info(symbol,
                                  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                                             &kernel_object);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to get kernel object\n");
      _exit(1);
    }

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

    err = hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(kernel_args),
                                                0, &resources.kernarg_buffer);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to allocate kernarg buffer\n");
      _exit(1);
    }

    memcpy(resources.kernarg_buffer, &kernel_args, sizeof(kernel_args));

    // Create completion signal
    err = hsa_signal_create(1, 0, nullptr, &resources.signal);
    if (err != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "CHILD: Failed to create completion signal\n");
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
    aql.kernarg_address = resources.kernarg_buffer;
    aql.completion_signal = resources.signal;

    const uint32_t queue_mask = resources.queue->size - 1;
    uint64_t index = hsa_queue_load_write_index_relaxed(resources.queue);
    hsa_queue_store_write_index_relaxed(resources.queue, index + 1);

    // Write packet to queue
    hsa_kernel_dispatch_packet_t* queue_packet =
      &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
                          resources.queue->base_address))[index & queue_mask];

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
    aql_header |=
              HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql_header |=
              HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    __atomic_store_n(reinterpret_cast<uint32_t*>(queue_packet),
                     aql_header | (aql.setup << 16), __ATOMIC_RELEASE);

    // Ring doorbell
    hsa_signal_store_screlease(resources.queue->doorbell_signal, index);
    fprintf(stderr, "CHILD: Kernel dispatched, waiting for completion...\n");

    // Wait for completion (or fault)
    hsa_signal_wait_scacquire(resources.signal, HSA_SIGNAL_CONDITION_LT, 1,
                              UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Should not reach here if fault occurs - destructor will cleanup
    fprintf(stderr, "CHILD: Kernel completed without fault! This should not happen with nullptr access.\n");
    _exit(0);
  }

  // Parent process - wait for child with timeout
  int status;
  int timeout_ms = 10000;  // 10 seconds
  int elapsed = 0;

  while (elapsed < timeout_ms) {
    pid_t result = waitpid(pid, &status, WNOHANG);
    if (result == pid) {
      // Child finished - check if it exited normally or faulted
      if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        // Child exited normally without fault
        return -2;
      }
      // Child crashed/faulted (expected)
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
    std::cout << "    Child process timeout, killing...\n";
  }
  kill(pid, SIGKILL);
  waitpid(pid, &status, 0);
  usleep(100000);  // Give time for core dump
  return pid;
}

bool GpuCoreDumpTest::VerifyCoreDumpFile(const std::string& filename) {
  std::error_code ec;
  if (!std::filesystem::exists(filename, ec)) {
    if (verbosity() > 0) {
      if (ec) {
        std::cout << "    Error checking file " << filename << ": " << ec.message() << '\n';
      } else {
        std::cout << "    Core dump file not found: " << filename << '\n';
      }
    }
    return false;
  }

  if (access(filename.c_str(), R_OK) != 0) {
    if (verbosity() > 0) {
      std::cout << "    Core dump file not readable: " << filename << '\n';
    }
    return false;
  }

  if (!IsValidGPUCoreDump(filename)) {
    if (verbosity() > 0) {
      std::cout << "    File is not a valid GPU core dump: " << filename << '\n';
    }
    return false;
  }

  if (verbosity() > 0) {
    struct stat st;
    stat(filename.c_str(), &st);
    std::cout << "    Core dump verified: " << filename
              << " (size: " << st.st_size << " bytes)\n";
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
  if (!prerequisites_met_) {
    std::cout << "SKIPPED: Prerequisites not met for GPU core dump tests" << std::endl;
    return;
  }

  if (verbosity() > 0) {
    std::cout <<
          "  Testing default pattern (kernel core_pattern + .gpu suffix)...\n";
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
  setenv("HSA_COREDUMP_SHOW_PROGRESS", "1", 1);

  std::string expected;

  if (kernel_pattern.empty()) {
    expected = "gpucore.%p.gpu";
  } else if (kernel_pattern[0] == '|') {
    if (verbosity() > 0) {
      std::cout <<
            "    Kernel uses pipe pattern - testing graceful fault handling\n";
    }

    pid_t child_pid = RunFaultingKernelInChild();
    if (child_pid < 0) {
      FAIL() << "Failed to run test in child process";
      return;
    }

    if (verbosity() > 0) {
      std::cout <<
                  "    GPU fault handled successfully (pipe pattern in use)\n";
    }
    return;
  } else {
    expected = kernel_pattern;
    // Replace /proc prefix with /tmp for testing (proc paths become invalid after child exits)
    // Only replace if pattern STARTS with /proc (position 0)
    if (expected.find("/proc") == 0) {
      // Pattern starts with /proc
      size_t cwd_pos = expected.find("/cwd");
      if (cwd_pos != std::string::npos) {
        // Replace /proc/%P/cwd (or /proc/%p/cwd) with /tmp
        expected.replace(0, cwd_pos + 4, "/tmp");
      } else {
        // Just replace /proc with /tmp
        expected.replace(0, 5, "/tmp");
      }
      // Override kernel pattern for this test
      setenv("HSA_COREDUMP_PATTERN", expected.c_str(), 1);
      if (verbosity() > 0) {
        std::cout << "    Kernel pattern starts with /proc, overriding with /tmp for test\n";
      }
      // Don't add .gpu suffix - runtime won't add it for custom patterns
    } else {
      // For non-/proc patterns, runtime adds .gpu suffix
      expected += ".gpu";
    }
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

  // Cleanup - unset if we overrode it for /proc patterns
  unsetenv("HSA_COREDUMP_PATTERN");
}

void GpuCoreDumpTest::TestCustomPattern(void) {
  if (!prerequisites_met_) {
    std::cout << "SKIPPED: Prerequisites not met for GPU core dump tests" << std::endl;
    return;
  }

  if (verbosity() > 0) {
    std::cout << "  Testing custom pattern (HSA_COREDUMP_PATTERN)...\n";
  }

  std::string pattern = test_dir_ + "/custom_gpu_core.%p";
  setenv("HSA_COREDUMP_PATTERN", pattern.c_str(), 1);
  setenv("HSA_COREDUMP_SHOW_PROGRESS", "1", 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid == -2) {
    std::cout << "NOTE: Child completed without GPU fault - "
              << "environment may not support fault triggering"
              << std::endl;
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }
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

bool GpuCoreDumpTest::ValidateNoteSegment(int fd, uint64_t offset,
                                                              uint64_t size) {
  if (verbosity() > 0) {
    std::cout << "    Validating PT_NOTE segment at offset " << offset << "\n";
  }

  // Read note header
  if (lseek(fd, offset, SEEK_SET) == -1) {
    if (verbosity() > 0) {
      std::cout << "    Failed to seek to note offset\n";
    }
    return false;
  }

  uint32_t namesz, descsz, type;
  if (read(fd, &namesz, 4) != 4 ||
      read(fd, &descsz, 4) != 4 ||
      read(fd, &type, 4) != 4) {
    if (verbosity() > 0) {
      std::cout << "    Failed to read note header\n";
    }
    return false;
  }

  // Validate note header
  if (namesz != 7) {  // "AMDGPU\0"
    if (verbosity() > 0) {
      std::cout << "    Invalid namesz: " << namesz << " (expected 7)\n";
    }
    return false;
  }

  if (type != NT_AMDGPU_CORE_STATE) {
    if (verbosity() > 0) {
      std::cout << "    Invalid note type: " << type
                << " (expected " << NT_AMDGPU_CORE_STATE << ")\n";
    }
    return false;
  }

  // Read and validate name
  char name[8];
  if (read(fd, name, 8) != 8) {
    if (verbosity() > 0) {
      std::cout << "    Failed to read note name\n";
    }
    return false;
  }

  if (strncmp(name, "AMDGPU", 6) != 0) {
    if (verbosity() > 0) {
      std::cout << "    Invalid note name: " << std::string(name, 6) << '\n';
    }
    return false;
  }

  // Read and validate descriptor structure
  uint64_t note_version;
  if (read(fd, &note_version, 8) != 8) {
    if (verbosity() > 0) {
      std::cout << "    Failed to read note version\n";
    }
    return false;
  }

  // Note version - currently hardcoded to 1 in amd_core_dump.cpp
  // This version tracks the PT_NOTE descriptor format and would only change
  // if the snapshot data structure (runtime/agent/queue info) is modified
  if (note_version != 1) {
    if (verbosity() > 0) {
      std::cout << "    Invalid note version: " <<
                                            note_version << " (expected 1)\n";
    }
    return false;
  }

  uint32_t version_major, version_minor;
  if (read(fd, &version_major, 4) != 4 || read(fd, &version_minor, 4) != 4) {
    if (verbosity() > 0) {
      std::cout << "    Failed to read KMT version\n";
    }
    return false;
  }

  uint64_t runtime_info_size;
  if (read(fd, &runtime_info_size, 8) != 8) {
    if (verbosity() > 0) {
      std::cout << "    Failed to read runtime_info_size\n";
    }
    return false;
  }

  if (runtime_info_size == 0) {
    if (verbosity() > 0) {
      std::cout << "    Invalid runtime_info_size: 0\n";
    }
    return false;
  }

  uint32_t n_agents, agent_entry_size;
  if (read(fd, &n_agents, 4) != 4 || read(fd, &agent_entry_size, 4) != 4) {
    if (verbosity() > 0) {
      std::cout << "    Failed to read agent info\n";
    }
    return false;
  }

  if (n_agents == 0) {
    if (verbosity() > 0) {
      std::cout << "    Invalid n_agents: 0 (expected at least 1 GPU)\n";
    }
    return false;
  }

  if (agent_entry_size == 0) {
    if (verbosity() > 0) {
      std::cout << "    Invalid agent_entry_size: 0\n";
    }
    return false;
  }

  if (verbosity() > 0) {
    std::cout << "    PT_NOTE validation passed:\n";
    std::cout << "      Note version: " << note_version << '\n';
    std::cout << "      KMT version: " << version_major << "." <<
                                                        version_minor << '\n';
    std::cout << "      Runtime info size: " << runtime_info_size << '\n';
    std::cout << "      Number of agents: " << n_agents << '\n';
    std::cout << "      Agent entry size: " << agent_entry_size << '\n';
  }

  return true;
}

bool GpuCoreDumpTest::ValidateLoadSegment(int fd, uint64_t file_offset,
                                          uint64_t vaddr, uint64_t size,
                                                            pid_t child_pid) {
  if (verbosity() > 0) {
    std::cout << "    Validating PT_LOAD segment:\n";
    std::cout << "      File offset: 0x" << std::hex << file_offset <<
                                                            std::dec << '\n';
    std::cout << "      Virtual addr: 0x" << std::hex << vaddr <<
                                                            std::dec << '\n';
    std::cout << "      Size: " << size << " bytes\n";
  }

  // Sample size - check first 4KB or segment size, whichever is smaller
  size_t sample_size = std::min(size, (uint64_t)4096);
  std::vector<uint8_t> data(sample_size);

  // Read from core dump file at specified offset
  if (lseek(fd, file_offset, SEEK_SET) == -1) {
    if (verbosity() > 0) {
      std::cout << "    Failed to seek to file offset\n";
    }
    return false;
  }

  ssize_t bytes_read = read(fd, data.data(), sample_size);
  if (bytes_read != (ssize_t)sample_size) {
    if (verbosity() > 0) {
      std::cout << "    Failed to read from core file (read " <<
                                                    bytes_read << " bytes)\n";
    }
    return false;
  }

  // Validate that data was actually written (not all zeros
  // from posix_fallocate)
  bool all_zeros = std::all_of(data.begin(), data.end(),
                                [](uint8_t b) { return b == 0; });
  if (all_zeros) {
    if (verbosity() > 0) {
      std::cout << "    FAIL: Segment data is all zeros (not written)\n";
    }
    return false;
  }

  // Check for variation in the data
  bool has_variation = false;
  for (size_t i = 1; i < data.size(); i++) {
    if (data[i] != data[0]) {
      has_variation = true;
      break;
    }
  }

  // All 0xFF is a valid pattern for uninitialized GPU memory
  bool all_ff = std::all_of(data.begin(), data.end(), 
                            [](uint8_t b) { return b == 0xFF; });

  // Pass if: has variation OR all 0xFF (valid GPU memory patterns)
  // Fail if: no variation AND not all 0xFF (possible error in writing
  // core dump data)
  if (!has_variation && !all_ff) {
    if (verbosity() > 0) {
      std::cout <<
          "    FAIL: Segment data has suspicious repeating pattern: 0x" <<
                        std::hex << (int)data[0] << std::dec << " repeated\n";
    }
    return false;
  }

  if (verbosity() > 0) {
    if (all_ff) {
      std::cout << "    PT_LOAD validation passed: Segment contains 0xFF "
                                "pattern (valid uninitialized GPU memory)\n";
    } else {
      std::cout << "    PT_LOAD validation passed: Segment contains non-zero"
                                                      " data with variation\n";
    }
  }

  return true;
}

void GpuCoreDumpTest::TestCoreDumpContentIntegrity(void) {
  if (!prerequisites_met_) {
    std::cout << "SKIPPED: Prerequisites not met for GPU core dump tests" << std::endl;
    return;
  }

  if (verbosity() > 0) {
    std::cout <<
        "  Testing core dump content integrity (headers and VRAM data)...\n";
  }

  std::string pattern = test_dir_ + "/integrity_test.%p";
  setenv("HSA_COREDUMP_PATTERN", pattern.c_str(), 1);
  setenv("HSA_COREDUMP_SHOW_PROGRESS", "1", 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid == -2) {
    std::cout << "NOTE: Child completed without GPU fault - "
              << "environment may not support fault triggering"
              << std::endl;
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  // Find the core dump file
  std::string glob_pattern = PatternToGlob(pattern, child_pid);
  std::string core_file = FindMatchingCoreDump(glob_pattern);

  if (core_file.empty()) {
    FAIL() << "No core dump found matching pattern: " << glob_pattern;
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  if (verbosity() > 0) {
    std::cout << "    Found core dump: " << core_file << '\n';
  }

  // Open and parse the core dump
  int fd = open(core_file.c_str(), O_RDONLY);
  if (fd == -1) {
    FAIL() << "Failed to open core dump file: " << core_file;
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  // Read ELF header
  Elf64_Ehdr ehdr;
  if (read(fd, &ehdr, sizeof(ehdr)) != sizeof(ehdr)) {
    close(fd);
    FAIL() << "Failed to read ELF header";
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  // Verify it's a valid ELF core dump
  EXPECT_EQ(ehdr.e_ident[EI_MAG0], ELFMAG0);
  EXPECT_EQ(ehdr.e_ident[EI_MAG1], ELFMAG1);
  EXPECT_EQ(ehdr.e_ident[EI_MAG2], ELFMAG2);
  EXPECT_EQ(ehdr.e_ident[EI_MAG3], ELFMAG3);
  EXPECT_EQ(ehdr.e_type, ET_CORE);

  // Read program headers
  std::vector<Elf64_Phdr> phdrs(ehdr.e_phnum);
  if (pread(fd, phdrs.data(), ehdr.e_phnum * sizeof(Elf64_Phdr), ehdr.e_phoff)
                             != (ssize_t)(ehdr.e_phnum * sizeof(Elf64_Phdr))) {
    close(fd);
    FAIL() << "Failed to read program headers";
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  if (verbosity() > 0) {
    std::cout << "    Found " << ehdr.e_phnum << " program headers\n";
  }

  bool found_note = false;
  bool found_load = false;
  int note_failures = 0;
  int segments_with_data = 0;  // Count segments with actual non-zero data

  // Validate each segment
  for (const auto& phdr : phdrs) {
    if (phdr.p_type == PT_NOTE) {
      found_note = true;
      if (!ValidateNoteSegment(fd, phdr.p_offset, phdr.p_filesz)) {
        note_failures++;
      }
    } else if (phdr.p_type == PT_LOAD) {
      found_load = true;
      // ValidateLoadSegment returns true if segment has non-zero data
      // Some segments can legitimately be all zeros
      // (unmapped/unused GPU memory)
      if (ValidateLoadSegment(fd, phdr.p_offset, phdr.p_vaddr,
                                                  phdr.p_filesz, child_pid)) {
        segments_with_data++;
      }
    }
  }

  close(fd);

  // Report results
  EXPECT_TRUE(found_note) << "No PT_NOTE segment found in core dump";
  EXPECT_TRUE(found_load) << "No PT_LOAD segments found in core dump";
  EXPECT_EQ(note_failures, 0) << "PT_NOTE segment validation failed";
  EXPECT_GT(segments_with_data, 0) << "No PT_LOAD segments contain non-zero"
                              " data - core dump may not be working correctly";

  if (verbosity() > 0) {
    std::cout << "    Validation summary: " << segments_with_data <<
                                      " PT_LOAD segments with non-zero data\n";
    if (note_failures == 0 && segments_with_data > 0) {
      std::cout << "    Core dump validation passed!\n";
    }
  }

  unsetenv("HSA_COREDUMP_PATTERN");
}

void GpuCoreDumpTest::TestDisableFlag(void) {
  if (!prerequisites_met_) {
    std::cout << "SKIPPED: Prerequisites not met for GPU core dump tests" << std::endl;
    return;
  }

  if (verbosity() > 0) {
    std::cout <<
          "  Testing disable flag (HSA_DISABLE_COREDUMP_ON_EXCEPTION=1)...\n";
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

  EXPECT_TRUE(actual_file.empty()) <<
                      "Core dump should not have been created when disabled";

  if (verbosity() > 0) {
    if (actual_file.empty()) {
      std::cout << "    Correctly prevented core dump creation\n";
    }
  }

  if (!actual_file.empty()) {
    unlink(actual_file.c_str());
  }

  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");
  unsetenv("HSA_COREDUMP_PATTERN");
}

void GpuCoreDumpTest::TestPatternSubstitution(void) {
  if (!prerequisites_met_) {
    std::cout << "SKIPPED: Prerequisites not met for GPU core dump tests" << std::endl;
    return;
  }

  if (verbosity() > 0) {
    std::cout << "  Testing pattern substitution (%p, %e, %t)...\n";
  }

  // Test pattern with multiple specifiers including timestamp
  std::string pattern = test_dir_ + "/core.%p.%e.%t.dump";
  setenv("HSA_COREDUMP_PATTERN", pattern.c_str(), 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid == -2) {
    std::cout << "NOTE: Child completed without GPU fault - "
              << "environment may not support fault triggering"
              << std::endl;
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }
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
  if (!prerequisites_met_) {
    std::cout << "SKIPPED: Prerequisites not met for GPU core dump tests" << std::endl;
    return;
  }

  if (verbosity() > 0) {
    std::cout << "  Testing invalid path handling...\n";
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

  EXPECT_TRUE(actual_file.empty()) <<
                          "Core dump should not be created with invalid path";

  if (verbosity() > 0) {
    if (actual_file.empty()) {
      std::cout << "    Correctly handled invalid path\n";
    }
  }

  unsetenv("HSA_COREDUMP_PATTERN");
}

void GpuCoreDumpTest::TestPipePattern(void) {
  if (!prerequisites_met_) {
    std::cout << "SKIPPED: Prerequisites not met for GPU core dump tests" << std::endl;
    return;
  }

  if (verbosity() > 0) {
    std::cout << "  Testing pipe pattern (using tee to capture output)...\n";
  }

  std::string output_file = test_dir_ + "/pipe_output.core";
  std::string pattern = "|sh -c \"cat > " + output_file + "\"";

  setenv("HSA_COREDUMP_PATTERN", pattern.c_str(), 1);
  setenv("HSA_COREDUMP_SHOW_PROGRESS", "1", 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  pid_t child_pid = RunFaultingKernelInChild();
  if (child_pid == -2) {
    std::cout << "NOTE: Child completed without GPU fault - "
              << "environment may not support fault triggering"
              << std::endl;
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }
  if (child_pid < 0) {
    FAIL() << "Failed to run test in child process";
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  // Give cat time to finish writing
  usleep(500000);  // 500ms

  // Verify the output file exists
  std::error_code ec;
  if (!std::filesystem::exists(output_file, ec)) {
    if (ec) {
      FAIL() << "Error checking pipe output file: " << ec.message();
    } else {
      FAIL() << "Pipe output file not created: " << output_file;
    }
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  if (verbosity() > 0) {
    struct stat st;
    stat(output_file.c_str(), &st);
    std::cout << "    Pipe output file created: " << output_file
              << " (size: " << st.st_size << " bytes)\n";
  }

  // Validate the piped core dump has same structure as file-based dump
  int fd = open(output_file.c_str(), O_RDONLY);
  if (fd == -1) {
    FAIL() << "Failed to open pipe output file";
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  // Read and validate ELF header
  Elf64_Ehdr ehdr;
  if (read(fd, &ehdr, sizeof(ehdr)) != sizeof(ehdr)) {
    close(fd);
    FAIL() << "Failed to read ELF header from pipe output";
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  EXPECT_EQ(ehdr.e_ident[EI_MAG0], ELFMAG0);
  EXPECT_EQ(ehdr.e_ident[EI_MAG1], ELFMAG1);
  EXPECT_EQ(ehdr.e_ident[EI_MAG2], ELFMAG2);
  EXPECT_EQ(ehdr.e_ident[EI_MAG3], ELFMAG3);
  EXPECT_EQ(ehdr.e_type, ET_CORE);

  // Read program headers
  std::vector<Elf64_Phdr> phdrs(ehdr.e_phnum);
  if (pread(fd, phdrs.data(), ehdr.e_phnum * sizeof(Elf64_Phdr), ehdr.e_phoff)
      != (ssize_t)(ehdr.e_phnum * sizeof(Elf64_Phdr))) {
    close(fd);
    FAIL() << "Failed to read program headers from pipe output";
    unsetenv("HSA_COREDUMP_PATTERN");
    return;
  }

  // Validate we have PT_NOTE and PT_LOAD segments
  bool found_note = false;
  bool found_load = false;
  int note_failures = 0;
  int segments_with_data = 0;

  for (const auto& phdr : phdrs) {
    if (phdr.p_type == PT_NOTE) {
      found_note = true;
      if (!ValidateNoteSegment(fd, phdr.p_offset, phdr.p_filesz)) {
        note_failures++;
      }
    } else if (phdr.p_type == PT_LOAD) {
      found_load = true;
      if (ValidateLoadSegment(fd, phdr.p_offset, phdr.p_vaddr,
                                                  phdr.p_filesz, child_pid)) {
        segments_with_data++;
      }
    }
  }

  close(fd);

  // Validate results
  EXPECT_TRUE(found_note) << "No PT_NOTE segment in pipe output";
  EXPECT_TRUE(found_load) << "No PT_LOAD segments in pipe output";
  EXPECT_EQ(note_failures, 0) << "PT_NOTE validation failed in pipe output";
  EXPECT_GT(segments_with_data, 0) <<
                            "No PT_LOAD segments contain data in pipe output";

  if (verbosity() > 0) {
    std::cout << "    Pipe pattern validation: " << segments_with_data
              << " PT_LOAD segments with non-zero data\n";
    if (note_failures == 0 && segments_with_data > 0) {
      std::cout << "    Pipe pattern test passed!\n";
    }
  }

  // Cleanup
  unlink(output_file.c_str());
  unsetenv("HSA_COREDUMP_PATTERN");
}
