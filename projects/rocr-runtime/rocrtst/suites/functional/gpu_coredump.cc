/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * 
 * SPDX-License-Identifier: NCSA
 */

#include <fcntl.h>
#include <sys/resource.h>
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

#define VECTOR_SIZE 256

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    const char* msg = 0; \
    hsa_status_string(err, &msg); \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
                          __FILE__ << ". Call returned " << err << std::endl; \
    std::cout << msg << std::endl; \
    return (err); \
  } \
}

namespace {
  void set_core_limit() {
    // Set ulimit -c to 100MB (sufficient for GPU core dumps)
    struct rlimit rlim;
    rlim.rlim_cur = 100 * 1024 * 1024;  // 100MB
    rlim.rlim_max = 100 * 1024 * 1024;
    setrlimit(RLIMIT_CORE, &rlim);
  }
}

GpuCoreDumpTest::GpuCoreDumpTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("GPU Core Dump Configuration Tests");
  set_description("Tests for configurable GPU core dump functionality including "
                  "custom patterns, format specifiers, and disable flag.");
  set_kernel_file_name("vector_add_memory_fault_kernels.hsaco");
  set_kernel_name("vector_add_memory_fault");
  
  // Save original ulimit
  getrlimit(RLIMIT_CORE, &original_rlimit_);
}

GpuCoreDumpTest::~GpuCoreDumpTest(void) {
  // Restore original ulimit
  setrlimit(RLIMIT_CORE, &original_rlimit_);
}

void GpuCoreDumpTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_agent_t* gpu_dev = gpu_device1();

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Create a queue
  hsa_queue_t* q = nullptr;
  rocrtst::CreateQueue(*gpu_dev, &q);
  ASSERT_NE(q, nullptr);
  set_main_queue(q);

  err = rocrtst::LoadKernelFromObjFile(this, gpu_dev);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  // Allocate buffers for the kernel
  err = hsa_amd_memory_pool_allocate(cpu_pool(),
                                   VECTOR_SIZE*sizeof(int),
                                   0, reinterpret_cast<void**>(&a_buffer_));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(2, ag_list, nullptr, a_buffer_);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_allocate(cpu_pool(),
                                   VECTOR_SIZE*sizeof(int),
                                   0, reinterpret_cast<void**>(&b_buffer_));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(2, ag_list, nullptr, b_buffer_);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_allocate(cpu_pool(),
                                   VECTOR_SIZE*sizeof(int),
                                   0, reinterpret_cast<void**>(&c_buffer_));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(2, ag_list, nullptr, c_buffer_);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_allocate(cpu_pool(),
                                   VECTOR_SIZE*sizeof(int),
                                   0, reinterpret_cast<void**>(&d_buffer_));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(2, ag_list, nullptr, d_buffer_);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_allocate(cpu_pool(),
                                   VECTOR_SIZE*sizeof(int),
                                   0, reinterpret_cast<void**>(&e_buffer_));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(2, ag_list, nullptr, e_buffer_);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Initialize buffers
  for (int i = 0; i < VECTOR_SIZE; i++) {
    reinterpret_cast<int*>(a_buffer_)[i] = i;
    reinterpret_cast<int*>(b_buffer_)[i] = i * 2;
    reinterpret_cast<int*>(c_buffer_)[i] = 0;
    reinterpret_cast<int*>(d_buffer_)[i] = 0;
    reinterpret_cast<int*>(e_buffer_)[i] = 0;
  }

  // Set up kernel arguments
  struct __attribute__((aligned(16))) kernel_args_t {
    const int *a;
    const int *b;
    const int *c;
    int *d;
    int *e;
  } kernel_args;

  kernel_args.a = reinterpret_cast<int*>(a_buffer_);
  kernel_args.b = reinterpret_cast<int*>(b_buffer_);
  kernel_args.c = reinterpret_cast<int*>(c_buffer_);
  kernel_args.d = reinterpret_cast<int*>(d_buffer_);
  kernel_args.e = reinterpret_cast<int*>(e_buffer_);

  err = rocrtst::AllocAndSetKernArgs(this, &kernel_args, sizeof(kernel_args));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Create temporary directory for test dumps
  test_dir_ = "/tmp/rocr_coredump_test_" + std::to_string(getpid());
  mkdir(test_dir_.c_str(), 0755);
}

void GpuCoreDumpTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void GpuCoreDumpTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void GpuCoreDumpTest::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  return;
}

void GpuCoreDumpTest::Close() {
  hsa_status_t err;

  if (a_buffer_) {
    err = hsa_amd_memory_pool_free(a_buffer_);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }
  if (b_buffer_) {
    err = hsa_amd_memory_pool_free(b_buffer_);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }
  if (c_buffer_) {
    err = hsa_amd_memory_pool_free(c_buffer_);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }
  if (d_buffer_) {
    err = hsa_amd_memory_pool_free(d_buffer_);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }
  if (e_buffer_) {
    err = hsa_amd_memory_pool_free(e_buffer_);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }

  // Clean up test directory and its contents
  if (!test_dir_.empty()) {
    CleanupCoreDumps(test_dir_ + "/*");
    rmdir(test_dir_.c_str());
  }
  
  TestBase::Close();
}

bool GpuCoreDumpTest::VerifyCoreDumpFile(const std::string& filename) {
  // Check if file exists
  if (access(filename.c_str(), F_OK) != 0) {
    if (verbosity() > 0) {
      std::cout << "    Core dump file not found: " << filename << std::endl;
    }
    return false;
  }
  
  // Check if file is readable
  if (access(filename.c_str(), R_OK) != 0) {
    if (verbosity() > 0) {
      std::cout << "    Core dump file not readable: " << filename << std::endl;
    }
    return false;
  }
  
  // Check if it's a valid GPU core dump
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
  
  // Verify it's a 64-bit ELF
  if (ehdr.e_ident[EI_CLASS] != ELFCLASS64) {
    return false;
  }
  
  // Verify it's a core dump
  if (ehdr.e_type != ET_CORE) {
    return false;
  }
  
  // Verify it's an AMDGPU core dump (EM_AMDGPU = 224)
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

void GpuCoreDumpTest::DispatchFaultingKernel() {
  // Override grid/workgroup sizes for our kernel
  aql().workgroup_size_x = 64;
  aql().workgroup_size_y = 1;
  aql().workgroup_size_z = 1;
  aql().grid_size_x = VECTOR_SIZE;
  aql().grid_size_y = 1;
  aql().grid_size_z = 1;

  uint64_t index;
  hsa_kernel_dispatch_packet_t *queue_aql_packet = rocrtst::WriteAQLToQueue(this, &index);
  
  uint32_t aql_header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  
  rocrtst::AtomicSetPacketHeader(aql_header, aql().setup, queue_aql_packet);
  hsa_signal_store_screlease(main_queue()->doorbell_signal, index);
  
  // Wait for completion (or fault) - ROCr will catch the GPU fault
  hsa_signal_wait_scacquire(aql().completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                            UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  
  // Reset signal for next dispatch
  hsa_signal_store_screlease(aql().completion_signal, 1);
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
  
  if (kernel_pattern.empty() || kernel_pattern[0] == '|') {
    if (verbosity() > 0) {
      std::cout << "    Skipping: kernel pattern is empty or pipe pattern" << std::endl;
    }
    return;
  }
  
  // Unset HSA_COREDUMP_FILE to use default
  unsetenv("HSA_COREDUMP_FILE");
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");
  
  set_core_limit(); 

  // Dispatch the faulting kernel
  DispatchFaultingKernel();
  
  // Expected pattern is kernel pattern + .gpu
  std::string expected = kernel_pattern + ".gpu";
  size_t pos = expected.find("%p");
  if (pos != std::string::npos) {
    expected.replace(pos, 2, std::to_string(getpid()));
  }
  
  bool success = VerifyCoreDumpFile(expected);
  EXPECT_TRUE(success);
  
  // Cleanup
  if (success) {
    unlink(expected.c_str());
  }
}

void GpuCoreDumpTest::TestCustomPattern(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing custom pattern (HSA_COREDUMP_FILE)..." << std::endl;
  }
  
  std::string custom_pattern = test_dir_ + "/custom_gpu_core." + std::to_string(getpid());
  setenv("HSA_COREDUMP_FILE", custom_pattern.c_str(), 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  set_core_limit(); 

  // Dispatch the faulting kernel
  DispatchFaultingKernel();
  
  bool success = VerifyCoreDumpFile(custom_pattern);
  EXPECT_TRUE(success);
  
  // Cleanup
  if (success) {
    unlink(custom_pattern.c_str());
  }
  
  unsetenv("HSA_COREDUMP_FILE");
}

void GpuCoreDumpTest::TestDisableFlag(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing disable flag (HSA_DISABLE_COREDUMP_ON_EXCEPTION=1)..." << std::endl;
  }
  
  std::string test_file = test_dir_ + "/should_not_exist." + std::to_string(getpid());
  setenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION", "1", 1);
  setenv("HSA_COREDUMP_FILE", test_file.c_str(), 1);

  set_core_limit(); 

  // Dispatch the faulting kernel
  DispatchFaultingKernel();
  
  // Verify NO core dump was created
  bool file_exists = (access(test_file.c_str(), F_OK) == 0);
  EXPECT_FALSE(file_exists) << "Core dump should not have been created when disabled";
  
  if (verbosity() > 0) {
    if (!file_exists) {
      std::cout << "    Correctly prevented core dump creation" << std::endl;
    }
  }
  
  // Cleanup just in case
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
  
  // Test pattern with PID specifier
  std::string pattern = test_dir_ + "/core.%p.dump";
  setenv("HSA_COREDUMP_FILE", pattern.c_str(), 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  set_core_limit(); 

  // Dispatch the faulting kernel
  DispatchFaultingKernel();
  
  // Verify with substituted PID
  std::string expected = test_dir_ + "/core." + std::to_string(getpid()) + ".dump";
  bool success = VerifyCoreDumpFile(expected);
  EXPECT_TRUE(success);
  
  // Cleanup
  if (success) {
    unlink(expected.c_str());
  }
  
  unsetenv("HSA_COREDUMP_FILE");
}

void GpuCoreDumpTest::TestInvalidPath(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing invalid path handling..." << std::endl;
  }
  
  // Use a path that doesn't exist
  std::string invalid_pattern = "/nonexistent_dir_12345/core." + std::to_string(getpid());
  setenv("HSA_COREDUMP_FILE", invalid_pattern.c_str(), 1);
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");

  set_core_limit(); 

  // Dispatch the faulting kernel
  DispatchFaultingKernel();
  
  // Verify NO core dump was created (path is invalid)
  bool file_exists = (access(invalid_pattern.c_str(), F_OK) == 0);
  EXPECT_FALSE(file_exists) << "Core dump should not be created with invalid path";
  
  if (verbosity() > 0) {
    if (!file_exists) {
      std::cout << "    Correctly handled invalid path" << std::endl;
    }
  }
  
  unsetenv("HSA_COREDUMP_FILE");
}

#undef RET_IF_HSA_ERR
