/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <fcntl.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <elf.h>
#include <glob.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdlib>

#include "suites/functional/gpu_coredump.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

// Simple kernel that causes a memory fault
static const char* kFaultKernel = 
  ".amdgcn_target \"amdgcn-amd-amdhsa--gfx900\"\n"
  ".text\n"
  ".globl fault_kernel\n"
  ".p2align 8\n"
  ".type fault_kernel,@function\n"
  "fault_kernel:\n"
  "  .amd_kernel_code_t\n"
  "    enable_sgpr_kernarg_segment_ptr = 1\n"
  "    is_ptr64 = 1\n"
  "    compute_pgm_rsrc1_vgprs = 0\n"
  "    compute_pgm_rsrc1_sgprs = 0\n"
  "    compute_pgm_rsrc2_user_sgpr = 2\n"
  "    kernarg_segment_byte_size = 8\n"
  "  .end_amd_kernel_code_t\n"
  "  s_load_dwordx2 s[0:1], s[0:1], 0x0\n"
  "  s_waitcnt lgkmcnt(0)\n"
  "  v_mov_b32 v0, 0xDEADBEEF\n"  // Invalid address
  "  v_mov_b32 v1, 0xDEADBEEF\n"
  "  flat_store_dword v[0:1], v0\n"  // This will cause a memory fault
  "  s_endpgm\n"
  ".Lfunc_end0:\n"
  "  .size fault_kernel, .Lfunc_end0-fault_kernel\n";

GpuCoreDumpTest::GpuCoreDumpTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("GPU Core Dump Configuration Tests");
  set_description("Tests for configurable GPU core dump functionality including "
                  "custom patterns, format specifiers, and disable flag.");
}

GpuCoreDumpTest::~GpuCoreDumpTest(void) {
}

void GpuCoreDumpTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Create temporary directory for test dumps
  test_dir_ = "/tmp/rocr_coredump_test_" + std::to_string(getpid());
  std::string mkdir_cmd = "mkdir -p " + test_dir_;
  ASSERT_EQ(0, system(mkdir_cmd.c_str()));
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
  // Clean up test directory
  if (!test_dir_.empty()) {
    std::string rm_cmd = "rm -rf " + test_dir_;
    system(rm_cmd.c_str());
  }
  
  TestBase::Close();
}

void GpuCoreDumpTest::TriggerGPUFault() {
  // This function runs in the child process and should cause a GPU fault
  // We'll trigger a memory access violation by accessing invalid GPU memory
  
  hsa_status_t err;
  
  // Initialize HSA
  err = hsa_init();
  if (err != HSA_STATUS_SUCCESS) {
    _exit(1);
  }
  
  // Get GPU agent
  hsa_agent_t gpu_agent = {0};
  auto find_gpu = [](hsa_agent_t agent, void* data) -> hsa_status_t {
    hsa_device_type_t device_type;
    hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (status == HSA_STATUS_SUCCESS && device_type == HSA_DEVICE_TYPE_GPU) {
      *((hsa_agent_t*)data) = agent;
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  };
  
  err = hsa_iterate_agents(find_gpu, &gpu_agent);
  if (gpu_agent.handle == 0) {
    hsa_shut_down();
    _exit(1);
  }
  
  // Allocate some GPU memory and intentionally access invalid memory
  // to trigger a fault
  hsa_region_t global_region = {0};
  auto find_global = [](hsa_region_t region, void* data) -> hsa_status_t {
    hsa_region_segment_t segment;
    hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (segment == HSA_REGION_SEGMENT_GLOBAL) {
      hsa_region_global_flag_t flags;
      hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
      if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
        *((hsa_region_t*)data) = region;
        return HSA_STATUS_INFO_BREAK;
      }
    }
    return HSA_STATUS_SUCCESS;
  };
  
  err = hsa_agent_iterate_regions(gpu_agent, find_global, &global_region);
  if (global_region.handle == 0) {
    hsa_shut_down();
    _exit(1);
  }
  
  // Allocate memory
  void* ptr = nullptr;
  err = hsa_memory_allocate(global_region, 1024, &ptr);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_shut_down();
    _exit(1);
  }
  
  // Create a queue
  hsa_queue_t* queue = nullptr;
  err = hsa_queue_create(gpu_agent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 
                         UINT32_MAX, UINT32_MAX, &queue);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_memory_free(ptr);
    hsa_shut_down();
    _exit(1);
  }
  
  // Trigger fault by accessing invalid memory address
  // We'll write to an invalid GPU address which should cause a memory fault
  volatile uint64_t* bad_addr = (volatile uint64_t*)0xDEADBEEF00000000UL;
  *bad_addr = 0x42;  // This should trigger a GPU memory fault
  
  // Should not reach here
  hsa_queue_destroy(queue);
  hsa_memory_free(ptr);
  hsa_shut_down();
  _exit(0);
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

bool GpuCoreDumpTest::RunFaultTest(const char* env_var, const char* env_value,
                                   const std::string& expected_pattern) {
  // Set ulimit -c unlimited for this process
  struct rlimit rlim;
  rlim.rlim_cur = RLIM_INFINITY;
  rlim.rlim_max = RLIM_INFINITY;
  if (setrlimit(RLIMIT_CORE, &rlim) != 0) {
    if (verbosity() > 0) {
      std::cout << "    Failed to set ulimit -c unlimited" << std::endl;
    }
    return false;
  }
  
  pid_t pid = fork();
  if (pid < 0) {
    if (verbosity() > 0) {
      std::cout << "    Fork failed" << std::endl;
    }
    return false;
  }
  
  if (pid == 0) {
    // Child process - set environment and trigger fault
    if (env_var && env_value) {
      setenv(env_var, env_value, 1);
    }
    
    // Trigger the GPU fault
    TriggerGPUFault();
    
    // Should not reach here
    _exit(0);
  } else {
    // Parent process - wait for child to crash
    int status;
    pid_t result = waitpid(pid, &status, 0);
    
    if (result != pid) {
      if (verbosity() > 0) {
        std::cout << "    waitpid failed" << std::endl;
      }
      return false;
    }
    
    // Give the runtime a moment to finish writing the core dump
    usleep(500000);  // 500ms
    
    // Check if child was signaled (crashed)
    if (!WIFSIGNALED(status)) {
      if (verbosity() > 0) {
        std::cout << "    Child process did not crash as expected" << std::endl;
      }
      return false;
    }
    
    // Verify the core dump file
    std::string pattern_with_pid = expected_pattern;
    size_t pos = pattern_with_pid.find("%p");
    if (pos != std::string::npos) {
      pattern_with_pid.replace(pos, 2, std::to_string(pid));
    }
    
    return VerifyCoreDumpFile(pattern_with_pid);
  }
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
  
  if (kernel_pattern.empty() || kernel_pattern[0] == '|') {
    if (verbosity() > 0) {
      std::cout << "    Skipping: kernel pattern is empty or pipe pattern" << std::endl;
    }
    return;
  }
  
  // Unset HSA_COREDUMP_FILE to use default
  unsetenv("HSA_COREDUMP_FILE");
  unsetenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION");
  
  // Expected pattern is kernel pattern + .gpu
  std::string expected = kernel_pattern + ".gpu";
  
  bool success = RunFaultTest(nullptr, nullptr, expected);
  EXPECT_TRUE(success);
  
  // Cleanup
  CleanupCoreDumps(expected);
}

void GpuCoreDumpTest::TestCustomPattern(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing custom pattern (HSA_COREDUMP_FILE)..." << std::endl;
  }
  
  std::string custom_pattern = test_dir_ + "/custom_gpu_core.%p";
  
  bool success = RunFaultTest("HSA_COREDUMP_FILE", custom_pattern.c_str(), custom_pattern);
  EXPECT_TRUE(success);
  
  // Cleanup
  CleanupCoreDumps(test_dir_ + "/custom_gpu_core.*");
}

void GpuCoreDumpTest::TestDisableFlag(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing disable flag (HSA_DISABLE_COREDUMP_ON_EXCEPTION=1)..." << std::endl;
  }
  
  pid_t pid = fork();
  if (pid < 0) {
    FAIL() << "Fork failed";
    return;
  }
  
  if (pid == 0) {
    // Child process
    setenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION", "1", 1);
    setenv("HSA_COREDUMP_FILE", (test_dir_ + "/should_not_exist.%p").c_str(), 1);
    
    TriggerGPUFault();
    _exit(0);
  } else {
    // Parent process
    int status;
    waitpid(pid, &status, 0);
    usleep(500000);  // Give time for any dump to be written
    
    // Verify NO core dump was created
    std::string expected_file = test_dir_ + "/should_not_exist." + std::to_string(pid);
    bool file_exists = (access(expected_file.c_str(), F_OK) == 0);
    
    EXPECT_FALSE(file_exists) << "Core dump should not have been created when disabled";
    
    if (verbosity() > 0) {
      if (!file_exists) {
        std::cout << "    Correctly prevented core dump creation" << std::endl;
      }
    }
    
    // Cleanup just in case
    if (file_exists) {
      unlink(expected_file.c_str());
    }
  }
}

void GpuCoreDumpTest::TestPatternSubstitution(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing pattern substitution (%p, %e, %t)..." << std::endl;
  }
  
  // Test pattern with multiple specifiers
  std::string pattern = test_dir_ + "/core.%e.%p.dump";
  
  bool success = RunFaultTest("HSA_COREDUMP_FILE", pattern.c_str(), pattern);
  EXPECT_TRUE(success);
  
  // Cleanup
  CleanupCoreDumps(test_dir_ + "/core.*.dump");
}

void GpuCoreDumpTest::TestInvalidPath(void) {
  if (verbosity() > 0) {
    std::cout << "  Testing invalid path handling..." << std::endl;
  }
  
  // Use a path that doesn't exist and can't be created
  std::string invalid_pattern = "/nonexistent_dir_12345/core.%p";
  
  pid_t pid = fork();
  if (pid < 0) {
    FAIL() << "Fork failed";
    return;
  }
  
  if (pid == 0) {
    // Child process
    setenv("HSA_COREDUMP_FILE", invalid_pattern.c_str(), 1);
    
    // Redirect stderr to /dev/null to suppress expected error messages
    int null_fd = open("/dev/null", O_WRONLY);
    if (null_fd >= 0) {
      dup2(null_fd, STDERR_FILENO);
      close(null_fd);
    }
    
    TriggerGPUFault();
    _exit(0);
  } else {
    // Parent process
    int status;
    waitpid(pid, &status, 0);
    usleep(500000);
    
    // Verify NO core dump was created (path is invalid)
    std::string expected_file = "/nonexistent_dir_12345/core." + std::to_string(pid);
    bool file_exists = (access(expected_file.c_str(), F_OK) == 0);
    
    EXPECT_FALSE(file_exists) << "Core dump should not be created with invalid path";
    
    if (verbosity() > 0) {
      if (!file_exists) {
        std::cout << "    Correctly handled invalid path" << std::endl;
      }
    }
  }
}
