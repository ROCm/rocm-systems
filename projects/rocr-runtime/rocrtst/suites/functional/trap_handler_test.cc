/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @file trap_handler_test.cc
 * @brief Comprehensive tests for ROCr trap handler functionality
 *
 * This test suite validates that the ROCr trap handler correctly:
 * - Detects various GPU exceptions (traps, memory violations, illegal instructions)
 * - Reports exceptions via queue error callbacks
 * - Maps hardware exception codes to correct HSA status values
 *
 * Tests run on all available GPU agents to ensure coverage across GFX generations.
 */

#include "suites/functional/trap_handler_test.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/platform_filter.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

// Number of work items for test kernels
static const uint32_t kNumWorkItems = 64;

// Timeout for waiting on trap (milliseconds)
static const uint64_t kTrapTimeoutMs = 5000;

// Queue error callback function
static void TrapErrorCallback(hsa_status_t status, hsa_queue_t* source, void* data) {
  TrapTestData* test_data = reinterpret_cast<TrapTestData*>(data);
  if (test_data == nullptr) {
    std::cerr << "ERROR: Trap callback received null data pointer" << std::endl;
    return;
  }

  test_data->received_status.store(status, std::memory_order_release);
  test_data->trap_triggered.store(true, std::memory_order_release);

  // Verify queue pointer if available
  if (test_data->queue_pointer != nullptr) {
    hsa_queue_t* expected_queue = *(test_data->queue_pointer);
    if (expected_queue != nullptr) {
      if (source == nullptr) {
        std::cerr << "ERROR: Queue source is NULL in callback" << std::endl;
        test_data->queue_mismatch.store(true, std::memory_order_release);
      } else if (source->id != expected_queue->id) {
        std::cerr << "ERROR: Queue ID mismatch in callback. "
                  << "Expected: " << expected_queue->id << " Got: " << source->id << std::endl;
        test_data->queue_mismatch.store(true, std::memory_order_release);
      }
    }
  }
}

TrapHandlerTest::TrapHandlerTest() : TestBase() {
  set_num_iteration(1);
  set_title("ROCr Trap Handler Tests");
  set_description(
      "Comprehensive test suite for ROCr trap handler functionality. "
      "Tests S_TRAP instructions, memory violations, illegal instructions, "
      "and math exceptions across all GPU agents.");
  set_kernel_file_name("trap_handler_test_kernels.hsaco");
}

TrapHandlerTest::~TrapHandlerTest() {}

void TrapHandlerTest::SetUp() {
  hsa_status_t err;

  TestBase::SetUp();
  if (test_skipped_) return;

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
}

void TrapHandlerTest::Run() {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void TrapHandlerTest::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

void TrapHandlerTest::DisplayResults() const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  std::cout << "\n=== Trap Handler Test Results ===" << std::endl;
  std::cout << "Passed: " << tests_passed_ << std::endl;
  std::cout << "Failed: " << tests_failed_ << std::endl;
  std::cout << "=================================" << std::endl;
}

void TrapHandlerTest::Close() { TestBase::Close(); }

bool TrapHandlerTest::RunTrapTest(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent,
                                  const char* kernel_name, hsa_status_t expected_status,
                                  bool pass_null_ptr, int divisor_value) {
  hsa_status_t err;
  hsa_queue_t* queue = nullptr;
  hsa_signal_t signal = {0};
  bool test_passed = true;

  // Get GPU name for logging
  char gpu_name[64] = {0};
  hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_NAME, gpu_name);

  std::cout << "  Testing " << kernel_name << " on " << gpu_name << "... ";

  // Initialize test data
  TrapTestData test_data;
  test_data.queue_pointer = &queue;

  // Get queue size
  uint32_t queue_size = 0;
  err = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "FAILED (cannot get queue size)" << std::endl;
    return false;
  }

  // Create queue with error callback. Use SINGLE since this is a single-producer test.
  err = hsa_queue_create(gpuAgent, queue_size, HSA_QUEUE_TYPE_SINGLE, TrapErrorCallback, &test_data,
                         0, 0, &queue);
  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "FAILED (cannot create queue)" << std::endl;
    return false;
  }

  // Find memory pools
  hsa_amd_memory_pool_t kernarg_pool;
  err = hsa_amd_agent_iterate_memory_pools(cpuAgent, rocrtst::GetKernArgMemoryPool, &kernarg_pool);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_queue_destroy(queue);
    std::cout << "FAILED (no kernarg pool)" << std::endl;
    return false;
  }

  hsa_amd_memory_pool_t global_pool;
  err = hsa_amd_agent_iterate_memory_pools(cpuAgent, rocrtst::GetGlobalMemoryPool, &global_pool);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_queue_destroy(queue);
    std::cout << "FAILED (no global pool)" << std::endl;
    return false;
  }

  // Allocate output buffer
  int* out_buffer = nullptr;
  err = hsa_amd_memory_pool_allocate(global_pool, kNumWorkItems * sizeof(int), 0,
                                     reinterpret_cast<void**>(&out_buffer));
  if (err != HSA_STATUS_SUCCESS) {
    hsa_queue_destroy(queue);
    std::cout << "FAILED (cannot allocate output buffer)" << std::endl;
    return false;
  }

  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, out_buffer);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_memory_free(out_buffer);
    hsa_queue_destroy(queue);
    std::cout << "FAILED (cannot allow GPU access)" << std::endl;
    return false;
  }

  memset(out_buffer, 0, kNumWorkItems * sizeof(int));

  // Allocate pointer buffer for memory violation test
  int* ptr_buffer = nullptr;
  if (!pass_null_ptr) {
    err = hsa_amd_memory_pool_allocate(global_pool, kNumWorkItems * sizeof(int), 0,
                                       reinterpret_cast<void**>(&ptr_buffer));
    if (err != HSA_STATUS_SUCCESS) {
      hsa_memory_free(out_buffer);
      hsa_queue_destroy(queue);
      std::cout << "FAILED (cannot allocate ptr buffer)" << std::endl;
      return false;
    }
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, ptr_buffer);
    if (err != HSA_STATUS_SUCCESS) {
      hsa_memory_free(ptr_buffer);
      hsa_memory_free(out_buffer);
      hsa_queue_destroy(queue);
      std::cout << "FAILED (cannot allow GPU access to ptr buffer)" << std::endl;
      return false;
    }
    memset(ptr_buffer, 0x42, kNumWorkItems * sizeof(int));
  }

  // Kernel arguments - allocate max needed size and zero-initialize
  // Different kernels have different argument layouts:
  //   - Single pointer kernels: trap_abort, trap_debugger, trap_generic,
  //                             trap_illegal_instruction, trap_aperture_violation, trap_none
  //     ABI: [ptr (8 bytes)]
  //   - Memory violation kernel: trap_memory_violation(__global int *ptr, __global int *out)
  //     ABI: [ptr (8 bytes), out (8 bytes)]
  //   - Math exception kernel: trap_math_exception(__global int *out, int divisor)
  //     ABI: [out (8 bytes), divisor (4 bytes), pad (4 bytes)]
  const size_t max_kernarg_size = 32;  // Enough for any kernel layout

  void* kern_args = nullptr;
  err = hsa_amd_memory_pool_allocate(kernarg_pool, max_kernarg_size, 0, &kern_args);
  if (err != HSA_STATUS_SUCCESS) {
    if (ptr_buffer) hsa_memory_free(ptr_buffer);
    hsa_memory_free(out_buffer);
    hsa_queue_destroy(queue);
    std::cout << "FAILED (cannot allocate kernargs)" << std::endl;
    return false;
  }

  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, kern_args);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_memory_free(kern_args);
    if (ptr_buffer) hsa_memory_free(ptr_buffer);
    hsa_memory_free(out_buffer);
    hsa_queue_destroy(queue);
    std::cout << "FAILED (cannot allow kernarg access)" << std::endl;
    return false;
  }

  // Zero-initialize kernarg buffer
  memset(kern_args, 0, max_kernarg_size);

  // Set kernel arguments based on kernel type
  bool is_memory_violation = (strstr(kernel_name, "memory_violation") != nullptr);
  bool is_math_exception = (strstr(kernel_name, "math_exception") != nullptr);

  if (is_memory_violation) {
    // trap_memory_violation(__global int *ptr, __global int *out)
    // ABI layout: [ptr at offset 0, out at offset 8]
    void** args = reinterpret_cast<void**>(kern_args);
    args[0] = pass_null_ptr ? nullptr : ptr_buffer;  // ptr
    args[1] = out_buffer;                            // out
  } else if (is_math_exception) {
    // trap_math_exception(__global int *out, int divisor)
    // ABI layout: [out at offset 0, divisor at offset 8]
    struct __attribute__((packed)) MathExceptionArgs {
      void* out;
      int divisor;
    };
    MathExceptionArgs* math_args = reinterpret_cast<MathExceptionArgs*>(kern_args);
    math_args->out = out_buffer;
    math_args->divisor = divisor_value;
  } else {
    // Single pointer kernels: trap_abort, trap_debugger, trap_generic, etc.
    // ABI layout: [out at offset 0]
    void** args = reinterpret_cast<void**>(kern_args);
    args[0] = out_buffer;
  }

  // Load kernel
  set_kernel_name(kernel_name);
  err = rocrtst::LoadKernelFromObjFile(this, &gpuAgent);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_memory_free(kern_args);
    if (ptr_buffer) hsa_memory_free(ptr_buffer);
    hsa_memory_free(out_buffer);
    hsa_queue_destroy(queue);
    std::cout << "FAILED (kernel not found: " << kernel_name << ")" << std::endl;
    return false;  // Kernel loading failure is a test failure
  }

  // Create completion signal
  err = hsa_signal_create(1, 0, NULL, &signal);
  if (err != HSA_STATUS_SUCCESS) {
    hsa_memory_free(kern_args);
    if (ptr_buffer) hsa_memory_free(ptr_buffer);
    hsa_memory_free(out_buffer);
    hsa_queue_destroy(queue);
    std::cout << "FAILED (cannot create signal)" << std::endl;
    return false;
  }

  // Create and dispatch AQL packet
  hsa_kernel_dispatch_packet_t aql;
  memset(&aql, 0, sizeof(aql));

  aql.setup = 1;
  aql.workgroup_size_x = kNumWorkItems;
  aql.workgroup_size_y = 1;
  aql.workgroup_size_z = 1;
  aql.grid_size_x = kNumWorkItems;
  aql.grid_size_y = 1;
  aql.grid_size_z = 1;
  aql.private_segment_size = 0;
  aql.group_segment_size = 0;
  aql.kernel_object = kernel_object();
  aql.kernarg_address = kern_args;
  aql.completion_signal = signal;

  const uint32_t queue_mask = queue->size - 1;
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);
  hsa_queue_store_write_index_relaxed(queue, index + 1);

  hsa_kernel_dispatch_packet_t* queue_packet =
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address) + (index & queue_mask);

  // Copy packet (except header)
  memcpy(reinterpret_cast<char*>(queue_packet) + 4, reinterpret_cast<char*>(&aql) + 4,
         sizeof(aql) - 4);

  // Set header atomically
  uint32_t aql_header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  aql_header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

  __atomic_store_n(reinterpret_cast<uint32_t*>(queue_packet), aql_header | (aql.setup << 16),
                   __ATOMIC_RELEASE);

  // Ring doorbell
  hsa_signal_store_relaxed(queue->doorbell_signal, index);

  // Wait for completion or trap
  // If we expect a trap, the signal may never complete
  bool wait_for_trap = (expected_status != HSA_STATUS_SUCCESS);
  hsa_signal_value_t completion = 1;

  if (wait_for_trap) {
    // Poll for trap callback with timeout
    auto start = std::chrono::steady_clock::now();
    while (!test_data.trap_triggered.load(std::memory_order_acquire)) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      if (elapsed > kTrapTimeoutMs) {
        break;
      }
      // Brief sleep to avoid busy-waiting
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // If trap callback not received, wait briefly for kernel completion to avoid
    // freeing resources while dispatch may still be in-flight
    if (!test_data.trap_triggered.load(std::memory_order_acquire)) {
      hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, 500000000ULL /* 500ms */,
                                HSA_WAIT_STATE_BLOCKED);
    }
  } else {
    // Wait for normal completion
    completion = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                           kTrapTimeoutMs * 1000000ULL, HSA_WAIT_STATE_BLOCKED);
  }

  // Check for queue ID mismatch (applies to all test types)
  if (test_data.queue_mismatch.load(std::memory_order_acquire)) {
    std::cout << "FAILED (queue ID mismatch in callback)" << std::endl;
    test_passed = false;
  }

  // Verify results
  if (test_passed && expected_status != HSA_STATUS_SUCCESS) {
    // We expected a trap
    if (!test_data.trap_triggered.load(std::memory_order_acquire)) {
      std::cout << "FAILED (trap not triggered)" << std::endl;
      test_passed = false;
    } else {
      hsa_status_t received = test_data.received_status.load(std::memory_order_acquire);
      if (received != expected_status) {
        // Some flexibility: memory faults may report different codes
        bool status_acceptable = false;
        hsa_status_t mem_fault = static_cast<hsa_status_t>(HSA_STATUS_ERROR_MEMORY_FAULT);
        hsa_status_t aperture_viol =
            static_cast<hsa_status_t>(HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION);
        if (expected_status == mem_fault || expected_status == aperture_viol) {
          // Accept either memory fault type
          status_acceptable = (received == mem_fault || received == aperture_viol);
        }
        if (!status_acceptable) {
          std::cout << "FAILED (wrong status: expected 0x" << std::hex << expected_status
                    << " got 0x" << received << std::dec << ")" << std::endl;
          test_passed = false;
        } else {
          std::cout << "PASSED (alternate status 0x" << std::hex << received << std::dec << ")"
                    << std::endl;
        }
      } else {
        std::cout << "PASSED" << std::endl;
      }
    }
  } else if (test_passed) {
    // We expected normal completion
    if (test_data.trap_triggered.load(std::memory_order_acquire)) {
      std::cout << "FAILED (unexpected trap)" << std::endl;
      test_passed = false;
    } else if (completion != 0) {
      std::cout << "FAILED (kernel did not complete)" << std::endl;
      test_passed = false;
    } else {
      std::cout << "PASSED" << std::endl;
    }
  }

  // Cleanup - destroy queue first to ensure no pending work before freeing resources
  hsa_queue_destroy(queue);
  hsa_signal_destroy(signal);
  hsa_memory_free(kern_args);
  if (ptr_buffer) hsa_memory_free(ptr_buffer);
  hsa_memory_free(out_buffer);

  return test_passed;
}

void TrapHandlerTest::RunTestOnAllGPUs(const char* kernel_name, hsa_status_t expected_status,
                                       bool pass_null_ptr, int divisor_value,
                                       bool skip_gfx11_below5) {
  hsa_status_t err;

  // Find all CPU agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(cpus.size(), 0u);

  // Find all GPU agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(gpus.size(), 0u);

  // Run test on each GPU
  uint32_t tests_failed_before = tests_failed_;
  size_t gpus_tested = 0;
  for (size_t i = 0; i < gpus.size(); ++i) {
    char gpu_name[64] = {0};
    hsa_agent_get_info(gpus[i], HSA_AGENT_INFO_NAME, gpu_name);
    // gfx110X does not service s_trap 2 (abort), so the callback never fires.
    if (skip_gfx11_below5 && std::string(gpu_name).compare(0, 6, "gfx110") == 0) {
      std::cout << "  Testing " << kernel_name << " on " << gpu_name
                << "... SKIPPED (gfx110X)" << std::endl;
      continue;
    }

    ++gpus_tested;
    bool passed =
        RunTrapTest(cpus[0], gpus[i], kernel_name, expected_status, pass_null_ptr, divisor_value);
    if (passed) {
      tests_passed_++;
    } else {
      tests_failed_++;
    }
  }

  // If every present GPU was skipped, record the skip so the test is reported
  // as SKIPPED rather than PASSED. The bundled gtest predates GTEST_SKIP(), so
  // use the same SkippedTestTracker mechanism as the core dump tests.
  if (skip_gfx11_below5 && gpus_tested == 0) {
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string testName = std::string(info->test_case_name()) + "." + info->name();
    std::string reason =
        std::string(kernel_name) + " is not supported on any present GPU (gfx110X)";
    std::cout << "[ SKIPPED ] " << reason << std::endl;
    rocrtst::SkippedTestTracker::getInstance().recordSkip(testName, reason);
    return;
  }

  // Fail the gtest if any GPU failed this test
  EXPECT_EQ(tests_failed_, tests_failed_before)
      << "Test " << kernel_name << " failed on one or more GPUs";
}

void TrapHandlerTest::TestTrapAbort() {
  std::cout << "\n--- Test: S_TRAP Abort (s_trap 2) ---" << std::endl;
  RunTestOnAllGPUs("trap_abort", HSA_STATUS_ERROR_EXCEPTION, false /* pass_null_ptr */,
                   1 /* divisor_value */, true /* skip_gfx11_below5 */);
}

void TrapHandlerTest::TestTrapDebugger() {
  std::cout << "\n--- Test: Debugger Trap (__builtin_debugtrap) ---" << std::endl;
  RunTestOnAllGPUs("trap_debugger", HSA_STATUS_ERROR_EXCEPTION);
}

void TrapHandlerTest::TestTrapGeneric() {
  std::cout << "\n--- Test: Generic Trap (__builtin_trap) ---" << std::endl;
  RunTestOnAllGPUs("trap_generic", HSA_STATUS_ERROR_EXCEPTION);
}

void TrapHandlerTest::TestTrapMemoryViolation() {
  std::cout << "\n--- Test: Memory Violation (NULL dereference) ---" << std::endl;
  RunTestOnAllGPUs("trap_memory_violation",
                   static_cast<hsa_status_t>(HSA_STATUS_ERROR_MEMORY_FAULT),
                   true /* pass_null_ptr */);
}

void TrapHandlerTest::TestTrapIllegalInstruction() {
  std::cout << "\n--- Test: Illegal Instruction ---" << std::endl;
  RunTestOnAllGPUs("trap_illegal_instruction",
                   static_cast<hsa_status_t>(HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION));
}

void TrapHandlerTest::TestTrapMathException() {
  std::cout << "\n--- Test: Math Exception (divide by zero) ---" << std::endl;
  // Note: Integer divide by zero behavior varies by GFX generation
  // Some may not trap at all. We try with divisor=0.
  RunTestOnAllGPUs("trap_math_exception", HSA_STATUS_ERROR_EXCEPTION, false /* pass_null_ptr */,
                   0 /* divisor */);
}

void TrapHandlerTest::TestTrapApertureViolation() {
  std::cout << "\n--- Test: Aperture Violation ---" << std::endl;
  RunTestOnAllGPUs("trap_aperture_violation",
                   static_cast<hsa_status_t>(HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION));
}

void TrapHandlerTest::TestNoTrap() {
  std::cout << "\n--- Test: No Trap (control test) ---" << std::endl;
  RunTestOnAllGPUs("trap_none", HSA_STATUS_SUCCESS);
}
