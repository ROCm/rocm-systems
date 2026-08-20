/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_TRAP_HANDLER_TEST_H_
#define ROCRTST_SUITES_FUNCTIONAL_TRAP_HANDLER_TEST_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

#include <atomic>

/**
 * @brief Test data structure passed to queue error callback
 *
 * Contains synchronization primitives and result storage for
 * validating trap handler behavior.
 */
struct TrapTestData {
  std::atomic<bool> trap_triggered{false};
  std::atomic<hsa_status_t> received_status{HSA_STATUS_SUCCESS};
  std::atomic<bool> queue_mismatch{false};
  hsa_queue_t** queue_pointer{nullptr};
};

/**
 * @brief Comprehensive trap handler test suite
 *
 * Tests the ROCr trap handler's ability to correctly handle various
 * GPU exceptions and report them via queue error callbacks.
 *
 * Test coverage:
 * - S_TRAP instructions (abort, debug, generic)
 * - Memory violations (NULL pointer, aperture violation)
 * - Illegal instruction execution
 * - Math exceptions (divide by zero)
 *
 * Each test verifies:
 * 1. Queue error callback is invoked
 * 2. Correct HSA status code is reported
 * 3. Queue pointer in callback matches expected queue
 */
class TrapHandlerTest : public TestBase {
 public:
  TrapHandlerTest();
  virtual ~TrapHandlerTest();

  // TestBase interface
  virtual void SetUp() override;
  virtual void Run() override;
  virtual void Close() override;
  virtual void DisplayResults() const override;
  virtual void DisplayTestInfo(void) override;

  /**
   * @brief Test S_TRAP 2 (abort trap)
   *
   * Dispatches kernel that executes s_trap 2 instruction.
   * Expects: HSA_STATUS_ERROR_EXCEPTION via callback
   */
  void TestTrapAbort();

  /**
   * @brief Test S_TRAP 3 (debugger trap)
   *
   * Dispatches kernel that executes __builtin_debugtrap().
   * When debugger is not attached, expects queue error.
   * Expects: HSA_STATUS_ERROR_EXCEPTION via callback
   */
  void TestTrapDebugger();

  /**
   * @brief Test S_TRAP 1 (generic trap / __builtin_trap)
   *
   * Dispatches kernel that executes __builtin_trap().
   * Expects: HSA_STATUS_ERROR_EXCEPTION via callback
   */
  void TestTrapGeneric();

  /**
   * @brief Test memory violation (NULL pointer dereference)
   *
   * Dispatches kernel with NULL pointer argument.
   * Expects: HSA_STATUS_ERROR_MEMORY_FAULT via callback
   */
  void TestTrapMemoryViolation();

  /**
   * @brief Test illegal instruction execution
   *
   * Dispatches kernel that executes an illegal opcode.
   * Expects: HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION via callback
   */
  void TestTrapIllegalInstruction();

  /**
   * @brief Test math exception (divide by zero)
   *
   * Dispatches kernel with divisor=0.
   * Note: Behavior varies by GFX generation - some may not trap.
   * Expects: HSA_STATUS_ERROR_EXCEPTION via callback (if supported)
   */
  void TestTrapMathException();

  /**
   * @brief Test aperture violation (invalid address access)
   *
   * Dispatches kernel that accesses invalid memory address.
   * Expects: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION via callback
   */
  void TestTrapApertureViolation();

  /**
   * @brief Control test - no trap
   *
   * Dispatches kernel that completes normally without trap.
   * Verifies test infrastructure works correctly.
   * Expects: No callback, kernel completes successfully
   */
  void TestNoTrap();

 private:
  /**
   * @brief Run a single trap test with specified kernel and expectations
   *
   * @param cpuAgent CPU agent for memory allocation
   * @param gpuAgent GPU agent for kernel dispatch
   * @param kernel_name Name of kernel function in .hsaco file
   * @param expected_status Expected HSA status in callback (HSA_STATUS_SUCCESS means no callback
   * expected)
   * @param pass_null_ptr If true, pass NULL as first kernel argument
   * @param divisor_value Value for divisor argument (math exception test)
   * @return true if test passed, false otherwise
   */
  bool RunTrapTest(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent, const char* kernel_name,
                   hsa_status_t expected_status, bool pass_null_ptr = false, int divisor_value = 1);

  /**
   * @brief Helper to iterate over all GPU agents and run test
   *
   * @param kernel_name Kernel to dispatch
   * @param expected_status Expected callback status
   * @param pass_null_ptr Pass NULL argument flag
   * @param divisor_value Divisor for math test
   */
  void RunTestOnAllGPUs(const char* kernel_name, hsa_status_t expected_status,
                        bool pass_null_ptr = false, int divisor_value = 1,
                        bool skip_gfx11_below5 = false);

  // Test results tracking
  int tests_passed_{0};
  int tests_failed_{0};
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_TRAP_HANDLER_TEST_H_
