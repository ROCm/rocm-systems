/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @file trap_handler_test_kernels.cl
 * @brief Test kernels for validating ROCr trap handler functionality
 *
 * These kernels intentionally trigger various GPU exceptions to test
 * the trap handler's ability to:
 * - Report queue errors via callback
 * - Map hardware exceptions to correct HSA status codes
 * - Handle S_TRAP instructions with different trap IDs
 */

/**
 * Test 1: S_TRAP Abort (trap_id = 2)
 *
 * Triggers s_trap 2 which maps to EC_QUEUE_WAVE_ABORT.
 * Expected HSA status: HSA_STATUS_ERROR_EXCEPTION
 */
__kernel void trap_abort(__global int *out) {
  int gid = get_global_id(0);
  out[gid] = gid;

  // s_trap 2 causes queue abort (EC_QUEUE_WAVE_ABORT)
  __asm__ volatile("s_trap 2");
}

/**
 * Test 2: Debugger Trap (trap_id = 3 / llvm.debugtrap)
 *
 * Triggers __builtin_debugtrap() which generates s_trap 3.
 * When debugger is not attached, this should cause a queue error.
 * Expected HSA status: HSA_STATUS_ERROR_EXCEPTION
 */
__kernel void trap_debugger(__global int *out) {
  int gid = get_global_id(0);
  out[gid] = gid;

  // __builtin_debugtrap generates s_trap 3
  __builtin_debugtrap();
}

/**
 * Test 3: Generic Trap (__builtin_trap)
 *
 * Triggers __builtin_trap() which generates s_trap 1.
 * This is the standard trap used for assertions and unreachable code.
 * Expected HSA status: HSA_STATUS_ERROR_EXCEPTION
 */
__kernel void trap_generic(__global int *out) {
  int gid = get_global_id(0);
  out[gid] = gid;

  // __builtin_trap generates s_trap 1
  __builtin_trap();
}

/**
 * Test 4: Memory Violation (NULL pointer dereference)
 *
 * Dereferences a NULL pointer to trigger XNACK_ERROR or MEM_VIOL.
 * Expected HSA status: HSA_STATUS_ERROR_MEMORY_FAULT
 *
 * Note: The kernel argument 'ptr' should be passed as NULL from host.
 */
__kernel void trap_memory_violation(__global int *ptr, __global int *out) {
  int gid = get_global_id(0);

  // Write to output first so we can verify partial execution
  out[gid] = gid;

  // Dereference NULL pointer - triggers memory violation
  // ptr should be NULL when called for this test
  volatile int x = ptr[gid];
  out[gid] = x;
}

/**
 * Test 5: Illegal Instruction
 *
 * Executes an undefined instruction encoding to trigger
 * illegal instruction exception.
 * Expected HSA status: HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION
 *
 * Note: We use .long to emit bytes directly into the instruction stream.
 * 0xB0FF0000 has bits[31:24]=0xB0 which is not a valid instruction class.
 */
__kernel void trap_illegal_instruction(__global int *out) {
  int gid = get_global_id(0);
  out[gid] = gid;

  // Emit an undefined instruction encoding
  // This encoding is not a valid instruction on AMD GPUs
  // Note: 0xB0FF0000 has bits[31:24]=0xB0 which is not a valid instruction class
  __asm__ volatile(".long 0xB0FF0000");
}

/**
 * Test 6: Math Exception (Integer divide by zero)
 *
 * Triggers integer divide by zero exception.
 * Note: GPU integer division by zero behavior varies by architecture.
 * On some GFX generations this may not trap (returns 0 or MAX_INT).
 * Expected HSA status: HSA_STATUS_ERROR_EXCEPTION (if trapping enabled)
 *
 * @param divisor Should be passed as 0 from host to trigger exception
 */
__kernel void trap_math_exception(__global int *out, int divisor) {
  int gid = get_global_id(0);

  // Store gid first
  out[gid] = gid;

  // Integer divide by zero
  // divisor should be 0 when called for this test
  volatile int result = gid / divisor;
  out[gid] = result;
}

/**
 * Test 7: Aperture Violation (out of bounds access)
 *
 * Accesses memory outside valid aperture range.
 * Expected HSA status: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
 *
 * Note: Uses a wild pointer to access invalid GPU memory.
 */
__kernel void trap_aperture_violation(__global int *out) {
  int gid = get_global_id(0);
  out[gid] = gid;

  // Access a wild pointer address outside valid GPU memory
  // This address is intentionally invalid
  volatile __global int *wild_ptr = (__global int *)0xDEADBEEF00000000UL;
  volatile int x = wild_ptr[gid];
  out[gid] = x;
}

/**
 * Helper kernel: No trap (control test)
 *
 * Simple kernel that completes without triggering any trap.
 * Used as a control to verify the test infrastructure works.
 */
__kernel void trap_none(__global int *out) {
  int gid = get_global_id(0);
  out[gid] = gid * 2;
}
