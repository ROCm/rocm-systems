/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef _HOST_AMO_TESTER_HPP_
#define _HOST_AMO_TESTER_HPP_

#include "tester.hpp"

/**
 * @file host_amo_tester.hpp
 *
 * Functional tests for host-side AMO, fence, and quiet in IPC non-MPI mode
 * (JIRA-419). Unlike all other functional testers, these call the host API
 * directly from the CPU — no GPU kernel is launched.
 *
 * Test coverage:
 *   HostAMOFAdd    — rocshmem_ctx_int_atomic_fetch_add via default context
 *   HostAMOFCswap  — rocshmem_ctx_int_atomic_compare_swap via default context
 *   HostAMOFenceQuiet — rocshmem_ctx_fence + rocshmem_ctx_quiet do not crash
 */
class HostAMOTester : public Tester {
 public:
  explicit HostAMOTester(TesterArguments args);
  virtual ~HostAMOTester();

  // Override execute() entirely: no GPU kernel, CPU-side timing.
  void execute() override;

 protected:
  void resetBuffers(uint64_t size) override;

  // Not used — execute() is overridden and drives the test directly.
  void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                    uint64_t size) override {}

  void verifyResults(uint64_t size) override;

 private:
  int* dest_{nullptr};   // symmetric int buffer; PE 0 targets PE 1's copy
};

#endif  // _HOST_AMO_TESTER_HPP_
