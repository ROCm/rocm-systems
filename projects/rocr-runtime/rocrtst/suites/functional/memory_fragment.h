/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ROCRTST_SUITES_FUNCTIONAL_MEMORY_FRAGMENT_H_
#define ROCRTST_SUITES_FUNCTIONAL_MEMORY_FRAGMENT_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

// This suite validates ROCr's fragment sub-allocator for fine-grain system
// memory. Small allocations are carved out of a larger backing buffer
// ("fragments"); the tests below confirm that:
//   * distinct fragments never overlap and never clobber each other, even
//     across free/re-allocate churn that recycles backing storage
//     (FragmentOverlapTest);
//   * freshly handed-out system fragments are zero-initialized to match
//     dedicated buffer-object semantics, including recycled fragments
//     (FragmentZeroInitTest);
//   * GPU writes to a sub-allocated fine-grain system fragment are coherent
//     and visible to the CPU after a system-scope release
//     (FragmentCoherenceTest).
class MemoryFragment : public TestBase {
 public:
  MemoryFragment();

  // @Brief: Destructor for test case of MemoryFragment
  virtual ~MemoryFragment();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrieve the resource
  virtual void Close();

  // @Brief: Display results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  // @Brief: Verify sub-allocated system fragments never overlap or clobber
  // each other, including across free/re-allocate churn.
  void FragmentOverlapTest(void);

  // @Brief: Verify freshly handed-out (and recycled) system fragments are
  // zero-initialized.
  void FragmentZeroInitTest(void);

  // @Brief: Verify GPU writes to a sub-allocated fine-grain system fragment
  // are coherent and visible to the CPU.
  void FragmentCoherenceTest(void);

 private:
  void FragmentCoherenceTest(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent);

  // Overlap/clobber worker for a single pool. When grant_cpu is set (device
  // memory), CPU access to each fragment is requested before host access.
  // light selects a smaller workload for slower (e.g. PCIe-BAR) paths.
  void FragmentOverlapOnPool(const char* label, hsa_amd_memory_pool_t pool,
                             bool grant_cpu, hsa_agent_t cpu_agent, bool light);

  // GPU->CPU coherence worker for a single "under test" pool. When grant_cpu
  // is set (device memory), CPU access to the fragment is requested.
  void FragmentCoherenceOnPool(const char* label, hsa_agent_t cpuAgent,
                               hsa_agent_t gpuAgent,
                               hsa_amd_memory_pool_t under_test_pool,
                               bool grant_cpu);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_MEMORY_FRAGMENT_H_
