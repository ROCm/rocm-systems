/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/
#ifndef ROCRTST_SUITES_PERFORMANCE_AGENT_PRELOAD_H_
#define ROCRTST_SUITES_PERFORMANCE_AGENT_PRELOAD_H_

#include <vector>

#include "suites/test_common/test_base.h"
#include "common/base_rocr.h"
#include "common/common.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

class AgentPreloadTest : public TestBase {
 public:
  // @Brief: Constructor
  AgentPreloadTest();

  // @Brief: Destructor
  virtual ~AgentPreloadTest(void);

  // @Brief: Set up the environment for the test
  virtual void SetUp(void);

  // @Brief: Run the test case
  virtual void Run(void);

  // @Brief: Display results
  virtual void DisplayResults(void) const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  // @Brief: Clean up and close the runtime
  virtual void Close(void);

 private:
  // @Brief: Measure profiling enable latency without preload
  double MeasureProfilingEnableLatencyWithoutPreload();

  // @Brief: Measure profiling enable latency with preload
  double MeasureProfilingEnableLatencyWithPreload();

  // @Brief: Measure first async copy latency without blit preload
  double MeasureFirstAsyncCopyLatencyWithoutPreload();

  // @Brief: Measure first async copy latency with blit preload
  double MeasureFirstAsyncCopyLatencyWithPreload();

  // GPU agents to test
  std::vector<hsa_agent_t> gpu_agents_;

  // CPU agents for profiling enable
  std::vector<hsa_agent_t> cpu_agents_;

  // Latency without preload (microseconds)
  double latency_without_preload_us_;

  // Latency with preload (microseconds)
  double latency_with_preload_us_;

  // Latency improvement (microseconds)
  double latency_improvement_us_;

  // Blit copy latency without preload (microseconds)
  double blit_latency_without_preload_us_;

  // Blit copy latency with preload (microseconds)
  double blit_latency_with_preload_us_;

  // Blit latency improvement (microseconds)
  double blit_latency_improvement_us_;
};

#endif  // ROCRTST_SUITES_PERFORMANCE_AGENT_PRELOAD_H_
