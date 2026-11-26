/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
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
#ifndef ROCRTST_SUITES_FUNCTIONAL_AGENT_PROPS_H_
#define ROCRTST_SUITES_FUNCTIONAL_AGENT_PROPS_H_

#include <string>
#include <vector>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class AgentPropTest : public TestBase {
 public:
    AgentPropTest();

  // @Brief: Destructor for test case of AgentPropTest
  virtual ~AgentPropTest();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrive the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  // @Brief: Query UUID property of agents of a ROCm platform
  void QueryAgentUUID();

  // @Brief: Query Clock Counter property of agents of a ROCm platform
  void QueryAgentClockCounters();

  // @Brief: A GPU with no shader engines, no SIMDs per CU, or no CU in a
  // shader engine is an invalid topology. Cluster limits use that topology.
  void QueryGpuShaderEngineTopology();

  // @Brief: Query every HSA_SYSTEM_INFO_* and HSA_AMD_SYSTEM_INFO_* attribute
  // through hsa_system_get_info and check the values for consistency. Also
  // checks that an unknown attribute and a NULL value are rejected.
  void QuerySystemInfo();

  // @Brief: Query every hsa_agent_get_info attribute that a GPU agent
  // implements, and walk the agent's regions, caches, ISAs and memory pools.
  // Also checks that an unknown attribute and a NULL value are rejected.
  void QueryGpuAgentInfo();

 private:
  // Capture value for all agents on system
  std::vector<std::string> propList_;

  void QueryAgentProp(hsa_agent_t agent, hsa_agent_info_t prop);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_AGENT_PROPS_H_
