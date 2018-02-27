/******************************************************************************

Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************/

#ifndef TEST_CTRL_TEST_HSA_H_
#define TEST_CTRL_TEST_HSA_H_

#include "ctrl/test_aql.h"
#include "ctrl/test_kernel.h"
#include "util/hsa_rsrc_factory.h"
#include "util/perf_timer.h"

// Class implements HSA test
class TestHsa : public TestAql {
 public:
  // Instantiate HSA resources
  static HsaRsrcFactory* HsaInstantiate(const uint32_t agent_ind = agent_id_);
  static void HsaShutdown();
  static void SetQueue(hsa_queue_t* queue) { hsa_queue_ = queue; }
  static uint32_t HsaAgentId() { return agent_id_; }

  // Constructor
  explicit TestHsa(TestKernel* test) : test_(test), name_(test->Name()) {
    total_time_taken_ = 0;
    setup_time_taken_ = 0;
    dispatch_time_taken_ = 0;
    hsa_exec_ = {};
  }

  // Get methods for Agent Info, HAS queue, HSA Resourcse Manager
  const AgentInfo* GetAgentInfo() { return agent_info_; }
  hsa_queue_t* GetQueue() { return hsa_queue_; }
  HsaRsrcFactory* GetRsrcFactory() { return hsa_rsrc_; }

  // Initialize application environment including setting
  // up of various configuration parameters based on
  // command line arguments
  // @return bool true on success and false on failure
  bool Initialize(int argc, char** argv);

  // Setup application parameters for exectuion
  // @return bool true on success and false on failure
  bool Setup();

  // Run the BinarySearch kernel
  // @return bool true on success and false on failure
  bool Run();

  // Verify against reference implementation
  // @return bool true on success and false on failure
  bool VerifyResults();

  // Print to console the time taken to execute kernel
  void PrintTime();

  // Release resources e.g. memory allocations
  // @return bool true on success and false on failure
  bool Cleanup();

 private:
  typedef TestKernel::mem_descr_t mem_descr_t;
  typedef TestKernel::mem_map_t mem_map_t;
  typedef TestKernel::mem_it_t mem_it_t;

  // Test object
  TestKernel* test_;

  // Path of Brig file
  std::string brig_path_obj_;

  // Used to track time taken to run the sample
  double total_time_taken_;
  double setup_time_taken_;
  double dispatch_time_taken_;

  // Handle of signal
  hsa_signal_t hsa_signal_;

  // Handle of Kernel Code Descriptor
  hsa_executable_symbol_t kernel_code_desc_;

  // Instance of timer object
  uint32_t setup_timer_idx_;
  uint32_t dispatch_timer_idx_;
  PerfTimer hsa_timer_;

  // Instance of Hsa Resources Factory
  static HsaRsrcFactory* hsa_rsrc_;

  // GPU id
  static uint32_t agent_id_;

  // Handle to an Hsa Gpu Agent
  static const AgentInfo* agent_info_;

  // Handle to an Hsa Queue
  static hsa_queue_t* hsa_queue_;

  // Test kernel name
  std::string name_;

  // Kernel executable
  hsa_executable_t hsa_exec_;
};

#endif  // TEST_CTRL_TEST_HSA_H_
