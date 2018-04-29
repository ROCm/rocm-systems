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

#include "ctrl/test_hsa.h"

#include <atomic>

#include "util/test_assert.h"
#include "util/helper_funcs.h"
#include "util/hsa_rsrc_factory.h"

HsaRsrcFactory* TestHsa::hsa_rsrc_ = NULL;
const AgentInfo* TestHsa::agent_info_ = NULL;
hsa_queue_t* TestHsa::hsa_queue_ = NULL;
uint32_t TestHsa::agent_id_ = 0;

HsaRsrcFactory* TestHsa::HsaInstantiate(const uint32_t agent_ind) {
  // Instantiate an instance of Hsa Resources Factory
  if (hsa_rsrc_ == NULL) {
    agent_id_ = agent_ind;

    hsa_rsrc_ = HsaRsrcFactory::Create();

    // Print properties of the agents
    hsa_rsrc_->PrintGpuAgents("> GPU agents");

    // Create an instance of Gpu agent
    if (!hsa_rsrc_->GetGpuAgentInfo(agent_ind, &agent_info_)) {
      agent_info_ = NULL;
      std::cerr << "> error: agent[" << agent_ind << "] is not found" << std::endl;
      return NULL;
    }
    std::clog << "> Using agent[" << agent_ind << "] : " << agent_info_->name << std::endl;

    // Create an instance of Aql Queue
    if (hsa_queue_ == NULL) {
      uint32_t num_pkts = 128;
      if (hsa_rsrc_->CreateQueue(agent_info_, num_pkts, &hsa_queue_) == false) {
        hsa_queue_ = NULL;
        TEST_ASSERT(false);
      }
    }
  }
  return hsa_rsrc_;
}

void TestHsa::HsaShutdown() {
  if (hsa_queue_ != NULL) {
    hsa_queue_destroy(hsa_queue_);
    hsa_queue_ = NULL;
  }
  if (hsa_rsrc_) hsa_rsrc_->Destroy();
}

bool TestHsa::Initialize(int arg_cnt, char** arg_list) {
  std::clog << "TestHsa::Initialize :" << std::endl;

  // Instantiate a Timer object
  setup_timer_idx_ = hsa_timer_.CreateTimer();
  dispatch_timer_idx_ = hsa_timer_.CreateTimer();

  if (HsaInstantiate(agent_id_) == NULL) {
    TEST_ASSERT(false);
    return false;
  }

  // Obtain handle of signal
  hsa_rsrc_->CreateSignal(1, &hsa_signal_);

  // Obtain the code object file name
  std::string agentName(agent_info_->name);
  if (agentName.compare(0, 4, "gfx8") == 0) {
    brig_path_obj_.append("gfx8");
  } else if (agentName.compare(0, 4, "gfx9") == 0) {
    brig_path_obj_.append("gfx9");
  } else {
    TEST_ASSERT(false);
    return false;
  }
  brig_path_obj_.append("_" + name_ + ".hsaco");

  return true;
}

bool TestHsa::Setup() {
  std::clog << "TestHsa::setup :" << std::endl;

  // Start the timer object
  hsa_timer_.StartTimer(setup_timer_idx_);

  mem_map_t& mem_map = test_->GetMemMap();
  for (mem_it_t it = mem_map.begin(); it != mem_map.end(); ++it) {
    mem_descr_t& des = it->second;
    switch (des.id) {
      case TestKernel::LOCAL_DES_ID:
        des.ptr = hsa_rsrc_->AllocateLocalMemory(agent_info_, des.size);
        break;
      case TestKernel::KERNARG_DES_ID:
        des.ptr = hsa_rsrc_->AllocateKernArgMemory(agent_info_, des.size);
        if (des.ptr) memset(des.ptr, 0, des.size);
        break;
      case TestKernel::SYS_DES_ID:
        des.ptr = hsa_rsrc_->AllocateSysMemory(agent_info_, des.size);
        if (des.ptr) memset(des.ptr, 0, des.size);
        break;
      case TestKernel::NULL_DES_ID:
        des.ptr = NULL;
        break;
      default:
        break;
    };
    TEST_ASSERT(des.ptr != NULL);
    if (des.ptr == NULL) return false;
  }
  test_->Init();

  // Load and Finalize Kernel Code Descriptor
  char* brig_path = (char*)brig_path_obj_.c_str();
  bool suc =  hsa_rsrc_->LoadAndFinalize(agent_info_, brig_path, name_.c_str(), &hsa_exec_, &kernel_code_desc_);
  if (suc == false) {
    std::cerr << "Error in loading and finalizing Kernel" << std::endl;
    return false;
  }

  // Stop the timer object
  hsa_timer_.StopTimer(setup_timer_idx_);
  setup_time_taken_ = hsa_timer_.ReadTimer(setup_timer_idx_);
  total_time_taken_ = setup_time_taken_;

  return true;
}

bool TestHsa::Run() {
  std::clog << "TestHsa::run :" << std::endl;

  const uint32_t work_group_size = 64;
  const uint32_t work_grid_size = test_->GetGridSize();
  uint32_t group_segment_size = 0;
  uint32_t private_segment_size = 0;
  const size_t kernarg_segment_size = test_->GetKernargSize();
  uint64_t code_handle = 0;

  // Retrieve the amount of group memory needed
  hsa_executable_symbol_get_info(
      kernel_code_desc_, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_segment_size);

  // Retrieve the amount of private memory needed
  hsa_executable_symbol_get_info(kernel_code_desc_,
                                 HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                 &private_segment_size);

  // Check the kernel args size
  size_t size_info = 0;
  hsa_executable_symbol_get_info(
      kernel_code_desc_, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &size_info);
  TEST_ASSERT(kernarg_segment_size == size_info);
  if (kernarg_segment_size != size_info) return false;

  // Retrieve handle of the code block
  hsa_executable_symbol_get_info(kernel_code_desc_, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                 &code_handle);

  // Initialize the dispatch packet.
  hsa_kernel_dispatch_packet_t aql;
  memset(&aql, 0, sizeof(aql));
  // Set the packet's type, barrier bit, acquire and release fences
  aql.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  aql.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  aql.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  // Populate Aql packet with default values
  aql.setup = 1;
  aql.grid_size_x = work_grid_size;
  aql.grid_size_y = 1;
  aql.grid_size_z = 1;
  aql.workgroup_size_x = work_group_size;
  aql.workgroup_size_y = 1;
  aql.workgroup_size_z = 1;
  // Bind the kernel code descriptor and arguments
  aql.kernel_object = code_handle;
  aql.kernarg_address = test_->GetKernargPtr();
  aql.group_segment_size = group_segment_size;
  aql.private_segment_size = private_segment_size;
  // Initialize Aql packet with handle of signal
  hsa_signal_store_relaxed(hsa_signal_, 1);
  aql.completion_signal = hsa_signal_;

  std::clog << "> Executing kernel: \"" << name_ << "\"" << std::endl;

  // Start the timer object
  hsa_timer_.StartTimer(dispatch_timer_idx_);

  // Submit AQL packet to the queue
  const uint64_t que_idx = hsa_rsrc_->Submit(hsa_queue_, &aql);

  std::clog << "> Waiting on kernel dispatch signal, que_idx=" << que_idx << std::endl;

  // Wait on the dispatch signal until the kernel is finished.
  // Update wait condition to HSA_WAIT_STATE_ACTIVE for Polling
  if (hsa_signal_wait_scacquire(
    hsa_signal_,
    HSA_SIGNAL_CONDITION_LT,
    1,
    UINT64_MAX,
    HSA_WAIT_STATE_BLOCKED) != 0)
  {
    TEST_ASSERT("signal_wait failed");
  }

  std::clog << "> DONE, que_idx=" << que_idx << std::endl;

  // Stop the timer object
  hsa_timer_.StopTimer(dispatch_timer_idx_);
  dispatch_time_taken_ = hsa_timer_.ReadTimer(dispatch_timer_idx_);
  total_time_taken_ += dispatch_time_taken_;

  return true;
}

bool TestHsa::VerifyResults() {
  bool cmp = false;
  void* output = NULL;
  const uint32_t size = test_->GetOutputSize();
  bool suc = false;

  // Copy local kernel output buffers from local memory into host memory
  if (test_->IsOutputLocal()) {
    output = hsa_rsrc_->AllocateSysMemory(agent_info_, size);
    suc = hsa_rsrc_->Memcpy(agent_info_, output, test_->GetOutputPtr(), size);
    if (!suc) std::clog << "> VerifyResults: Memcpy failed" << std::endl << std::flush;
  } else {
    output = test_->GetOutputPtr();;
    suc = true;
  }

  if ((output != NULL) && suc) {
    // Print the test output
    test_->PrintOutput(output);
    // Compare the results and see if they match
    cmp = (memcmp(output, test_->GetRefOut(), size) == 0);
  }

  if (test_->IsOutputLocal() && (output != NULL)) hsa_rsrc_->FreeMemory(output);

  return cmp;
}

void TestHsa::PrintTime() {
  std::clog << "Time taken for Setup by " << this->name_ << " : " << this->setup_time_taken_
            << std::endl;
  std::clog << "Time taken for Dispatch by " << this->name_ << " : " << this->dispatch_time_taken_
            << std::endl;
  std::clog << "Time taken in Total by " << this->name_ << " : " << this->total_time_taken_
            << std::endl;
}

bool TestHsa::Cleanup() {
  hsa_executable_destroy(hsa_exec_);
  hsa_signal_destroy(hsa_signal_);
  return true;
}
