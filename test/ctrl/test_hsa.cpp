/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#include "ctrl/test_hsa.h"

#include <atomic>

#include <dlfcn.h>  // for dladdr

#include "util/test_assert.h"
#include "util/helper_funcs.h"
#include "util/hsa_rsrc_factory.h"

#include "src/utils/filesystem.hpp"

namespace fs = rocprofiler::common::filesystem;

HsaRsrcFactory* TestHsa::hsa_rsrc_ = NULL;

HsaRsrcFactory* TestHsa::HsaInstantiate() {
  // Instantiate an instance of Hsa Resources Factory
  if (hsa_rsrc_ == NULL) {
    hsa_rsrc_ = HsaRsrcFactory::Create();
    // Print properties of the agents
    hsa_rsrc_->PrintGpuAgents("> GPU agents");
  }
  return hsa_rsrc_;
}

void TestHsa::HsaShutdown() {
  if (hsa_rsrc_) hsa_rsrc_->Destroy();
}

bool TestHsa::Initialize(int /*arg_cnt*/, char** /*arg_list*/) {
  std::clog << "TestHsa::Initialize :" << std::endl;

  // Instantiate a Timer object
  setup_timer_idx_ = hsa_timer_.CreateTimer();
  dispatch_timer_idx_ = hsa_timer_.CreateTimer();

  if (hsa_rsrc_ == NULL) {
    TEST_ASSERT(false);
    return false;
  }

  // Create an instance of Gpu agent
  if (agent_info_ == NULL) {
    const uint32_t agent_id = 0;
    if (!hsa_rsrc_->GetGpuAgentInfo(agent_id, &agent_info_)) {
      agent_info_ = NULL;
      std::cerr << "> error: agent[" << agent_id << "] is not found" << std::endl;
      return false;
    }
  }
  std::clog << "> Using agent[" << agent_info_->dev_index << "] : " << agent_info_->name
            << std::endl;

  // Create an instance of Aql Queue
  if (hsa_queue_ == NULL) {
    const uint32_t num_pkts = 128;
    if (hsa_rsrc_->CreateQueue(agent_info_, num_pkts, &hsa_queue_) == false) {
      hsa_queue_ = NULL;
      TEST_ASSERT(false);
    }
    my_queue_ = true;
  }

  // Obtain handle of signal
  hsa_rsrc_->CreateSignal(1, &hsa_signal_);

  // Obtain the code object file name
  std::string agentName(agent_info_->name);
  const char* hsaco_obj_files_path_str = getenv("HSACO_OBJ_FILES_PATH");
  fs::path hsaco_obj_files_path;
  Dl_info dl_info;

  if(hsaco_obj_files_path_str) {
    hsaco_obj_files_path = fs::path(hsaco_obj_files_path_str);
    brig_path_obj_.append(hsaco_obj_files_path.remove_filename());

  } else if (dladdr(reinterpret_cast<const void*>(TestHsa::HsaShutdown), &dl_info) != 0) {

    hsaco_obj_files_path = fs::canonical(fs::path(dl_info.dli_fname));
    brig_path_obj_.append(hsaco_obj_files_path.remove_filename().parent_path().parent_path().parent_path());
    brig_path_obj_ += "/share/rocprofiler/tests-v1/";
  }
  brig_path_obj_.append(agentName);
  brig_path_obj_.append("_" + name_ + ".hsaco");

  return true;
}

bool TestHsa::Setup() {
  std::clog << "TestHsa::setup :" << std::endl;

  // Start the timer object
  hsa_timer_.StartTimer(setup_timer_idx_);

  // Load and Finalize Kernel Code Descriptor
  const char* brig_path = brig_path_obj_.c_str();
  bool suc = hsa_rsrc_->LoadAndFinalize(agent_info_, brig_path, symb_.c_str(), &hsa_exec_,
                                        &kernel_code_desc_);
  if (suc == false) {
    std::cerr << "Error in loading and finalizing Kernel" << std::endl;
    return false;
  }

  mem_map_t& mem_map = test_->GetMemMap();
  for (mem_it_t it = mem_map.begin(); it != mem_map.end(); ++it) {
    mem_descr_t& des = it->second;
    if (des.size == 0) continue;

    switch (des.id) {
      case TestKernel::LOCAL_DES_ID:
        des.ptr = hsa_rsrc_->AllocateLocalMemory(agent_info_, des.size);
        break;
      case TestKernel::KERNARG_DES_ID: {
        // Check the kernel args size
        const size_t kernarg_size = des.size;
        size_t size_info = 0;
        const hsa_status_t status = hsa_executable_symbol_get_info(
            kernel_code_desc_, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &size_info);
        TEST_ASSERT(status == HSA_STATUS_SUCCESS);
        size_info = kernarg_size;
        const bool kernarg_missmatch = (kernarg_size > size_info);
        if (kernarg_missmatch) {
          std::cout << "kernarg_size = " << kernarg_size << ", size_info = " << size_info
                    << std::flush << std::endl;
          TEST_ASSERT(!kernarg_missmatch);
          break;
        }
        // ALlocate kernarg memory
        des.size = size_info;
        des.ptr = hsa_rsrc_->AllocateKernArgMemory(agent_info_, size_info);
        if (des.ptr) memset(des.ptr, 0, size_info);
        break;
      }
      case TestKernel::SYS_DES_ID:
        des.ptr = hsa_rsrc_->AllocateSysMemory(agent_info_, des.size);
        if (des.ptr) memset(des.ptr, 0, des.size);
        break;
      case TestKernel::NULL_DES_ID:
        des.ptr = NULL;
        break;
      default:
        break;
    }
    TEST_ASSERT(des.ptr != NULL);
    if (des.ptr == NULL) return false;
  }
  test_->Init();

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
  uint64_t code_handle = 0;

  // Retrieve the amount of group memory needed
  hsa_executable_symbol_get_info(
      kernel_code_desc_, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_segment_size);

  // Retrieve the amount of private memory needed
  hsa_executable_symbol_get_info(kernel_code_desc_,
                                 HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                 &private_segment_size);


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

  std::clog << "> Waiting on kernel dispatch signal, que_idx=" << que_idx << std::endl
            << std::flush;

  // Wait on the dispatch signal until the kernel is finished.
  // Update wait condition to HSA_WAIT_STATE_ACTIVE for Polling
  if (hsa_signal_wait_relaxed(hsa_signal_, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED) != 0) {
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

  if (size == 0) return true;

  // Copy local kernel output buffers from local memory into host memory
  if (test_->IsOutputLocal()) {
    output = hsa_rsrc_->AllocateSysMemory(agent_info_, size);
    suc = hsa_rsrc_->Memcpy(agent_info_, output, test_->GetOutputPtr(), size);
    if (!suc) std::clog << "> VerifyResults: Memcpy failed" << std::endl << std::flush;
  } else {
    output = test_->GetOutputPtr();
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
  if (my_queue_) hsa_queue_destroy(hsa_queue_);
  hsa_queue_ = NULL;
  agent_info_ = NULL;
   return true;
}
