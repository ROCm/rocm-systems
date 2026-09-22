// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include "ctrl/test_hsa.h"

#include <atomic>
#include <cassert>

#include "util/helper_funcs.h"
#include "util/hsa_rsrc_factory.h"
#include "util/test_assert.h"

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
      uint32_t num_pkts = 1024;
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
  if (agentName.find(":") != std::string::npos) {
    agentName = agentName.substr(0, agentName.find(":"));
  }
  brig_path_obj_.append(agentName + "_" + name_ + ".hsaco");

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

  std::clog << "> Waiting on kernel dispatch signal, que_idx=" << que_idx << std::endl;

  // Wait on the dispatch signal until the kernel is finished.
  // Update wait condition to HSA_WAIT_STATE_ACTIVE for Polling
  if (hsa_signal_wait_scacquire(hsa_signal_, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                HSA_WAIT_STATE_BLOCKED) != 0) {
    TEST_ASSERT(false);
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
    output = test_->GetOutputPtr();
    suc = true;
  }

  if ((output != NULL) && suc) {
    // Print the test output
    test_->PrintOutput(output);
    // Compare the results and see if they match
    cmp = (memcmp(output, test_->GetRefOut(), size) >= 0);
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

bool TestHsa::RunSdma(size_t sdma_size) {
  std::cout << "Run SDMA test ..." << std::endl;
  const AgentInfo* cpu_agent{nullptr};
  hsa_rsrc_->GetCpuAgentInfo(0, &cpu_agent);
  const AgentInfo* gpu_agent{nullptr};
  hsa_rsrc_->GetGpuAgentInfo(0, &gpu_agent);

  // allocate SDMA buffers: src_buf, dest_buf and gpu_buf.
  void* src_buf = hsa_rsrc_->AllocateSysMemory(gpu_agent, sdma_size);
  assert(src_buf != nullptr);
  void* dest_buf = hsa_rsrc_->AllocateSysMemory(gpu_agent, sdma_size);
  assert(dest_buf != nullptr);
  void* gpu_buf = hsa_rsrc_->AllocateLocalMemory(gpu_agent, sdma_size);
  assert(gpu_buf != nullptr);

  for (size_t i = 0; i < sdma_size; ++i) {
    ((char*)src_buf)[i] = i;
    ((char*)dest_buf)[sdma_size - 1 - i] = i & 0xFF;
  }

  for (size_t i = 0; i < 10; ++i)
    std::cout << i << ": src_buf = " << (unsigned)(((char*)src_buf)[i] & 0xFF)
              << ", dest_buf = " << (unsigned)(((char*)dest_buf)[i] & 0xFF) << std::endl;

  hsa_status_t status;

  hsa_signal_t completion_signal;
  status = hsa_signal_create(1, 0, NULL, &completion_signal);
  CHECK_STATUS("hsa_signal_create", status);

  // SDMA src_buf -> gpu_buf
  status = hsa_amd_memory_async_copy(gpu_buf, gpu_agent->dev_id, src_buf, cpu_agent->dev_id,
                                     sdma_size, 0, nullptr, completion_signal);
  CHECK_STATUS("hsa_amd_memory_async_copy(...): src_buf -> gpu_buf", status);

  while (1) {
    const hsa_signal_value_t signal_value = hsa_signal_wait_scacquire(
        completion_signal, HSA_SIGNAL_CONDITION_LT, 1, 5000000, HSA_WAIT_STATE_BLOCKED);
    if (signal_value == 0) {
      break;
    } else {
      CHECK_STATUS("hsa_signal_wait_scacquire(): src_buf -> gpu_buf", HSA_STATUS_ERROR);
    }
  }
  status = hsa_signal_destroy(completion_signal);
  CHECK_STATUS("hsa_signal_destroy()", status);

  // SDMA gpu_buf -> dest_buf
  hsa_signal_t completion_signal1;
  status = hsa_signal_create(1, 0, NULL, &completion_signal1);
  CHECK_STATUS("hsa_signal_create", status);

  status = hsa_amd_memory_async_copy(dest_buf, cpu_agent->dev_id, gpu_buf, gpu_agent->dev_id,
                                     sdma_size, 0, nullptr, completion_signal1);
  CHECK_STATUS("hsa_amd_memory_async_copy(...): gpu_buf -> dest_buf", status);

  while (1) {
    const hsa_signal_value_t signal_value = hsa_signal_wait_scacquire(
        completion_signal1, HSA_SIGNAL_CONDITION_LT, 1, 500000, HSA_WAIT_STATE_BLOCKED);
    if (signal_value == 0) {
      break;
    } else {
      CHECK_STATUS("hsa_signal_wait_scacquire(): gpu_buf -> dest_buf", HSA_STATUS_ERROR);
    }
  }
  status = hsa_signal_destroy(completion_signal1);
  CHECK_STATUS("hsa_signal_destroy()", status);

  // check copy results
  for (size_t i = 0; i < sdma_size; ++i) {
    assert(((char*)src_buf)[i] == ((char*)dest_buf)[i]);
  }

  std::cout << std::endl;

  // print out some dma data.
  for (size_t i = 0; i < 10; ++i)
    std::cout << i << ": src_buf = " << (int)((char*)src_buf)[i]
              << ", dest_buf = " << (int)((char*)dest_buf)[i] << std::endl;

  return true;
}
