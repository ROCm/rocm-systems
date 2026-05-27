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


#include "hsa/hsa_ext_amd.h"
#include "aqlprofile-sdk/aql_profile_v2.h"
#include <stdlib.h>

#include <string>
#include <thread>

#include "ctrl/run_kernel.h"
#include "pgen/test_pgen_pcsmp.h"
#include "pgen/test_pgen_pmc.h"
#include "pgen/test_pgen_spm.h"
#include "pgen/test_pgen_sqtt.h"
#include "simple_convolution/simple_convolution.h"

#include <cstdlib>
#include <cstring>

namespace {

// GPU memory callbacks for aqlprofile_pmc_create_packets (see also integration/counter.cpp).
hsa_status_t new_tests_mem_alloc(void** ptr, uint64_t size,
                                 aqlprofile_buffer_desc_flags_t flags, void* userdata) {
  auto& rsrc = HsaRsrcFactory::Instance();
  auto* gpu = static_cast<const AgentInfo*>(userdata);
  if (flags.memory_hint == AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED) {
    *ptr = rsrc.AllocateKernArgMemory(gpu, static_cast<size_t>(size));
  } else {
    *ptr = rsrc.AllocateSysMemory(gpu, static_cast<size_t>(size));
  }
  if (*ptr) {
    hsa_amd_memory_fill(*ptr, 0u, size / sizeof(uint32_t));
  }
  return *ptr ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

void new_tests_mem_dealloc(void* ptr, void* /*userdata*/) {
  HsaRsrcFactory::FreeMemory(ptr);
}

hsa_status_t new_tests_mem_copy(void* dst, const void* src, size_t size, void* /*userdata*/) {
  if (size == 0) return HSA_STATUS_SUCCESS;
  return hsa_memory_copy(dst, src, size);
}

aqlprofile_agent_info_t make_agent_info(const AgentInfo* gpu) {
  aqlprofile_agent_info_t info{};
  info.agent_gfxip = gpu->gfxip;
  info.xcc_num = gpu->xcc_num;
  info.se_num = gpu->se_num;
  info.cu_num = gpu->cu_num;
  info.shader_arrays_per_se = gpu->shader_arrays_per_se;
  return info;
}

aqlprofile_agent_info_v1_t make_agent_info_v1(const AgentInfo* gpu) {
  aqlprofile_agent_info_v1_t info{};
  info.agent_gfxip = gpu->gfxip;
  info.xcc_num = gpu->xcc_num;
  info.se_num = gpu->se_num;
  info.cu_num = gpu->cu_num;
  info.shader_arrays_per_se = gpu->shader_arrays_per_se;
  info.domain = gpu->domain;
  info.location_id = gpu->bdf_id;
  return info;
}

bool alloc_kernel_memory(HsaRsrcFactory* rsrc, const AgentInfo* gpu,
                         simple_convolution& kernel,
                         hsa_executable_symbol_t kern_desc) {
  auto& mem_map = kernel.GetMemMap();
  for (auto& [buf_id, des] : mem_map) {
    switch (des.id) {
      case TestKernel::LOCAL_DES_ID:
        des.ptr = rsrc->AllocateLocalMemory(gpu, des.size);
        break;
      case TestKernel::KERNARG_DES_ID: {
        size_t actual_size = des.size;
        size_t info_size = 0;
        hsa_executable_symbol_get_info(kern_desc,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &info_size);
        if (des.size > info_size) actual_size = info_size;
        des.size = static_cast<uint32_t>(actual_size);
        des.ptr = rsrc->AllocateKernArgMemory(gpu, actual_size);
        if (des.ptr) memset(des.ptr, 0, actual_size);
        break;
      }
      case TestKernel::SYS_DES_ID:
        des.ptr = rsrc->AllocateSysMemory(gpu, des.size);
        if (des.ptr) memset(des.ptr, 0, des.size);
        break;
      case TestKernel::NULL_DES_ID:
        des.ptr = nullptr;
        break;
      default:
        break;
    }
    if (des.ptr == nullptr && des.id != TestKernel::NULL_DES_ID) {
      std::cerr << "AQLPROFILE_NEW_TESTS: memory allocation failed" << std::endl;
      return false;
    }
  }
  kernel.Init();
  return true;
}

struct PmcCreateResult {
  aqlprofile_handle_t handle;
  aqlprofile_pmc_aql_packets_t packets;
  hsa_status_t status;
};

PmcCreateResult create_pmc_packets(aqlprofile_agent_handle_t agent,
                                    const AgentInfo* gpu) {
  PmcCreateResult ret{};
  aqlprofile_pmc_event_flags_t ev_flags{};
  ev_flags.raw = 0;
  aqlprofile_pmc_event_t events_v2[1] = {};
  events_v2[0].block_index = 0;
  events_v2[0].event_id = 4;
  events_v2[0].flags = ev_flags;
  events_v2[0].block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;

  aqlprofile_pmc_profile_t profile{};
  profile.agent = agent;
  profile.events = events_v2;
  profile.event_count = 1;

  ret.status = aqlprofile_pmc_create_packets(
      &ret.handle, &ret.packets, profile,
      new_tests_mem_alloc, new_tests_mem_dealloc,
      new_tests_mem_copy, const_cast<AgentInfo*>(gpu));
  return ret;
}

}  // namespace

struct KernelSetupResult {
  simple_convolution conv;
  hsa_executable_t hsa_exec;
  hsa_executable_symbol_t kernel_code_desc;
  uint32_t group_segment_size;
  uint32_t private_segment_size;
  uint64_t code_handle;
};

bool setup_kernel_and_properties(HsaRsrcFactory* rsrc, const AgentInfo* gpu,
                                  aqlprofile_handle_t pmc_handle,
                                  hsa_signal_t kernel_signal, hsa_signal_t prof_signal,
                                  hsa_queue_t* queue,
                                  KernelSetupResult* kernelSetupResult) {
  std::string agent_name(gpu->name);
  size_t colon = agent_name.find(':');
  if (colon != std::string::npos) agent_name.resize(colon);
  std::string hsaco_path = agent_name + "_" + kernelSetupResult->conv.Name() + ".hsaco";

  if (!rsrc->LoadAndFinalize(gpu, hsaco_path.c_str(), kernelSetupResult->conv.SymbName().c_str(),
                              &kernelSetupResult->hsa_exec, &kernelSetupResult->kernel_code_desc)) {
    std::cerr << "AQLPROFILE_NEW_TESTS: LoadAndFinalize failed for " << hsaco_path << std::endl;
    return false;
  }

  if (!alloc_kernel_memory(rsrc, gpu, kernelSetupResult->conv, kernelSetupResult->kernel_code_desc)) {
    hsa_executable_destroy(kernelSetupResult->hsa_exec);
    return false;
  }

  hsa_executable_symbol_get_info(kernelSetupResult->kernel_code_desc,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &kernelSetupResult->group_segment_size);
  hsa_executable_symbol_get_info(kernelSetupResult->kernel_code_desc,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &kernelSetupResult->private_segment_size);
  hsa_executable_symbol_get_info(kernelSetupResult->kernel_code_desc,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelSetupResult->code_handle);
  return true;
}

hsa_kernel_dispatch_packet_t build_kernel_dispatch_packet(KernelSetupResult& kr,
                                                           hsa_signal_t kernel_signal) {
  const uint32_t work_group_size = 64;
  const uint32_t work_grid_size = kr.conv.GetGridSize();
  hsa_kernel_dispatch_packet_t aql{};
  memset(&aql, 0, sizeof(aql));
  aql.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  aql.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  aql.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  aql.setup = 1;
  aql.grid_size_x = work_grid_size;
  aql.grid_size_y = 1;
  aql.grid_size_z = 1;
  aql.workgroup_size_x = work_group_size;
  aql.workgroup_size_y = 1;
  aql.workgroup_size_z = 1;
  aql.kernel_object = kr.code_handle;
  aql.kernarg_address = kr.conv.GetKernargPtr();
  aql.group_segment_size = kr.group_segment_size;
  aql.private_segment_size = kr.private_segment_size;
  hsa_signal_store_relaxed(kernel_signal, 1);
  aql.completion_signal = kernel_signal;
  return aql;
}

// memcpy packet contents to the queue.
//Attach HSA signal to the packet.
//Ring doorbell by incrementing the queue write_index
//Wait for completion by watching the packet signal
void explicit_submit(const void* packet, hsa_signal_t signal, hsa_queue_t* queue) {
    constexpr uint32_t slot_size_b = HsaRsrcFactory::CMD_SLOT_SIZE_B;
    constexpr hsa_signal_value_t signal_init_val = 1;
    hsa_signal_store_relaxed(signal, signal_init_val);
    const uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    while ((write_idx - hsa_queue_load_read_index_relaxed(queue)) >= queue->size)
      std::this_thread::yield();
    const uint32_t slot_idx = static_cast<uint32_t>(write_idx % queue->size);
    uint32_t* queue_slot = reinterpret_cast<uint32_t*>(
        reinterpret_cast<uintptr_t>(queue->base_address) + (slot_idx * slot_size_b));
    const uint32_t* slot_data = reinterpret_cast<const uint32_t*>(packet);
    memcpy(&queue_slot[1], &slot_data[1], slot_size_b - sizeof(uint32_t));
    std::atomic<uint32_t>* header =
        reinterpret_cast<std::atomic<uint32_t>*>(&queue_slot[0]);
    header->store(slot_data[0], std::memory_order_release);
    hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
    return;
};

void submit_with_signal(hsa_ext_amd_aql_pm4_packet_t* pkt,
                         hsa_signal_t sig, hsa_queue_t* q) {
  pkt->header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
  pkt->completion_signal = sig;
  explicit_submit(pkt, sig, q);
  constexpr hsa_signal_value_t signal_init_val = 1;
  hsa_signal_store_relaxed(sig, signal_init_val);
}

char** pmc_argv(unsigned argc, const hsa_ven_amd_aqlprofile_event_t* events) {
  const int argv_pmc_size = 32;
  static unsigned argc_pmc = 0;
  static char* argv_arr = NULL;
  static char** argv_pmc = NULL;

  if (argc > argc_pmc) {
    argc_pmc = argc;
    argv_arr = reinterpret_cast<char*>(realloc(argv_arr, argc_pmc * argv_pmc_size));
    if (argv_pmc) delete argv_pmc;
    argv_pmc = new char*[argc + 1];
  }
  for (unsigned i = 0; i < argc; ++i) {
    char* argv_ptr = argv_arr + (i * argv_pmc_size);
    snprintf(argv_ptr, argv_pmc_size, "%d:%d:%d", events[i].block_name, events[i].block_index,
             events[i].counter_id);
    argv_pmc[i] = argv_ptr;
  }
  argv_pmc[argc] = NULL;
  return argv_pmc;
}

// --- Collect and print profiling results ---
struct PmcResult {
  uint32_t block_name;
  uint32_t block_index;
  uint32_t event_id;
  uint64_t value;
};

bool run_new_test_flow() {
    HsaRsrcFactory* rsrc = TestHsa::HsaInstantiate();
    const AgentInfo* gpu = nullptr;
    if (!rsrc || !rsrc->GetGpuAgentInfo(TestHsa::HsaAgentId(), &gpu) || gpu == nullptr) {
      std::cerr << "AQLPROFILE_NEW_TESTS: GPU agent not available" << std::endl;
      TestHsa::HsaShutdown();
      return 1;
    }

    //register agent.
    aqlprofile_agent_info_v1_t agent_info_v1 = make_agent_info_v1(gpu);
    aqlprofile_agent_handle_t agent_v1{};
    if (aqlprofile_register_agent_info(&agent_v1, &agent_info_v1, AQLPROFILE_AGENT_VERSION_V1) !=
        HSA_STATUS_SUCCESS) {
      std::cerr << "AQLPROFILE_NEW_TESTS: aqlprofile_register_agent_info failed" << std::endl;
      TestHsa::HsaShutdown();
      return 1;
    }

    //Create a queue.
    hsa_queue_t* queue = nullptr;
    if (!rsrc->CreateQueue(gpu, 1024, &queue)) {
      std::cerr << "AQLPROFILE_NEW_TESTS: CreateQueue failed" << std::endl;
      TestHsa::HsaShutdown();
      return 1;
    }

    auto pmc = create_pmc_packets(agent_v1, gpu);
    auto& pmc_handle = pmc.handle;
    auto& packets = pmc.packets;
    hsa_status_t st = pmc.status;

    // --- Create completion signals ---
    hsa_signal_t kernel_signal{}, prof_signal{};
    hsa_signal_create(1, 0, nullptr, &kernel_signal);
    hsa_signal_create(1, 0, nullptr, &prof_signal);

    if (st == HSA_STATUS_SUCCESS) {

        KernelSetupResult kr{};
        if (!setup_kernel_and_properties(rsrc, gpu, pmc_handle, kernel_signal, prof_signal,
                                          queue, &kr)) {
          aqlprofile_pmc_delete_packets(pmc_handle);
          hsa_signal_destroy(kernel_signal);
          hsa_signal_destroy(prof_signal);
          hsa_queue_destroy(queue);
          TestHsa::HsaShutdown();
          return 1;
        }

        // --- Build kernel dispatch AQL packet ---
        hsa_kernel_dispatch_packet_t aql = build_kernel_dispatch_packet(kr, kernel_signal);

        // --- Submit profiling start packet ---
        std::vector<PmcResult> pmc_results;
        auto pmc_data_cb = [](aqlprofile_pmc_event_t ev, uint64_t /*counter_id*/,
                               uint64_t val, void* userdata) -> hsa_status_t {
          static_cast<std::vector<PmcResult>*>(userdata)->push_back(
              {static_cast<uint32_t>(ev.block_name), ev.block_index, ev.event_id, val});
          return HSA_STATUS_SUCCESS;
        };
        aqlprofile_pmc_iterate_data(pmc_handle, pmc_data_cb, &pmc_results);

        auto total = 0;
        for (auto& pmc_result : pmc_results) {
          //std::cout << "block_name " << pmc_result.block_name 
          //          << " block_index " << pmc_result.block_index 
          //          << " event_id " << pmc_result.event_id
          //          << " value " << pmc_result.value 
          //           <<std::endl;
          total += pmc_result.value;
        }
        TEST_ASSERT_EQUAL(total, 0, "counter_result_value_beforedoorbell");

        submit_with_signal(&packets.start_packet, prof_signal, queue);

        // --- Submit kernel dispatch ---
        aql.completion_signal = kernel_signal;
        explicit_submit(&aql, kernel_signal, queue);

        // --- Submit profiling read + stop packets ---
        submit_with_signal(&packets.read_packet, prof_signal, queue);
        submit_with_signal(&packets.stop_packet, prof_signal, queue);

        aqlprofile_pmc_iterate_data(pmc_handle, pmc_data_cb, &pmc_results);

        // --- Verify results ---
        std::cout << "=== AQLProfile PMC Results (aqlprofile SDK v2) ===" << std::endl;
        auto total1 = 0;
        for (auto& pmc_result : pmc_results) {
          std::cout << "  block=" << pmc_result.block_name
                    << " idx=" << pmc_result.block_index
                    << " event=" << pmc_result.event_id
                    << " value=" << pmc_result.value << std::endl;
          total1 += pmc_result.value;
        }
        std::cout << "===================================================" << std::endl;
        std::cout << "total1 = " << total1 << std::endl;
        TEST_ASSERT_EQUAL(total1, 614, "counter_result_value_afterdoorbell"); /*614 also matches with older version of this test*/

        // Allocate host memory, copy GPU result, and verify against reference
        void* output = rsrc->AllocateSysMemory(gpu, kr.conv.GetOutputSize());
        if (rsrc->Memcpy(gpu, output, kr.conv.GetOutputPtr(), kr.conv.GetOutputSize())) {
          bool pass = (memcmp(output, kr.conv.GetRefOut(), kr.conv.GetOutputSize()) == 0);
          std::cout << ">>> Verification: " << (pass ? "PASS" : "FAIL") << std::endl;
          if (!pass) kr.conv.PrintOutput(output);
        } else {
          std::cerr << "AQLPROFILE_NEW_TESTS: Memcpy for verify failed" << std::endl;
        }
        rsrc->FreeMemory(output);

        // --- Cleanup kernel resources ---
        //TODO: This code crashes for some reason. Not sure why
        //for (auto& [buf_id, des] : kr.conv.GetMemMap()) {
        //  if (des.ptr) rsrc->FreeMemory(des.ptr);
        //}
        hsa_executable_destroy(kr.hsa_exec);
        aqlprofile_pmc_delete_packets(pmc_handle);

    } else {
        std::cerr << "AQLPROFILE_NEW_TESTS: aqlprofile_pmc_create_packets failed" << std::endl;
    }
    hsa_signal_destroy(kernel_signal);
    hsa_signal_destroy(prof_signal);

    hsa_queue_destroy(queue);
    TestHsa::HsaShutdown();
    return (st == HSA_STATUS_SUCCESS) ? 0 : 1;
}

int main(int argc, char* argv[]) {
  bool ret_val = false;
  const bool pmc_enable = (getenv("AQLPROFILE_PMC") != NULL);
  const bool pmc_priv_enable = (getenv("AQLPROFILE_PMC_PRIV") != NULL);
  const bool sdma_enable = (getenv("AQLPROFILE_SDMA") != NULL);
  const bool sqtt_enable = (getenv("AQLPROFILE_SQTT") != NULL);
  const bool pcsmp_enable = (getenv("AQLPROFILE_PCSMP") != NULL);
  const bool scan_enable = (getenv("AQLPROFILE_SCAN") != NULL);
  const bool trace_enable = (getenv("AQLPROFILE_TRACE") != NULL);
  const bool spm_enable = (getenv("AQLPROFILE_SPM") != NULL);
  const bool new_tests = (getenv("AQLPROFILE_NEW_TESTS") != NULL);
  int scan_step = 1;
  const char* step_env = getenv("AQLPROFILE_SCAN_STEP");
  if (step_env != NULL) {
    int step = atoi(step_env);
    if (step <= 0) {
      std::cerr << "Error in setting environment variable AQLPROFILE_SCAN_STEP=" << step_env
                << ", it should be greater than or equal to 1." << std::endl;
      return 1;
    }
    scan_step = step;
  }

  const char* spm_loop_env = getenv("AQLPROFILE_SPM_LOOPS");
  int spm_loops = spm_loop_env ? atoi(spm_loop_env) : 1;
  if (!spm_loops) spm_loops = 1;

  if (!trace_enable) {
    std::clog.rdbuf(NULL);
  }
  if (scan_enable) {
    std::cerr.rdbuf(NULL);
  }

  TestHsa::HsaInstantiate();
  const hsa_ven_amd_aqlprofile_event_t* events_arr;

  if (new_tests) {
      auto res = run_new_test_flow();
      return res;
  }

  //Older V1 based Test flow starts here. This will be deprecated soon.
  // Run simple convolution test
  if (pmc_enable) {
    if (argc > 1) {
      ret_val = RunKernel<simple_convolution, TestPGenPmc<RUN_MODE> >(argc - 1, argv + 1);
    } else if (!scan_enable) {
      int events_count = 0;
      if (TestHsa::HsaAgentName() == "gfx9") {
        const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 2 /*CYCLES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 3 /*BUSY_CYCLES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 4 /*WAVES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 14 /*ITEMS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 47 /*WAVE_READY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 1 /*CYCLE*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 3 /*REQ*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 22 /*WRITEBACK*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 0 /*ALWAYS_COUNT*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 8 /*ME1_STALL_WAIT_ON_RCIU_READ*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 0},  /*CYCLE*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 2},  /*BANK0_PTE_CACHE_HITS*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 7},  /*PDE0_CACHE_REQS*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 8},  /*PDE0_CACHE_HITS*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 13}, /*BANK0_4K_PTE_CACHE_MISSES*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 14}, /*BANK0_BIGK_PTE_CACHE_HITS*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 15}, /*BANK0_BIGK_PTE_CACHE_MISSES*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, 0, 0},   /*CYCLE*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, 0, 2},   /*BANK0_REQUESTS*/
        };
        events_count = sizeof(events_arr1) / sizeof(hsa_ven_amd_aqlprofile_event_t);
        events_arr = events_arr1;
      } else if (TestHsa::HsaAgentName() == "gfx12") {
        const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CHA, 0, 25 /*ALWAYS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CHA, 0, 0 /*BUSY*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CHC, 0, 0 /*ALWAYS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CHC, 0, 1 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 25 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, 0, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, 0, 24 /*BUSY*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CPG, 0, 0 /*ALWAYS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CPG, 0, 51 /*BUSY*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GC_UTCL2, 0, 1},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GC_VML2, 0, 5},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, 0, 3},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, 0, 4},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR, 0, 6},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR, 0, 22},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A, 0, 1 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A, 0, 2 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C, 0, 1 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C, 0, 2 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, 0, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, 0, 2 /*GUI_ACTIVE*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_RLC, 0, 2},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_RLC, 0, 5},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 0, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 0, 2 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 1, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 1, 2 /*BUSY*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GC_UTCL1, 0, 1},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GC_UTCL1, 0, 2},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GCEA_SE, 0, 3},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GCEA_SE, 0, 4},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GRBMH, 0, 0 /*ALWAYS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GRBMH, 0, 19},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 46 /*CSN_BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 47 /*CSN_NUM_THREADGROUPS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_SQG, 0,14 /*ALWAYS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_SQG, 0, 15 /*BUSY*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_SQG, 0, 19 /*WAVES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A, 0, 21 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A, 0, 0 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C, 0, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C, 0, 1 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 2 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 3 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 4 /*WAVES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0, 15 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0, 1 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0, 96 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0, 10 /*REQ_READ*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0, 14 /*REQ_WRITE*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 1, 96 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 1, 10 /*REQ_READ*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 1, 14 /*REQ_WRITE*/},
        };
        events_count = sizeof(events_arr1) / sizeof(hsa_ven_amd_aqlprofile_event_t);
        events_arr = events_arr1;
      } else {
        const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 4 /*WAVES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 14 /*ITEMS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 47 /*WAVE_READY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 1 /*CYCLE*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 3 /*REQS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 22 /*WRITEBACK*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 0 /*ALWAYS_COUNT*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 8 /*ME1_STALL_WAIT_ON_RCIU_READ*/},
        };
        events_count = sizeof(events_arr1) / sizeof(hsa_ven_amd_aqlprofile_event_t);
        events_arr = events_arr1;
      }
      ret_val = RunKernel<simple_convolution, TestPGenPmc<RUN_MODE> >(
          events_count, pmc_argv(events_count, events_arr));
    } else {
      const int block_index_max = 16;
      const int event_id_max = 128;
      for (unsigned i = 0; i < AQLPROFILE_BLOCKS_NUMBER; ++i) {
        for (unsigned j = 0; j < block_index_max; ++j) {
          for (unsigned k = 0; k <= event_id_max; k += scan_step) {
            fflush(stdout);
            fprintf(stderr, " %d %d %d                 \r", i, j, k);
            fflush(stderr);
            hsa_ven_amd_aqlprofile_event_t event = {(hsa_ven_amd_aqlprofile_block_name_t)i, j, k};
            if (!RunKernel<simple_convolution, TestPGenPmc<RUN_MODE> >(1, pmc_argv(1, &event))) {
              if (k == 0) {
                k = event_id_max + 1;
                if (j == 0) j = block_index_max + 1;
              }
              continue;
            }
          }
        }
      }
    }
  } else if (sdma_enable) {
    int events_count = 0;
    const hsa_ven_amd_aqlprofile_event_t events_sdma[] = {
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 0, 17 /*MC_WR_COUNT*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 0, 19 /*MC_RD_COUNT*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 1, 17 /*MC_WR_COUNT*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 1, 19 /*MC_RD_COUNT*/},
    };
    events_count = sizeof(events_sdma) / sizeof(hsa_ven_amd_aqlprofile_event_t);
    ret_val = RunKernel<simple_convolution, TestPGenPmc<SETUP_MODE> >(
        events_count, pmc_argv(events_count, events_sdma));
  } else if (pmc_priv_enable) {
    int events_count = 0;
    if (TestHsa::HsaAgentName() == "gfx9") {
      const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 0},  /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 2},  /*BANK0_PTE_CACHE_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 7},  /*PDE0_CACHE_REQS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 8},  /*PDE0_CACHE_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 13}, /*BANK0_4K_PTE_CACHE_MISSES*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 14}, /*BANK0_BIGK_PTE_CACHE_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 15}, /*BANK0_BIGK_PTE_CACHE_MISSES*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 0},  /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, 0, 0},   /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, 0, 2},   /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 0},     /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 2},     /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 7},     /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 8},     /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, 0, 0},    /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, 0, 2},    /*REQS_PER_CLIENT_GROUP*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 0},     /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 2},     /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 7},     /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 8},     /**/
      };
      events_count = sizeof(events_arr1) / sizeof(hsa_ven_amd_aqlprofile_event_t);
      events_arr = events_arr1;
    } else {
      const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 0},  /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 2},  /*BANK0_PTE_CACHE_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB, 0, 0},   /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB, 0, 1},   /*CORRECTABLE_GECC_ERR_CNT_CHAN0*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB, 0, 2},   /*CORRECTABLE_GECC_ERR_CNT_CHAN1*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB, 0, 3},   /*UNCORRECTABLE_GECC_ERR_CNT_CHAN0*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB, 0, 0},   /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB, 0, 1},   /*ACPG_WRRET_VLD*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB, 0, 2},   /*ACPO_WRRET_VLD*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB, 0, 3},   /*IH_WRRET_VLD*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR, 0, 0},  /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR, 0, 1},  /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR, 0, 2},  /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR, 0, 3},  /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM, 0, 0}, /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM, 0, 1}, /*TLB0_REQS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM, 0, 2}, /*TLB0_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM, 0, 3}, /*TLB0_MISSES*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 0},     /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 24},    /*ATCL2_L1_REQAS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 25},    /*ATCL2_BANK0_REQS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 26},    /*ATCL2_BANK0_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 0},     /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 2},     /*RD_REQS_IN*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 7},     /*WR_REQ_QUEUE2_IN*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 8},     /*WR_REQ_QUEUE3_IN*/
      };
      events_count = sizeof(events_arr1) / sizeof(hsa_ven_amd_aqlprofile_event_t);
      events_arr = events_arr1;
    }
    ret_val = RunKernel<simple_convolution, TestPGenPmc<RUN_MODE> >(
        events_count, pmc_argv(events_count, events_arr));
  } else if (sqtt_enable) {
    ret_val = RunKernel<simple_convolution, TestPGenSqtt>(argc, argv);
  } else if (pcsmp_enable && TestHsa::HsaAgentName().substr(0, 4) != "gfx1") {
    ret_val = RunKernel<simple_convolution, TestPGenPcsmp>(argc, argv);
  } else if (spm_enable) {
    int events_count = 0;
    const hsa_ven_amd_aqlprofile_event_t events_spm[] = {
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 2 /*CYCLES*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 4 /*WAVES*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 14 /*ITEMS*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 48 /*CSN_BUSY*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 49 /*CSN_NUM_THREADGROUPS*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 51 /*CSN_EVENT_WAVE*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 47 /*CSN_WINDOW_VALID*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 0 /*ALWAYS_COUNT*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 8 /*SEL_ME1_STALL_WAIT_ON_RCIU_READ*/},
    };
    events_count = sizeof(events_spm) / sizeof(hsa_ven_amd_aqlprofile_event_t);

    ret_val = RunKernel<simple_convolution, TestPGenSpm>(
        events_count, pmc_argv(events_count, events_spm), spm_loops);
  } else {
    ret_val = RunKernel<simple_convolution, TestAql>(argc, argv);
  }
  TestHsa::HsaShutdown();

  return (ret_val) ? 0 : 1;
}
