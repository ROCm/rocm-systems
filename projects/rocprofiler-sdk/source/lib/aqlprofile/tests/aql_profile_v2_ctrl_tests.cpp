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

namespace
{
// GPU memory callbacks for aqlprofile_pmc_create_packets (see also integration/counter.cpp).
hsa_status_t
new_tests_mem_alloc(void** ptr, uint64_t size, aqlprofile_buffer_desc_flags_t flags, void* userdata)
{
    auto& rsrc = HsaRsrcFactory::Instance();
    auto* gpu  = static_cast<const AgentInfo*>(userdata);
    if(flags.memory_hint == AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED)
    {
        *ptr = rsrc.AllocateKernArgMemory(gpu, static_cast<size_t>(size));
    }
    else
    {
        *ptr = rsrc.AllocateSysMemory(gpu, static_cast<size_t>(size));
    }
    if(*ptr)
    {
        hsa_amd_memory_fill(*ptr, 0u, size / sizeof(uint32_t));
    }
    return *ptr ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

void
new_tests_mem_dealloc(void* ptr, void* /*userdata*/)
{
    HsaRsrcFactory::FreeMemory(ptr);
}

hsa_status_t
new_tests_mem_copy(void* dst, const void* src, size_t size, void* /*userdata*/)
{
    if(size == 0) return HSA_STATUS_SUCCESS;
    return hsa_memory_copy(dst, src, size);
}

aqlprofile_agent_info_t
make_agent_info(const AgentInfo* gpu)
{
    aqlprofile_agent_info_t info{};
    info.agent_gfxip          = gpu->gfxip;
    info.xcc_num              = gpu->xcc_num;
    info.se_num               = gpu->se_num;
    info.cu_num               = gpu->cu_num;
    info.shader_arrays_per_se = gpu->shader_arrays_per_se;
    return info;
}

aqlprofile_agent_info_v1_t
make_agent_info_v1(const AgentInfo* gpu)
{
    aqlprofile_agent_info_v1_t info{};
    info.agent_gfxip          = gpu->gfxip;
    info.xcc_num              = gpu->xcc_num;
    info.se_num               = gpu->se_num;
    info.cu_num               = gpu->cu_num;
    info.shader_arrays_per_se = gpu->shader_arrays_per_se;
    info.domain               = gpu->domain;
    info.location_id          = gpu->bdf_id;
    return info;
}

bool
alloc_kernel_memory(HsaRsrcFactory*         rsrc,
                    const AgentInfo*        gpu,
                    SimpleConvolution&      kernel,
                    hsa_executable_symbol_t kern_desc)
{
    auto& mem_map = kernel.GetMemMap();
    for(auto& [buf_id, des] : mem_map)
    {
        switch(des.id)
        {
            case TestKernel::LOCAL_DES_ID:
                des.ptr = rsrc->AllocateLocalMemory(gpu, des.size);
                break;
            case TestKernel::KERNARG_DES_ID:
            {
                size_t actual_size = des.size;
                size_t info_size   = 0;
                hsa_executable_symbol_get_info(
                    kern_desc, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &info_size);
                if(des.size > info_size) actual_size = info_size;
                des.size = static_cast<uint32_t>(actual_size);
                des.ptr  = rsrc->AllocateKernArgMemory(gpu, actual_size);
                if(des.ptr) memset(des.ptr, 0, actual_size);
                break;
            }
            case TestKernel::SYS_DES_ID:
                des.ptr = rsrc->AllocateSysMemory(gpu, des.size);
                if(des.ptr) memset(des.ptr, 0, des.size);
                break;
            case TestKernel::NULL_DES_ID: des.ptr = nullptr; break;
            default: break;
        }
        if(des.ptr == nullptr && des.id != TestKernel::NULL_DES_ID)
        {
            std::cerr << "AQLPROFILE_NEW_TESTS: memory allocation failed" << std::endl;
            return false;
        }
    }
    kernel.Init();
    return true;
}

// Mirror alloc_kernel_memory
void
dealloc_kernel_memory(HsaRsrcFactory* rsrc, SimpleConvolution& kernel)
{
    for(auto& [buf_id, des] : kernel.GetMemMap())
    {
        switch(des.id)
        {
            case TestKernel::LOCAL_DES_ID:
            case TestKernel::KERNARG_DES_ID:
            case TestKernel::SYS_DES_ID:
                if(des.ptr) rsrc->FreeMemory(des.ptr);
                break;
            case TestKernel::NULL_DES_ID: break;
            default: break;
        }
    }
}

struct PmcCreateResult
{
    aqlprofile_handle_t          handle;
    aqlprofile_pmc_aql_packets_t packets;
    hsa_status_t                 status;
};

PmcCreateResult
create_pmc_packets(aqlprofile_agent_handle_t agent, const AgentInfo* gpu)
{
    PmcCreateResult              ret{};
    aqlprofile_pmc_event_flags_t ev_flags{};
    ev_flags.raw                        = 0;
    aqlprofile_pmc_event_t events_v2[1] = {};
    events_v2[0].block_index            = 0;
    events_v2[0].event_id               = 4;
    events_v2[0].flags                  = ev_flags;
    events_v2[0].block_name             = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;

    aqlprofile_pmc_profile_t profile{};
    profile.agent       = agent;
    profile.events      = events_v2;
    profile.event_count = 1;

    ret.status = aqlprofile_pmc_create_packets(&ret.handle,
                                               &ret.packets,
                                               profile,
                                               new_tests_mem_alloc,
                                               new_tests_mem_dealloc,
                                               new_tests_mem_copy,
                                               const_cast<AgentInfo*>(gpu));
    return ret;
}

}  // namespace

struct KernelSetupResult
{
    SimpleConvolution       conv;
    hsa_executable_t        hsa_exec;
    hsa_executable_symbol_t kernel_code_desc;
    uint32_t                group_segment_size;
    uint32_t                private_segment_size;
    uint64_t                code_handle;
};

bool
setup_kernel_and_properties(HsaRsrcFactory*     rsrc,
                            const AgentInfo*    gpu,
                            aqlprofile_handle_t pmc_handle,
                            hsa_signal_t        kernel_signal,
                            hsa_signal_t        prof_signal,
                            hsa_queue_t*        queue,
                            KernelSetupResult*  kernelSetupResult)
{
    std::string agent_name(gpu->name);
    size_t      colon = agent_name.find(':');
    if(colon != std::string::npos) agent_name.resize(colon);
    std::string hsaco_path     = agent_name + "_" + kernelSetupResult->conv.Name() + ".hsaco";
    char*       hsaco_fullpath = realpath(hsaco_path.c_str(), NULL);
    std::cout << "Loading hsaco: " << (hsaco_fullpath ? hsaco_fullpath : hsaco_path.c_str())
              << std::endl;
    free(hsaco_fullpath);

    if(!rsrc->LoadAndFinalize(gpu,
                              hsaco_path.c_str(),
                              kernelSetupResult->conv.SymbName().c_str(),
                              &kernelSetupResult->hsa_exec,
                              &kernelSetupResult->kernel_code_desc))
    {
        std::cerr << "AQLPROFILE_NEW_TESTS: LoadAndFinalize failed for " << hsaco_path << std::endl;
        return false;
    }

    if(!alloc_kernel_memory(
           rsrc, gpu, kernelSetupResult->conv, kernelSetupResult->kernel_code_desc))
    {
        hsa_executable_destroy(kernelSetupResult->hsa_exec);
        return false;
    }

    hsa_executable_symbol_get_info(kernelSetupResult->kernel_code_desc,
                                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                   &kernelSetupResult->group_segment_size);
    hsa_executable_symbol_get_info(kernelSetupResult->kernel_code_desc,
                                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                   &kernelSetupResult->private_segment_size);
    hsa_executable_symbol_get_info(kernelSetupResult->kernel_code_desc,
                                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                   &kernelSetupResult->code_handle);
    return true;
}

hsa_kernel_dispatch_packet_t
build_kernel_dispatch_packet(KernelSetupResult& kr, hsa_signal_t kernel_signal)
{
    const uint32_t               work_group_size = 64;
    const uint32_t               work_grid_size  = kr.conv.GetGridSize();
    hsa_kernel_dispatch_packet_t aql{};
    memset(&aql, 0, sizeof(aql));
    aql.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    aql.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    aql.setup                = 1;
    aql.grid_size_x          = work_grid_size;
    aql.grid_size_y          = 1;
    aql.grid_size_z          = 1;
    aql.workgroup_size_x     = work_group_size;
    aql.workgroup_size_y     = 1;
    aql.workgroup_size_z     = 1;
    aql.kernel_object        = kr.code_handle;
    aql.kernarg_address      = kr.conv.GetKernargPtr();
    aql.group_segment_size   = kr.group_segment_size;
    aql.private_segment_size = kr.private_segment_size;
    hsa_signal_store_relaxed(kernel_signal, 1);
    aql.completion_signal = kernel_signal;
    return aql;
}

// memcpy packet contents to the queue.
// Attach HSA signal to the packet.
// Ring doorbell by incrementing the queue write_index
// Wait for completion by watching the packet signal
void
explicit_submit(const void* packet, hsa_signal_t signal, hsa_queue_t* queue)
{
    constexpr uint32_t           slot_size_b     = HsaRsrcFactory::CMD_SLOT_SIZE_B;
    constexpr hsa_signal_value_t signal_init_val = 1;
    hsa_signal_store_relaxed(signal, signal_init_val);
    const uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    while((write_idx - hsa_queue_load_read_index_relaxed(queue)) >= queue->size)
        std::this_thread::yield();
    const uint32_t slot_idx   = static_cast<uint32_t>(write_idx % queue->size);
    uint32_t*      queue_slot = reinterpret_cast<uint32_t*>(
        reinterpret_cast<uintptr_t>(queue->base_address) + (slot_idx * slot_size_b));
    const uint32_t* slot_data = reinterpret_cast<const uint32_t*>(packet);
    memcpy(&queue_slot[1], &slot_data[1], slot_size_b - sizeof(uint32_t));
    std::atomic<uint32_t>* header = reinterpret_cast<std::atomic<uint32_t>*>(&queue_slot[0]);
    header->store(slot_data[0], std::memory_order_release);
    hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);
    hsa_signal_wait_scacquire(
        signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    return;
};

void
submit_with_signal(hsa_ext_amd_aql_pm4_packet_t* pkt, hsa_signal_t sig, hsa_queue_t* q)
{
    pkt->header            = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    pkt->completion_signal = sig;
    explicit_submit(pkt, sig, q);
    constexpr hsa_signal_value_t signal_init_val = 1;
    hsa_signal_store_relaxed(sig, signal_init_val);
}

char**
pmc_argv(unsigned argc, const hsa_ven_amd_aqlprofile_event_t* events)
{
    const int       argv_pmc_size = 32;
    static unsigned argc_pmc      = 0;
    static char*    argv_arr      = NULL;
    static char**   argv_pmc      = NULL;

    if(argc > argc_pmc)
    {
        argc_pmc = argc;
        argv_arr = reinterpret_cast<char*>(realloc(argv_arr, argc_pmc * argv_pmc_size));
        if(argv_pmc) delete argv_pmc;
        argv_pmc = new char*[argc + 1];
    }
    for(unsigned i = 0; i < argc; ++i)
    {
        char* argv_ptr = argv_arr + (i * argv_pmc_size);
        snprintf(argv_ptr,
                 argv_pmc_size,
                 "%d:%d:%d",
                 events[i].block_name,
                 events[i].block_index,
                 events[i].counter_id);
        argv_pmc[i] = argv_ptr;
    }
    argv_pmc[argc] = NULL;
    return argv_pmc;
}

// --- Collect and print profiling results ---
struct PmcResult
{
    uint32_t block_name;
    uint32_t block_index;
    uint32_t event_id;
    uint64_t value;
};

bool
run_new_test_flow()
{
    HsaRsrcFactory*  rsrc = &HsaRsrcFactory::Instance();
    const AgentInfo* gpu  = nullptr;
    if(!rsrc->GetGpuAgentInfo(0, &gpu) || gpu == nullptr)
    {
        std::cerr << "AQLPROFILE_NEW_TESTS: GPU agent not available" << std::endl;
        HsaRsrcFactory::Destroy();
        return 1;
    }

    // register agent.
    aqlprofile_agent_info_v1_t agent_info_v1 = make_agent_info_v1(gpu);
    aqlprofile_agent_handle_t  agent_v1{};
    if(aqlprofile_register_agent_info(&agent_v1, &agent_info_v1, AQLPROFILE_AGENT_VERSION_V1) !=
       HSA_STATUS_SUCCESS)
    {
        std::cerr << "AQLPROFILE_NEW_TESTS: aqlprofile_register_agent_info failed" << std::endl;
        HsaRsrcFactory::Destroy();
        return 1;
    }

    // Create a queue.
    hsa_queue_t* queue = nullptr;
    if(!rsrc->CreateQueue(gpu, 1024, &queue))
    {
        std::cerr << "AQLPROFILE_NEW_TESTS: CreateQueue failed" << std::endl;
        HsaRsrcFactory::Destroy();
        return 1;
    }

    auto         pmc        = create_pmc_packets(agent_v1, gpu);
    auto&        pmc_handle = pmc.handle;
    auto&        packets    = pmc.packets;
    hsa_status_t st         = pmc.status;

    // --- Create completion signals ---
    hsa_signal_t kernel_signal{}, prof_signal{};
    hsa_signal_create(1, 0, nullptr, &kernel_signal);
    hsa_signal_create(1, 0, nullptr, &prof_signal);

    if(st == HSA_STATUS_SUCCESS)
    {
        KernelSetupResult kr{};
        if(!setup_kernel_and_properties(
               rsrc, gpu, pmc_handle, kernel_signal, prof_signal, queue, &kr))
        {
            aqlprofile_pmc_delete_packets(pmc_handle);
            hsa_signal_destroy(kernel_signal);
            hsa_signal_destroy(prof_signal);
            hsa_queue_destroy(queue);
            HsaRsrcFactory::Destroy();
            return 1;
        }

        // --- Build kernel dispatch AQL packet ---
        hsa_kernel_dispatch_packet_t aql = build_kernel_dispatch_packet(kr, kernel_signal);

        // --- Submit profiling start packet ---
        std::vector<PmcResult> pmc_results;
        auto                   pmc_data_cb = [](aqlprofile_pmc_event_t ev,
                              uint64_t /*counter_id*/,
                              uint64_t val,
                              void*    userdata) -> hsa_status_t {
            static_cast<std::vector<PmcResult>*>(userdata)->push_back(
                {static_cast<uint32_t>(ev.block_name), ev.block_index, ev.event_id, val});
            return HSA_STATUS_SUCCESS;
        };
        aqlprofile_pmc_iterate_data(pmc_handle, pmc_data_cb, &pmc_results);

        auto total = 0;
        for(auto& pmc_result : pmc_results)
        {
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
        for(auto& pmc_result : pmc_results)
        {
            std::cout << "  block=" << pmc_result.block_name << " idx=" << pmc_result.block_index
                      << " event=" << pmc_result.event_id << " value=" << pmc_result.value
                      << std::endl;
            total1 += pmc_result.value;
        }
        std::cout << "===================================================" << std::endl;
        std::cout << "total1 = " << total1 << std::endl;
        TEST_ASSERT(total1 > 0);

        // Allocate host memory, copy GPU result, and verify against reference
        void* output = rsrc->AllocateSysMemory(gpu, kr.conv.GetOutputSize());
        if(rsrc->Memcpy(gpu, output, kr.conv.GetOutputPtr(), kr.conv.GetOutputSize()))
        {
            bool pass = (memcmp(output, kr.conv.GetRefOut(), kr.conv.GetOutputSize()) == 0);
            std::cout << ">>> Verification: " << (pass ? "PASS" : "FAIL") << std::endl;
            if(!pass) kr.conv.PrintOutput(output);
        }
        else
        {
            std::cerr << "AQLPROFILE_NEW_TESTS: Memcpy for verify failed" << std::endl;
        }
        rsrc->FreeMemory(output);

        // --- Cleanup kernel resources ---
        dealloc_kernel_memory(rsrc, kr.conv);
        hsa_executable_destroy(kr.hsa_exec);
        aqlprofile_pmc_delete_packets(pmc_handle);
    }
    else
    {
        std::cerr << "AQLPROFILE_NEW_TESTS: aqlprofile_pmc_create_packets failed" << std::endl;
    }
    hsa_signal_destroy(kernel_signal);
    hsa_signal_destroy(prof_signal);

    hsa_queue_destroy(queue);
    HsaRsrcFactory::Destroy();
    return (st == HSA_STATUS_SUCCESS) ? 0 : 1;
}

int
main(int argc, char* argv[])
{
    bool       ret_val      = false;
    const bool trace_enable = (getenv("AQLPROFILE_TRACE") != NULL);
    const bool scan_enable  = (getenv("AQLPROFILE_SCAN") != NULL);

    if(scan_enable)
    {
        std::cerr.rdbuf(NULL);
    }

    hsa_init();
    HsaRsrcFactory::Instance();
    const hsa_ven_amd_aqlprofile_event_t* events_arr;

    auto res = run_new_test_flow();
    HsaRsrcFactory::Destroy();
    hsa_shut_down();
    return res;
}
