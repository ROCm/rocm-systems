// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "common/filesystem.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace fs = common::fs;

#define RET_IF_HSA_ERR(err)                                                                        \
    {                                                                                              \
        if((err) != HSA_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            char  err_val[12];                                                                     \
            char* err_str = nullptr;                                                               \
            if(hsa_status_string(err, (const char**) &err_str) != HSA_STATUS_SUCCESS)              \
            {                                                                                      \
                sprintf(&(err_val[0]), "%#x", (uint32_t) err);                                     \
                err_str = &(err_val[0]);                                                           \
            }                                                                                      \
            printf("hsa api call failure at: %s:%d\n", __FILE__, __LINE__);                        \
            printf("Call returned %s\n", err_str);                                                 \
            abort();                                                                               \
        }                                                                                          \
    }

struct Device
{
    struct Memory
    {
        hsa_amd_memory_pool_t pool;
        bool                  fine;
        bool                  kernarg;
        size_t                size;
        size_t                granule;
    };

    hsa_agent_t                     agent;
    char                            name[64];
    std::vector<Memory>             pools;
    uint32_t                        fine;
    uint32_t                        coarse;
    static std::vector<hsa_agent_t> all_devices;
};

class MQDependencyTest
{
public:
    MQDependencyTest() { hsa_init(); }
    ~MQDependencyTest() { hsa_shut_down(); }

    static std::vector<Device> cpu;
    static std::vector<Device> gpu;
    static Device::Memory      kernarg;

    struct CodeObject
    {
        hsa_file_t               file         = 0;
        hsa_code_object_reader_t code_obj_rdr = {};
        hsa_executable_t         executable   = {};
    };

    struct Kernel
    {
        uint64_t handle        = 0;
        uint32_t scratch       = 0;
        uint32_t group         = 0;
        uint32_t kernarg_size  = 0;
        uint32_t kernarg_align = 0;
    };

    union AqlHeader
    {
        struct
        {
            uint16_t type     : 8;
            uint16_t barrier  : 1;
            uint16_t acquire  : 2;
            uint16_t release  : 2;
            uint16_t reserved : 3;
        };
        uint16_t raw = 0;
    };

    struct BarrierValue
    {
        AqlHeader          header            = {};
        uint8_t            AmdFormat         = 0;
        uint8_t            reserved          = 0;
        uint32_t           reserved1         = 0;
        hsa_signal_t       signal            = {};
        hsa_signal_value_t value             = 0;
        hsa_signal_value_t mask              = 0;
        uint32_t           cond              = 0;
        uint32_t           reserved2         = 0;
        uint64_t           reserved3         = 0;
        uint64_t           reserved4         = 0;
        hsa_signal_t       completion_signal = {};
    };

    union Aql
    {
        AqlHeader                    header;
        hsa_kernel_dispatch_packet_t dispatch;
        hsa_barrier_and_packet_t     barrier_and;
        hsa_barrier_or_packet_t      barrier_or;
        BarrierValue                 barrier_value = {};
    };

    struct OCLHiddenArgs
    {
        uint64_t offset_x      = 0;
        uint64_t offset_y      = 0;
        uint64_t offset_z      = 0;
        void*    printf_buffer = nullptr;
        void*    enqueue       = nullptr;
        void*    enqueue2      = nullptr;
        void*    multi_grid    = nullptr;
    };

    static bool load_code_object(const std::string& filename,
                                 hsa_agent_t        agent,
                                 CodeObject&        code_object)
    {
        hsa_status_t err;
        code_object.file = open(filename.c_str(), O_RDONLY);
        if(code_object.file == -1)
        {
            fprintf(stderr, "%s:%s\n", "Could not load code object", filename.c_str());
            abort();
            return false;
        }

        err = hsa_code_object_reader_create_from_file(code_object.file, &code_object.code_obj_rdr);
        RET_IF_HSA_ERR(err);

        err = hsa_executable_create_alt(HSA_PROFILE_FULL,
                                        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                        nullptr,
                                        &code_object.executable);
        RET_IF_HSA_ERR(err);

        err = hsa_executable_load_agent_code_object(
            code_object.executable, agent, code_object.code_obj_rdr, nullptr, nullptr);
        if(err != HSA_STATUS_SUCCESS) return false;

        err = hsa_executable_freeze(code_object.executable, nullptr);
        RET_IF_HSA_ERR(err);

        return true;
    }

    static bool get_kernel(const CodeObject&  code_object,
                           const std::string& kernel,
                           hsa_agent_t        agent,
                           Kernel&            kern)
    {
        hsa_executable_symbol_t symbol;
        hsa_status_t            err = hsa_executable_get_symbol_by_name(
            code_object.executable, kernel.c_str(), &agent, &symbol);
        if(err != HSA_STATUS_SUCCESS)
        {
            err = hsa_executable_get_symbol_by_name(
                code_object.executable, (kernel + ".kd").c_str(), &agent, &symbol);
            if(err != HSA_STATUS_SUCCESS)
            {
                return false;
            }
        }
        printf("\nkernel-name: %s\n", kernel.c_str());
        err = hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kern.handle);
        RET_IF_HSA_ERR(err);

        return true;
    }

    // Not for parallel insertion.
    static bool submit_packet(hsa_queue_t* queue, Aql& pkt)
    {
        size_t mask = queue->size - 1;
        Aql*   ring = static_cast<Aql*>(queue->base_address);

        uint64_t write = hsa_queue_load_write_index_relaxed(queue);
        uint64_t read  = hsa_queue_load_read_index_relaxed(queue);
        if(write - read + 1 > queue->size) return false;

        Aql& dst = ring[write & mask];

        uint16_t header = pkt.header.raw;
        pkt.header.raw  = dst.header.raw;
        dst             = pkt;
        __atomic_store_n(&dst.header.raw, header, __ATOMIC_RELEASE);
        pkt.header.raw = header;

        hsa_queue_store_write_index_release(queue, write + 1);
        hsa_signal_store_screlease(queue->doorbell_signal, write);

        return true;
    }

    static void* hsa_malloc(size_t size, const Device::Memory& mem)
    {
#define LOCAL_HSA_AMD_INTERFACE_VERSION                                                            \
    (10000 * HSA_AMD_INTERFACE_VERSION_MAJOR) + (100 * HSA_AMD_INTERFACE_VERSION_MINOR)

#if LOCAL_HSA_AMD_INTERFACE_VERSION >= 10700
        constexpr auto hsa_amd_memory_pool_executable_flag = HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG;
#elif LOCAL_HSA_AMD_INTERFACE_VERSION == 10600
        constexpr auto hsa_amd_memory_pool_executable_flag = (1 << 2);
#else
        constexpr auto hsa_amd_memory_pool_executable_flag = 0;
#endif

        void*        ret;
        hsa_status_t err =
            hsa_amd_memory_pool_allocate(mem.pool, size, hsa_amd_memory_pool_executable_flag, &ret);
        RET_IF_HSA_ERR(err);

        err = hsa_amd_agents_allow_access(
            Device::all_devices.size(), Device::all_devices.data(), nullptr, ret);
        RET_IF_HSA_ERR(err);
        return ret;
    }

    static void* hsa_malloc(size_t size, const Device& dev, bool fine)
    {
        uint32_t index = fine ? dev.fine : dev.coarse;
        assert(index != -1u && "Memory type unavailable.");
        return hsa_malloc(size, dev.pools[index]);
    }

    static bool device_discovery()
    {
        hsa_status_t err;

        err = hsa_iterate_agents(
            [](hsa_agent_t agent, void*) {
                hsa_status_t error;

                Device dev;
                dev.agent = agent;

                dev.fine   = -1u;
                dev.coarse = -1u;

                error = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, dev.name);
                RET_IF_HSA_ERR(error)

                hsa_device_type_t type;
                error = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
                RET_IF_HSA_ERR(error)

                error = hsa_amd_agent_iterate_memory_pools(
                    agent,
                    [](hsa_amd_memory_pool_t pool, void* data) {
                        auto&        pools = *reinterpret_cast<std::vector<Device::Memory>*>(data);
                        hsa_status_t status;

                        bool allowed = false;
                        status       = hsa_amd_memory_pool_get_info(
                            pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allowed);

                        if(!allowed) return HSA_STATUS_SUCCESS;

                        hsa_amd_segment_t segment;
                        status = hsa_amd_memory_pool_get_info(
                            pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
                        RET_IF_HSA_ERR(status)

                        if(segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

                        uint32_t flags;
                        status = hsa_amd_memory_pool_get_info(
                            pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
                        RET_IF_HSA_ERR(status)

                        Device::Memory mem;
                        mem.pool = pool;
                        mem.fine = ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) != 0u);
                        mem.kernarg =
                            ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0u);

                        status = hsa_amd_memory_pool_get_info(
                            pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &mem.size);
                        RET_IF_HSA_ERR(status)

                        status = hsa_amd_memory_pool_get_info(
                            pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &mem.granule);
                        RET_IF_HSA_ERR(status)

                        pools.push_back(mem);
                        return HSA_STATUS_SUCCESS;
                    },
                    static_cast<void*>(&dev.pools));

                if(!dev.pools.empty())
                {
                    for(size_t i = 0; i < dev.pools.size(); i++)
                    {
                        if(dev.pools[i].fine && dev.pools[i].kernarg && dev.fine == -1u)
                            dev.fine = i;
                        if(dev.pools[i].fine && !dev.pools[i].kernarg) dev.fine = i;
                        if(!dev.pools[i].fine) dev.coarse = i;
                    }

                    if(type == HSA_DEVICE_TYPE_CPU)
                        cpu.push_back(dev);
                    else
                        gpu.push_back(dev);

                    Device::all_devices.push_back(dev.agent);
                }

                return HSA_STATUS_SUCCESS;
            },
            nullptr);

        []() {
            for(auto& dev : cpu)
            {
                for(auto& mem : dev.pools)
                {
                    if(mem.fine && mem.kernarg)
                    {
                        kernarg = mem;
                        return;
                    }
                }
            }
        }();
        RET_IF_HSA_ERR(err);

        if(cpu.empty() || gpu.empty() || kernarg.pool.handle == 0) return false;
        return true;
    }

    void search_hasco(const fs::path& directory, std::string& filename)
    {
        for(const auto& entry : fs::directory_iterator(directory))
        {
            if(fs::is_regular_file(entry))
            {
                if(entry.path().filename() == filename)
                {
                    filename = entry.path();
                }
            }
            else if(fs::is_directory(entry))
            {
                search_hasco(entry, filename);  // Recursive call for subdirectories
            }
        }
    }
};
