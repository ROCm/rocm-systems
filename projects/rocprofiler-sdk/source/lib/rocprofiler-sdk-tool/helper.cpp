// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "helper.hpp"
#include "config.hpp"
#include "rocprofiler-sdk/fwd.h"

#include <glog/logging.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace
{
using amd_compute_pgm_rsrc_three32_t = uint32_t;

// AMD Compute Program Resource Register Three.
enum amd_compute_gfx9_pgm_rsrc_three_t
{
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_ACCUM_OFFSET, 0, 5),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TG_SPLIT, 16, 1)
};

enum amd_compute_gfx10_gfx11_pgm_rsrc_three_t
{
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_SHARED_VGPR_COUNT, 0, 4),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_INST_PREF_SIZE, 4, 6),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TRAP_ON_START, 10, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TRAP_ON_END, 11, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_IMAGE_OP, 31, 1)
};

// Kernel code properties.
enum amd_kernel_code_property_t
{
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER,
                                     0,
                                     1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, 1, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, 2, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR,
                                     3,
                                     1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID, 4, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT, 5, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE,
                                     6,
                                     1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_RESERVED0, 7, 3),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                                     10,
                                     1),  // GFX10+
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK, 11, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_RESERVED1, 12, 4),
};

std::unordered_map<rocprofiler_address_t, const char*> kernel_descriptor_name_map;

std::mutex kernel_properties_correlation_mutex;
std::unordered_map<uint64_t, rocprofiler_tool_kernel_properties_t>
    kernel_properties_correlation_map;

uint32_t
arch_vgpr_count(const std::string_view& name, const kernel_descriptor_t& kernel_code)
{
    std::string info_name(name.data(), name.size());
    if(strcmp(name.data(), "gfx90a") == 0 || strncmp(name.data(), "gfx94", 5) == 0)
        return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc3,
                                 AMD_COMPUTE_PGM_RSRC_THREE_ACCUM_OFFSET) +
                1) *
               4;

    return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                             AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT) +
            1) *
           (AMD_HSA_BITS_GET(kernel_code.kernel_code_properties,
                             AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32)
                ? 8
                : 4);
}

uint32_t
accum_vgpr_count(const std::string_view& name, const kernel_descriptor_t& kernel_code)
{
    std::string info_name(name.data(), name.size());
    if(strcmp(info_name.c_str(), "gfx908") == 0) return arch_vgpr_count(name, kernel_code);
    if(strcmp(info_name.c_str(), "gfx90a") == 0 || strncmp(info_name.c_str(), "gfx94", 5) == 0)
        return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                                 AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT) +
                1) *
                   8 -
               arch_vgpr_count(name, kernel_code);

    return 0;
}

uint32_t
sgpr_count(const std::string_view& name, const kernel_descriptor_t& kernel_code)
{
    // GFX10 and later always allocate 128 sgprs.

    // TODO(srnagara): Recheck the extraction of gfxip from gpu name

    const char*  name_data       = name.data();
    const size_t gfxip_label_len = std::min(name.size() - 2, size_t{63});
    if(gfxip_label_len > 0 && strnlen(name_data, gfxip_label_len + 1) >= gfxip_label_len)
    {
        auto gfxip = std::vector<char>{};
        gfxip.resize(gfxip_label_len + 1, '\0');
        memcpy(gfxip.data(), name_data, gfxip_label_len);
        // TODO(srnagara): Check if it is hardcoded
        if(std::stoi(&gfxip.at(3)) >= 10) return 128;
        return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                                 AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT) /
                    2 +
                1) *
               16;
    }
    return 0;
}

const auto&
GetLoaderTable()
{
    static const auto _v = []() {
        using hsa_loader_table_t = hsa_ven_amd_loader_1_01_pfn_t;
        auto _tbl                = hsa_loader_table_t{};
        memset(&_tbl, 0, sizeof(hsa_loader_table_t));
        hsa_system_get_major_extension_table(
            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_loader_table_t), &_tbl);
        return _tbl;
    }();
    return _v;
}

const kernel_descriptor_t*
GetKernelCode(uint64_t kernel_object)
{
    const kernel_descriptor_t* kernel_code = nullptr;
    if(GetLoaderTable().hsa_ven_amd_loader_query_host_address == nullptr) return kernel_code;
    hsa_status_t status = GetLoaderTable().hsa_ven_amd_loader_query_host_address(
        reinterpret_cast<const void*>(kernel_object),  // NOLINT(performance-no-int-to-ptr)
        reinterpret_cast<const void**>(&kernel_code));
    if(HSA_STATUS_SUCCESS != status)
    {
        kernel_code = reinterpret_cast<kernel_descriptor_t*>(  // NOLINT(performance-no-int-to-ptr)
            kernel_object);
    }
    return kernel_code;
}
}  // namespace

void
SetKernelProperties(uint64_t correlation_id, rocprofiler_tool_kernel_properties_t kernel_properties)
{
    std::lock_guard<std::mutex> kernel_properties_correlation_map_lock(
        kernel_properties_correlation_mutex);
    kernel_properties_correlation_map[correlation_id] = std::move(kernel_properties);
}

rocprofiler_tool_kernel_properties_t
GetKernelProperties(uint64_t correlation_id)
{
    std::lock_guard<std::mutex> kernel_properties_correlation_map_lock(
        kernel_properties_correlation_mutex);
    auto it = kernel_properties_correlation_map.find(correlation_id);
    if(it == kernel_properties_correlation_map.end())
    {
        std::cout << "kernel properties not found" << std::endl;
        abort();
    }
    return it->second;
}

void
populate_kernel_properties_data(rocprofiler_tool_kernel_properties_t* kernel_properties,
                                const hsa_kernel_dispatch_packet_t*   dispatch_packet)
{
    const uint64_t kernel_object = dispatch_packet->kernel_object;

    const kernel_descriptor_t* kernel_code = GetKernelCode(kernel_object);
    uint64_t                   grid_size =
        dispatch_packet->grid_size_x * dispatch_packet->grid_size_y * dispatch_packet->grid_size_z;
    if(grid_size > UINT32_MAX) abort();
    kernel_properties->grid_size = grid_size;
    uint64_t workgroup_size      = dispatch_packet->workgroup_size_x *
                              dispatch_packet->workgroup_size_y * dispatch_packet->workgroup_size_z;
    if(workgroup_size > UINT32_MAX) abort();
    kernel_properties->workgroup_size = (uint32_t) workgroup_size;
    kernel_properties->lds_size       = dispatch_packet->group_segment_size;
    kernel_properties->scratch_size   = dispatch_packet->private_segment_size;
    kernel_properties->arch_vgpr_count =
        arch_vgpr_count(kernel_properties->gpu_agent.name, *kernel_code);
    kernel_properties->accum_vgpr_count =
        accum_vgpr_count(kernel_properties->gpu_agent.name, *kernel_code);
    kernel_properties->sgpr_count = sgpr_count(kernel_properties->gpu_agent.name, *kernel_code);
    kernel_properties->wave_size =
        AMD_HSA_BITS_GET(kernel_code->kernel_code_properties,
                         AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32)
            ? 32
            : 64;
    kernel_properties->signal_handle = dispatch_packet->completion_signal.handle;
}

rocprofiler_tool_buffer_name_info_t
get_buffer_id_names()
{
    static auto supported = std::unordered_set<rocprofiler_buffer_tracing_kind_t>{
        ROCPROFILER_BUFFER_TRACING_HSA_API,
        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
        ROCPROFILER_BUFFER_TRACING_MARKER_API};

    auto cb_name_info = rocprofiler_tool_buffer_name_info_t{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_buffer_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<rocprofiler_tool_buffer_name_info_t*>(data_v);

            if(supported.count(kindv) > 0)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query buffer failed");
                if(name) name_info_v->operation_names[kindv][operation] = name;
            }

            return 0;
        };

    //
    //  callback for each kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_buffer_tracing_kind_t kind, void* data) {
        //  store the buffer kind name
        auto*       name_info_v = static_cast<rocprofiler_tool_buffer_name_info_t*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_name(kind, &name, nullptr),
                         "query buffer failed");

        if(name) name_info_v->kind_names[kind] = name;

        if(supported.count(kind) > 0)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_buffer_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "query buffer failed");
        }

        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_buffer_tracing_kinds(tracing_kind_cb,
                                                              static_cast<void*>(&cb_name_info)),
                     "iterate_buffer failed");

    return cb_name_info;
}
