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

#include <glog/logging.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <unordered_map>

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

std::mutex                                             kernel_descriptor_name_map_mutex;
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
SetKernelDescriptorName(rocprofiler_address_t kernel_descriptor, const char* kernel_name)
{
    std::lock_guard<std::mutex> kernel_descriptor_name_map_lock(kernel_descriptor_name_map_mutex);
    kernel_descriptor_name_map[kernel_descriptor] = kernel_name;
}

void
SetKernelProperties(uint64_t correlation_id, rocprofiler_tool_kernel_properties_t kernel_properties)
{
    std::lock_guard<std::mutex> kernel_properties_correlation_map_lock(
        kernel_properties_correlation_mutex);
    kernel_properties_correlation_map[correlation_id] = kernel_properties;
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
const char*
GetKernelDescriptorName(rocprofiler_address_t kernel_descriptor)
{
    std::lock_guard<std::mutex> kernel_descriptor_name_map_lock(kernel_descriptor_name_map_mutex);
    auto                        it = kernel_descriptor_name_map.find(kernel_descriptor);
    if(it == kernel_descriptor_name_map.end())
    {
        std::cout << "kernel name not found" << std::endl;
        abort();
    }
    return it->second;
}

std::vector<std::string>
GetCounterNames()
{
    std::vector<std::string> counters;
    const char*              line_c_str = getenv("ROCPROFILER_COUNTERS");
    if(line_c_str)
    {
        std::string line = line_c_str;
        // skip commented lines
        auto found = line.find_first_not_of(" \t");
        if(found != std::string::npos)
        {
            if(line[found] == '#') return {};
        }
        if(line.find("pmc") == std::string::npos) return counters;
        char                   seperator = ' ';
        std::string::size_type prev_pos = 0, pos = line.find(seperator, prev_pos);
        prev_pos = ++pos;
        if(pos != std::string::npos)
        {
            while((pos = line.find(seperator, pos)) != std::string::npos)
            {
                std::string substring(line.substr(prev_pos, pos - prev_pos));
                if(substring.length() > 0 && substring != ":")
                {
                    counters.push_back(substring);
                }
                prev_pos = ++pos;
            }
            if(!line.substr(prev_pos, pos - prev_pos).empty())
            {
                counters.push_back(line.substr(prev_pos, pos - prev_pos));
            }
        }
    }
    return counters;
}

void
populate_kernel_properties_data(rocprofiler_tool_kernel_properties_t* kernel_properties,
                                const hsa_kernel_dispatch_packet_t    dispatch_packet)
{
    const uint64_t kernel_object = dispatch_packet.kernel_object;

    const kernel_descriptor_t* kernel_code = GetKernelCode(kernel_object);
    uint64_t                   grid_size =
        dispatch_packet.grid_size_x * dispatch_packet.grid_size_y * dispatch_packet.grid_size_z;
    if(grid_size > UINT32_MAX) abort();
    kernel_properties->grid_size = grid_size;
    uint64_t workgroup_size = dispatch_packet.workgroup_size_x * dispatch_packet.workgroup_size_y *
                              dispatch_packet.workgroup_size_z;
    if(workgroup_size > UINT32_MAX) abort();
    kernel_properties->workgroup_size = (uint32_t) workgroup_size;
    kernel_properties->lds_size       = dispatch_packet.group_segment_size;
    kernel_properties->scratch_size   = dispatch_packet.private_segment_size;
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
    kernel_properties->signal_handle = dispatch_packet.completion_signal.handle;
}

std::string
cxa_demangle(std::string_view _mangled_name, int* _status)
{
    constexpr size_t buffer_len = 4096;
    // return the mangled since there is no buffer
    if(_mangled_name.empty())
    {
        *_status = -2;
        return std::string{};
    }

    auto _demangled_name = std::string{_mangled_name};

    // PARAMETERS to __cxa_demangle
    //  mangled_name:
    //      A nullptr-terminated character string containing the name to be demangled.
    //  buffer:
    //      A region of memory, allocated with malloc, of *length bytes, into which the
    //      demangled name is stored. If output_buffer is not long enough, it is expanded
    //      using realloc. output_buffer may instead be nullptr; in that case, the demangled
    //      name is placed in a region of memory allocated with malloc.
    //  _buflen:
    //      If length is non-nullptr, the length of the buffer containing the demangled name
    //      is placed in *length.
    //  status:
    //      *status is set to one of the following values
    size_t _demang_len = 0;
    char*  _demang = abi::__cxa_demangle(_demangled_name.c_str(), nullptr, &_demang_len, _status);
    switch(*_status)
    {
        //  0 : The demangling operation succeeded.
        // -1 : A memory allocation failure occurred.
        // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
        // -3 : One of the arguments is invalid.
        case 0:
        {
            if(_demang) _demangled_name = std::string{_demang};
            break;
        }
        case -1:
        {
            char _msg[buffer_len];
            ::memset(_msg, '\0', buffer_len * sizeof(char));
            ::snprintf(_msg,
                       buffer_len,
                       "memory allocation failure occurred demangling %s",
                       _demangled_name.c_str());
            ::perror(_msg);
            break;
        }
        case -2: break;
        case -3:
        {
            char _msg[buffer_len];
            ::memset(_msg, '\0', buffer_len * sizeof(char));
            ::snprintf(_msg,
                       buffer_len,
                       "Invalid argument in: (\"%s\", nullptr, nullptr, %p)",
                       _demangled_name.c_str(),
                       (void*) _status);
            ::perror(_msg);
            break;
        }
        default: break;
    };

    // if it "demangled" but the length is zero, set the status to -2
    if(_demang_len == 0 && *_status == 0) *_status = -2;

    // free allocated buffer
    ::free(_demang);
    return _demangled_name;
}
