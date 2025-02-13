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

#include "lib/rocprofiler-sdk/code_object/hip/code_object.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <amd_comgr/amd_comgr.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace code_object
{
namespace hip
{
constexpr auto kernels_metadata_lookup       = "amdhsa.kernels";
constexpr auto kernel_name_metadata_lookup   = ".name";
constexpr auto kernel_symbol_metadata_lookup = ".symbol";

#define CHECK_RETURN_HSA(call)                                                                     \
    {                                                                                              \
        if(hsa_status_t status = (call); status != HSA_STATUS_SUCCESS)                             \
        {                                                                                          \
            const char* reason = "<unknown-error-reason>";                                         \
            if(rocprofiler::hsa::get_core_table())                                                 \
                rocprofiler::hsa::get_core_table()->hsa_status_string_fn(status, &reason);         \
            ROCP_INFO << #call << " returned error code " << status << " :: " << reason;           \
            return status;                                                                         \
        }                                                                                          \
    }

#define CHECK_WARNING_COMGR(call)                                                                  \
    if(amd_comgr_status_s status = (call); status != AMD_COMGR_STATUS_SUCCESS)                     \
    {                                                                                              \
        const char* reason = "";                                                                   \
        amd_comgr_status_string(status, &reason);                                                  \
        ROCP_INFO << #call << " failed with error code " << status << " :: " << reason;            \
    }

#define CHECK_WARNING_COMGR_EXT(call, ...)                                                         \
    if(amd_comgr_status_s status = (call); status != AMD_COMGR_STATUS_SUCCESS)                     \
    {                                                                                              \
        const char* reason = "";                                                                   \
        amd_comgr_status_string(status, &reason);                                                  \
        ROCP_INFO << #call << " failed with error code " << status << " :: " << reason             \
                  << " :: " << __VA_ARGS__;                                                        \
    }

#define CHECK_RETURN_COMGR(call)                                                                   \
    if(amd_comgr_status_s status = (call); status != AMD_COMGR_STATUS_SUCCESS)                     \
    {                                                                                              \
        const char* reason = "";                                                                   \
        amd_comgr_status_string(status, &reason);                                                  \
        ROCP_INFO << #call << " returned error code " << status << " :: " << reason;               \
        return AMD_COMGR_STATUS_ERROR;                                                             \
    }

#define CHECK_RETURN_COMGR_EXT(call, ...)                                                          \
    if(amd_comgr_status_s status = (call); status != AMD_COMGR_STATUS_SUCCESS)                     \
    {                                                                                              \
        const char* reason = "";                                                                   \
        amd_comgr_status_string(status, &reason);                                                  \
        ROCP_INFO << #call << " returned error code " << status << " :: " << reason                \
                  << " :: " << __VA_ARGS__;                                                        \
        return AMD_COMGR_STATUS_ERROR;                                                             \
    }

hsa_status_t
get_isa_info(hsa_isa_t isa, void* data)
{
    size_t name_len = 0;
    CHECK_RETURN_HSA(rocprofiler::hsa::get_core_table()->hsa_isa_get_info_alt_fn(
        isa, HSA_ISA_INFO_NAME_LENGTH, &name_len));

    ROCP_INFO << "isa name length: " << name_len;

    if(name_len > 0)
    {
        auto name = std::string(name_len, '\0');
        CHECK_RETURN_HSA(rocprofiler::hsa::get_core_table()->hsa_isa_get_info_alt_fn(
            isa, HSA_ISA_INFO_NAME, name.data()));
        name = name.substr(0, name.find_first_of('\0'));

        ROCP_INFO << "found isa: " << name;

        auto* info = static_cast<isa_names_t*>(data);
        CHECK_NOTNULL(info)->emplace_back(common::get_string_entry(name));
    }

    return HSA_STATUS_SUCCESS;
}

comgr_code_object_vec_t
get_isa_offsets(hsa_agent_t hsa_agent, const void* fat_bin)
{
    auto isas       = isa_names_t{};
    auto hsa_status = rocprofiler::hsa::get_core_table()->hsa_agent_iterate_isas_fn(
        hsa_agent, get_isa_info, &isas);

    if(isas.empty())
    {
        ROCP_INFO << "failed to get ISAs for agent-"
                  << CHECK_NOTNULL(agent::get_rocprofiler_agent(hsa_agent))->node_id
                  << " :: " << rocprofiler::hsa::get_hsa_status_string(hsa_status);
        return comgr_code_object_vec_t{};
    }

    auto query_list = comgr_code_object_vec_t{};
    for(auto& isa : isas)
        query_list.emplace_back(amd_comgr_code_object_info_t{isa->c_str(), 0, 0});

    auto data_object = amd_comgr_data_t{0};
    CHECK_WARNING_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &data_object));
    CHECK_WARNING_COMGR(
        amd_comgr_set_data(data_object, 4096, reinterpret_cast<const char*>(fat_bin)));
    CHECK_WARNING_COMGR(
        amd_comgr_lookup_code_object(data_object, query_list.data(), query_list.size()));
    CHECK_WARNING_COMGR(amd_comgr_release_data(data_object));

    return query_list;
}

amd_comgr_status_t
get_node_string(const amd_comgr_metadata_node_t& node, std::string* value)
{
    size_t size = 0;
    CHECK_RETURN_COMGR(amd_comgr_get_metadata_string(node, &size, nullptr));
    CHECK_NOTNULL(value)->resize(size, '\0');
    CHECK_RETURN_COMGR(amd_comgr_get_metadata_string(node, &size, value->data()));
    *value = value->substr(0, value->find_first_of('\0'));
    ROCP_INFO << "found node string: " << *value;
    return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t
get_device_name_kernel_symbols_mapping(const amd_comgr_metadata_node_t key,
                                       const amd_comgr_metadata_node_t value,
                                       void*                           data)
{
    std::string key_str{};
    CHECK_RETURN_COMGR(get_node_string(key, &key_str));
    if(key_str != kernel_symbol_metadata_lookup) return AMD_COMGR_STATUS_SUCCESS;

    // More meta data information can be extracted from binary image here
    std::string* kernel_symbol = static_cast<std::string*>(data);
    CHECK_RETURN_COMGR(get_node_string(value, kernel_symbol));
    return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t
get_kernels_meta_node(const amd_comgr_code_object_info_t& isa_offset,
                      const void*                         fat_bin,
                      amd_comgr_metadata_node_t*          kernels_metadata)
{
    auto binary_data = amd_comgr_data_t{0};
    CHECK_WARNING_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &binary_data));

    void* bin_offset = static_cast<char*>(const_cast<void*>(fat_bin)) + isa_offset.offset;
    CHECK_RETURN_COMGR_EXT(
        amd_comgr_set_data(binary_data, isa_offset.size, static_cast<const char*>(bin_offset)),
        "binary_data=" << binary_data.handle << ", isa=(" << isa_offset.isa << ", "
                       << isa_offset.size << ", " << isa_offset.offset << "), fat_bin=" << fat_bin);

    auto binary_metadata = amd_comgr_metadata_node_t{};
    CHECK_WARNING_COMGR(amd_comgr_get_data_metadata(binary_data, &binary_metadata));
    CHECK_WARNING_COMGR(
        amd_comgr_metadata_lookup(binary_metadata, kernels_metadata_lookup, kernels_metadata));

    return AMD_COMGR_STATUS_SUCCESS;
}

kernel_symbol_hip_device_map_t
get_kernel_symbol_device_name_map(const amd_comgr_code_object_info_t& isa_offset,
                                  const void*                         fat_bin)
{
    auto   kernel_sym_device_func_map = kernel_symbol_hip_device_map_t{};
    auto   kernels_metadata           = amd_comgr_metadata_node_t{0};
    size_t num_kernels{0};

    if(get_kernels_meta_node(isa_offset, CHECK_NOTNULL(fat_bin), &kernels_metadata) !=
       AMD_COMGR_STATUS_SUCCESS)
        return kernel_sym_device_func_map;

    CHECK_WARNING_COMGR(amd_comgr_get_metadata_list_size(kernels_metadata, &num_kernels));
    for(size_t i = 0; i < num_kernels; i++)
    {
        auto kernel_node      = amd_comgr_metadata_node_t{};
        auto kernel_name_meta = amd_comgr_metadata_node_t{};

        CHECK_WARNING_COMGR(amd_comgr_index_list_metadata(kernels_metadata, i, &kernel_node));
        CHECK_WARNING_COMGR(
            amd_comgr_metadata_lookup(kernel_node, kernel_name_metadata_lookup, &kernel_name_meta));

        auto kernel_meta_name = std::string{};
        if(get_node_string(kernel_name_meta, &kernel_meta_name) != AMD_COMGR_STATUS_SUCCESS ||
           kernel_meta_name.empty())
            continue;

        ROCP_INFO << "found kernel meta name: " << kernel_meta_name;

        auto kernel_symbol = std::string{};
        CHECK_WARNING_COMGR(amd_comgr_iterate_map_metadata(
            kernel_node, get_device_name_kernel_symbols_mapping, &kernel_symbol));
        if(!kernel_symbol.empty())
        {
            ROCP_INFO << "found kernel symbol mapping: " << kernel_symbol << " -> "
                      << kernel_meta_name;
            kernel_sym_device_func_map.emplace(kernel_symbol, kernel_meta_name);
        }
    }
    return kernel_sym_device_func_map;
}
}  // namespace hip
}  // namespace code_object
}  // namespace rocprofiler
