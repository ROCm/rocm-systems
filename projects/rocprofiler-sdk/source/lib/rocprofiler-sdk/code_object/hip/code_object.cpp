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
#include "lib/common/scope_destructor.hpp"
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

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <optional>
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

namespace
{
bool
comgr_success(amd_comgr_status_s status, const char* call)
{
    if(status == AMD_COMGR_STATUS_SUCCESS) return true;

    const char* reason = "";
    amd_comgr_status_string(status, &reason);
    ROCP_INFO << call << " failed with error code " << status << " :: " << reason;
    return false;
}

void
destroy_metadata(amd_comgr_metadata_node_t metadata)
{
    if(metadata.handle != 0) CHECK_WARNING_COMGR(amd_comgr_destroy_metadata(metadata));
}

void
release_data(amd_comgr_data_t data)
{
    if(data.handle != 0) CHECK_WARNING_COMGR(amd_comgr_release_data(data));
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

namespace
{
constexpr auto kernel_args_metadata_lookup           = ".args";
constexpr auto kernel_arg_offset_metadata_lookup     = ".offset";
constexpr auto kernel_arg_size_metadata_lookup       = ".size";
constexpr auto kernel_arg_value_kind_metadata_lookup = ".value_kind";
constexpr auto kernel_arg_segment_metadata_lookup    = ".kernarg_segment_size";
constexpr auto hidden_arg_value_kind_prefix          = "hidden";
constexpr auto global_buffer_arg_value_kind          = "global_buffer";
constexpr auto global_buffer_static_arg_value_kind   = "global_buffer_static";
constexpr auto by_value_arg_value_kind               = "by_value";

// A by-value argument this small cannot be told apart from an integer or floating-point scalar, and
// a scalar whose bits happen to look like an address must not be mistaken for one.
constexpr uint64_t by_value_struct_min_size = 16;

// The size of an argument holding an address.
constexpr uint64_t pointer_arg_size = 8;

// The value of a metadata key. COMGR renders numeric metadata as text, so this reads strings and
// numbers alike; nullopt when the key is absent or does not hold text.
std::optional<std::string>
metadata_string(amd_comgr_metadata_node_t node, const char* key)
{
    auto value = amd_comgr_metadata_node_t{};
    if(amd_comgr_metadata_lookup(node, key, &value) != AMD_COMGR_STATUS_SUCCESS)
        return std::nullopt;
    auto value_fini = common::scope_destructor{[value]() { destroy_metadata(value); }};

    auto out = std::string{};
    if(get_node_string(value, &out) != AMD_COMGR_STATUS_SUCCESS) return std::nullopt;
    return out;
}

std::optional<uint64_t>
metadata_uint64(amd_comgr_metadata_node_t node, const char* key)
{
    const auto text = metadata_string(node, key);
    if(!text || text->empty()) return std::nullopt;

    char*      end   = nullptr;
    const auto value = static_cast<uint64_t>(std::strtoull(text->c_str(), &end, 10));
    if(end == text->c_str() || *end != '\0') return std::nullopt;
    return value;
}

// The ranges of one kernel's kernarg segment that can hold a pointer the kernel writes through:
// its global_buffer arguments and the by-value arguments large enough to be a struct. When an
// argument is not described well enough to tell, every byte the kernel declares itself (everything
// before the hidden arguments the runtime appends) is a candidate. nullopt when the kernel has no
// argument list to read.
std::optional<std::vector<kernarg_region_t>>
get_kernel_pointer_regions(amd_comgr_metadata_node_t kernel_node)
{
    const auto segment_size = metadata_uint64(kernel_node, kernel_arg_segment_metadata_lookup);
    if(!segment_size || *segment_size > std::numeric_limits<uint32_t>::max()) return std::nullopt;

    auto args = amd_comgr_metadata_node_t{};
    if(amd_comgr_metadata_lookup(kernel_node, kernel_args_metadata_lookup, &args) !=
       AMD_COMGR_STATUS_SUCCESS)
        return std::nullopt;
    auto args_fini = common::scope_destructor{[args]() { destroy_metadata(args); }};

    auto num_args = size_t{0};
    if(amd_comgr_get_metadata_list_size(args, &num_args) != AMD_COMGR_STATUS_SUCCESS)
        return std::nullopt;

    auto regions      = std::vector<kernarg_region_t>{};
    auto hidden_start = *segment_size;
    auto described    = true;
    for(size_t i = 0; i < num_args; i++)
    {
        auto arg = amd_comgr_metadata_node_t{};
        if(amd_comgr_index_list_metadata(args, i, &arg) != AMD_COMGR_STATUS_SUCCESS)
        {
            described = false;
            continue;
        }
        auto arg_fini = common::scope_destructor{[arg]() { destroy_metadata(arg); }};

        const auto value_kind = metadata_string(arg, kernel_arg_value_kind_metadata_lookup);
        const auto offset     = metadata_uint64(arg, kernel_arg_offset_metadata_lookup);
        const auto size       = metadata_uint64(arg, kernel_arg_size_metadata_lookup);
        if(!value_kind || !offset || !size || *offset + *size > *segment_size)
        {
            described = false;
            continue;
        }

        if(value_kind->rfind(hidden_arg_value_kind_prefix, 0) == 0)
            hidden_start = std::min(hidden_start, *offset);
        else if(*value_kind == global_buffer_arg_value_kind ||
                *value_kind == global_buffer_static_arg_value_kind)
            regions.push_back(kernarg_region_t{static_cast<uint32_t>(*offset),
                                               static_cast<uint32_t>(pointer_arg_size)});
        else if(*value_kind == by_value_arg_value_kind)
        {
            if(*size >= by_value_struct_min_size)
                regions.push_back(
                    kernarg_region_t{static_cast<uint32_t>(*offset), static_cast<uint32_t>(*size)});
        }
        else
            described = false;
    }

    if(!described) return std::vector<kernarg_region_t>{{0u, static_cast<uint32_t>(hidden_start)}};
    return regions;
}
}  // namespace

// Everything this library derives from the amdhsa.kernels list of one code object, read in a
// single pass: the kernel-symbol-to-device-function map, and which kernarg bytes of each kernel can
// hold a pointer.
code_object_metadata_t
get_metadata_from_kernels_node(amd_comgr_metadata_node_t kernels_metadata)
{
    auto   metadata    = code_object_metadata_t{};
    size_t num_kernels = 0;

    if(!comgr_success(amd_comgr_get_metadata_list_size(kernels_metadata, &num_kernels),
                      "amd_comgr_get_metadata_list_size"))
        return metadata;

    for(size_t i = 0; i < num_kernels; i++)
    {
        auto kernel_node = amd_comgr_metadata_node_t{};
        if(!comgr_success(amd_comgr_index_list_metadata(kernels_metadata, i, &kernel_node),
                          "amd_comgr_index_list_metadata"))
            continue;
        auto kernel_node_fini =
            common::scope_destructor{[kernel_node]() { destroy_metadata(kernel_node); }};

        auto kernel_name_meta = amd_comgr_metadata_node_t{};
        if(!comgr_success(amd_comgr_metadata_lookup(
                              kernel_node, kernel_name_metadata_lookup, &kernel_name_meta),
                          "amd_comgr_metadata_lookup"))
            continue;
        auto kernel_name_meta_fini =
            common::scope_destructor{[kernel_name_meta]() { destroy_metadata(kernel_name_meta); }};

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
            metadata.kernel_symbol_device_map.emplace(kernel_symbol, kernel_meta_name);
        }

        // Recorded under every name the entry is known by: the loader reports the kernel
        // descriptor ("<symbol>.kd") for newer code objects and the function itself for older
        // ones, and the metadata carries both spellings.
        if(auto regions = get_kernel_pointer_regions(kernel_node))
        {
            if(!kernel_symbol.empty())
                metadata.kernarg_pointer_regions.emplace(kernel_symbol, *regions);
            metadata.kernarg_pointer_regions.emplace(kernel_meta_name, std::move(*regions));
        }
    }

    return metadata;
}

amd_comgr_status_t
get_kernels_meta_node(const amd_comgr_code_object_info_t& isa_offset,
                      const void*                         fat_bin,
                      amd_comgr_data_t*                   binary_data,
                      amd_comgr_metadata_node_t*          binary_metadata,
                      amd_comgr_metadata_node_t*          kernels_metadata)
{
    CHECK_RETURN_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, binary_data));

    const char* bin_offset = static_cast<const char*>(fat_bin) + isa_offset.offset;
    CHECK_RETURN_COMGR_EXT(amd_comgr_set_data(*binary_data, isa_offset.size, bin_offset),
                           "binary_data=" << binary_data->handle << ", isa=(" << isa_offset.isa
                                          << ", " << isa_offset.size << ", " << isa_offset.offset
                                          << "), fat_bin=" << fat_bin);

    CHECK_RETURN_COMGR(amd_comgr_get_data_metadata(*binary_data, binary_metadata));
    CHECK_RETURN_COMGR(
        amd_comgr_metadata_lookup(*binary_metadata, kernels_metadata_lookup, kernels_metadata));

    return AMD_COMGR_STATUS_SUCCESS;
}
}  // namespace

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

kernel_symbol_hip_device_map_t
get_kernel_symbol_device_name_map(const amd_comgr_code_object_info_t& isa_offset,
                                  const void*                         fat_bin)
{
    auto binary_data      = amd_comgr_data_t{};
    auto binary_metadata  = amd_comgr_metadata_node_t{};
    auto kernels_metadata = amd_comgr_metadata_node_t{};

    if(get_kernels_meta_node(
           isa_offset, CHECK_NOTNULL(fat_bin), &binary_data, &binary_metadata, &kernels_metadata) !=
       AMD_COMGR_STATUS_SUCCESS)
    {
        destroy_metadata(kernels_metadata);
        destroy_metadata(binary_metadata);
        release_data(binary_data);
        return kernel_symbol_hip_device_map_t{};
    }

    auto binary_data_fini =
        common::scope_destructor{[binary_data]() { release_data(binary_data); }};
    auto binary_metadata_fini =
        common::scope_destructor{[binary_metadata]() { destroy_metadata(binary_metadata); }};
    auto kernels_metadata_fini =
        common::scope_destructor{[kernels_metadata]() { destroy_metadata(kernels_metadata); }};

    return get_metadata_from_kernels_node(kernels_metadata).kernel_symbol_device_map;
}

code_object_metadata_t
get_metadata_from_executable(const void* executable, size_t executable_size)
{
    auto metadata = code_object_metadata_t{};
    if(executable == nullptr || executable_size == 0) return metadata;

    auto binary_data = amd_comgr_data_t{};
    if(!comgr_success(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &binary_data),
                      "amd_comgr_create_data"))
        return metadata;
    auto binary_data_fini =
        common::scope_destructor{[binary_data]() { release_data(binary_data); }};

    if(!comgr_success(
           amd_comgr_set_data(binary_data, executable_size, static_cast<const char*>(executable)),
           "amd_comgr_set_data"))
        return metadata;

    auto binary_metadata = amd_comgr_metadata_node_t{};
    if(!comgr_success(amd_comgr_get_data_metadata(binary_data, &binary_metadata),
                      "amd_comgr_get_data_metadata"))
        return metadata;
    auto binary_metadata_fini =
        common::scope_destructor{[binary_metadata]() { destroy_metadata(binary_metadata); }};

    auto kernels_metadata = amd_comgr_metadata_node_t{};
    if(!comgr_success(
           amd_comgr_metadata_lookup(binary_metadata, kernels_metadata_lookup, &kernels_metadata),
           "amd_comgr_metadata_lookup"))
        return metadata;
    auto kernels_metadata_fini =
        common::scope_destructor{[kernels_metadata]() { destroy_metadata(kernels_metadata); }};

    return get_metadata_from_kernels_node(kernels_metadata);
}

kernel_symbol_hip_device_map_t
get_kernel_symbol_device_name_map_from_executable(const void* executable, size_t executable_size)
{
    return get_metadata_from_executable(executable, executable_size).kernel_symbol_device_map;
}
}  // namespace hip
}  // namespace code_object
}  // namespace rocprofiler
