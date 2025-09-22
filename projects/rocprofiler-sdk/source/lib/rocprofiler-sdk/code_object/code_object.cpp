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

#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/code_object/hip/code_object.hpp"
#include "lib/rocprofiler-sdk/code_object/hsa/code_object.hpp"
#include "lib/rocprofiler-sdk/code_object/hsa/kernel_symbol.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/cxx/utility.hpp>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace code_object
{
namespace
{
using context_t              = context::context;
using context_array_t        = common::container::small_vector<const context_t*>;
using external_corr_id_map_t = std::unordered_map<const context_t*, rocprofiler_user_data_t>;

template <size_t OpIdx>
struct code_object_info;

#define SPECIALIZE_CODE_OBJECT_INFO(OPERATION)                                                     \
    template <>                                                                                    \
    struct code_object_info<ROCPROFILER_CODE_OBJECT_##OPERATION>                                   \
    {                                                                                              \
        static constexpr auto operation_idx = ROCPROFILER_CODE_OBJECT_##OPERATION;                 \
        static constexpr auto name          = "CODE_OBJECT_" #OPERATION;                           \
    };

SPECIALIZE_CODE_OBJECT_INFO(NONE)
SPECIALIZE_CODE_OBJECT_INFO(LOAD)
SPECIALIZE_CODE_OBJECT_INFO(DEVICE_KERNEL_SYMBOL_REGISTER)
SPECIALIZE_CODE_OBJECT_INFO(HOST_KERNEL_SYMBOL_REGISTER)

#undef SPECIALIZE_CODE_OBJECT_INFO

template <size_t Idx, size_t... IdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return code_object_info<Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t Idx, size_t... IdxTail>
uint32_t
id_by_name(const char* name, std::index_sequence<Idx, IdxTail...>)
{
    if(std::string_view{code_object_info<Idx>::name} == std::string_view{name})
        return code_object_info<Idx>::operation_idx;
    if constexpr(sizeof...(IdxTail) > 0)
        return id_by_name(name, std::index_sequence<IdxTail...>{});
    else
        return ROCPROFILER_CODE_OBJECT_NONE;
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_CODE_OBJECT_LAST)) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, code_object_info<Idx>::operation_idx), ...);
}

template <size_t... Idx>
void
get_names(std::vector<const char*>& _name_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, const char* _v) {
        if(_v != nullptr && strnlen(_v, 1) > 0) _vec.emplace_back(_v);
    };

    (_emplace(_name_list, code_object_info<Idx>::name), ...);
}
}  // namespace

// check out the assembly here... this compiles to a switch statement
const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_CODE_OBJECT_LAST>{});
}

uint32_t
id_by_name(const char* name)
{
    return id_by_name(name, std::make_index_sequence<ROCPROFILER_CODE_OBJECT_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_CODE_OBJECT_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_CODE_OBJECT_LAST>{});
    return _data;
}

std::vector<const char*>
get_names()
{
    auto _data = std::vector<const char*>{};
    _data.reserve(ROCPROFILER_CODE_OBJECT_LAST);
    get_names(_data, std::make_index_sequence<ROCPROFILER_CODE_OBJECT_LAST>{});
    return _data;
}

namespace
{
using hsa_loader_table_t             = hsa_ven_amd_loader_1_01_pfn_t;
using context_t                      = context::context;
using user_data_t                    = rocprofiler_user_data_t;
using context_array_t                = context::context_array_t;
using context_user_data_map_t        = std::unordered_map<const context_t*, user_data_t>;
using amd_compute_pgm_rsrc_three32_t = uint32_t;

struct kernel_descriptor_t
{
    uint8_t  reserved0[16];
    int64_t  kernel_code_entry_byte_offset = 0;
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3      = 0;
    uint32_t compute_pgm_rsrc1      = 0;
    uint32_t compute_pgm_rsrc2      = 0;
    uint16_t kernel_code_properties = 0;
    uint8_t  reserved2[6];
};

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

uint32_t
arch_vgpr_count(std::string_view name, kernel_descriptor_t kernel_code)
{
    if(name == "gfx90a" || name.find("gfx94") == 0)
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
accum_vgpr_count(std::string_view name, kernel_descriptor_t kernel_code)
{
    if(name == "gfx908")
        return arch_vgpr_count(name, kernel_code);
    else if(name == "gfx90a" || name.find("gfx94") == 0)
        return ((AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                                  AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT) +
                 1) *
                8) -
               arch_vgpr_count(name, kernel_code);

    bool emplaced = false;
    {
        static auto warned = std::unordered_set<std::string>{};
        static auto mtx    = std::mutex{};
        auto        lk     = std::unique_lock<std::mutex>{mtx};
        emplaced           = warned.emplace(name).second;
    }

    ROCP_INFO_IF(emplaced) << "Missing support for accum_vgpr_count for " << name;
    return 0;
}

uint32_t
sgpr_count(std::string_view name, kernel_descriptor_t kernel_code)
{
    // GFX10 and later always allocate 128 sgprs.
    constexpr uint32_t gfx10_sgprs = 128;

    auto begp = name.find_first_of("0123456789");
    if(!name.empty() && begp != std::string_view::npos)
    {
        auto endp      = name.find_first_not_of("0123456789", begp);
        auto lenp      = (endp - begp) + 1;
        auto gfxip_str = name.substr(begp, lenp);
        auto gfxip_n   = int32_t{0};
        if(!gfxip_str.empty()) gfxip_n = std::stoi(std::string{gfxip_str});

        if(gfxip_n >= 1000)
        {
            return gfx10_sgprs;
        }
        else
        {
            return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                                     AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT) /
                        2 +
                    1) *
                   16;
        }
    }

    bool emplaced = false;
    {
        static auto warned = std::unordered_set<std::string>{};
        static auto mtx    = std::mutex{};
        auto        lk     = std::unique_lock<std::mutex>{mtx};
        emplaced           = warned.emplace(name).second;
    }

    ROCP_INFO_IF(emplaced) << "Missing support for sgpr_count for " << name;

    return 0;
}

hsa_loader_table_t&
get_loader_table()
{
    static auto _v = []() {
        auto _val = hsa_loader_table_t{};
        memset(&_val, 0, sizeof(hsa_loader_table_t));
        return _val;
    }();
    return _v;
}

auto*&
get_status_string_function()
{
    static decltype(::hsa_status_string)* _v = nullptr;
    return _v;
}

std::string_view
get_status_string(hsa_status_t _status)
{
    const char* _msg = nullptr;
    if(get_status_string_function() &&
       get_status_string_function()(_status, &_msg) == HSA_STATUS_SUCCESS && _msg)
        return std::string_view{_msg};

    return std::string_view{"(unknown HSA error)"};
}

const kernel_descriptor_t*
get_kernel_descriptor(uint64_t kernel_object)
{
    const kernel_descriptor_t* kernel_code = nullptr;
    if(get_loader_table().hsa_ven_amd_loader_query_host_address == nullptr) return kernel_code;
    hsa_status_t status = get_loader_table().hsa_ven_amd_loader_query_host_address(
        reinterpret_cast<const void*>(kernel_object),  // NOLINT(performance-no-int-to-ptr)
        reinterpret_cast<const void**>(&kernel_code));
    if(status == HSA_STATUS_SUCCESS) return kernel_code;

    ROCP_WARNING << "hsa_ven_amd_loader_query_host_address(kernel_object=" << kernel_object
                 << ") returned " << status << ": " << get_status_string(status);

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<kernel_descriptor_t*>(kernel_object);
}

auto&
get_code_object_id()
{
    static auto _v = std::atomic<uint64_t>{};
    return _v;
}

auto&
get_kernel_symbol_id()
{
    static auto _v = std::atomic<uint64_t>{};
    return _v;
}

auto&
get_host_function_id()
{
    static auto _v = std::atomic<uint64_t>{};
    return _v;
}

using kernel_object_map_t        = std::unordered_map<uint64_t, uint64_t>;
using executable_array_t         = std::vector<hsa_executable_t>;
using code_object_unload_array_t = std::vector<hsa::code_object_unload>;

std::vector<hsa::code_object_unload>
shutdown(hsa_executable_t executable);

bool is_shutdown = false;

auto*
get_executables()
{
    static auto*& _v = common::static_object<common::Synchronized<executable_array_t>>::construct();
    return _v;
}

auto*
get_code_objects()
{
    static auto*& _v =
        common::static_object<common::Synchronized<code_object_array_t>>::construct();
    return _v;
}

auto*
get_kernel_object_map()
{
    static auto*& _v =
        common::static_object<common::Synchronized<kernel_object_map_t>>::construct();
    return _v;
}

auto*
get_hip_register_data()
{
    static auto*& _v =
        common::static_object<common::Synchronized<hip::hip_register_data>>::construct();
    return _v;
}

hsa_status_t
executable_iterate_agent_symbols_load_callback(hsa_executable_t        executable,
                                               hsa_agent_t             agent,
                                               hsa_executable_symbol_t symbol,
                                               void*                   args)
{
#define ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(...)                                                     \
    {                                                                                              \
        auto _status = core_table.hsa_executable_symbol_get_info_fn(symbol, __VA_ARGS__);          \
        ROCP_ERROR_IF(_status != HSA_STATUS_SUCCESS)                                               \
            << "core_table.hsa_executable_symbol_get_info_fn(hsa_executable_symbol_t{.handle="     \
            << symbol.handle << "}, " << #__VA_ARGS__ << " failed";                                \
        if(_status != HSA_STATUS_SUCCESS) return _status;                                          \
    }

    auto& core_table = *::rocprofiler::hsa::get_core_table();
    auto* code_obj_v = static_cast<hsa::code_object*>(args);
    auto  symbol_v   = hsa::kernel_symbol{};
    auto& data       = symbol_v.rocp_data;

    symbol_v.hsa_executable = executable;
    symbol_v.hsa_agent      = agent;
    symbol_v.hsa_symbol     = symbol;

    auto exists = std::any_of(code_obj_v->symbols.begin(),
                              code_obj_v->symbols.end(),
                              [&symbol_v](auto& itr) { return (itr && symbol_v == *itr); });

    // if there is an existing matching kernel symbol, return success and move onto next symbol
    if(exists) return HSA_STATUS_SUCCESS;

    ROCP_FATAL_IF(data.size == 0) << "kernel symbol did not properly initialized the size field "
                                     "upon construction (this is likely a compiler bug)";

    auto type = hsa_symbol_kind_t{};
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &type);

    if(type != HSA_SYMBOL_KIND_KERNEL) return HSA_STATUS_SUCCESS;

    // set the code object id
    data.code_object_id = code_obj_v->rocp_data.code_object_id;

    // compute the kernel name length
    constexpr auto name_length_max = std::numeric_limits<uint32_t>::max();
    uint32_t       _name_length    = 0;
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &_name_length);

    ROCP_CI_LOG_IF(WARNING, _name_length > name_length_max / 2)
        << "kernel symbol name length is extremely large: " << _name_length;

    // set the kernel name
    if(_name_length > 0 && _name_length < name_length_max)
    {
        auto _name = std::string(_name_length + 1, '\0');
        ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_NAME, _name.data());

        symbol_v.name = common::get_string_entry(_name.substr(0, _name.find_first_of('\0')));
    }
    data.kernel_name = (symbol_v.name) ? symbol_v.name->c_str() : nullptr;

    // these should all be self-explanatory
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                      &data.kernel_object);
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                      &data.kernarg_segment_size);
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                                      &data.kernarg_segment_alignment);
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                      &data.group_segment_size);
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                      &data.private_segment_size);

    // This works for gfx9 but may not for Navi arch
    const auto* kernel_descript = get_kernel_descriptor(data.kernel_object);
    if(CHECK_NOTNULL(code_obj_v) && CHECK_NOTNULL(kernel_descript))
    {
        data.kernel_code_entry_byte_offset = kernel_descript->kernel_code_entry_byte_offset;
        data.kernel_address.handle = data.kernel_object + data.kernel_code_entry_byte_offset;

        if(const auto* rocp_agent = agent::get_agent(code_obj_v->rocp_data.rocp_agent);
           CHECK_NOTNULL(rocp_agent))
        {
            data.arch_vgpr_count  = arch_vgpr_count(rocp_agent->name, *kernel_descript);
            data.accum_vgpr_count = accum_vgpr_count(rocp_agent->name, *kernel_descript);
            data.sgpr_count       = sgpr_count(rocp_agent->name, *kernel_descript);
        }
    }

    // if we have reached this point (i.e. there were no HSA errors returned within macro) then we
    // generate a unique kernel symbol id
    data.kernel_id = ++get_kernel_symbol_id();

    CHECK_NOTNULL(get_kernel_object_map())
        ->wlock(
            [](kernel_object_map_t& object_map, uint64_t _kern_obj, uint64_t _kern_id) {
                object_map[_kern_obj] = _kern_id;
            },
            data.kernel_object,
            data.kernel_id);

    code_obj_v->symbols.emplace_back(std::make_unique<hsa::kernel_symbol>(std::move(symbol_v)));

    return HSA_STATUS_SUCCESS;

#undef ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO
}

hsa_status_t
executable_iterate_agent_symbols_unload_callback(hsa_executable_t        executable,
                                                 hsa_agent_t             agent,
                                                 hsa_executable_symbol_t symbol,
                                                 void*                   args)
{
    auto symbol_v           = hsa::kernel_symbol{};
    symbol_v.hsa_executable = executable;
    symbol_v.hsa_agent      = agent;
    symbol_v.hsa_symbol     = symbol;

    auto* code_obj_v = static_cast<hsa::code_object_unload*>(args);
    CHECK_NOTNULL(code_obj_v);
    CHECK_NOTNULL(code_obj_v->object);

    for(const auto& itr : code_obj_v->object->symbols)
    {
        if(itr && *itr == symbol_v) code_obj_v->symbols.emplace_back(itr.get());
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
code_object_load_callback(hsa_executable_t         executable,
                          hsa_loaded_code_object_t loaded_code_object,
                          void*                    cb_data)
{
#define ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(...)                                              \
    {                                                                                              \
        auto _status = loader_table.hsa_ven_amd_loader_loaded_code_object_get_info(                \
            loaded_code_object, __VA_ARGS__);                                                      \
        ROCP_ERROR_IF(_status != HSA_STATUS_SUCCESS)                                               \
            << "loader_table.hsa_ven_amd_loader_loaded_code_object_get_info(loaded_code_object, "  \
            << #__VA_ARGS__ << " failed";                                                          \
        if(_status != HSA_STATUS_SUCCESS) return _status;                                          \
    }

    auto&    loader_table  = get_loader_table();
    auto     code_obj_v    = hsa::code_object{};
    auto&    data          = code_obj_v.rocp_data;
    uint32_t _storage_type = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_NONE;

    ROCP_FATAL_IF(data.size == 0) << "code object did not properly initialized the size field upon "
                                     "construction (this is likely a compiler bug)";

    code_obj_v.hsa_executable  = executable;
    code_obj_v.hsa_code_object = loaded_code_object;

    auto* code_obj_vec = static_cast<code_object_array_t*>(cb_data);
    auto exists = std::any_of(code_obj_vec->begin(), code_obj_vec->end(), [&code_obj_v](auto& itr) {
        return (itr && code_obj_v == *itr);
    });

    // if there is an existing matching code object, check for any new symbols and then return
    // success and move onto next code object
    if(exists)
    {
        for(auto& itr : *code_obj_vec)
        {
            if(itr && *itr == code_obj_v)
            {
                ::rocprofiler::hsa::get_core_table()->hsa_executable_iterate_agent_symbols_fn(
                    executable,
                    data.hsa_agent,
                    executable_iterate_agent_symbols_load_callback,
                    itr.get());
            }
        }

        return HSA_STATUS_SUCCESS;
    }

    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(
        HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE, &_storage_type);

    ROCP_FATAL_IF(_storage_type >= ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_LAST)
        << "HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE returned an "
           "unsupported code object storage type. Expected 0=none, 1=file, or 2=memory but "
           "received a value of "
        << _storage_type;

    data.storage_type = static_cast<rocprofiler_code_object_storage_type_t>(_storage_type);

    if(_storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE,
            &data.storage_file);
    }
    else if(_storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
        ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
            &data.memory_base);
        ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
            &data.memory_size);
    }
    else if(_storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE)
    {
        ROCP_WARNING << "Code object storage type of none was ignored";
        return HSA_STATUS_SUCCESS;
    }

    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                                             &data.load_base);

    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                                             &data.load_size);

    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
                                             &data.load_delta);

    constexpr auto uri_length_max = std::numeric_limits<uint32_t>::max();
    auto           _uri_length    = uint32_t{0};
    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
                                             &_uri_length);

    ROCP_CI_LOG_IF(WARNING, _uri_length > uri_length_max / 2)
        << "code object uri length is extremely large: " << _uri_length;

    if(_uri_length > 0 && _uri_length < uri_length_max)
    {
        auto _uri = std::string(_uri_length + 1, '\0');
        ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI,
                                                 _uri.data());

        code_obj_v.uri = common::get_string_entry(_uri);
    }
    data.uri = (code_obj_v.uri) ? code_obj_v.uri->data() : nullptr;

    auto _hsa_agent = hsa_agent_t{};
    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
                                             &data.hsa_agent);

    const auto* _rocp_agent = agent::get_rocprofiler_agent(data.hsa_agent);
    if(!_rocp_agent)
    {
        ROCP_ERROR << "hsa agent (handle=" << _hsa_agent.handle
                   << ") did not map to a rocprofiler agent";
        return HSA_STATUS_ERROR_INVALID_AGENT;
    }
    data.rocp_agent = _rocp_agent->id;

    // if we have reached this point (i.e. there were no HSA errors returned within macro) then we
    // generate a unique code object id
    data.code_object_id = ++get_code_object_id();

    auto _status = ::rocprofiler::hsa::get_core_table()->hsa_executable_iterate_agent_symbols_fn(
        executable, data.hsa_agent, executable_iterate_agent_symbols_load_callback, &code_obj_v);

    if(_status == HSA_STATUS_SUCCESS)
    {
        code_obj_vec->emplace_back(std::make_unique<hsa::code_object>(std::move(code_obj_v)));
    }
    else
    {
        ROCP_ERROR << "hsa_executable_iterate_agent_symbols failed for " << data.uri;
    }

    return _status;

#undef ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO
}

hsa_status_t
code_object_unload_callback(hsa_executable_t         executable,
                            hsa_loaded_code_object_t loaded_code_object,
                            void*                    args)
{
    auto code_obj_v            = hsa::code_object{};
    code_obj_v.hsa_executable  = executable;
    code_obj_v.hsa_code_object = loaded_code_object;

    auto* code_obj_arr = static_cast<code_object_unload_array_t*>(args);

    CHECK_NOTNULL(code_obj_arr);

    ROCP_TRACE << "[inp] executable=" << executable.handle
               << ", code_object=" << loaded_code_object.handle << " vs. "
               << (CHECK_NOTNULL(get_code_objects())->rlock([](const auto& data) {
                      return data.size();
                  }));

    CHECK_NOTNULL(get_code_objects())->rlock([&](const code_object_array_t& arr) {
        for(const auto& itr : arr)
        {
            ROCP_TRACE << "[cmp] executable=" << itr->hsa_executable.handle
                       << ", code_object=" << itr->hsa_code_object.handle;
            if(itr->hsa_executable.handle == executable.handle &&
               itr->hsa_code_object.handle == loaded_code_object.handle)
            // if(itr && *itr == code_obj_v)
            {
                auto& _last =
                    code_obj_arr->emplace_back(hsa::code_object_unload{.object = itr.get()});

                if(auto agent = agent::get_hsa_agent(itr->rocp_data.agent_id); agent)
                    ::rocprofiler::hsa::get_core_table()->hsa_executable_iterate_agent_symbols_fn(
                        executable,
                        *agent,
                        executable_iterate_agent_symbols_unload_callback,
                        &_last);
            }
        }
    });

    return HSA_STATUS_SUCCESS;
}

std::vector<hsa::code_object_unload>
get_unloaded_code_objects(hsa_executable_t executable)
{
    auto _unloaded = std::vector<hsa::code_object_unload>{};

    if(!is_shutdown && get_loader_table().hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
        get_loader_table().hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
            executable, code_object_unload_callback, &_unloaded);

    return _unloaded;
}

auto&
get_freeze_function()
{
    static decltype(::hsa_executable_freeze)* _v = nullptr;
    return _v;
}

auto&
get_destroy_function()
{
    static decltype(::hsa_executable_destroy)* _v = nullptr;
    return _v;
}

auto&
get_hip_register_fatbinary_function()
{
    static decltype(::std::declval<HipCompilerDispatchTable>().__hipRegisterFatBinary_fn) _v =
        nullptr;
    return _v;
}

auto&
get_hip_register_function_function()
{
    static decltype(::std::declval<HipCompilerDispatchTable>().__hipRegisterFunction_fn) _v =
        nullptr;
    return _v;
}

bool
initialize_hip_binary_data()
{
    static bool is_initialized =
        CHECK_NOTNULL(get_hip_register_data())->wlock([](hip::hip_register_data& data) {
            ROCP_INFO_IF(!data.fat_binary) << "No binary registered for HIP";
            if(!data.fat_binary) return false;
            std::vector<const rocprofiler_agent_t*> rocp_agents = rocprofiler::agent::get_agents();
            for(const auto* rocp_agent : rocp_agents)
            {
                if(rocp_agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
                auto hsa_agent = agent::get_hsa_agent(rocp_agent);
                if(!hsa_agent.has_value()) continue;
                for(auto& isa : hip::get_isa_offsets(hsa_agent.value(), data.fat_binary))
                {
                    auto kernel_symbols_name_map =
                        hip::get_kernel_symbol_device_name_map(isa, data.fat_binary);
                    // many to one mapping as the same kernel symbols can be found in multiple code
                    // objects
                    if(!kernel_symbols_name_map.empty())
                        data.kernel_symbol_device_map.insert(kernel_symbols_name_map.begin(),
                                                             kernel_symbols_name_map.end());
                }
            }
            return true;
        });
    return is_initialized;
}

hsa_status_t
executable_freeze(hsa_executable_t executable, const char* options)
{
    hsa_status_t status = CHECK_NOTNULL(get_freeze_function())(executable, options);
    if(status != HSA_STATUS_SUCCESS) return status;

    // before iterating code-object populate the host function map from registered binary
    bool is_initialized = initialize_hip_binary_data();
    ROCP_INFO_IF(!is_initialized) << "hip mapping data not initialized";

    ROCP_INFO << "running " << __FUNCTION__ << " (executable=" << executable.handle << ")...";
    CHECK_NOTNULL(get_executables())->wlock([executable](executable_array_t& data) {
        data.emplace_back(executable);
    });

    auto* code_obj_vec = get_code_objects();
    CHECK_NOTNULL(code_obj_vec)->wlock([executable](code_object_array_t& _vec) {
        get_loader_table().hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
            executable, code_object_load_callback, &_vec);
    });

    constexpr auto CODE_OBJECT_KIND = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
    constexpr auto CODE_OBJECT_LOAD = ROCPROFILER_CODE_OBJECT_LOAD;
    constexpr auto CODE_OBJECT_KERNEL_SYMBOL =
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
    constexpr auto CODE_OBJECT_HOST_SYMBOL = ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER;

    auto&& context_filter = [](const context_t* ctx) {
        return (ctx->callback_tracer && ctx->callback_tracer->domains(CODE_OBJECT_KIND) &&
                (ctx->callback_tracer->domains(CODE_OBJECT_KIND, CODE_OBJECT_LOAD) ||
                 ctx->callback_tracer->domains(CODE_OBJECT_KIND, CODE_OBJECT_KERNEL_SYMBOL)));
    };

    static thread_local auto ctxs = context_array_t{};
    context::get_active_contexts(ctxs, std::move(context_filter));

    if(!ctxs.empty())
    {
        code_obj_vec->rlock([](const code_object_array_t& data) {
            auto tidx = common::get_tid();
            // set the contexts for each code object
            for(const auto& ditr : data)
                ditr->contexts = ctxs;

            for(const auto& ditr : data)
            {
                for(const auto* citr : ditr->contexts)
                {
                    if(citr->callback_tracer->domains(CODE_OBJECT_KIND, CODE_OBJECT_LOAD))
                    {
                        if(!ditr->beg_notified)
                        {
                            auto co_data = ditr->rocp_data;
                            auto record  = rocprofiler_callback_tracing_record_t{
                                .context_id     = rocprofiler_context_id_t{citr->context_idx},
                                .thread_id      = tidx,
                                .correlation_id = rocprofiler_correlation_id_t{},
                                .kind           = CODE_OBJECT_KIND,
                                .operation      = CODE_OBJECT_LOAD,
                                .phase          = ROCPROFILER_CALLBACK_PHASE_LOAD,
                                .payload        = static_cast<void*>(&co_data)};

                            // invoke callback
                            auto& cb_data =
                                citr->callback_tracer->callback_data.at(CODE_OBJECT_KIND);
                            auto& user_data = ditr->user_data[citr];
                            cb_data.callback(record, &user_data, cb_data.data);
                        }
                    }

                    for(const auto& sitr : ditr->symbols)
                    {
                        if(sitr && citr->callback_tracer->domains(CODE_OBJECT_KIND,
                                                                  CODE_OBJECT_KERNEL_SYMBOL))
                        {
                            if(!sitr->beg_notified)
                            {
                                auto sym_data = sitr->rocp_data;
                                auto record   = rocprofiler_callback_tracing_record_t{
                                    .context_id     = rocprofiler_context_id_t{citr->context_idx},
                                    .thread_id      = tidx,
                                    .correlation_id = rocprofiler_correlation_id_t{},
                                    .kind           = CODE_OBJECT_KIND,
                                    .operation      = CODE_OBJECT_KERNEL_SYMBOL,
                                    .phase          = ROCPROFILER_CALLBACK_PHASE_LOAD,
                                    .payload        = static_cast<void*>(&sym_data)};

                                // invoke callback
                                auto& cb_data =
                                    citr->callback_tracer->callback_data.at(CODE_OBJECT_KIND);
                                auto& user_data = sitr->user_data[citr];
                                cb_data.callback(record, &user_data, cb_data.data);

                                std::string device_name =
                                    CHECK_NOTNULL(get_hip_register_data())
                                        ->rlock([sym_data](
                                                    const hip::hip_register_data& register_data) {
                                            const auto& sym_map =
                                                register_data.kernel_symbol_device_map;
                                            const auto it = sym_map.find(*CHECK_NOTNULL(
                                                common::get_string_entry(sym_data.kernel_name)));
                                            if(it != sym_map.end()) return it->second;
                                            return std::string();
                                        });
                                // Does not have a host function, skip
                                if(device_name.empty()) continue;
                                auto host_data =
                                    CHECK_NOTNULL(get_hip_register_data())
                                        ->rlock([device_name](
                                                    const hip::hip_register_data& register_data) {
                                            // Add check for out of range here
                                            const auto it =
                                                register_data.host_function_map.find(device_name);
                                            if(it == register_data.host_function_map.end())
                                            {
                                                return rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t{};
                                            }
                                            return it->second;
                                        });
                                // when kernel_symbol_device_map kernels are not present in
                                // host_function_map, skip.
                                if(host_data.device_function == nullptr) continue;
                                host_data.code_object_id   = sym_data.code_object_id;
                                host_data.kernel_id        = sym_data.kernel_id;
                                host_data.host_function_id = ++get_host_function_id();
                                auto hip_record            = rocprofiler_callback_tracing_record_t{
                                    .context_id     = rocprofiler_context_id_t{citr->context_idx},
                                    .thread_id      = tidx,
                                    .correlation_id = rocprofiler_correlation_id_t{},
                                    .kind           = CODE_OBJECT_KIND,
                                    .operation      = CODE_OBJECT_HOST_SYMBOL,
                                    .phase          = ROCPROFILER_CALLBACK_PHASE_LOAD,
                                    .payload        = static_cast<void*>(&host_data)};

                                // invoke callback
                                cb_data.callback(hip_record, &user_data, cb_data.data);
                            }
                        }
                    }
                }
            }

            for(const auto& ditr : data)
            {
                ditr->beg_notified = true;
                for(auto& sitr : ditr->symbols)
                    sitr->beg_notified = true;
            }
        });
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_destroy(hsa_executable_t executable)
{
    if(is_shutdown) return HSA_STATUS_SUCCESS;

    auto _unloaded = shutdown(executable);

    if(get_kernel_object_map())
    {
        CHECK_NOTNULL(get_kernel_object_map())->wlock([_unloaded](kernel_object_map_t& data) {
            for(const auto& uitr : _unloaded)
            {
                for(const auto& sitr : uitr.symbols)
                {
                    data.erase(sitr->rocp_data.kernel_id);
                }
            }
        });
    }

    if(get_code_objects())
    {
        CHECK_NOTNULL(get_code_objects())->wlock([executable](code_object_array_t& data) {
            for(auto& itr : data)
            {
                if(itr->hsa_executable.handle == executable.handle) itr.reset();
            }
            data.erase(std::remove_if(
                           data.begin(), data.end(), [](auto& itr) { return (itr == nullptr); }),
                       data.end());
        });
    }

    if(get_executables())
    {
        CHECK_NOTNULL(get_executables())->wlock([executable](executable_array_t& data) {
            data.erase(std::remove_if(data.begin(),
                                      data.end(),
                                      [executable](hsa_executable_t itr) {
                                          return (itr.handle == executable.handle);
                                      }),
                       data.end());
        });
    }

    return CHECK_NOTNULL(get_destroy_function())(executable);
}

void**
hip_register_fat_binary(const void* data)
{
    const hip::hip_fat_binary_wrapper* fbwrapper =
        reinterpret_cast<const hip::hip_fat_binary_wrapper*>(data);
    ROCP_ERROR_IF((fbwrapper->magic != hip::HIP_FAT_MAGIC || fbwrapper->version != 1))
        << "register fat binary failed";
    CHECK_NOTNULL(get_hip_register_data())->wlock([fbwrapper](hip::hip_register_data& reg_data) {
        reg_data.fat_binary = fbwrapper->binary;
    });
    return CHECK_NOTNULL(get_hip_register_fatbinary_function())(data);
}

void
hip_register_function(void**       modules,
                      const void*  host_function,
                      char*        device_function,
                      const char*  device_name,
                      unsigned int thread_limit,
                      uint3*       thread_id,
                      uint3*       block_id,
                      dim3*        block_dim,
                      dim3*        grid_dim,
                      int*         workgroup_size)
{
    auto convert_to_dim3 = [](auto* val) {
        return (val) ? rocprofiler_dim3_t{.x = val->x, .y = val->y, .z = val->z}
                     : rocprofiler_dim3_t{0, 0, 0};
    };

    CHECK_NOTNULL(get_hip_register_data())->wlock([&](hip::hip_register_data& data) {
        const std::string* d_func      = common::get_string_entry(device_function);
        auto               host_symbol = common::init_public_api_struct(hip::host_symbol_data_t{});
        host_symbol.host_function.ptr  = const_cast<void*>(host_function);
        host_symbol.modules.ptr        = modules;
        host_symbol.device_function    = d_func->c_str();
        host_symbol.thread_limit       = thread_limit;
        host_symbol.thread_ids         = convert_to_dim3(thread_id);
        host_symbol.block_ids          = convert_to_dim3(block_id);
        host_symbol.block_dims         = convert_to_dim3(block_dim);
        host_symbol.grid_dims          = convert_to_dim3(grid_dim);
        host_symbol.workgroup_size     = (workgroup_size) ? *workgroup_size : 0;
        data.host_function_map.emplace(*CHECK_NOTNULL(d_func), host_symbol);
    });
    CHECK_NOTNULL(get_hip_register_function_function())
    (modules,
     host_function,
     device_function,
     device_name,
     thread_limit,
     thread_id,
     block_id,
     block_dim,
     grid_dim,
     workgroup_size);
}

std::vector<hsa::code_object_unload>
shutdown(hsa_executable_t executable)
{
    ROCP_INFO << "running " << __FUNCTION__ << " (executable=" << executable.handle << ")...";

    auto _unloaded = code_object::get_unloaded_code_objects(executable);

    constexpr auto CODE_OBJECT_KIND = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
    constexpr auto CODE_OBJECT_LOAD = ROCPROFILER_CODE_OBJECT_LOAD;
    constexpr auto CODE_OBJECT_KERNEL_SYMBOL =
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;

    auto tidx = common::get_tid();
    for(auto& itr : _unloaded)
    {
        ROCP_FATAL_IF(itr.object == nullptr);
        for(const auto* citr : itr.object->contexts)
        {
            if(citr->callback_tracer->domains(CODE_OBJECT_KIND, CODE_OBJECT_LOAD))
            {
                if(!itr.object->end_notified)
                {
                    auto record = rocprofiler_callback_tracing_record_t{
                        .context_id     = rocprofiler_context_id_t{citr->context_idx},
                        .thread_id      = tidx,
                        .correlation_id = rocprofiler_correlation_id_t{},
                        .kind           = CODE_OBJECT_KIND,
                        .operation      = CODE_OBJECT_LOAD,
                        .phase          = ROCPROFILER_CALLBACK_PHASE_UNLOAD,
                        .payload        = static_cast<void*>(&itr.object->rocp_data)};

                    // invoke callback
                    auto& cb_data   = citr->callback_tracer->callback_data.at(CODE_OBJECT_KIND);
                    auto& user_data = itr.object->user_data.at(citr);
                    cb_data.callback(record, &user_data, cb_data.data);
                }
            }

            // generate callbacks for kernel symbols after the callback for code object
            // unloading so the code object unload can be used to flush the buffer before the
            // symbol information is removed
            if(citr->callback_tracer->domains(CODE_OBJECT_KIND, CODE_OBJECT_KERNEL_SYMBOL))
            {
                for(auto& sitr : itr.symbols)
                {
                    if(!sitr->end_notified)
                    {
                        auto record = rocprofiler_callback_tracing_record_t{
                            .context_id     = rocprofiler_context_id_t{citr->context_idx},
                            .thread_id      = tidx,
                            .correlation_id = rocprofiler_correlation_id_t{},
                            .kind           = CODE_OBJECT_KIND,
                            .operation      = CODE_OBJECT_KERNEL_SYMBOL,
                            .phase          = ROCPROFILER_CALLBACK_PHASE_UNLOAD,
                            .payload        = static_cast<void*>(&sitr->rocp_data)};

                        // invoke callback
                        auto& cb_data   = citr->callback_tracer->callback_data.at(CODE_OBJECT_KIND);
                        auto& user_data = sitr->user_data.at(citr);
                        cb_data.callback(record, &user_data, cb_data.data);
                    }
                }
            }
        }
    }

    for(auto& itr : _unloaded)
    {
        itr.object->end_notified = true;
        for(auto& sitr : itr.symbols)
            sitr->end_notified = true;
    }

    return _unloaded;
}
}  // namespace

void
initialize(HsaApiTable* table)
{
    auto& core_table = *table->core_;

    get_status_string_function() = core_table.hsa_status_string_fn;

    auto _status = core_table.hsa_system_get_major_extension_table_fn(
        HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_loader_table_t), &get_loader_table());

    ROCP_ERROR_IF(_status != HSA_STATUS_SUCCESS)
        << "hsa_system_get_major_extension_table failed: " << get_status_string(_status);

    if(_status == HSA_STATUS_SUCCESS)
    {
        get_freeze_function()                = CHECK_NOTNULL(core_table.hsa_executable_freeze_fn);
        get_destroy_function()               = CHECK_NOTNULL(core_table.hsa_executable_destroy_fn);
        core_table.hsa_executable_freeze_fn  = executable_freeze;
        core_table.hsa_executable_destroy_fn = executable_destroy;
        ROCP_FATAL_IF(get_freeze_function() == core_table.hsa_executable_freeze_fn)
            << "infinite recursion";
        ROCP_FATAL_IF(get_destroy_function() == core_table.hsa_executable_destroy_fn)
            << "infinite recursion";
    }
}

void
initialize(HipCompilerDispatchTable* table)
{
    get_hip_register_fatbinary_function() = CHECK_NOTNULL(table->__hipRegisterFatBinary_fn);
    get_hip_register_function_function()  = CHECK_NOTNULL(table->__hipRegisterFunction_fn);
    table->__hipRegisterFatBinary_fn      = hip_register_fat_binary;
    table->__hipRegisterFunction_fn       = hip_register_function;
    ROCP_FATAL_IF(get_hip_register_fatbinary_function() == table->__hipRegisterFatBinary_fn)
        << "infinite recursion";
    ROCP_FATAL_IF(get_hip_register_function_function() == table->__hipRegisterFunction_fn)
        << "infinite recursion";
}

uint64_t
get_kernel_id(uint64_t kernel_object)
{
    return CHECK_NOTNULL(get_kernel_object_map())
        ->rlock(
            [](const kernel_object_map_t& object_map, uint64_t _kern_obj) -> uint64_t {
                auto itr = object_map.find(_kern_obj);
                return (itr == object_map.end()) ? 0 : itr->second;
            },
            kernel_object);
}

void
finalize()
{
    if(is_shutdown || !get_executables() || !get_code_objects()) return;

    CHECK_NOTNULL(get_executables())->rlock([](const executable_array_t& edata) {
        auto tmp = edata;
        std::reverse(tmp.begin(), tmp.end());
        for(auto itr : tmp)
            shutdown(itr);
    });

    CHECK_NOTNULL(get_code_objects())->wlock([](code_object_array_t& data) { data.clear(); });

    is_shutdown = true;
}

void
iterate_loaded_code_objects(code_object_iterator_t&& func)
{
    if(is_shutdown || !get_executables() || !get_code_objects()) return;
    CHECK_NOTNULL(get_code_objects())
        ->rlock(
            [](const code_object_array_t& data, code_object_iterator_t&& func_v) {
                for(const auto& itr : data)
                {
                    if(itr) func_v(*itr);
                }
            },
            std::move(func));
}
}  // namespace code_object
}  // namespace rocprofiler
