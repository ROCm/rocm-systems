// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#define GNU_SOURCE 1

#include <rocprofiler-register/rocprofiler-register.h>

#include "details/dl.hpp"
#include "details/environment.hpp"
#include "glog/logging.h"

#include <array>
#include <atomic>
#include <bitset>
#include <filesystem>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <dlfcn.h>

extern "C" {
#pragma weak rocprofiler_configure
#pragma weak rocprofiler_set_api_table
#pragma weak rocprofiler_register_import_hip
#pragma weak rocprofiler_register_import_hip_static
#pragma weak rocprofiler_register_import_hsa
#pragma weak rocprofiler_register_import_hsa_static
#pragma weak rocprofiler_register_import_roctx
#pragma weak rocprofiler_register_import_roctx_static

extern rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*);

extern int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t);

extern uint32_t
rocprofiler_register_import_hip(void);

extern uint32_t
rocprofiler_register_import_hsa(void);

extern uint32_t
rocprofiler_register_import_roctx(void);

extern uint32_t
rocprofiler_register_import_hip_static(void);

extern uint32_t
rocprofiler_register_import_hsa_static(void);

extern uint32_t
rocprofiler_register_import_roctx_static(void);
}

namespace
{
namespace fs = ::std::filesystem;
using namespace rocprofiler_register;
using rocprofiler_set_api_table_t = decltype(::rocprofiler_set_api_table)*;
using bitset_t = std::bitset<sizeof(rocprofiler_register_library_indentifier_t::handle)>;

static_assert(sizeof(bitset_t) ==
                  sizeof(rocprofiler_register_library_indentifier_t::handle),
              "bitset should be same at uint64_t");

int rocprofiler_register_verbose = common::get_env("ROCPROFILER_REGISTER_VERBOSE", 0);
constexpr int  rocprofiler_register_info_level     = 2;
constexpr auto rocprofiler_lib_name                = "librocprofiler64.so";
constexpr auto rocprofiler_lib_register_entrypoint = "rocprofiler_set_api_table";
constexpr auto rocprofiler_register_lib_name =
    "librocprofiler-register.so." ROCPROFILER_REGISTER_SOVERSION;

enum rocp_reg_supported_library  // NOLINT(performance-enum-size)
{
    ROCP_REG_HSA = 0,
    ROCP_REG_HIP,
    ROCP_REG_ROCTX,
    ROCP_REG_LAST,
};

template <size_t>
struct supported_library_trait
{
    static constexpr bool              specialized  = false;
    static constexpr auto              value        = ROCP_REG_LAST;
    static constexpr const char* const common_name  = nullptr;
    static constexpr const char* const symbol_name  = nullptr;
    static constexpr const char* const library_name = nullptr;
};

#define ROCP_REG_DEFINE_LIBRARY_TRAITS(ENUM, NAME, SYM_NAME, LIB_NAME)                   \
    template <>                                                                          \
    struct supported_library_trait<ENUM>                                                 \
    {                                                                                    \
        static constexpr bool specialized  = true;                                       \
        static constexpr auto value        = ENUM;                                       \
        static constexpr auto common_name  = NAME;                                       \
        static constexpr auto symbol_name  = SYM_NAME;                                   \
        static constexpr auto library_name = LIB_NAME;                                   \
    };

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_HSA,
                               "hsa",
                               "rocprofiler_register_import_hsa",
                               "libhsa-runtime64.so.[2-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_HIP,
                               "hip",
                               "rocprofiler_register_import_hip",
                               "libamdhip64.so.[6-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_ROCTX,
                               "roctx",
                               "rocprofiler_register_import_roctx",
                               "libroctx64.so.[4-9]($|\\.[0-9\\.]+)")

auto
get_this_library_path()
{
    auto _this_lib_path = binary::get_linked_path(rocprofiler_register_lib_name,
                                                  { RTLD_NOLOAD | RTLD_LAZY });
    LOG_IF(FATAL, !_this_lib_path)
        << rocprofiler_register_lib_name
        << " could not locate itself in the list of loaded libraries";
    return fs::path{ *_this_lib_path }.parent_path().string();
}

struct rocp_import
{
    rocp_reg_supported_library library_idx  = ROCP_REG_LAST;
    std::string_view           common_name  = {};
    std::string_view           symbol_name  = {};
    std::string_view           library_name = {};
};

template <size_t... Idx>
auto rocp_reg_get_imports(std::index_sequence<Idx...>)
{
    auto _data        = std::vector<rocp_import>{};
    auto _import_scan = [&_data](auto _info) {
        if(_info.specialized)
        {
            _data.emplace_back(rocp_import{
                _info.value, _info.common_name, _info.symbol_name, _info.library_name });
        }
    };

    (_import_scan(supported_library_trait<Idx>{}), ...);
    return _data;
}

auto
rocp_reg_scan_for_tools()
{
    auto _rocp_reg_lib = common::get_env("ROCPROFILER_REGISTER_LIBRARY", std::string{});
    bool _force_tool =
        common::get_env("ROCPROFILER_REGISTER_FORCE_LOAD", !_rocp_reg_lib.empty());
    bool _found_tool = (rocprofiler_configure != nullptr || _force_tool);

    static void*                       rocprofiler_lib_handle    = nullptr;
    static rocprofiler_set_api_table_t rocprofiler_lib_config_fn = nullptr;

    if(_force_tool)
    {
        if(rocprofiler_lib_handle && rocprofiler_lib_config_fn)
            return std::make_pair(rocprofiler_lib_handle, rocprofiler_lib_config_fn);

        if(_rocp_reg_lib.empty()) _rocp_reg_lib = rocprofiler_lib_name;

        auto _rocp_reg_lib_path       = fs::path{ _rocp_reg_lib };
        auto _rocp_reg_lib_path_fname = _rocp_reg_lib_path.filename();
        auto _rocp_reg_lib_path_abs =
            (_rocp_reg_lib_path.is_absolute())
                ? _rocp_reg_lib_path
                : (fs::path{ get_this_library_path() } / _rocp_reg_lib_path_fname);

        // check to see if the rocprofiler library is already loaded
        rocprofiler_lib_handle =
            dlopen(_rocp_reg_lib_path.c_str(), RTLD_NOLOAD | RTLD_LAZY);

        // try to load with the given path
        if(!rocprofiler_lib_handle)
        {
            rocprofiler_lib_handle =
                dlopen(_rocp_reg_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
        }

        // try to load with the absoulte path
        if(!rocprofiler_lib_handle)
        {
            _rocp_reg_lib_path = _rocp_reg_lib_path_abs;
            rocprofiler_lib_handle =
                dlopen(_rocp_reg_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
        }

        // try to load with the basename path
        if(!rocprofiler_lib_handle)
        {
            _rocp_reg_lib_path = _rocp_reg_lib_path_fname;
            rocprofiler_lib_handle =
                dlopen(_rocp_reg_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
        }

        if(rocprofiler_register_verbose >= rocprofiler_register_info_level)
            LOG(INFO) << "loaded " << _rocp_reg_lib_path_fname.string() << " library at "
                      << _rocp_reg_lib_path.string();

        LOG_IF(FATAL, rocprofiler_lib_handle == nullptr)
            << _rocp_reg_lib << " failed to load\n";

        *(void**) (&rocprofiler_lib_config_fn) =
            dlsym(rocprofiler_lib_handle, rocprofiler_lib_register_entrypoint);

        LOG_IF(FATAL, rocprofiler_lib_config_fn == nullptr)
            << _rocp_reg_lib << " did not contain '"
            << rocprofiler_lib_register_entrypoint << "' symbol\n";
    }
    else if(_found_tool && rocprofiler_set_api_table)
    {
        rocprofiler_lib_config_fn = &rocprofiler_set_api_table;
    }

    return std::make_pair(rocprofiler_lib_handle, rocprofiler_lib_config_fn);
}

constexpr auto library_seq       = std::make_index_sequence<ROCP_REG_LAST>{};
auto           global_mutex      = std::recursive_mutex{};
auto           import_info       = rocp_reg_get_imports(library_seq);
auto           instance_counters = std::array<std::atomic_uint64_t, ROCP_REG_LAST>{};
}  // namespace

extern "C" {
rocprofiler_register_error_code_t
rocprofiler_register_library_api_table(
    const char*                                 common_name,
    rocprofiler_register_import_func_t          import_func,
    uint32_t                                    lib_version,
    void**                                      api_tables,
    uint64_t                                    api_table_length,
    rocprofiler_register_library_indentifier_t* register_id)
{
    if(api_table_length < 1) return ROCP_REG_BAD_API_TABLE_LENGTH;

    (void) lib_version;
    (void) api_tables;

    auto _lk = std::unique_lock<std::recursive_mutex>{ global_mutex, std::defer_lock };
    if(_lk.owns_lock()) return ROCP_REG_DEADLOCK;

    auto _scan_result = rocp_reg_scan_for_tools();

    rocp_import* _import_match = nullptr;
    for(auto& itr : import_info)
    {
        if(itr.common_name == common_name)
        {
            _import_match = &itr;
            break;
        }
    }

    // not a supported library name
    if(!_import_match || _import_match->library_idx == ROCP_REG_LAST)
        return ROCP_REG_UNSUPPORTED_API;

    if(import_func != nullptr &&
       common::get_env<bool>("ROCPROFILER_REGISTER_SECURE", false))
    {
        auto _import_func_addr  = reinterpret_cast<uintptr_t>(import_func);
        auto _segment_addresses = binary::get_segment_addresses();
        auto _in_address_range  = [](uintptr_t                                 _addr,
                                    const std::vector<binary::address_range>& _range) {
            for(auto ritr : _range)
            {
                if(_addr >= ritr.start && _addr < ritr.last) return true;
            }
            return false;
        };

        // check that the address of the import function is within the expected library
        // name
        bool _valid_addr = false;
        for(const auto& itr : _segment_addresses)
        {
            if(_in_address_range(_import_func_addr, itr.ranges))
            {
                if(std::regex_search(fs::path{ itr.filepath }.filename().string(),
                                     std::regex{ _import_match->library_name.data() }))
                {
                    _valid_addr = true;
                }
            }
        }

        // the library provided
        if(!_valid_addr) return ROCP_REG_INVALID_API_ADDRESS;
    }

    constexpr auto offset_factor = 64 / std::max<size_t>(ROCP_REG_LAST, 8);
    // if ROCP_REG_LAST > 8, then we can no longer encode 8 instances per lib
    // because we ran out of bits (i.e. max of 8 * 8 = 64)
    static_assert((offset_factor * ROCP_REG_LAST) <= sizeof(uint64_t) * 8,
                  "ROCP_REG_LAST has exceeded the max allowable size");

    // too many instances of the same library
    if(instance_counters.at(_import_match->library_idx) >= offset_factor)
        return ROCP_REG_EXCESS_API_INSTANCES;

    auto  _instance_val = instance_counters.at(_import_match->library_idx)++;
    auto& _bits         = *reinterpret_cast<bitset_t*>(&register_id->handle);
    _bits = bitset_t{ (offset_factor * _import_match->library_idx) + _instance_val };

    if(_bits.to_ulong() != register_id->handle)
        throw std::runtime_error("error encoding register_id");

    // rocprofiler library is dlopened and we have the functor to pass the API data
    auto _activate_rocprofiler = (_scan_result.second != nullptr);

    if(_activate_rocprofiler)
    {
        auto _ret = _scan_result.second(
            common_name, lib_version, _instance_val, api_tables, api_table_length);
        if(_ret != 0) return ROCP_REG_ROCPROFILER_ERROR;
    }
    else
    {
        return ROCP_REG_NO_TOOLS;
    }

    return ROCP_REG_SUCCESS;
}

const char*
rocprofiler_register_error_string(rocprofiler_register_error_code_t _ec)
{
    switch(_ec)
    {
        case ROCP_REG_SUCCESS: return "rocprofiler_register_success";
        case ROCP_REG_NO_TOOLS: return "rocprofiler_register_no_tools";
        case ROCP_REG_DEADLOCK: return "rocprofiler_register_deadlock";
        case ROCP_REG_BAD_API_TABLE_LENGTH:
            return "rocprofiler_register_bad_api_table_length";
        case ROCP_REG_UNSUPPORTED_API: return "rocprofiler_register_unsupported_api";
        case ROCP_REG_INVALID_API_ADDRESS:
            return "rocprofiler_register_invalid_api_address";
        case ROCP_REG_ROCPROFILER_ERROR: return "rocprofiler_register_rocprofiler_error";
        case ROCP_REG_EXCESS_API_INSTANCES:
            return "rocprofiler_register_excess_api_instances";
        case ROCP_REG_ERROR_CODE_END: return "rocprofiler_register_unknown_error";
    }
    return "rocprofiler_register_unknown_error";
}
}
