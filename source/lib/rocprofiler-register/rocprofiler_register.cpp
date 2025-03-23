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
#include "details/filesystem.hpp"
#include "details/logging.hpp"
#include "details/scope_destructor.hpp"

#include <fmt/format.h>
#include <glog/logging.h>

#include <array>
#include <atomic>
#include <bitset>
#include <fstream>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <dlfcn.h>
#include <unistd.h>

extern "C" {
#pragma weak rocprofiler_configure
#pragma weak rocprofiler_set_api_table
#pragma weak rocprofiler_register_import_hip
#pragma weak rocprofiler_register_import_hip_static
#pragma weak rocprofiler_register_import_hip_compiler
#pragma weak rocprofiler_register_import_hip_compiler_static
#pragma weak rocprofiler_register_import_hsa
#pragma weak rocprofiler_register_import_hsa_static
#pragma weak rocprofiler_register_import_roctx
#pragma weak rocprofiler_register_import_roctx_static

typedef struct rocprofiler_client_id_t
{
    const char*    name;    ///< clients should set this value for debugging
    const uint32_t handle;  ///< internal handle
} rocprofiler_client_id_t;

typedef void (*rocprofiler_client_finalize_t)(rocprofiler_client_id_t);

typedef int (*rocprofiler_tool_initialize_t)(rocprofiler_client_finalize_t finalize_func,
                                             void*                         tool_data);

typedef void (*rocprofiler_tool_finalize_t)(void* tool_data);

typedef struct rocprofiler_tool_configure_result_t
{
    size_t                        size;        ///< in case of future extensions
    rocprofiler_tool_initialize_t initialize;  ///< context creation
    rocprofiler_tool_finalize_t   finalize;    ///< cleanup
    void* tool_data;  ///< data to provide to init and fini callbacks
} rocprofiler_tool_configure_result_t;

extern rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*);

extern int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t);

extern uint32_t
rocprofiler_register_import_hip(void);

extern uint32_t
rocprofiler_register_import_hip_compiler(void);

extern uint32_t
rocprofiler_register_import_hsa(void);

extern uint32_t
rocprofiler_register_import_roctx(void);

extern uint32_t
rocprofiler_register_import_hip_static(void);

extern uint32_t
rocprofiler_register_import_hip_compiler_static(void);

extern uint32_t
rocprofiler_register_import_hsa_static(void);

extern uint32_t
rocprofiler_register_import_roctx_static(void);
}

namespace
{
using namespace rocprofiler_register;
using rocprofiler_set_api_table_t = decltype(::rocprofiler_set_api_table)*;
using rocp_set_api_table_data_t   = std::tuple<void*, rocprofiler_set_api_table_t>;
using bitset_t = std::bitset<sizeof(rocprofiler_register_library_indentifier_t::handle)>;

static_assert(sizeof(bitset_t) ==
                  sizeof(rocprofiler_register_library_indentifier_t::handle),
              "bitset should be same at uint64_t");

constexpr auto rocprofiler_lib_name                = "librocprofiler-sdk.so";
constexpr auto rocprofiler_lib_register_entrypoint = "rocprofiler_set_api_table";
constexpr auto rocprofiler_register_lib_name =
    "librocprofiler-register.so." ROCPROFILER_REGISTER_SOVERSION;

enum rocp_reg_supported_library  // NOLINT(performance-enum-size)
{
    ROCP_REG_HSA = 0,
    ROCP_REG_HIP,
    ROCP_REG_ROCTX,
    ROCP_REG_HIP_COMPILER,
    ROCP_REG_RCCL,
    ROCP_REG_ROCDECODE,
    ROCP_REG_ROCJPEG,
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

template <size_t Idx>
struct rocp_reg_error_message;

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

#define ROCP_REG_DEFINE_ERROR_MESSAGE(ENUM, MSG)                                         \
    template <>                                                                          \
    struct rocp_reg_error_message<ENUM>                                                  \
    {                                                                                    \
        static constexpr auto value = MSG;                                               \
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

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_HIP_COMPILER,
                               "hip_compiler",
                               "rocprofiler_register_import_hip_compiler",
                               "libamdhip64.so.[6-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_RCCL,
                               "rccl",
                               "rocprofiler_register_import_rccl",
                               "librccl.so.[6-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_ROCDECODE,
                               "rocdecode",
                               "rocprofiler_register_import_rocdecode",
                               "librocdecode.so.[0-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_ROCJPEG,
                               "rocjpeg",
                               "rocprofiler_register_import_rocjpeg",
                               "librocjpeg.so.[0-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_SUCCESS, "Success")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_NO_TOOLS, "rocprofiler-register found no tools")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_DEADLOCK, "rocprofiler-register deadlocked")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_BAD_API_TABLE_LENGTH,
                              "Library passed an invalid number of API tables")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_UNSUPPORTED_API, "Library's API is not supported")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_INVALID_API_ADDRESS,
                              "Invalid API address (secure mode enabled)")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_ROCPROFILER_ERROR,
                              "Unspecified rocprofiler-register error")
ROCP_REG_DEFINE_ERROR_MESSAGE(
    ROCP_REG_EXCESS_API_INSTANCES,
    "Too many instances of the same library API were registered")

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

template <size_t Idx, size_t... Tail>
constexpr auto
rocprofiler_register_error_string(rocprofiler_register_error_code_t _ec,
                                  std::index_sequence<Idx, Tail...>)
{
    if(_ec == Idx) return rocp_reg_error_message<Idx>::value;

    if constexpr(sizeof...(Tail) > 0)
    {
        return rocprofiler_register_error_string(_ec, std::index_sequence<Tail...>{});
    }
    else
    {
        return "rocprofiler_register_unknown_error";
    }
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

rocp_set_api_table_data_t
rocp_load_rocprofiler_lib(std::string _rocp_reg_lib);

struct rocp_scan_data
{
    void*                       handle           = nullptr;
    rocprofiler_set_api_table_t set_api_table_fn = nullptr;
};

rocp_scan_data
rocp_reg_scan_for_tools()
{
    auto* _configure_func = dlsym(RTLD_DEFAULT, "rocprofiler_configure");
    auto  _rocp_tool_libs = common::get_env("ROCP_TOOL_LIBRARIES", std::string{});
    auto  _rocp_reg_lib = common::get_env("ROCPROFILER_REGISTER_LIBRARY", std::string{});
    bool  _force_tool =
        common::get_env("ROCPROFILER_REGISTER_FORCE_LOAD",
                        !_rocp_reg_lib.empty() || !_rocp_tool_libs.empty());
    bool _found_tool =
        (rocprofiler_configure != nullptr || _configure_func != nullptr || _force_tool);

    static void*                       rocprofiler_lib_handle    = nullptr;
    static rocprofiler_set_api_table_t rocprofiler_lib_config_fn = nullptr;

    if(_found_tool)
    {
        if(rocprofiler_lib_handle && rocprofiler_lib_config_fn)
            return rocp_scan_data{ rocprofiler_lib_handle, rocprofiler_lib_config_fn };

        if(_rocp_reg_lib.empty()) _rocp_reg_lib = rocprofiler_lib_name;

        std::tie(rocprofiler_lib_handle, rocprofiler_lib_config_fn) =
            rocp_load_rocprofiler_lib(_rocp_reg_lib);

        LOG_IF(FATAL, !rocprofiler_lib_config_fn)
            << rocprofiler_lib_register_entrypoint << " not found. Tried to dlopen "
            << _rocp_reg_lib;
    }
    else if(_found_tool && rocprofiler_set_api_table)
    {
        rocprofiler_lib_config_fn = &rocprofiler_set_api_table;
    }

    return rocp_scan_data{ rocprofiler_lib_handle, rocprofiler_lib_config_fn };
}

rocp_set_api_table_data_t
rocp_load_rocprofiler_lib(std::string _rocp_reg_lib)
{
    void*                       rocprofiler_lib_handle    = nullptr;
    rocprofiler_set_api_table_t rocprofiler_lib_config_fn = nullptr;

    if(rocprofiler_set_api_table) rocprofiler_lib_config_fn = &rocprofiler_set_api_table;

    // return if found via LD_PRELOAD
    if(rocprofiler_lib_config_fn)
        return std::make_tuple(rocprofiler_lib_handle, rocprofiler_lib_config_fn);

    // look to see if entrypoint function is already a symbol
    *(void**) (&rocprofiler_lib_config_fn) =
        dlsym(RTLD_DEFAULT, rocprofiler_lib_register_entrypoint);

    // return if found via RTLD_DEFAULT
    if(rocprofiler_lib_config_fn)
        return std::make_tuple(rocprofiler_lib_handle, rocprofiler_lib_config_fn);

    if(_rocp_reg_lib.empty()) _rocp_reg_lib = rocprofiler_lib_name;

    auto _rocp_reg_lib_path       = fs::path{ _rocp_reg_lib };
    auto _rocp_reg_lib_path_fname = _rocp_reg_lib_path.filename();
    auto _rocp_reg_lib_path_abs =
        (_rocp_reg_lib_path.is_absolute())
            ? _rocp_reg_lib_path
            : (fs::path{ get_this_library_path() } / _rocp_reg_lib_path_fname);

    // check to see if the rocprofiler library is already loaded
    rocprofiler_lib_handle = dlopen(_rocp_reg_lib_path.c_str(), RTLD_NOLOAD | RTLD_LAZY);

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

    LOG(INFO) << "loaded " << _rocp_reg_lib_path_fname.string() << " library at "
              << _rocp_reg_lib_path.string();

    LOG_IF(WARNING, rocprofiler_lib_handle == nullptr)
        << _rocp_reg_lib << " failed to load\n";

    *(void**) (&rocprofiler_lib_config_fn) =
        dlsym(rocprofiler_lib_handle, rocprofiler_lib_register_entrypoint);

    LOG_IF(WARNING, rocprofiler_lib_config_fn == nullptr)
        << _rocp_reg_lib << " did not contain '" << rocprofiler_lib_register_entrypoint
        << "' symbol\n";

    return std::make_tuple(rocprofiler_lib_handle, rocprofiler_lib_config_fn);
}

struct registered_library_api_table
{
    bool                               propagated     = false;
    const char*                        common_name    = nullptr;
    rocprofiler_register_import_func_t import_func    = nullptr;
    uint32_t                           lib_version    = 0;
    std::vector<void*>                 api_tables     = {};
    uint64_t                           instance_value = 0;
};

constexpr auto instance_bits     = sizeof(uint64_t) * 8;  // bits in instance_counters
constexpr auto max_instances     = instance_bits * ROCP_REG_LAST;
constexpr auto library_seq       = std::make_index_sequence<ROCP_REG_LAST>{};
auto           global_count      = std::atomic<uint32_t>{ 0 };
auto           import_info       = rocp_reg_get_imports(library_seq);
auto           instance_counters = std::array<std::atomic_uint64_t, ROCP_REG_LAST>{};
auto           registered =
    std::array<std::optional<registered_library_api_table>, max_instances>{};

struct scoped_count
{
    scoped_count()
    : value{ ++global_count }
    { }

    ~scoped_count() { --global_count; }

    scoped_count(const scoped_count&)     = delete;
    scoped_count(scoped_count&&) noexcept = delete;
    scoped_count& operator=(const scoped_count&) = delete;
    scoped_count& operator=(scoped_count&&) noexcept = delete;

    uint32_t value = 0;
};

std::optional<registered_library_api_table>*
rocp_add_registered_library_api_table(const char*                        common_name,
                                      rocprofiler_register_import_func_t import_func,
                                      uint32_t                           lib_version,
                                      void**                             api_tables,
                                      uint64_t                           api_tables_len,
                                      uint64_t                           instance_val)
{
    LOG(INFO) << fmt::format("rocprofiler-register library api table registration:\n\t-"
                             "name: {}\n\t- version: {}\n\t- # tables: {}",
                             common_name,
                             lib_version,
                             api_tables_len);

    for(auto& itr : registered)
    {
        if(!itr)
        {
            auto _tables = std::vector<void*>{};
            _tables.reserve(api_tables_len);
            for(uint64_t i = 0; i < api_tables_len; ++i)
                _tables.emplace_back(api_tables[i]);

            itr = registered_library_api_table{
                false,       common_name,        import_func,
                lib_version, std::move(_tables), instance_val
            };
            return &itr;
        }
    }

    return nullptr;
}

rocprofiler_register_error_code_t
rocp_invoke_registrations(bool invoke_all)
{
    auto _count = scoped_count{};
    if(_count.value > 1) return ROCP_REG_DEADLOCK;

    for(auto& itr : registered)
    {
        if(itr && (!itr->propagated || invoke_all))
        {
            auto _scan_result = rocp_reg_scan_for_tools();

            // rocprofiler_set_api_table has been found and we have pass the API data
            auto _activate_rocprofiler = (_scan_result.set_api_table_fn != nullptr);

            if(_activate_rocprofiler)
            {
                auto _ret = _scan_result.set_api_table_fn(itr->common_name,
                                                          itr->lib_version,
                                                          itr->instance_value,
                                                          itr->api_tables.data(),
                                                          itr->api_tables.size());
                if(_ret != 0) return ROCP_REG_ROCPROFILER_ERROR;
                itr->propagated = true;
            }
        }
    }

    return ROCP_REG_SUCCESS;
}
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

    rocprofiler_register::logging::initialize();

    // rocprofiler-register is disabled via environment
    if(!common::get_env("ROCPROFILER_REGISTER_ENABLED", true))
    {
        LOG(INFO) << "rocprofiler-register disabled via ROCPROFILER_REGISTER_ENABLED=0";
        return ROCP_REG_NO_TOOLS;
    }

    auto _count = scoped_count{};
    if(_count.value > 1) return ROCP_REG_DEADLOCK;

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

    auto* reginfo = rocp_add_registered_library_api_table(common_name,
                                                          import_func,
                                                          lib_version,
                                                          api_tables,
                                                          api_table_length,
                                                          _instance_val);

    LOG_IF(WARNING, !reginfo) << fmt::format(
        "rocprofiler-register failed to create registration info for "
        "{} version {} (instance {})",
        common_name,
        lib_version,
        _instance_val);

    if(_bits.to_ulong() != register_id->handle)
        throw std::runtime_error("error encoding register_id");

    // rocprofiler library is dlopened and we have the functor to pass the API data
    auto _activate_rocprofiler = (_scan_result.set_api_table_fn != nullptr);

    if(_activate_rocprofiler)
    {
        auto _ret = _scan_result.set_api_table_fn(
            common_name, lib_version, _instance_val, api_tables, api_table_length);
        if(_ret != 0) return ROCP_REG_ROCPROFILER_ERROR;

        if(reginfo) (*reginfo)->propagated = true;
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
    return rocprofiler_register_error_string(
        _ec, std::make_index_sequence<ROCP_REG_ERROR_CODE_END>{});
}

rocprofiler_register_error_code_t
rocprofiler_register_iterate_registration_info(
    rocprofiler_register_registration_info_cb_t callback,
    void*                                       data)
{
    for(const auto& itr : registered)
    {
        if(itr)
        {
            auto _info = rocprofiler_register_registration_info_t{
                .size             = sizeof(rocprofiler_register_registration_info_t),
                .common_name      = itr->common_name,
                .lib_version      = itr->lib_version,
                .api_table_length = itr->api_tables.size()
            };
            // invoke callback and break if the caller does not return zero
            if(callback(&_info, data) != ROCP_REG_SUCCESS) break;
        }
    }

    return ROCP_REG_SUCCESS;
}

rocprofiler_register_error_code_t
rocprofiler_register_invoke_nonpropagated_registrations() ROCPROFILER_REGISTER_PUBLIC_API;

//
//  This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_invoke_nonpropagated_registrations()
{
    return rocp_invoke_registrations(false);
}

rocprofiler_register_error_code_t
rocprofiler_register_invoke_all_registrations() ROCPROFILER_REGISTER_PUBLIC_API;

//
//  This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_invoke_all_registrations()
{
    return rocp_invoke_registrations(true);
}
}
