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

#include "details/checked_lock.hpp"
#include "details/dl.hpp"
#include "details/environment.hpp"
#include "details/filesystem.hpp"
#include "details/library_config.hpp"
#include "details/logging.hpp"
#include "details/scope_destructor.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <dlfcn.h>
#include <unistd.h>

namespace
{
using rocprofiler_register_library_api_table_func_t =
    decltype(::rocprofiler_register_library_api_table)*;
}

extern "C" {
#pragma weak rocprofiler_configure
#pragma weak rocprofiler_set_api_table
#pragma weak rocprofiler_load_attachment_tool
#pragma weak rocprofiler_attach
#pragma weak rocprofiler_detach
#pragma weak rocprofiler_attach_initialize
#pragma weak rocprofiler_attach_set_api_table
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

extern int
rocprofiler_load_attachment_tool(const char*);

extern int
rocprofiler_attach(void);

extern int
rocprofiler_detach(void);

extern int rocprofiler_attach_initialize(rocprofiler_register_library_api_table_func_t);

extern int
rocprofiler_attach_set_api_table(const char*,
                                 uint64_t,
                                 uint64_t,
                                 void**,
                                 uint64_t,
                                 rocprofiler_register_library_api_table_func_t);

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
using rocprofiler_set_api_table_t        = decltype(::rocprofiler_set_api_table)*;
using rocprofiler_load_attachment_tool_t = decltype(::rocprofiler_load_attachment_tool)*;
using rocprofiler_attach_initialize_t    = decltype(::rocprofiler_attach_initialize)*;
using rocprofiler_attach_set_api_table_t = decltype(::rocprofiler_attach_set_api_table)*;
using rocprofiler_attach_func_t          = decltype(::rocprofiler_attach)*;
using rocprofiler_detach_func_t          = decltype(::rocprofiler_detach)*;
using rocp_set_api_table_data_t          = std::tuple<void*,
                                             rocprofiler_set_api_table_t,
                                             rocprofiler_load_attachment_tool_t,
                                             rocprofiler_attach_func_t,
                                             rocprofiler_detach_func_t>;

using bitset_t = std::bitset<sizeof(rocprofiler_register_library_indentifier_t::handle)>;

static_assert(sizeof(bitset_t) ==
                  sizeof(rocprofiler_register_library_indentifier_t::handle),
              "bitset should be same at uint64_t");

constexpr auto shared_library_prefix =
    std::string_view{ ROCPROFILER_REGISTER_SHARED_LIBRARY_PREFIX };
constexpr auto shared_library_suffix =
    std::string_view{ ROCPROFILER_REGISTER_SHARED_LIBRARY_SUFFIX };

std::string
get_unversioned_library_name(std::string_view base_name)
{
    return fmt::format("{}{}{}", shared_library_prefix, base_name, shared_library_suffix);
}

std::string
get_versioned_library_name(std::string_view base_name, std::string_view soversion)
{
#if defined(_WIN32)
    static_cast<void>(soversion);
    return get_unversioned_library_name(base_name);
#elif defined(__APPLE__)
    return fmt::format(
        "{}{}.{}{}", shared_library_prefix, base_name, soversion, shared_library_suffix);
#else
    return fmt::format(
        "{}{}{}.{}", shared_library_prefix, base_name, shared_library_suffix, soversion);
#endif
}

std::string
get_versioned_library_name(std::string_view base_name, uint32_t soversion)
{
    return get_versioned_library_name(base_name, std::to_string(soversion));
}

enum class load_library_kind : uint8_t
{
    sdk,
    attach,
};

template <load_library_kind>
struct load_library_trait;

#define ROCPROFILER_REGISTER_DEFINE_LOAD_LIBRARY_TRAIT(                                   \
    KIND, BASE_NAME, MIN_SOVERSION, MAX_SOVERSION, ENTRYPOINT)                            \
    template <>                                                                           \
    struct load_library_trait<load_library_kind::KIND>                                    \
    {                                                                                     \
        static constexpr auto        base_name           = std::string_view{ BASE_NAME }; \
        static constexpr uint32_t    min_soversion       = MIN_SOVERSION;                 \
        static constexpr uint32_t    max_soversion       = MAX_SOVERSION;                 \
        static constexpr const char* required_entrypoint = ENTRYPOINT;                    \
                                                                                          \
        static_assert(min_soversion <= max_soversion);                                    \
    }

// These ranges enumerate known-compatible SONAME fallbacks. Libraries found through
// weak symbols, RTLD_DEFAULT, or the unversioned name retain the existing symbol-based
// compatibility behavior.
// rocprofiler_set_api_table has the same signature in SDK SOVERSIONs 0 and 1.
ROCPROFILER_REGISTER_DEFINE_LOAD_LIBRARY_TRAIT(sdk,
                                               "rocprofiler-sdk",
                                               0,
                                               1,
                                               "rocprofiler_set_api_table");
// The attachment helper and its registration entry point were introduced in ABI 1.
ROCPROFILER_REGISTER_DEFINE_LOAD_LIBRARY_TRAIT(attach,
                                               "rocprofiler-sdk-attach",
                                               1,
                                               1,
                                               "rocprofiler_attach_initialize");

#undef ROCPROFILER_REGISTER_DEFINE_LOAD_LIBRARY_TRAIT

using rocprofiler_sdk_load_trait    = load_library_trait<load_library_kind::sdk>;
using rocprofiler_attach_load_trait = load_library_trait<load_library_kind::attach>;

template <typename TraitT>
std::vector<std::string>
get_default_library_candidates()
{
    auto candidates    = std::vector<std::string>{};
    auto append_unique = [&candidates](std::string name) {
        if(std::find(candidates.begin(), candidates.end(), name) == candidates.end())
            candidates.emplace_back(std::move(name));
    };

    // Preserve the existing unversioned lookup, then support runtime-only installations
    // which provide only known-compatible SONAMEs.
    append_unique(get_unversioned_library_name(TraitT::base_name));
    for(auto soversion = TraitT::max_soversion;; --soversion)
    {
        append_unique(get_versioned_library_name(TraitT::base_name, soversion));
        if(soversion == TraitT::min_soversion) break;
    }

    return candidates;
}

constexpr auto rocprofiler_lib_register_entrypoint =
    rocprofiler_sdk_load_trait::required_entrypoint;
constexpr auto rocprofiler_attach_lib_register_entrypoint =
    rocprofiler_attach_load_trait::required_entrypoint;
constexpr auto rocprofiler_attach_lib_set_api_table_entrypoint =
    "rocprofiler_attach_set_api_table";
constexpr auto rocprofiler_lib_load_attachment_tool_entrypoint =
    "rocprofiler_load_attachment_tool";
constexpr auto rocprofiler_lib_is_initialized_entrypoint = "rocprofiler_is_initialized";
constexpr auto rocprofiler_lib_attach_entrypoint         = "rocprofiler_attach";
constexpr auto rocprofiler_lib_detach_entrypoint         = "rocprofiler_detach";

const auto rocprofiler_register_lib_name =
    get_versioned_library_name("rocprofiler-register", ROCPROFILER_REGISTER_SOVERSION);

enum rocp_reg_supported_library  // NOLINT(performance-enum-size)
{
    ROCP_REG_HSA = 0,
    ROCP_REG_HIP,
    ROCP_REG_ROCTX,
    ROCP_REG_HIP_COMPILER,
    ROCP_REG_RCCL,
    ROCP_REG_ROCDECODE,
    ROCP_REG_ROCJPEG,
    ROCP_REG_ROCATTACH,
    ROCP_REG_HIPFILE,
    ROCP_REG_ROCSHMEM,
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

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_ROCSHMEM,
                               "rocshmem",
                               "rocprofiler_register_import_rocshmem",
                               "librocshmem.so.[0-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_ROCATTACH,
                               "rocattach",
                               "rocprofiler_register_import_attach",
                               "librocprofiler-sdk-attach.so.[0-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_HIPFILE,
                               "hipFile",
                               "rocprofiler_register_import_hipFile",
                               "libhipfile.so.[0-9]($|\\.[0-9\\.]+)")

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
ROCP_REG_DEFINE_ERROR_MESSAGE(
    ROCP_REG_INVALID_ARGUMENT,
    "rocprofiler-register API function was provided an invalid argument")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_ATTACHMENT_NOT_AVAILABLE,
                              "rocprofiler-register attach was invoked, but the "
                              "attachment library was never loaded.")

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

struct loaded_library
{
    void* handle     = nullptr;
    void* entrypoint = nullptr;
};

struct opened_library
{
    void*       handle = nullptr;
    std::string path   = {};
};

opened_library open_library_local(std::string_view);

loaded_library
load_library(const std::vector<std::string>&,
             const char*,
             bool prefer_colocated = false,
             bool warn_on_failure  = true);

rocp_set_api_table_data_t
rocp_load_rocprofiler_lib(const std::string& _rocp_reg_lib);

struct rocp_scan_data
{
    void*                              handle                  = nullptr;
    rocprofiler_set_api_table_t        set_api_table_fn        = nullptr;
    rocprofiler_load_attachment_tool_t load_attachment_tool_fn = nullptr;
    rocprofiler_attach_func_t          attach_fn               = nullptr;
    rocprofiler_detach_func_t          detach_fn               = nullptr;
};

auto existing_scanned_data = rocp_scan_data{};

bool
is_attachment_enabled()
{
#if defined(ROCP_REG_DEFAULT_ATTACHMENT) && ROCP_REG_DEFAULT_ATTACHMENT != 0
    constexpr auto default_attachment_enabled = true;
#else
    constexpr auto default_attachment_enabled = false;
#endif

    return common::get_env("ROCP_TOOL_ATTACH", default_attachment_enabled);
}

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

    static void*                              rocprofiler_lib_handle    = nullptr;
    static rocprofiler_set_api_table_t        rocprofiler_lib_config_fn = nullptr;
    static rocprofiler_load_attachment_tool_t rocprofiler_lib_load_attachment_tool_fn =
        nullptr;
    static rocprofiler_attach_func_t rocprofiler_lib_attach_fn = nullptr;
    static rocprofiler_detach_func_t rocprofiler_lib_detach_fn = nullptr;

    if(_found_tool)
    {
        if(rocprofiler_lib_handle && rocprofiler_lib_config_fn)
            return rocp_scan_data{ rocprofiler_lib_handle,
                                   rocprofiler_lib_config_fn,
                                   rocprofiler_lib_load_attachment_tool_fn,
                                   rocprofiler_lib_attach_fn,
                                   rocprofiler_lib_detach_fn };

        std::tie(rocprofiler_lib_handle,
                 rocprofiler_lib_config_fn,
                 rocprofiler_lib_load_attachment_tool_fn,
                 rocprofiler_lib_attach_fn,
                 rocprofiler_lib_detach_fn) = rocp_load_rocprofiler_lib(_rocp_reg_lib);

        LOG_IF(FATAL, !rocprofiler_lib_config_fn)
            << rocprofiler_lib_register_entrypoint << " not found after trying "
            << ((_rocp_reg_lib.empty()) ? "the default rocprofiler-sdk library candidates"
                                        : _rocp_reg_lib);
    }
    return rocp_scan_data{ rocprofiler_lib_handle,
                           rocprofiler_lib_config_fn,
                           rocprofiler_lib_load_attachment_tool_fn,
                           rocprofiler_lib_attach_fn,
                           rocprofiler_lib_detach_fn };
}

rocp_scan_data
rocp_get_propagation_target()
{
    auto scan_result = rocp_reg_scan_for_tools();
    if(scan_result.set_api_table_fn == nullptr &&
       existing_scanned_data.set_api_table_fn != nullptr)
    {
        // Runtime attachment explicitly loads the SDK and is independent of startup
        // FORCE_LOAD policy. Reuse that exact SDK for anytime and future propagation.
        scan_result = existing_scanned_data;
    }
    return scan_result;
}

opened_library
open_library_local(std::string_view _rocp_reg_lib)
{
    void* rocprofiler_lib_handle = nullptr;

    if(_rocp_reg_lib.empty()) return {};

    auto _rocp_reg_lib_path        = fs::path{ _rocp_reg_lib };
    auto _rocp_reg_lib_is_absolute = _rocp_reg_lib_path.is_absolute();

    // check to see if the rocprofiler library is already loaded
    rocprofiler_lib_handle =
        dlopen(_rocp_reg_lib_path.c_str(), RTLD_NOLOAD | RTLD_LOCAL | RTLD_LAZY);

    if(rocprofiler_lib_handle)
    {
        LOG(INFO) << "found loaded " << _rocp_reg_lib << " library at "
                  << _rocp_reg_lib_path.string() << " (handle=" << rocprofiler_lib_handle
                  << ") via RTLD_NOLOAD | RTLD_LOCAL | RTLD_LAZY";
    }

    // Probe with local visibility so a candidate is not globally exposed before its
    // required entry point is validated.
    if(!rocprofiler_lib_handle)
    {
        rocprofiler_lib_handle =
            dlopen(_rocp_reg_lib_path.c_str(), RTLD_LOCAL | RTLD_LAZY);

        if(rocprofiler_lib_handle)
        {
            LOG(INFO) << "opened " << _rocp_reg_lib << " library at "
                      << _rocp_reg_lib_path.string()
                      << " (handle=" << rocprofiler_lib_handle
                      << ") via RTLD_LOCAL | RTLD_LAZY";
        }
    }

    // Try the same relative path from the rocprofiler-register installation.
    if(!rocprofiler_lib_handle && !_rocp_reg_lib_is_absolute)
    {
        _rocp_reg_lib_path = fs::path{ get_this_library_path() } / _rocp_reg_lib_path;
        rocprofiler_lib_handle =
            dlopen(_rocp_reg_lib_path.c_str(), RTLD_LOCAL | RTLD_LAZY);
    }

    LOG_IF(INFO, rocprofiler_lib_handle != nullptr)
        << "locally opened " << _rocp_reg_lib << " library at "
        << _rocp_reg_lib_path.string() << " (handle=" << rocprofiler_lib_handle << ")";

    return opened_library{ rocprofiler_lib_handle, _rocp_reg_lib_path.string() };
}

std::string
format_library_candidates(const std::vector<std::string>& candidates)
{
    return fmt::format("{}", fmt::join(candidates.begin(), candidates.end(), ", "));
}

loaded_library
load_library(const std::vector<std::string>& candidates,
             const char*                     required_entrypoint,
             bool                            prefer_colocated,
             bool                            warn_on_failure)
{
    auto search_candidates = std::vector<std::string>{};
    auto append_unique     = [&search_candidates](std::string candidate) {
        if(std::find(search_candidates.begin(), search_candidates.end(), candidate) ==
           search_candidates.end())
            search_candidates.emplace_back(std::move(candidate));
    };

    if(prefer_colocated)
    {
        // Try every candidate from this installation before searching globally. Otherwise
        // a missing local unversioned name could select an unrelated system installation
        // before reaching a co-located SONAME.
        auto library_directory = fs::path{ get_this_library_path() };
        for(const auto& candidate : candidates)
        {
            auto path = fs::path{ candidate };
            if(path.is_absolute()) continue;

            auto colocated = library_directory / path;
            auto ec        = std::error_code{};
            if(fs::exists(colocated, ec) && !ec) append_unique(colocated.string());
        }
    }

    for(const auto& candidate : candidates)
        append_unique(candidate);

    for(const auto& candidate : search_candidates)
    {
        auto opened = open_library_local(candidate);
        if(!opened.handle) continue;

        dlerror();
        auto* entrypoint = dlsym(opened.handle, required_entrypoint);
        auto* error      = dlerror();
        if(error == nullptr && entrypoint != nullptr)
        {
            auto* handle =
                dlopen(opened.path.c_str(), RTLD_NOLOAD | RTLD_GLOBAL | RTLD_LAZY);
            if(!handle) handle = dlopen(opened.path.c_str(), RTLD_GLOBAL | RTLD_LAZY);

            void*       promoted_entrypoint = nullptr;
            const char* promotion_error     = nullptr;
            if(handle)
            {
                dlerror();
                promoted_entrypoint = dlsym(handle, required_entrypoint);
                promotion_error     = dlerror();
            }

            dlclose(opened.handle);
            if(!handle || promotion_error != nullptr || promoted_entrypoint == nullptr)
            {
                if(handle) dlclose(handle);
                continue;
            }

            LOG(INFO) << "selected " << candidate << " for entry point "
                      << required_entrypoint;
            return loaded_library{ handle, promoted_entrypoint };
        }

        dlclose(opened.handle);
    }

    LOG_IF(WARNING, warn_on_failure)
        << "failed to load a library containing '" << required_entrypoint
        << "'. Tried: " << format_library_candidates(search_candidates);
    return {};
}

void
resolve_sdk_companion_entrypoints(
    void*&                              sdk_handle,
    rocprofiler_set_api_table_t         set_api_table_fn,
    rocprofiler_load_attachment_tool_t& load_attachment_tool_fn,
    rocprofiler_attach_func_t&          attach_fn,
    rocprofiler_detach_func_t&          detach_fn)
{
    if(set_api_table_fn == nullptr) return;

    auto owner_info = Dl_info{};
    if(dladdr(reinterpret_cast<const void*>(set_api_table_fn), &owner_info) == 0 ||
       owner_info.dli_fname == nullptr || owner_info.dli_fbase == nullptr)
    {
        LOG(WARNING) << "Could not identify the rocprofiler-sdk that owns "
                     << rocprofiler_lib_register_entrypoint;
        return;
    }

    auto* owner_handle =
        dlopen(owner_info.dli_fname, RTLD_NOLOAD | RTLD_LOCAL | RTLD_LAZY);
    if(owner_handle == nullptr)
    {
        LOG(WARNING) << "Could not retain the rocprofiler-sdk that owns "
                     << rocprofiler_lib_register_entrypoint << ": " << dlerror();
        return;
    }

    if(sdk_handle == owner_handle)
    {
        // dlopen(RTLD_NOLOAD) acquired one additional reference; the existing handle
        // already pins this object.
        dlclose(owner_handle);
    }
    else
    {
        if(sdk_handle != nullptr) dlclose(sdk_handle);
        sdk_handle = owner_handle;
    }

    auto resolve_companion = [owner_handle, &owner_info](const char* symbol_name,
                                                         auto&       output) {
        auto* symbol = dlsym(owner_handle, symbol_name);
        auto  info   = Dl_info{};
        if(symbol != nullptr && dladdr(symbol, &info) != 0 &&
           info.dli_fbase == owner_info.dli_fbase)
        {
            *(void**) (&output) = symbol;
        }
        else
        {
            output = nullptr;
            LOG_IF(WARNING, symbol != nullptr)
                << "Ignoring " << symbol_name
                << " because it resolves from a different shared library than "
                << rocprofiler_lib_register_entrypoint;
        }
    };

    resolve_companion(rocprofiler_lib_load_attachment_tool_entrypoint,
                      load_attachment_tool_fn);
    resolve_companion(rocprofiler_lib_attach_entrypoint, attach_fn);
    resolve_companion(rocprofiler_lib_detach_entrypoint, detach_fn);
}

rocp_set_api_table_data_t
rocp_load_rocprofiler_lib(const std::string& _rocp_reg_lib)
{
    void*                              rocprofiler_lib_handle                  = nullptr;
    rocprofiler_set_api_table_t        rocprofiler_lib_config_fn               = nullptr;
    rocprofiler_load_attachment_tool_t rocprofiler_lib_load_attachment_tool_fn = nullptr;
    rocprofiler_attach_func_t          rocprofiler_lib_attach_fn               = nullptr;
    rocprofiler_detach_func_t          rocprofiler_lib_detach_fn               = nullptr;

    if(rocprofiler_set_api_table)
    {
        rocprofiler_lib_config_fn = &rocprofiler_set_api_table;
    }

    if(rocprofiler_lib_config_fn)
    {
        resolve_sdk_companion_entrypoints(rocprofiler_lib_handle,
                                          rocprofiler_lib_config_fn,
                                          rocprofiler_lib_load_attachment_tool_fn,
                                          rocprofiler_lib_attach_fn,
                                          rocprofiler_lib_detach_fn);
        return std::make_tuple(rocprofiler_lib_handle,
                               rocprofiler_lib_config_fn,
                               rocprofiler_lib_load_attachment_tool_fn,
                               rocprofiler_lib_attach_fn,
                               rocprofiler_lib_detach_fn);
    }

    // look to see if entrypoint function is already a symbol
    *(void**) (&rocprofiler_lib_config_fn) =
        dlsym(RTLD_DEFAULT, rocprofiler_lib_register_entrypoint);

    if(rocprofiler_lib_config_fn)
    {
        resolve_sdk_companion_entrypoints(rocprofiler_lib_handle,
                                          rocprofiler_lib_config_fn,
                                          rocprofiler_lib_load_attachment_tool_fn,
                                          rocprofiler_lib_attach_fn,
                                          rocprofiler_lib_detach_fn);
        return std::make_tuple(rocprofiler_lib_handle,
                               rocprofiler_lib_config_fn,
                               rocprofiler_lib_load_attachment_tool_fn,
                               rocprofiler_lib_attach_fn,
                               rocprofiler_lib_detach_fn);
    }

    auto use_default_candidates = _rocp_reg_lib.empty();
    auto candidates             = (use_default_candidates)
                                      ? get_default_library_candidates<rocprofiler_sdk_load_trait>()
                                      : std::vector<std::string>{ _rocp_reg_lib };
    auto loaded                 = load_library(
        candidates, rocprofiler_lib_register_entrypoint, use_default_candidates);
    rocprofiler_lib_handle                 = loaded.handle;
    *(void**) (&rocprofiler_lib_config_fn) = loaded.entrypoint;

    if(!rocprofiler_lib_handle)
        return std::make_tuple(rocprofiler_lib_handle,
                               rocprofiler_lib_config_fn,
                               rocprofiler_lib_load_attachment_tool_fn,
                               rocprofiler_lib_attach_fn,
                               rocprofiler_lib_detach_fn);

    resolve_sdk_companion_entrypoints(rocprofiler_lib_handle,
                                      rocprofiler_lib_config_fn,
                                      rocprofiler_lib_load_attachment_tool_fn,
                                      rocprofiler_lib_attach_fn,
                                      rocprofiler_lib_detach_fn);

    LOG_IF(INFO, rocprofiler_lib_config_fn != nullptr)
        << "Found " << rocprofiler_lib_register_entrypoint << " symbol";

    LOG_IF(INFO, rocprofiler_lib_load_attachment_tool_fn != nullptr)
        << "Found " << rocprofiler_lib_load_attachment_tool_entrypoint << " symbol";

    LOG_IF(INFO, rocprofiler_lib_attach_fn != nullptr)
        << "Found " << rocprofiler_lib_attach_entrypoint << " symbol";

    LOG_IF(INFO, rocprofiler_lib_detach_fn != nullptr)
        << "Found " << rocprofiler_lib_detach_entrypoint << " symbol";

    return std::make_tuple(rocprofiler_lib_handle,
                           rocprofiler_lib_config_fn,
                           rocprofiler_lib_load_attachment_tool_fn,
                           rocprofiler_lib_attach_fn,
                           rocprofiler_lib_detach_fn);
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
auto           import_info       = rocp_reg_get_imports(library_seq);
auto           instance_counters = std::array<std::atomic_uint64_t, ROCP_REG_LAST>{};
auto           registered =
    std::array<std::optional<registered_library_api_table>, max_instances>{};
// Serialises concurrent callers and detects (disallowed) recursive re-entry.
auto registration_mutex = common::checked_mutex{};
// Serialises process-attachment lifecycle operations without holding
// registration_mutex across SDK and tool callbacks.
auto attachment_operation_mutex    = common::checked_mutex{};
auto previous_attachment_tool_path = std::string{};

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

    constexpr auto rocattach_name =
        supported_library_trait<ROCP_REG_ROCATTACH>::common_name;
    const bool is_rocattach = (std::string_view{ common_name } == rocattach_name);

    auto _tables = std::vector<void*>{};
    _tables.reserve(api_tables_len);
    for(uint64_t i = 0; i < api_tables_len; ++i)
        _tables.emplace_back(api_tables[i]);

    auto _entry =
        registered_library_api_table{ false,       common_name,        import_func,
                                      lib_version, std::move(_tables), instance_val };

    if(is_rocattach)
    {
        // Insert rocattach at index 0 so that rocp_invoke_registrations always propagates
        // it before any other API table. Shift existing entries right to make room.
        auto* end = registered.begin();
        while(end != registered.end() && *end)
            ++end;
        if(end == registered.end()) return nullptr;
        for(auto* itr = end; itr != registered.begin(); --itr)
            *itr = std::move(*(itr - 1));
        registered[0] = std::move(_entry);
        return registered.data();
    }

    for(auto& itr : registered)
    {
        if(!itr)
        {
            itr = std::move(_entry);
            return &itr;
        }
    }

    return nullptr;
}

rocprofiler_register_error_code_t
rocp_propagate_registrations(bool invoke_all, const rocp_scan_data& scan_result)
{
    if(scan_result.set_api_table_fn == nullptr) return ROCP_REG_SUCCESS;

    existing_scanned_data = scan_result;
    for(auto& itr : registered)
    {
        if(itr && (!itr->propagated || invoke_all))
        {
            auto _ret = scan_result.set_api_table_fn(itr->common_name,
                                                     itr->lib_version,
                                                     itr->instance_value,
                                                     itr->api_tables.data(),
                                                     itr->api_tables.size());
            if(_ret != 0) return ROCP_REG_ROCPROFILER_ERROR;
            itr->propagated = true;
        }
    }

    return ROCP_REG_SUCCESS;
}

rocprofiler_register_error_code_t
rocp_invoke_registrations(bool invoke_all)
{
    auto _lk = common::checked_lock{ registration_mutex };
    if(_lk.recursive) return ROCP_REG_DEADLOCK;

    return rocp_propagate_registrations(invoke_all, rocp_get_propagation_target());
}

struct environment_variable_state
{
    std::string name    = {};
    std::string value   = {};
    bool        was_set = false;
};

using environment_snapshot_t = std::vector<environment_variable_state>;

environment_snapshot_t
load_environment_buffer(const char* environment_buffer)
{
    auto snapshot = environment_snapshot_t{};

    // environment_buffer is a null-character delimited list of name value pairs.
    // Each name and value is delimited separately.
    // The first 4 bytes contain a uint32_t count of pairs.

    if(!environment_buffer)
    {
        LOG(WARNING) << "Attachment was invoked with no environment variables provided "
                        "for what to trace.";
        return snapshot;
    }

    const uint32_t pair_count = *reinterpret_cast<const uint32_t*>(environment_buffer);
    const char*    position   = environment_buffer + sizeof(uint32_t);
    for(uint32_t pair_idx = 0; pair_idx < pair_count; ++pair_idx)
    {
        const char* name = position;
        position += strlen(name) + 1;
        const char* value = position;
        position += strlen(value) + 1;

        if(std::string_view{ name } == "ROCPROFILER_REGISTER_TOOL_ATTACHED")
        {
            LOG(INFO) << "Ignoring internal attachment marker from environment buffer";
            continue;
        }

        LOG(INFO) << "Attachment adding environment variable: " << name << "=" << value;
        const auto* old_value = std::getenv(name);
        snapshot.emplace_back(environment_variable_state{
            std::string{ name },
            (old_value != nullptr) ? std::string{ old_value } : std::string{},
            old_value != nullptr,
        });
        if(setenv(name, value, 1) != 0)
        {
            LOG(ERROR) << "Failed to set attachment environment variable " << name;
            snapshot.pop_back();
        }
    }

    return snapshot;
}

void
restore_environment(const environment_snapshot_t& snapshot)
{
    for(auto itr = snapshot.rbegin(); itr != snapshot.rend(); ++itr)
    {
        auto status = (itr->was_set) ? setenv(itr->name.c_str(), itr->value.c_str(), 1)
                                     : unsetenv(itr->name.c_str());
        LOG_IF(ERROR, status != 0)
            << "Failed to restore attachment environment variable " << itr->name;
    }
}

bool
is_attachment_library_registered()
{
    for(const auto& itr : registered)
    {
        if(itr.has_value() &&
           std::string_view{ itr->common_name } ==
               supported_library_trait<ROCP_REG_ROCATTACH>::common_name)
        {
            return true;
        }
    }
    return false;
}

constexpr auto offset_factor = 64 / std::max<size_t>(ROCP_REG_LAST, 8);

rocprofiler_register_error_code_t
register_functor(const char*                                 common_name,
                 rocprofiler_register_import_func_t          import_func,
                 uint32_t                                    lib_version,
                 void**                                      api_tables,
                 uint64_t                                    api_table_length,
                 rocprofiler_register_library_indentifier_t* register_id)
{
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

    return ROCP_REG_SUCCESS;
};

rocprofiler_register_error_code_t
initialize_attachment_hsa_interception(const loaded_library& attach_library,
                                       const char*           common_name,
                                       uint32_t              lib_version,
                                       uint64_t              instance_value,
                                       void**                api_tables,
                                       uint64_t              api_table_length)
{
    // If HSA is the first runtime, propagate the already-registered rocattach
    // table first. SDK client configuration can then determine whether proxy
    // interception is needed before HSA is modified.
    auto startup_scan = rocp_reg_scan_for_tools();
    if(startup_scan.set_api_table_fn != nullptr)
    {
        auto status = rocp_propagate_registrations(false, startup_scan);
        if(status != ROCP_REG_SUCCESS) return status;
    }

    using is_initialized_t = int (*)(int*);
    auto is_initialized_fn = is_initialized_t{};
    *(void**) (&is_initialized_fn) =
        dlsym(RTLD_DEFAULT, rocprofiler_lib_is_initialized_entrypoint);
    auto       sdk_init_status = int{ 0 };
    const auto sdk_is_initialized =
        (is_initialized_fn != nullptr && is_initialized_fn(&sdk_init_status) == 0 &&
         sdk_init_status > 0);
    if(sdk_is_initialized)
    {
        LOG(INFO) << "Skipping attachment-library HSA interception because "
                     "rocprofiler-sdk is already initialized; anytime initialization "
                     "will update SDK interception when the attachment client is loaded.";
        return ROCP_REG_SUCCESS;
    }

    auto set_api_table_fn = rocprofiler_attach_set_api_table_t{};
    *(void**) (&set_api_table_fn) =
        dlsym(attach_library.handle, rocprofiler_attach_lib_set_api_table_entrypoint);
    if(set_api_table_fn == nullptr)
    {
        LOG(ERROR) << "Attachment is enabled, but the attach library HSA entry point "
                      "was not found. Startup profiling remains available.";
        return ROCP_REG_SUCCESS;
    }

    auto status = set_api_table_fn(common_name,
                                   lib_version,
                                   instance_value,
                                   api_tables,
                                   api_table_length,
                                   &register_functor);
    LOG_IF(ERROR, status != 0)
        << "Attachment library HSA registration returned an error: " << status
        << ". Startup profiling remains available.";
    return ROCP_REG_SUCCESS;
}

rocprofiler_register_error_code_t
initialize_attachment_library(bool               attachment_enabled,
                              const rocp_import& import_info_entry,
                              const char*        common_name,
                              uint32_t           lib_version,
                              uint64_t           instance_value,
                              void**             api_tables,
                              uint64_t           api_table_length)
{
    if(!attachment_enabled) return ROCP_REG_SUCCESS;

    const auto attachment_explicitly_enabled =
        (std::getenv("ROCP_TOOL_ATTACH") != nullptr);
    static auto loaded_attach_library =
        load_library(get_default_library_candidates<rocprofiler_attach_load_trait>(),
                     rocprofiler_attach_lib_register_entrypoint,
                     true,
                     attachment_explicitly_enabled);
    if(!loaded_attach_library.handle)
    {
        static auto warning_once = std::once_flag{};
        std::call_once(warning_once, [attachment_explicitly_enabled]() {
            LOG_IF(WARNING, attachment_explicitly_enabled)
                << "Attachment is enabled, but the attach library was not found or "
                   "could not be loaded. Startup profiling remains available.";
            LOG_IF(INFO, !attachment_explicitly_enabled)
                << "Default attachment support is unavailable because the attach "
                   "library was not found. Startup profiling remains available.";
        });
        return ROCP_REG_SUCCESS;
    }

    auto initialize_fn         = rocprofiler_attach_initialize_t{};
    *(void**) (&initialize_fn) = loaded_attach_library.entrypoint;
    if(initialize_fn == nullptr)
    {
        LOG(ERROR) << "Attachment is enabled, but the attach library initialization "
                      "entry point was not found. Startup profiling remains available.";
        return ROCP_REG_SUCCESS;
    }

    auto status = initialize_fn(&register_functor);
    if(status != 0)
    {
        LOG(ERROR) << "Attachment library initialization returned an error: " << status
                   << ". Startup profiling remains available.";
        return ROCP_REG_SUCCESS;
    }

    static auto success_log_once = std::once_flag{};
    std::call_once(success_log_once,
                   []() { LOG(INFO) << "Successfully initialized attachment listener"; });

    if(import_info_entry.library_idx == ROCP_REG_HSA)
        return initialize_attachment_hsa_interception(loaded_attach_library,
                                                      common_name,
                                                      lib_version,
                                                      instance_value,
                                                      api_tables,
                                                      api_table_length);

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

    auto _lk = common::checked_lock{ registration_mutex };
    if(_lk.recursive) return ROCP_REG_DEADLOCK;

    const auto _attachment_enabled = is_attachment_enabled();

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

    // Initialize attachment before scanning for startup tools so rocattach is
    // always the first table propagated to rocprofiler-sdk.
    if(auto status = initialize_attachment_library(_attachment_enabled,
                                                   *_import_match,
                                                   common_name,
                                                   lib_version,
                                                   _instance_val,
                                                   api_tables,
                                                   api_table_length);
       status != ROCP_REG_SUCCESS)
        return status;

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

    auto _scan_result = rocp_get_propagation_target();
    if(_scan_result.set_api_table_fn != nullptr)
    {
        return rocp_propagate_registrations(false, _scan_result);
    }
    return ROCP_REG_NO_TOOLS;
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

//
//  This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_invoke_nonpropagated_registrations() ROCPROFILER_REGISTER_PUBLIC_API;

rocprofiler_register_error_code_t
rocprofiler_register_invoke_nonpropagated_registrations()
{
    return rocp_invoke_registrations(false);
}

//
//  This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_invoke_all_registrations() ROCPROFILER_REGISTER_PUBLIC_API;

// This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_invoke_prestore_loads() ROCPROFILER_REGISTER_PUBLIC_API;

rocprofiler_register_error_code_t
rocprofiler_register_invoke_all_registrations()
{
    return rocp_invoke_registrations(true);
}

rocprofiler_register_error_code_t
rocprofiler_register_attach(const char* environment_buffer,
                            const char* tool_lib_path) ROCPROFILER_REGISTER_PUBLIC_API;

rocprofiler_register_error_code_t
rocprofiler_register_detach() ROCPROFILER_REGISTER_PUBLIC_API;

//
//  This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_attach(const char* environment_buffer, const char* tool_lib_path)
{
    if(tool_lib_path == nullptr)
    {
        LOG(ERROR) << "rocprofiler_register_attach requires a tool library path";
        return ROCP_REG_INVALID_ARGUMENT;
    }

    auto operation_lk = common::checked_lock{ attachment_operation_mutex };
    if(operation_lk.recursive) return ROCP_REG_DEADLOCK;

    // If the attachment library has not been loaded when attach is called, tracing
    // that relies on proxy queues will fail (e.g. kernel tracing).
    // Log error and abort.
    auto attachment_available = false;
    {
        auto _lk = common::checked_lock{ registration_mutex };
        if(_lk.recursive) return ROCP_REG_DEADLOCK;
        attachment_available = is_attachment_library_registered();
    }
    if(!attachment_available)
    {
        LOG(ERROR)
            << "rocprofiler-register attach was invoked, but the rocprofiler-attach "
               "library was never loaded. Start the app with environment variable "
               "ROCP_TOOL_ATTACH=1 or build rocprofiler-register with cmake option "
               "ROCPROFILER_REGISTER_BUILD_DEFAULT_ATTACHMENT=ON";
        return ROCP_REG_ATTACHMENT_NOT_AVAILABLE;
    }

    if(!previous_attachment_tool_path.empty() &&
       previous_attachment_tool_path != tool_lib_path)
    {
        LOG(WARNING) << "rocprofiler_register_attach invoked with a different "
                        "tool_lib_path ("
                     << tool_lib_path << ") than a previous attach (previous="
                     << previous_attachment_tool_path << "). This is not supported.";
        return ROCP_REG_INVALID_ARGUMENT;
    }

    LOG(INFO) << "rocprofiler_register_attach started with tool_lib_path: "
              << tool_lib_path;

    // Successful sessions retain these values for same-configuration reattach.
    // Failed attempts must not poison the retained environment.
    auto environment_snapshot = load_environment_buffer(environment_buffer);
    auto attach_succeeded     = false;
    auto environment_scope    = common::scope_destructor{ [&]() {
        if(!attach_succeeded) restore_environment(environment_snapshot);
    } };

    // No previous tool library was attached
    if(previous_attachment_tool_path.empty())
    {
        auto _rocp_reg_lib =
            common::get_env("ROCPROFILER_REGISTER_LIBRARY", std::string{});
        auto [handle, set_api_table_fn, load_attachment_tool_fn, attach_fn, detach_fn] =
            rocp_load_rocprofiler_lib(_rocp_reg_lib);
        auto attach_scan_data = rocp_scan_data{
            handle, set_api_table_fn, load_attachment_tool_fn, attach_fn, detach_fn
        };
        {
            auto _lk = common::checked_lock{ registration_mutex };
            if(_lk.recursive) return ROCP_REG_DEADLOCK;
            existing_scanned_data = attach_scan_data;
        }

        if(attach_scan_data.load_attachment_tool_fn == nullptr)
        {
            LOG(ERROR)
                << rocprofiler_lib_load_attachment_tool_entrypoint
                << " was not found in rocprofiler-sdk. Runtime attachment requires a "
                   "rocprofiler-sdk version with anytime attachment support.";
            return ROCP_REG_NO_TOOLS;
        }

        const auto* marker_value = std::getenv("ROCPROFILER_REGISTER_TOOL_ATTACHED");
        const auto  had_marker   = (marker_value != nullptr);
        const auto  old_marker =
            (had_marker) ? std::string{ marker_value } : std::string{};
        auto marker_scope = common::scope_destructor{
            [had_marker, old_marker]() {
                if(had_marker)
                    setenv("ROCPROFILER_REGISTER_TOOL_ATTACHED", old_marker.c_str(), 1);
                else
                    unsetenv("ROCPROFILER_REGISTER_TOOL_ATTACHED");
            },
            []() { setenv("ROCPROFILER_REGISTER_TOOL_ATTACHED", "1", 1); }
        };

        auto _ret = attach_scan_data.load_attachment_tool_fn(tool_lib_path);
        if(_ret != 0)
        {
            LOG(ERROR) << "rocprofiler-sdk failed to load attachment tool "
                       << tool_lib_path << ": " << _ret;
            return ROCP_REG_ROCPROFILER_ERROR;
        }
        previous_attachment_tool_path = tool_lib_path;
    }

    auto attach_fn = rocprofiler_attach_func_t{};
    {
        auto _lk = common::checked_lock{ registration_mutex };
        if(_lk.recursive) return ROCP_REG_DEADLOCK;
        attach_fn = existing_scanned_data.attach_fn;
    }
    if(attach_fn == nullptr) return ROCP_REG_NO_TOOLS;

    LOG(INFO) << "rocprofiler-sdk attach starting...";
    auto _ret = attach_fn();

    LOG(INFO) << "rocprofiler-sdk attach completed.";

    attach_succeeded = (_ret == 0);
    return (attach_succeeded) ? ROCP_REG_SUCCESS : ROCP_REG_ROCPROFILER_ERROR;
}

//
//  This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_detach()
{
    auto operation_lk = common::checked_lock{ attachment_operation_mutex };
    if(operation_lk.recursive) return ROCP_REG_DEADLOCK;

    LOG(INFO) << "rocprofiler_register_detach started";

    auto attachment_available = false;
    {
        auto _lk = common::checked_lock{ registration_mutex };
        if(_lk.recursive) return ROCP_REG_DEADLOCK;
        attachment_available = is_attachment_library_registered();
    }
    if(!attachment_available)
    {
        LOG(ERROR)
            << "rocprofiler-register detach was invoked, but the rocprofiler-attach "
               "library was never loaded. Start the app with environment variable "
               "ROCP_TOOL_ATTACH=1 or build rocprofiler-register with cmake option "
               "ROCPROFILER_REGISTER_BUILD_DEFAULT_ATTACHMENT=ON";
        return ROCP_REG_ATTACHMENT_NOT_AVAILABLE;
    }

    auto detach_fn = rocprofiler_detach_func_t{};
    {
        auto _lk = common::checked_lock{ registration_mutex };
        if(_lk.recursive) return ROCP_REG_DEADLOCK;
        detach_fn = existing_scanned_data.detach_fn;
    }

    if(detach_fn)
    {
        LOG(INFO) << "rocprofiler-sdk detach starting...";
        auto _ret = detach_fn();
        LOG(INFO) << "rocprofiler-sdk detach completed.";
        return (_ret == 0) ? ROCP_REG_SUCCESS : ROCP_REG_ROCPROFILER_ERROR;
    }

    LOG(ERROR) << "detach entry point is NULL";
    return ROCP_REG_NO_TOOLS;
}
}
