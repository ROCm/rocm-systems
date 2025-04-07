// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#if !defined(ROCPROFSYS_DL_SOURCE)
#    define ROCPROFSYS_DL_SOURCE 1
#endif

#define ROCPROFSYS_COMMON_LIBRARY_NAME "dl"

#include <timemory/log/color.hpp>

#define ROCPROFSYS_COMMON_LIBRARY_LOG_START                                              \
    fprintf(stderr, "%s", ::tim::log::color::info());
#define ROCPROFSYS_COMMON_LIBRARY_LOG_END fprintf(stderr, "%s", ::tim::log::color::end());

#include "common/defines.h"
#include "common/delimit.hpp"
#include "common/environment.hpp"
#include "common/invoke.hpp"
#include "common/join.hpp"
#include "common/setup.hpp"
#include "dl/dl.hpp"
#include "rocprofiler-systems/categories.h"
#include "rocprofiler-systems/types.h"

#include <timemory/utility/filepath.hpp>

#include <cassert>
#include <chrono>
#include <gnu/libc-version.h>
#include <link.h>
#include <linux/limits.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#if !defined(ROCPROFSYS_USE_ROCM)
#    define ROCPROFSYS_USE_ROCM 0
#endif

#if ROCPROFSYS_USE_ROCM > 0
#    include <rocprofiler-sdk/registration.h>
#endif

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_DLSYM(VARNAME, HANDLE, FUNCNAME)                                      \
    if(HANDLE)                                                                           \
    {                                                                                    \
        *(void**) (&VARNAME) = dlsym(HANDLE, FUNCNAME);                                  \
        if(VARNAME == nullptr && _rocprofsys_dl_verbose >= _warn_verbose)                \
        {                                                                                \
            ROCPROFSYS_COMMON_LIBRARY_LOG_START                                          \
            fprintf(stderr, "[rocprof-sys][dl][pid=%i]> %s :: %s\n", getpid(), FUNCNAME, \
                    dlerror());                                                          \
            ROCPROFSYS_COMMON_LIBRARY_LOG_END                                            \
        }                                                                                \
        else if(_rocprofsys_dl_verbose > _info_verbose)                                  \
        {                                                                                \
            ROCPROFSYS_COMMON_LIBRARY_LOG_START                                          \
            fprintf(stderr, "[rocprof-sys][dl][pid=%i]> %s :: success\n", getpid(),      \
                    FUNCNAME);                                                           \
            ROCPROFSYS_COMMON_LIBRARY_LOG_END                                            \
        }                                                                                \
    }

//--------------------------------------------------------------------------------------//

using main_func_t = int (*)(int, char**, char**);
using init_func_t = void (*)(void);

std::ostream&
operator<<(std::ostream& _os, const SpaceHandle& _handle)
{
    _os << _handle.name;
    return _os;
}

namespace rocprofsys
{
namespace dl
{
namespace
{
inline int
get_rocprofsys_env()
{
    auto&& _debug = get_env("ROCPROFSYS_DEBUG", false);
    return get_env("ROCPROFSYS_VERBOSE", (_debug) ? 100 : 0);
}

inline int
get_rocprofsys_dl_env()
{
    return get_env("ROCPROFSYS_DL_DEBUG", false)
               ? 100
               : get_env("ROCPROFSYS_DL_VERBOSE", get_rocprofsys_env());
}

inline bool&
get_rocprofsys_is_preloaded()
{
    static bool _v = []() {
        auto&& _preload_libs = get_env("LD_PRELOAD", std::string{});
        return (_preload_libs.find("librocprof-sys-dl.so") != std::string::npos);
    }();
    return _v;
}

inline bool
get_rocprofsys_preload()
{
    static bool _v = []() {
        auto&& _preload      = get_env("ROCPROFSYS_PRELOAD", true);
        auto&& _preload_libs = get_env("LD_PRELOAD", std::string{});
        return (_preload &&
                _preload_libs.find("librocprof-sys-dl.so") != std::string::npos);
    }();
    return _v;
}

inline void
reset_rocprofsys_preload()
{
    auto&& _preload_libs = get_env("LD_PRELOAD", std::string{});
    if(_preload_libs.find("librocprof-sys-dl.so") != std::string::npos)
    {
        (void) get_rocprofsys_is_preloaded();
        (void) get_rocprofsys_preload();
        auto _modified_preload = std::string{};
        for(const auto& itr : delimit(_preload_libs, ":"))
        {
            if(itr.find("librocprof-sys") != std::string::npos) continue;
            _modified_preload += common::join("", ":", itr);
        }
        if(!_modified_preload.empty() && _modified_preload.find(':') == 0)
            _modified_preload = _modified_preload.substr(1);

        setenv("LD_PRELOAD", _modified_preload.c_str(), 1);
    }
}

inline pid_t
get_rocprofsys_root_pid()
{
    auto _pid = getpid();
    setenv("ROCPROFSYS_ROOT_PROCESS", std::to_string(_pid).c_str(), 0);
    return get_env("ROCPROFSYS_ROOT_PROCESS", _pid);
}

void
rocprofsys_preinit() ROCPROFSYS_INTERNAL_API;

void
rocprofsys_postinit(std::string exe = {}) ROCPROFSYS_INTERNAL_API;

pid_t _rocprofsys_root_pid = get_rocprofsys_root_pid();

// environment priority:
//  - ROCPROFSYS_DL_DEBUG
//  - ROCPROFSYS_DL_VERBOSE
//  - ROCPROFSYS_DEBUG
//  - ROCPROFSYS_VERBOSE
int _rocprofsys_dl_verbose = get_rocprofsys_dl_env();

// The docs for dlopen suggest that the combination of RTLD_LOCAL + RTLD_DEEPBIND
// (when available) helps ensure that the symbols in the instrumentation library
// librocprof-sys.so will use it's own symbols... not symbols that are potentially
// instrumented. However, this only applies to the symbols in librocprof-sys.so,
// which is NOT self-contained, i.e. symbols in timemory and the libs it links to
// (such as libpapi.so) are not protected by the deep-bind option. Additionally,
// it should be noted that DynInst does *NOT* add instrumentation by manipulating the
// dynamic linker (otherwise it would only be limited to shared libs) -- it manipulates
// the instructions in the binary so that a call to a function such as "main" actually
// calls "main_dyninst", which executes the instrumentation snippets around the actual
// "main" (this is the reason you need the dyninstAPI_RT library).
//
//  UPDATE:
//      Use of RTLD_DEEPBIND has been removed because it causes the dyninst
//      ProcControlAPI to segfault within pthread_cond_wait on certain executables.
//
// Here are the docs on the dlopen options used:
//
// RTLD_LAZY
//    Perform lazy binding. Only resolve symbols as the code that references them is
//    executed. If the symbol is never referenced, then it is never resolved. (Lazy
//    binding is only performed for function references; references to variables are
//    always immediately bound when the library is loaded.)
//
// RTLD_LOCAL
//    This is the converse of RTLD_GLOBAL, and the default if neither flag is specified.
//    Symbols defined in this library are not made available to resolve references in
//    subsequently loaded libraries.
//
// RTLD_DEEPBIND (since glibc 2.3.4)
//    Place the lookup scope of the symbols in this library ahead of the global scope.
//    This means that a self-contained library will use its own symbols in preference to
//    global symbols with the same name contained in libraries that have already been
//    loaded. This flag is not specified in POSIX.1-2001.
//
#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 4
auto        _rocprofsys_dl_dlopen_flags = RTLD_LAZY | RTLD_LOCAL;
const char* _rocprofsys_dl_dlopen_descr = "RTLD_LAZY | RTLD_LOCAL";
#else
auto        _rocprofsys_dl_dlopen_flags = RTLD_LAZY | RTLD_LOCAL;
const char* _rocprofsys_dl_dlopen_descr = "RTLD_LAZY | RTLD_LOCAL";
#endif

/// This class contains function pointers for rocprof-sys's instrumentation functions
struct ROCPROFSYS_INTERNAL_API indirect
{
    ROCPROFSYS_INLINE indirect(const std::string& _omnilib, const std::string& _userlib,
                               const std::string& _dllib)
    : m_omnilib{ common::path::find_path(_omnilib, _rocprofsys_dl_verbose) }
    , m_dllib{ common::path::find_path(_dllib, _rocprofsys_dl_verbose) }
    , m_userlib{ common::path::find_path(_userlib, _rocprofsys_dl_verbose) }
    {
        if(_rocprofsys_dl_verbose >= 1)
        {
            ROCPROFSYS_COMMON_LIBRARY_LOG_START
            fprintf(stderr, "[rocprof-sys][dl][pid=%i] %s resolved to '%s'\n", getpid(),
                    ::basename(_omnilib.c_str()), m_omnilib.c_str());
            fprintf(stderr, "[rocprof-sys][dl][pid=%i] %s resolved to '%s'\n", getpid(),
                    ::basename(_dllib.c_str()), m_dllib.c_str());
            fprintf(stderr, "[rocprof-sys][dl][pid=%i] %s resolved to '%s'\n", getpid(),
                    ::basename(_userlib.c_str()), m_userlib.c_str());
            ROCPROFSYS_COMMON_LIBRARY_LOG_END
        }

        auto _search_paths = common::join(':', common::path::dirname(_omnilib),
                                          common::path::dirname(_dllib));
        common::setup_environ(_rocprofsys_dl_verbose, _search_paths, _omnilib, _dllib);

        m_omnihandle = open(m_omnilib);
        m_userhandle = open(m_userlib);
        init();
    }

    ROCPROFSYS_INLINE ~indirect() { dlclose(m_omnihandle); }

    static ROCPROFSYS_INLINE void* open(const std::string& _lib)
    {
        auto* libhandle = dlopen(_lib.c_str(), _rocprofsys_dl_dlopen_flags);

        if(libhandle)
        {
            if(_rocprofsys_dl_verbose >= 2)
            {
                ROCPROFSYS_COMMON_LIBRARY_LOG_START
                fprintf(stderr,
                        "[rocprof-sys][dl][pid=%i] dlopen(\"%s\", %s) :: success\n",
                        getpid(), _lib.c_str(), _rocprofsys_dl_dlopen_descr);
                ROCPROFSYS_COMMON_LIBRARY_LOG_END
            }
        }
        else
        {
            if(_rocprofsys_dl_verbose >= 0)
            {
                perror("dlopen");
                ROCPROFSYS_COMMON_LIBRARY_LOG_START
                fprintf(stderr, "[rocprof-sys][dl][pid=%i] dlopen(\"%s\", %s) :: %s\n",
                        getpid(), _lib.c_str(), _rocprofsys_dl_dlopen_descr, dlerror());
                ROCPROFSYS_COMMON_LIBRARY_LOG_END
            }
        }

        dlerror();  // Clear any existing error

        return libhandle;
    }

    ROCPROFSYS_INLINE void init()
    {
        if(!m_omnihandle) m_omnihandle = open(m_omnilib);

        int _warn_verbose = 0;
        int _info_verbose = 2;
        // Initialize all pointers
        ROCPROFSYS_DLSYM(rocprofsys_init_library_f, m_omnihandle,
                         "rocprofsys_init_library");
        ROCPROFSYS_DLSYM(rocprofsys_init_tooling_f, m_omnihandle,
                         "rocprofsys_init_tooling");
        ROCPROFSYS_DLSYM(rocprofsys_init_f, m_omnihandle, "rocprofsys_init");
        ROCPROFSYS_DLSYM(rocprofsys_finalize_f, m_omnihandle, "rocprofsys_finalize");
        ROCPROFSYS_DLSYM(rocprofsys_set_env_f, m_omnihandle, "rocprofsys_set_env");
        ROCPROFSYS_DLSYM(rocprofsys_set_mpi_f, m_omnihandle, "rocprofsys_set_mpi");
        ROCPROFSYS_DLSYM(rocprofsys_push_trace_f, m_omnihandle, "rocprofsys_push_trace");
        ROCPROFSYS_DLSYM(rocprofsys_pop_trace_f, m_omnihandle, "rocprofsys_pop_trace");
        ROCPROFSYS_DLSYM(rocprofsys_push_region_f, m_omnihandle,
                         "rocprofsys_push_region");
        ROCPROFSYS_DLSYM(rocprofsys_pop_region_f, m_omnihandle, "rocprofsys_pop_region");
        ROCPROFSYS_DLSYM(rocprofsys_push_category_region_f, m_omnihandle,
                         "rocprofsys_push_category_region");
        ROCPROFSYS_DLSYM(rocprofsys_pop_category_region_f, m_omnihandle,
                         "rocprofsys_pop_category_region");
        ROCPROFSYS_DLSYM(rocprofsys_register_source_f, m_omnihandle,
                         "rocprofsys_register_source");
        ROCPROFSYS_DLSYM(rocprofsys_register_coverage_f, m_omnihandle,
                         "rocprofsys_register_coverage");
        ROCPROFSYS_DLSYM(rocprofsys_progress_f, m_omnihandle, "rocprofsys_progress");
        ROCPROFSYS_DLSYM(rocprofsys_annotated_progress_f, m_omnihandle,
                         "rocprofsys_annotated_progress");

        ROCPROFSYS_DLSYM(kokkosp_print_help_f, m_omnihandle, "kokkosp_print_help");
        ROCPROFSYS_DLSYM(kokkosp_parse_args_f, m_omnihandle, "kokkosp_parse_args");
        ROCPROFSYS_DLSYM(kokkosp_declare_metadata_f, m_omnihandle,
                         "kokkosp_declare_metadata");
        ROCPROFSYS_DLSYM(kokkosp_request_tool_settings_f, m_omnihandle,
                         "kokkosp_request_tool_settings");
        ROCPROFSYS_DLSYM(kokkosp_init_library_f, m_omnihandle, "kokkosp_init_library");
        ROCPROFSYS_DLSYM(kokkosp_finalize_library_f, m_omnihandle,
                         "kokkosp_finalize_library");
        ROCPROFSYS_DLSYM(kokkosp_begin_parallel_for_f, m_omnihandle,
                         "kokkosp_begin_parallel_for");
        ROCPROFSYS_DLSYM(kokkosp_end_parallel_for_f, m_omnihandle,
                         "kokkosp_end_parallel_for");
        ROCPROFSYS_DLSYM(kokkosp_begin_parallel_reduce_f, m_omnihandle,
                         "kokkosp_begin_parallel_reduce");
        ROCPROFSYS_DLSYM(kokkosp_end_parallel_reduce_f, m_omnihandle,
                         "kokkosp_end_parallel_reduce");
        ROCPROFSYS_DLSYM(kokkosp_begin_parallel_scan_f, m_omnihandle,
                         "kokkosp_begin_parallel_scan");
        ROCPROFSYS_DLSYM(kokkosp_end_parallel_scan_f, m_omnihandle,
                         "kokkosp_end_parallel_scan");
        ROCPROFSYS_DLSYM(kokkosp_begin_fence_f, m_omnihandle, "kokkosp_begin_fence");
        ROCPROFSYS_DLSYM(kokkosp_end_fence_f, m_omnihandle, "kokkosp_end_fence");
        ROCPROFSYS_DLSYM(kokkosp_push_profile_region_f, m_omnihandle,
                         "kokkosp_push_profile_region");
        ROCPROFSYS_DLSYM(kokkosp_pop_profile_region_f, m_omnihandle,
                         "kokkosp_pop_profile_region");
        ROCPROFSYS_DLSYM(kokkosp_create_profile_section_f, m_omnihandle,
                         "kokkosp_create_profile_section");
        ROCPROFSYS_DLSYM(kokkosp_destroy_profile_section_f, m_omnihandle,
                         "kokkosp_destroy_profile_section");
        ROCPROFSYS_DLSYM(kokkosp_start_profile_section_f, m_omnihandle,
                         "kokkosp_start_profile_section");
        ROCPROFSYS_DLSYM(kokkosp_stop_profile_section_f, m_omnihandle,
                         "kokkosp_stop_profile_section");
        ROCPROFSYS_DLSYM(kokkosp_allocate_data_f, m_omnihandle, "kokkosp_allocate_data");
        ROCPROFSYS_DLSYM(kokkosp_deallocate_data_f, m_omnihandle,
                         "kokkosp_deallocate_data");
        ROCPROFSYS_DLSYM(kokkosp_begin_deep_copy_f, m_omnihandle,
                         "kokkosp_begin_deep_copy");
        ROCPROFSYS_DLSYM(kokkosp_end_deep_copy_f, m_omnihandle, "kokkosp_end_deep_copy");
        ROCPROFSYS_DLSYM(kokkosp_profile_event_f, m_omnihandle, "kokkosp_profile_event");
        ROCPROFSYS_DLSYM(kokkosp_dual_view_sync_f, m_omnihandle,
                         "kokkosp_dual_view_sync");
        ROCPROFSYS_DLSYM(kokkosp_dual_view_modify_f, m_omnihandle,
                         "kokkosp_dual_view_modify");

#if ROCPROFSYS_USE_ROCM > 0
        ROCPROFSYS_DLSYM(rocprofiler_configure_f, m_omnihandle, "rocprofiler_configure");
#endif

#if ROCPROFSYS_USE_OMPT == 0
        _warn_verbose = 5;
#else
        ROCPROFSYS_DLSYM(ompt_start_tool_f, m_omnihandle, "ompt_start_tool");
#endif

        if(!m_userhandle) m_userhandle = open(m_userlib);
        _warn_verbose = 0;
        ROCPROFSYS_DLSYM(rocprofsys_user_configure_f, m_userhandle,
                         "rocprofsys_user_configure");

        if(rocprofsys_user_configure_f)
        {
            rocprofsys_user_callbacks_t _cb = {};
            _cb.start_trace                 = &rocprofsys_user_start_trace_dl;
            _cb.stop_trace                  = &rocprofsys_user_stop_trace_dl;
            _cb.start_thread_trace          = &rocprofsys_user_start_thread_trace_dl;
            _cb.stop_thread_trace           = &rocprofsys_user_stop_thread_trace_dl;
            _cb.push_region                 = &rocprofsys_user_push_region_dl;
            _cb.pop_region                  = &rocprofsys_user_pop_region_dl;
            _cb.progress                    = &rocprofsys_user_progress_dl;
            _cb.push_annotated_region       = &rocprofsys_user_push_annotated_region_dl;
            _cb.pop_annotated_region        = &rocprofsys_user_pop_annotated_region_dl;
            _cb.annotated_progress          = &rocprofsys_user_annotated_progress_dl;
            (*rocprofsys_user_configure_f)(ROCPROFSYS_USER_REPLACE_CONFIG, _cb, nullptr);
        }
    }

public:
    using user_cb_t = rocprofsys_user_callbacks_t;

    // librocprof-sys functions
    void (*rocprofsys_init_library_f)(void)                                    = nullptr;
    void (*rocprofsys_init_tooling_f)(void)                                    = nullptr;
    void (*rocprofsys_init_f)(const char*, bool, const char*)                  = nullptr;
    void (*rocprofsys_finalize_f)(void)                                        = nullptr;
    void (*rocprofsys_set_env_f)(const char*, const char*)                     = nullptr;
    void (*rocprofsys_set_mpi_f)(bool, bool)                                   = nullptr;
    void (*rocprofsys_register_source_f)(const char*, const char*, size_t, size_t,
                                         const char*)                          = nullptr;
    void (*rocprofsys_register_coverage_f)(const char*, const char*, size_t)   = nullptr;
    void (*rocprofsys_push_trace_f)(const char*)                               = nullptr;
    void (*rocprofsys_pop_trace_f)(const char*)                                = nullptr;
    int (*rocprofsys_push_region_f)(const char*)                               = nullptr;
    int (*rocprofsys_pop_region_f)(const char*)                                = nullptr;
    int (*rocprofsys_push_category_region_f)(rocprofsys_category_t, const char*,
                                             rocprofsys_annotation_t*, size_t) = nullptr;
    int (*rocprofsys_pop_category_region_f)(rocprofsys_category_t, const char*,
                                            rocprofsys_annotation_t*, size_t)  = nullptr;
    void (*rocprofsys_progress_f)(const char*)                                 = nullptr;
    void (*rocprofsys_annotated_progress_f)(const char*, rocprofsys_annotation_t*,
                                            size_t)                            = nullptr;

    // librocprof-sys-user functions
    int (*rocprofsys_user_configure_f)(int, user_cb_t, user_cb_t*) = nullptr;

    // KokkosP functions
    void (*kokkosp_print_help_f)(char*)                                       = nullptr;
    void (*kokkosp_parse_args_f)(int, char**)                                 = nullptr;
    void (*kokkosp_declare_metadata_f)(const char*, const char*)              = nullptr;
    void (*kokkosp_request_tool_settings_f)(const uint32_t,
                                            Kokkos_Tools_ToolSettings*)       = nullptr;
    void (*kokkosp_init_library_f)(const int, const uint64_t, const uint32_t,
                                   void*)                                     = nullptr;
    void (*kokkosp_finalize_library_f)()                                      = nullptr;
    void (*kokkosp_begin_parallel_for_f)(const char*, uint32_t, uint64_t*)    = nullptr;
    void (*kokkosp_end_parallel_for_f)(uint64_t)                              = nullptr;
    void (*kokkosp_begin_parallel_reduce_f)(const char*, uint32_t, uint64_t*) = nullptr;
    void (*kokkosp_end_parallel_reduce_f)(uint64_t)                           = nullptr;
    void (*kokkosp_begin_parallel_scan_f)(const char*, uint32_t, uint64_t*)   = nullptr;
    void (*kokkosp_end_parallel_scan_f)(uint64_t)                             = nullptr;
    void (*kokkosp_begin_fence_f)(const char*, uint32_t, uint64_t*)           = nullptr;
    void (*kokkosp_end_fence_f)(uint64_t)                                     = nullptr;
    void (*kokkosp_push_profile_region_f)(const char*)                        = nullptr;
    void (*kokkosp_pop_profile_region_f)()                                    = nullptr;
    void (*kokkosp_create_profile_section_f)(const char*, uint32_t*)          = nullptr;
    void (*kokkosp_destroy_profile_section_f)(uint32_t)                       = nullptr;
    void (*kokkosp_start_profile_section_f)(uint32_t)                         = nullptr;
    void (*kokkosp_stop_profile_section_f)(uint32_t)                          = nullptr;
    void (*kokkosp_allocate_data_f)(const SpaceHandle, const char*, const void* const,
                                    const uint64_t)                           = nullptr;
    void (*kokkosp_deallocate_data_f)(const SpaceHandle, const char*, const void* const,
                                      const uint64_t)                         = nullptr;
    void (*kokkosp_begin_deep_copy_f)(SpaceHandle, const char*, const void*, SpaceHandle,
                                      const char*, const void*, uint64_t)     = nullptr;
    void (*kokkosp_end_deep_copy_f)()                                         = nullptr;
    void (*kokkosp_profile_event_f)(const char*)                              = nullptr;
    void (*kokkosp_dual_view_sync_f)(const char*, const void* const, bool)    = nullptr;
    void (*kokkosp_dual_view_modify_f)(const char*, const void* const, bool)  = nullptr;

#if ROCPROFSYS_USE_ROCM > 0
    rocprofiler_tool_configure_result_t* (*rocprofiler_configure_f)(
        uint32_t, const char*, uint32_t, rocprofiler_client_id_t*) = nullptr;
#endif

    // OpenMP functions
#if defined(ROCPROFSYS_USE_OMPT) && ROCPROFSYS_USE_OMPT > 0
    ompt_start_tool_result_t* (*ompt_start_tool_f)(unsigned int, const char*);
#endif

    auto get_omni_library() const { return m_omnilib; }
    auto get_user_library() const { return m_userlib; }
    auto get_dl_library() const { return m_dllib; }

private:
    void*       m_omnihandle = nullptr;
    void*       m_userhandle = nullptr;
    std::string m_omnilib    = {};
    std::string m_dllib      = {};
    std::string m_userlib    = {};
};

inline indirect&
get_indirect() ROCPROFSYS_INTERNAL_API;

indirect&
get_indirect()
{
    rocprofsys_preinit_library();

    static auto  _libomni = get_env("ROCPROFSYS_LIBRARY", "librocprof-sys.so");
    static auto  _libuser = get_env("ROCPROFSYS_USER_LIBRARY", "librocprof-sys-user.so");
    static auto  _libdlib = get_env("ROCPROFSYS_DL_LIBRARY", "librocprof-sys-dl.so");
    static auto* _v       = new indirect{ _libomni, _libuser, _libdlib };
    return *_v;
}

auto&
get_inited()
{
    static bool* _v = new bool{ false };
    return *_v;
}

auto&
get_finied()
{
    static bool* _v = new bool{ false };
    return *_v;
}

auto&
get_active()
{
    static bool* _v = new bool{ false };
    return *_v;
}

auto&
get_enabled()
{
    static auto* _v = new std::atomic<bool>{ get_env("ROCPROFSYS_INIT_ENABLED", true) };
    return *_v;
}

auto&
get_thread_enabled()
{
    static thread_local bool _v = get_enabled();
    return _v;
}

auto&
get_thread_count()
{
    static thread_local int64_t _v = 0;
    return _v;
}

auto&
get_thread_status()
{
    static thread_local bool _v = false;
    return _v;
}

InstrumentMode&
get_instrumented()
{
    static auto _v = get_env("ROCPROFSYS_INSTRUMENT_MODE", InstrumentMode::None);
    return _v;
}

// ensure finalization is called
bool _rocprofsys_dl_fini = (std::atexit([]() {
                                if(get_active()) rocprofsys_finalize();
                            }),
                            true);
}  // namespace
}  // namespace dl
}  // namespace rocprofsys

//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_DL_INVOKE(...)                                                        \
    ::rocprofsys::common::invoke(__FUNCTION__, ::rocprofsys::dl::_rocprofsys_dl_verbose, \
                                 (::rocprofsys::dl::get_thread_status() = false),        \
                                 __VA_ARGS__)

#define ROCPROFSYS_DL_IGNORE(...)                                                        \
    ::rocprofsys::common::ignore(__FUNCTION__, ::rocprofsys::dl::_rocprofsys_dl_verbose, \
                                 __VA_ARGS__)

#define ROCPROFSYS_DL_INVOKE_STATUS(STATUS, ...)                                         \
    ::rocprofsys::common::invoke(__FUNCTION__, ::rocprofsys::dl::_rocprofsys_dl_verbose, \
                                 STATUS, __VA_ARGS__)

#define ROCPROFSYS_DL_LOG(LEVEL, ...)                                                    \
    if(::rocprofsys::dl::_rocprofsys_dl_verbose >= LEVEL)                                \
    {                                                                                    \
        fflush(stderr);                                                                  \
        ROCPROFSYS_COMMON_LIBRARY_LOG_START                                              \
        fprintf(stderr, "[rocprof-sys][" ROCPROFSYS_COMMON_LIBRARY_NAME "][%i] ",        \
                getpid());                                                               \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ROCPROFSYS_COMMON_LIBRARY_LOG_END                                                \
        fflush(stderr);                                                                  \
    }

using rocprofsys::dl::get_indirect;
namespace dl = rocprofsys::dl;

extern "C"
{
    void rocprofsys_preinit_library(void)
    {
        if(rocprofsys::common::get_env("ROCPROFSYS_MONOCHROME", tim::log::monochrome()))
            tim::log::monochrome() = true;
    }

    int rocprofsys_preload_library(void)
    {
        return (::rocprofsys::dl::get_rocprofsys_preload()) ? 1 : 0;
    }

    void rocprofsys_init_library(void)
    {
        ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_init_library_f);
    }

    void rocprofsys_init_tooling(void)
    {
        ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_init_tooling_f);
    }

    void rocprofsys_init(const char* a, bool b, const char* c)
    {
        if(dl::get_inited() && dl::get_finied())
        {
            ROCPROFSYS_DL_LOG(
                2, "%s(%s) ignored :: already initialized and finalized\n", __FUNCTION__,
                ::rocprofsys::join(::rocprofsys::QuoteStrings{}, ", ", a, b, c).c_str());
            return;
        }
        else if(dl::get_inited() && dl::get_active())
        {
            ROCPROFSYS_DL_LOG(
                2, "%s(%s) ignored :: already initialized and active\n", __FUNCTION__,
                ::rocprofsys::join(::rocprofsys::QuoteStrings{}, ", ", a, b, c).c_str());
            return;
        }

        if(dl::get_instrumented() < dl::InstrumentMode::PythonProfile)
            dl::rocprofsys_preinit();

        bool _invoked = false;
        ROCPROFSYS_DL_INVOKE_STATUS(_invoked, get_indirect().rocprofsys_init_f, a, b, c);

        if(_invoked)
        {
            dl::get_active()           = true;
            dl::get_inited()           = true;
            dl::_rocprofsys_dl_verbose = dl::get_rocprofsys_dl_env();

            if(dl::get_instrumented() >= dl::InstrumentMode::None &&
               dl::get_instrumented() < dl::InstrumentMode::PythonProfile)
            {
                dl::rocprofsys_postinit((c) ? std::string{ c } : std::string{});
            }
        }
    }

    void rocprofsys_finalize(void)
    {
        if(dl::get_inited() && dl::get_finied())
        {
            ROCPROFSYS_DL_LOG(2, "%s() ignored :: already initialized and finalized\n",
                              __FUNCTION__);
            return;
        }
        else if(dl::get_finied() && !dl::get_active())
        {
            ROCPROFSYS_DL_LOG(2, "%s() ignored :: already finalized but not active\n",
                              __FUNCTION__);
            return;
        }

        bool _invoked = false;
        ROCPROFSYS_DL_INVOKE_STATUS(_invoked, get_indirect().rocprofsys_finalize_f);
        if(_invoked)
        {
            dl::get_active() = false;
            dl::get_finied() = true;
        }
    }

    void rocprofsys_push_trace(const char* name)
    {
        if(!dl::get_active()) return;
        if(dl::get_thread_enabled())
        {
            ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_push_trace_f, name);
        }
        else
        {
            ++dl::get_thread_count();
        }
    }

    void rocprofsys_pop_trace(const char* name)
    {
        if(!dl::get_active()) return;
        if(dl::get_thread_enabled())
        {
            ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_pop_trace_f, name);
        }
        else
        {
            if(dl::get_thread_count()-- == 0) rocprofsys_user_start_thread_trace_dl();
        }
    }

    int rocprofsys_push_region(const char* name)
    {
        if(!dl::get_active()) return 0;
        if(dl::get_thread_enabled())
        {
            return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_push_region_f, name);
        }
        else
        {
            ++dl::get_thread_count();
        }
        return 0;
    }

    int rocprofsys_pop_region(const char* name)
    {
        if(!dl::get_active()) return 0;
        if(dl::get_thread_enabled())
        {
            return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_pop_region_f, name);
        }
        else
        {
            if(dl::get_thread_count()-- == 0) rocprofsys_user_start_thread_trace_dl();
        }
        return 0;
    }

    int rocprofsys_push_category_region(rocprofsys_category_t _category, const char* name,
                                        rocprofsys_annotation_t* _annotations,
                                        size_t                   _annotation_count)
    {
        if(!dl::get_active()) return 0;
        if(dl::get_thread_enabled())
        {
            return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_push_category_region_f,
                                        _category, name, _annotations, _annotation_count);
        }
        else
        {
            ++dl::get_thread_count();
        }
        return 0;
    }

    int rocprofsys_pop_category_region(rocprofsys_category_t _category, const char* name,
                                       rocprofsys_annotation_t* _annotations,
                                       size_t                   _annotation_count)
    {
        if(!dl::get_active()) return 0;
        if(dl::get_thread_enabled())
        {
            return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_pop_category_region_f,
                                        _category, name, _annotations, _annotation_count);
        }
        else
        {
            ++dl::get_thread_count();
        }
        return 0;
    }

    void rocprofsys_set_env(const char* a, const char* b)
    {
        if(dl::get_inited() && dl::get_active())
        {
            ROCPROFSYS_DL_IGNORE(2, "already initialized and active", a, b);
            return;
        }
        ROCPROFSYS_DL_LOG(2, "%s(%s, %s)\n", __FUNCTION__, a, b);
        setenv(a, b, 0);
        // ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_set_env_f, a, b);
    }

    void rocprofsys_set_mpi(bool a, bool b)
    {
        if(dl::get_inited() && dl::get_active())
        {
            ROCPROFSYS_DL_IGNORE(2, "already initialized and active", a, b);
            return;
        }
        ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_set_mpi_f, a, b);
    }

    void rocprofsys_register_source(const char* file, const char* func, size_t line,
                                    size_t address, const char* source)
    {
        ROCPROFSYS_DL_LOG(3, "%s(\"%s\", \"%s\", %zu, %zu, \"%s\")\n", __FUNCTION__, file,
                          func, line, address, source);
        ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_register_source_f, file, func,
                             line, address, source);
    }

    void rocprofsys_register_coverage(const char* file, const char* func, size_t address)
    {
        ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_register_coverage_f, file, func,
                             address);
    }

    int rocprofsys_user_start_trace_dl(void)
    {
        dl::get_enabled().store(true);
        return rocprofsys_user_start_thread_trace_dl();
    }

    int rocprofsys_user_stop_trace_dl(void)
    {
        dl::get_enabled().store(false);
        return rocprofsys_user_stop_thread_trace_dl();
    }

    int rocprofsys_user_start_thread_trace_dl(void)
    {
        dl::get_thread_enabled() = true;
        return 0;
    }

    int rocprofsys_user_stop_thread_trace_dl(void)
    {
        dl::get_thread_enabled() = false;
        return 0;
    }

    int rocprofsys_user_push_region_dl(const char* name)
    {
        if(!dl::get_active()) return 0;
        return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_push_region_f, name);
    }

    int rocprofsys_user_pop_region_dl(const char* name)
    {
        if(!dl::get_active()) return 0;
        return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_pop_region_f, name);
    }

    int rocprofsys_user_progress_dl(const char* name)
    {
        ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_progress_f, name);
        return 0;
    }

    int rocprofsys_user_push_annotated_region_dl(const char*              name,
                                                 rocprofsys_annotation_t* _annotations,
                                                 size_t _annotation_count)
    {
        if(!dl::get_active()) return 0;
        return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_push_category_region_f,
                                    ROCPROFSYS_CATEGORY_USER, name, _annotations,
                                    _annotation_count);
    }

    int rocprofsys_user_pop_annotated_region_dl(const char*              name,
                                                rocprofsys_annotation_t* _annotations,
                                                size_t _annotation_count)
    {
        if(!dl::get_active()) return 0;
        return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_pop_category_region_f,
                                    ROCPROFSYS_CATEGORY_USER, name, _annotations,
                                    _annotation_count);
    }

    int rocprofsys_user_annotated_progress_dl(const char*              name,
                                              rocprofsys_annotation_t* _annotations,
                                              size_t                   _annotation_count)
    {
        ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_annotated_progress_f, name,
                             _annotations, _annotation_count);
        return 0;
    }

    void rocprofsys_progress(const char* _name)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_progress_f, _name);
    }

    void rocprofsys_annotated_progress(const char*              _name,
                                       rocprofsys_annotation_t* _annotations,
                                       size_t                   _annotation_count)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofsys_annotated_progress_f, _name,
                                    _annotations, _annotation_count);
    }

    void rocprofsys_set_instrumented(int _mode)
    {
        ROCPROFSYS_DL_LOG(2, "%s(%i)\n", __FUNCTION__, _mode);
        auto _mode_v = static_cast<dl::InstrumentMode>(_mode);
        if(_mode_v < dl::InstrumentMode::None || _mode_v >= dl::InstrumentMode::Last)
        {
            ROCPROFSYS_DL_LOG(-127,
                              "%s(mode=%i) invoked with invalid instrumentation mode. "
                              "mode should be %i >= mode < %i\n",
                              __FUNCTION__, _mode,
                              static_cast<int>(dl::InstrumentMode::None),
                              static_cast<int>(dl::InstrumentMode::Last));
        }
        dl::get_instrumented() = _mode_v;
    }

    //----------------------------------------------------------------------------------//
    //
    //      KokkosP
    //
    //----------------------------------------------------------------------------------//

    void kokkosp_print_help(char* argv0)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_print_help_f, argv0);
    }

    void kokkosp_parse_args(int argc, char** argv)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_parse_args_f, argc, argv);
    }

    void kokkosp_declare_metadata(const char* key, const char* value)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_declare_metadata_f, key,
                                    value);
    }

    void kokkosp_request_tool_settings(const uint32_t             version,
                                       Kokkos_Tools_ToolSettings* settings)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_request_tool_settings_f,
                                    version, settings);
    }

    void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                              const uint32_t devInfoCount, void* deviceInfo)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_init_library_f, loadSeq,
                                    interfaceVer, devInfoCount, deviceInfo);
    }

    void kokkosp_finalize_library()
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_finalize_library_f);
    }

    void kokkosp_begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_begin_parallel_for_f, name,
                                    devid, kernid);
    }

    void kokkosp_end_parallel_for(uint64_t kernid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_end_parallel_for_f, kernid);
    }

    void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_begin_parallel_reduce_f, name,
                                    devid, kernid);
    }

    void kokkosp_end_parallel_reduce(uint64_t kernid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_end_parallel_reduce_f, kernid);
    }

    void kokkosp_begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_begin_parallel_scan_f, name,
                                    devid, kernid);
    }

    void kokkosp_end_parallel_scan(uint64_t kernid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_end_parallel_scan_f, kernid);
    }

    void kokkosp_begin_fence(const char* name, uint32_t devid, uint64_t* kernid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_begin_fence_f, name, devid,
                                    kernid);
    }

    void kokkosp_end_fence(uint64_t kernid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_end_fence_f, kernid);
    }

    void kokkosp_push_profile_region(const char* name)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_push_profile_region_f, name);
    }

    void kokkosp_pop_profile_region()
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_pop_profile_region_f);
    }

    void kokkosp_create_profile_section(const char* name, uint32_t* secid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_create_profile_section_f, name,
                                    secid);
    }

    void kokkosp_destroy_profile_section(uint32_t secid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_destroy_profile_section_f,
                                    secid);
    }

    void kokkosp_start_profile_section(uint32_t secid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_start_profile_section_f,
                                    secid);
    }

    void kokkosp_stop_profile_section(uint32_t secid)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_stop_profile_section_f, secid);
    }

    void kokkosp_allocate_data(const SpaceHandle space, const char* label,
                               const void* const ptr, const uint64_t size)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_allocate_data_f, space, label,
                                    ptr, size);
    }

    void kokkosp_deallocate_data(const SpaceHandle space, const char* label,
                                 const void* const ptr, const uint64_t size)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_deallocate_data_f, space,
                                    label, ptr, size);
    }

    void kokkosp_begin_deep_copy(SpaceHandle dst_handle, const char* dst_name,
                                 const void* dst_ptr, SpaceHandle src_handle,
                                 const char* src_name, const void* src_ptr, uint64_t size)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_begin_deep_copy_f, dst_handle,
                                    dst_name, dst_ptr, src_handle, src_name, src_ptr,
                                    size);
    }

    void kokkosp_end_deep_copy()
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_end_deep_copy_f);
    }

    void kokkosp_profile_event(const char* name)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_profile_event_f, name);
    }

    void kokkosp_dual_view_sync(const char* label, const void* const data, bool is_device)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_dual_view_sync_f, label, data,
                                    is_device);
    }

    void kokkosp_dual_view_modify(const char* label, const void* const data,
                                  bool is_device)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().kokkosp_dual_view_modify_f, label,
                                    data, is_device);
    }

    //----------------------------------------------------------------------------------//
    //
    //      ROCm
    //
    //----------------------------------------------------------------------------------//

#if ROCPROFSYS_USE_ROCM > 0
    rocprofiler_tool_configure_result_t* rocprofiler_configure(
        uint32_t version, const char* runtime_version, uint32_t priority,
        rocprofiler_client_id_t* client_id)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().rocprofiler_configure_f, version,
                                    runtime_version, priority, client_id);
    }
#endif

    //----------------------------------------------------------------------------------//
    //
    //      OMPT
    //
    //----------------------------------------------------------------------------------//
#if ROCPROFSYS_USE_OMPT > 0
    ompt_start_tool_result_t* ompt_start_tool(unsigned int omp_version,
                                              const char*  runtime_version)
    {
        return ROCPROFSYS_DL_INVOKE(get_indirect().ompt_start_tool_f, omp_version,
                                    runtime_version);
    }
#endif
}

namespace rocprofsys
{
namespace dl
{
namespace
{
bool
rocprofsys_preload() ROCPROFSYS_INTERNAL_API;

std::vector<std::string>
get_link_map(const char*,
             std::vector<int>&& = { (RTLD_LAZY | RTLD_NOLOAD) }) ROCPROFSYS_INTERNAL_API;

const char*
get_default_mode() ROCPROFSYS_INTERNAL_API;

void
verify_instrumented_preloaded() ROCPROFSYS_INTERNAL_API;

std::vector<std::string>
get_link_map(const char* _name, std::vector<int>&& _open_modes)
{
    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_name, _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    auto _chain = std::vector<std::string>{};
    if(_handle)
    {
        struct link_map* _link_map = nullptr;
        dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
        struct link_map* _next = _link_map->l_next;
        while(_next)
        {
            if(_next->l_name != nullptr && !std::string_view{ _next->l_name }.empty())
            {
                _chain.emplace_back(_next->l_name);
            }
            _next = _next->l_next;
        }

        if(_noload == false) dlclose(_handle);
    }
    return _chain;
}

const char*
get_default_mode()
{
    if(get_env("ROCPROFSYS_USE_CAUSAL", false)) return "causal";

    auto _link_map = get_link_map(nullptr);
    for(const auto& itr : _link_map)
    {
        if(itr.find("librocprof-sys-rt.so") != std::string::npos ||
           itr.find("libdyninstAPI_RT.so") != std::string::npos)
            return "trace";
    }

    return "sampling";
}

void
rocprofsys_preinit()
{
    switch(get_instrumented())
    {
        case InstrumentMode::None:
        case InstrumentMode::BinaryRewrite:
        case InstrumentMode::ProcessCreate:
        case InstrumentMode::ProcessAttach:
        {
            auto _use_mpip = get_env("ROCPROFSYS_USE_MPIP", false);
            auto _use_mpi  = get_env("ROCPROFSYS_USE_MPI", _use_mpip);
            auto _causal   = get_env("ROCPROFSYS_USE_CAUSAL", false);
            auto _mode     = get_env("ROCPROFSYS_MODE", get_default_mode());

            if(_use_mpi && !(_causal && _mode == "causal"))
            {
                // only make this call if true bc otherwise, if
                // false, it will disable the MPIP component and
                // we may intercept the MPI init call later.
                // If _use_mpi defaults to true above, calling this
                // will override can current env or config value for
                // ROCPROFSYS_USE_PID.
                rocprofsys_set_mpi(_use_mpi, dl::get_instrumented() ==
                                                 dl::InstrumentMode::ProcessAttach);
            }
            break;
        }
        case InstrumentMode::PythonProfile:
        case InstrumentMode::Last: break;
    }
}

void
rocprofsys_postinit(std::string _exe)
{
    InstrumentMode instrumentMode = get_instrumented();

    switch(instrumentMode)
    {
        case InstrumentMode::None:
        case InstrumentMode::BinaryRewrite:
        case InstrumentMode::ProcessCreate:
        case InstrumentMode::ProcessAttach:
        {
            if(_exe.empty())
                _exe = tim::filepath::readlink(join('/', "/proc", getpid(), "exe"));

            rocprofsys_init_tooling();
            if(_exe.empty())
                rocprofsys_push_trace("main");
            else
                rocprofsys_push_trace(basename(_exe.c_str()));
            break;
        }
        case InstrumentMode::PythonProfile:
        {
            rocprofsys_init_tooling();
            break;
        }
        case InstrumentMode::Last: break;
    }
}

bool
rocprofsys_preload()
{
    auto _preload = get_rocprofsys_is_preloaded() && get_rocprofsys_preload() &&
                    get_env("ROCPROFSYS_ENABLED", true);

    auto _link_map = get_link_map(nullptr);
    auto _instr_mode =
        get_env("ROCPROFSYS_INSTRUMENT_MODE", dl::InstrumentMode::BinaryRewrite);
    for(const auto& itr : _link_map)
    {
        if(itr.find("librocprof-sys-rt.so") != std::string::npos ||
           itr.find("libdyninstAPI_RT.so") != std::string::npos)
        {
            rocprofsys_set_instrumented(static_cast<int>(_instr_mode));
            break;
        }
    }

    verify_instrumented_preloaded();

    static bool _once = false;
    if(_once) return _preload;
    _once = true;

    if(_preload)
    {
        reset_rocprofsys_preload();
        rocprofsys_preinit_library();
    }

    return _preload;
}

void
verify_instrumented_preloaded()
{
    // if preloaded then we are fine
    if(get_rocprofsys_is_preloaded()) return;

    // value returned by get_instrumented is set by either:
    // - the search of the linked libraries
    // - via the instrumenter
    // if binary rewrite or runtime instrumentation, there is an opportunity for
    // LD_PRELOAD
    switch(dl::get_instrumented())
    {
        case dl::InstrumentMode::None:
        case dl::InstrumentMode::ProcessAttach:
        case dl::InstrumentMode::ProcessCreate:
        case dl::InstrumentMode::PythonProfile:
        {
            return;
        }
        case dl::InstrumentMode::BinaryRewrite:
        {
            break;
        }
        case dl::InstrumentMode::Last:
        {
            throw std::runtime_error(
                "Invalid instrumentation type: InstrumentMode::Last");
        }
    }

    static const char* _notice = R"notice(

        NNNNNNNN        NNNNNNNN     OOOOOOOOO     TTTTTTTTTTTTTTTTTTTTTTTIIIIIIIIII      CCCCCCCCCCCCCEEEEEEEEEEEEEEEEEEEEEE
        N:::::::N       N::::::N   OO:::::::::OO   T:::::::::::::::::::::TI::::::::I   CCC::::::::::::CE::::::::::::::::::::E
        N::::::::N      N::::::N OO:::::::::::::OO T:::::::::::::::::::::TI::::::::I CC:::::::::::::::CE::::::::::::::::::::E
        N:::::::::N     N::::::NO:::::::OOO:::::::OT:::::TT:::::::TT:::::TII::::::IIC:::::CCCCCCCC::::CEE::::::EEEEEEEEE::::E
        N::::::::::N    N::::::NO::::::O   O::::::OTTTTTT  T:::::T  TTTTTT  I::::I C:::::C       CCCCCC  E:::::E       EEEEEE
        N:::::::::::N   N::::::NO:::::O     O:::::O        T:::::T          I::::IC:::::C                E:::::E
        N:::::::N::::N  N::::::NO:::::O     O:::::O        T:::::T          I::::IC:::::C                E::::::EEEEEEEEEE
        N::::::N N::::N N::::::NO:::::O     O:::::O        T:::::T          I::::IC:::::C                E:::::::::::::::E
        N::::::N  N::::N:::::::NO:::::O     O:::::O        T:::::T          I::::IC:::::C                E:::::::::::::::E
        N::::::N   N:::::::::::NO:::::O     O:::::O        T:::::T          I::::IC:::::C                E::::::EEEEEEEEEE
        N::::::N    N::::::::::NO:::::O     O:::::O        T:::::T          I::::IC:::::C                E:::::E
        N::::::N     N:::::::::NO::::::O   O::::::O        T:::::T          I::::I C:::::C       CCCCCC  E:::::E       EEEEEE
        N::::::N      N::::::::NO:::::::OOO:::::::O      TT:::::::TT      II::::::IIC:::::CCCCCCCC::::CEE::::::EEEEEEEE:::::E
        N::::::N       N:::::::N OO:::::::::::::OO       T:::::::::T      I::::::::I CC:::::::::::::::CE::::::::::::::::::::E
        N::::::N        N::::::N   OO:::::::::OO         T:::::::::T      I::::::::I   CCC::::::::::::CE::::::::::::::::::::E
        NNNNNNNN         NNNNNNN     OOOOOOOOO           TTTTTTTTTTT      IIIIIIIIII      CCCCCCCCCCCCCEEEEEEEEEEEEEEEEEEEEEE

                                                     _    _  _____ ______
                                                    | |  | |/ ____|  ____|
                                                    | |  | | (___ | |__
                                                    | |  | |\___ \|  __|
                                                    | |__| |____) | |____
                                                     \____/|_____/|______|

                                                         __
                         _ __ ___   ___ _ __  _ __ ___  / _|      ___ _   _ ___       _ __ _   _ _ __
                        | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____/ __| | | / __|_____| '__| | | | '_ \
                        | | | (_) | (__| |_) | | | (_) |  _|_____\__ \ |_| \__ \_____| |  | |_| | | | |
                        |_|  \___/ \___| .__/|_|  \___/|_|       |___/\__, |___/     |_|   \__,_|_| |_|
                                       |_|                            |___/


    Due to a variety of edge cases we've encountered, ROCm Systems Profiller now requires that binary rewritten
    executables and libraries be launched with the 'rocprof-sys-run' executable.

    In order to launch the executable with 'rocprof-sys-run', prefix the current command with 'rocprof-sys-run'
    and a standalone double hyphen ('--').

    For MPI applications, place 'rocprof-sys-run --' after the MPI command.
    E.g.:

        <EXECUTABLE> <ARGS...>
        mpirun -n 2 <EXECUTABLE> <ARGS...>

    should be:

        rocprof-sys-run -- <EXECUTABLE> <ARGS...>
        mpirun -n 2 rocprof-sys-run -- <EXECUTABLE> <ARGS...>

    Note: the command-line arguments passed to 'rocprof-sys-run' (which are specified before the double hyphen) will override configuration variables
    and/or any configuration values specified to 'rocprof-sys-instrument' via the '--config' or '--env' options.
    E.g.:

        $ rocprof-sys-instrument -o ./sleep.inst --env ROCPROFSYS_SAMPLING_DELAY=5.0 -- sleep
        $ echo "ROCPROFSYS_SAMPLING_FREQ = 500" > rocprof-sys.cfg
        $ export ROCPROFSYS_CONFIG_FILE=rocprof-sys.cfg
        $ rocprof-sys-run --sampling-freq=100 --sampling-delay=1.0 -- ./sleep.inst 10

    In the first command, a default sampling delay of 5 seconds in embedded into the instrumented 'sleep.inst'.
    In the second command, the sampling frequency will be set to 500 interrupts per second when ROCm Systems Profiller reads the config file
    In the fourth command, the sampling frequency and sampling delay are overridden to 100 interrupts per second and 1 second, respectively, when sleep.inst runs

    Thanks for using ROCm Systems Profiler and happy optimizing!
    )notice";

    // emit notice
    std::cerr << _notice << std::endl;

    std::quick_exit(EXIT_FAILURE);
}

bool        _handle_preload = rocprofsys_preload();
main_func_t main_real       = nullptr;
init_func_t init_real       = nullptr;
}  // namespace
}  // namespace dl
}  // namespace rocprofsys

extern "C"
{
    void rocprofsys_main_init(void) ROCPROFSYS_INTERNAL_API;
    int  rocprofsys_main(int argc, char** argv, char** envp) ROCPROFSYS_INTERNAL_API;

    void rocprofsys_set_main_init(init_func_t) ROCPROFSYS_INTERNAL_API;
    void rocprofsys_set_main(main_func_t) ROCPROFSYS_INTERNAL_API;

    void rocprofsys_set_main_init(init_func_t _init_real)
    {
        ::rocprofsys::dl::init_real = _init_real;
    }

    void rocprofsys_set_main(main_func_t _main_real)
    {
        ::rocprofsys::dl::main_real = _main_real;
    }

    // void rocprofsys_main_init(int argc, char** argv, char** envp)
    // {
    //     ROCPROFSYS_DL_LOG(0, "%s\n", __FUNCTION__);
    //     using ::rocprofsys::common::get_env;
    //     using ::rocprofsys::dl::get_default_mode;

    //     // prevent re-entry
    //     static int _reentry = 0;
    //     if(_reentry > 0) return -1;
    //     _reentry = 1;

    //     int ret = 0;

    //     if(::rocprofsys::dl::init_real)
    //     {
    //         if(envp)
    //         {
    //             size_t _idx = 0;
    //             while(envp[_idx] != nullptr)
    //             {
    //                 auto _env_v = std::string_view{ envp[_idx++] };
    //                 if(_env_v.find("ROCPROFSYS") != 0 &&
    //                    _env_v.find("librocprof-sys") == std::string_view::npos)
    //                     continue;
    //                 auto _pos = _env_v.find('=');
    //                 if(_pos < _env_v.length())
    //                 {
    //                     auto _var = std::string{ _env_v }.substr(0, _pos);
    //                     auto _val = std::string{ _env_v }.substr(_pos + 1);
    //                     ROCPROFSYS_DL_LOG(1, "%s(%s, %s)\n", "rocprofsys_set_env",
    //                                       _var.c_str(), _val.c_str());
    //                     setenv(_var.c_str(), _val.c_str(), 0);
    //                 }
    //             }
    //         }

    //         ret = (*::rocprofsys::dl::init_real)(argc, argv, envp);
    //     }
    //     else
    //     {
    //         ROCPROFSYS_DL_LOG(
    //             0, "%s\n",
    //             "Unsuccessful wrapping of init: nullptr to real init function");
    //     }

    //     auto _mode = get_env("ROCPROFSYS_MODE", get_default_mode());
    //     rocprofsys_init(_mode.c_str(),
    //                     dl::get_instrumented() == dl::InstrumentMode::BinaryRewrite,
    //                     argv[0]);

    //     return ret;
    // }

    // int rocprofsys_main(int argc, char** argv, char** envp)
    // {
    //     ROCPROFSYS_DL_LOG(0, "%s\n", __FUNCTION__);

    //     // prevent re-entry
    //     static int _reentry = 0;
    //     if(_reentry > 0) return -1;
    //     _reentry = 1;

    //     if(!::rocprofsys::dl::main_real)
    //         throw std::runtime_error("[rocprof-sys][dl] Unsuccessful wrapping of main:
    //         "
    //                                  "nullptr to real main function");

    //     rocprofsys_push_trace(basename(argv[0]));

    //     int ret = (*::rocprofsys::dl::main_real)(argc, argv, envp);

    //     rocprofsys_pop_trace(basename(argv[0]));
    //     rocprofsys_finalize();

    //     return ret;
    // }

    void rocprofsys_main_init(void)
    {
        ROCPROFSYS_DL_LOG(0, "[%s].\n", __FUNCTION__);

        if(::rocprofsys::dl::init_real)
        {
            // Call real init function
            (*::rocprofsys::dl::init_real)();
        }
        else
        {
            ROCPROFSYS_DL_LOG(
                0, "Unsuccessful wrapping of init: real_init function is nullptr.\n");
        }
    }

    int rocprofsys_main(int argc, char** argv, char** envp)
    {
        ROCPROFSYS_DL_LOG(0, "%s\n", __FUNCTION__);
        using ::rocprofsys::common::get_env;
        using ::rocprofsys::dl::get_default_mode;

        // prevent re-entry
        static int _reentry = 0;
        if(_reentry > 0) return -1;
        _reentry = 1;

        if(!::rocprofsys::dl::main_real)
            throw std::runtime_error("[rocprof-sys][dl] Unsuccessful wrapping of main: "
                                     "real_main function is nullptr.");

        if(envp)
        {
            size_t _idx = 0;
            while(envp[_idx] != nullptr)
            {
                auto _env_v = std::string_view{ envp[_idx++] };
                if(_env_v.find("ROCPROFSYS") != 0 &&
                   _env_v.find("librocprof-sys") == std::string_view::npos)
                    continue;
                auto _pos = _env_v.find('=');
                if(_pos < _env_v.length())
                {
                    auto _var = std::string{ _env_v }.substr(0, _pos);
                    auto _val = std::string{ _env_v }.substr(_pos + 1);
                    ROCPROFSYS_DL_LOG(1, "%s(%s, %s)\n", "rocprofsys_set_env",
                                      _var.c_str(), _val.c_str());
                    setenv(_var.c_str(), _val.c_str(), 0);
                }
            }
        }

        auto _mode = get_env("ROCPROFSYS_MODE", get_default_mode());
        rocprofsys_init(_mode.c_str(),
                        dl::get_instrumented() == dl::InstrumentMode::BinaryRewrite,
                        argv[0]);

        int ret = (*::rocprofsys::dl::main_real)(argc, argv, envp);

        rocprofsys_pop_trace(basename(argv[0]));
        rocprofsys_finalize();

        return ret;
    }
}  // extern "C"
