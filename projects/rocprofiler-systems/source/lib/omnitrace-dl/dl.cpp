// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#if !defined(OMNITRACE_DL_SOURCE)
#    define OMNITRACE_DL_SOURCE 1
#endif

#define OMNITRACE_COMMON_LIBRARY_NAME "dl"

#include <timemory/log/color.hpp>

#define OMNITRACE_COMMON_LIBRARY_LOG_START                                               \
    fprintf(stderr, "%s", ::tim::log::color::info());
#define OMNITRACE_COMMON_LIBRARY_LOG_END fprintf(stderr, "%s", ::tim::log::color::end());

#include "common/defines.h"
#include "common/delimit.hpp"
#include "common/environment.hpp"
#include "common/invoke.hpp"
#include "common/join.hpp"
#include "common/setup.hpp"
#include "dl.hpp"

#include <cassert>
#include <gnu/libc-version.h>

//--------------------------------------------------------------------------------------//

#define OMNITRACE_DLSYM(VARNAME, HANDLE, FUNCNAME)                                       \
    if(HANDLE)                                                                           \
    {                                                                                    \
        *(void**) (&VARNAME) = dlsym(HANDLE, FUNCNAME);                                  \
        if(VARNAME == nullptr && _omnitrace_dl_verbose >= _warn_verbose)                 \
        {                                                                                \
            OMNITRACE_COMMON_LIBRARY_LOG_START                                           \
            fprintf(stderr, "[omnitrace][dl][pid=%i]> %s :: %s\n", getpid(), FUNCNAME,   \
                    dlerror());                                                          \
            OMNITRACE_COMMON_LIBRARY_LOG_END                                             \
        }                                                                                \
        else if(_omnitrace_dl_verbose > _info_verbose)                                   \
        {                                                                                \
            OMNITRACE_COMMON_LIBRARY_LOG_START                                           \
            fprintf(stderr, "[omnitrace][dl][pid=%i]> %s :: success\n", getpid(),        \
                    FUNCNAME);                                                           \
            OMNITRACE_COMMON_LIBRARY_LOG_END                                             \
        }                                                                                \
    }

//--------------------------------------------------------------------------------------//

std::ostream&
operator<<(std::ostream& _os, const SpaceHandle& _handle)
{
    _os << _handle.name;
    return _os;
}

namespace omnitrace
{
inline namespace dl
{
namespace
{
inline int
get_omnitrace_env()
{
    auto&& _debug = get_env("OMNITRACE_DEBUG", false);
    return get_env("OMNITRACE_VERBOSE", (_debug) ? 100 : 0);
}

inline int
get_omnitrace_dl_env()
{
    return get_env("OMNITRACE_DL_DEBUG", false)
               ? 100
               : get_env("OMNITRACE_DL_VERBOSE", get_omnitrace_env());
}

// environment priority:
//  - OMNITRACE_DL_DEBUG
//  - OMNITRACE_DL_VERBOSE
//  - OMNITRACE_DEBUG
//  - OMNITRACE_VERBOSE
int _omnitrace_dl_verbose = get_omnitrace_dl_env();

// The docs for dlopen suggest that the combination of RTLD_LOCAL + RTLD_DEEPBIND
// (when available) helps ensure that the symbols in the instrumentation library
// libomnitrace.so will use it's own symbols... not symbols that are potentially
// instrumented. However, this only applies to the symbols in libomnitrace.so,
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
auto        _omnitrace_dl_dlopen_flags = RTLD_LAZY | RTLD_LOCAL;
const char* _omnitrace_dl_dlopen_descr = "RTLD_LAZY | RTLD_LOCAL";
#else
auto        _omnitrace_dl_dlopen_flags = RTLD_LAZY | RTLD_LOCAL;
const char* _omnitrace_dl_dlopen_descr = "RTLD_LAZY | RTLD_LOCAL";
#endif

/// This class contains function pointers for omnitrace's instrumentation functions
struct OMNITRACE_HIDDEN_API indirect
{
    OMNITRACE_INLINE indirect(const std::string& _omnilib, const std::string& _userlib,
                              const std::string& _dllib)
    : m_omnilib{ common::path::find_path(_omnilib, _omnitrace_dl_verbose) }
    , m_dllib{ common::path::find_path(_dllib, _omnitrace_dl_verbose) }
    , m_userlib{ common::path::find_path(_userlib, _omnitrace_dl_verbose) }
    {
        if(_omnitrace_dl_verbose >= 1)
        {
            OMNITRACE_COMMON_LIBRARY_LOG_START
            fprintf(stderr, "[omnitrace][dl][pid=%i] %s resolved to '%s'\n", getpid(),
                    ::basename(_omnilib.c_str()), m_omnilib.c_str());
            fprintf(stderr, "[omnitrace][dl][pid=%i] %s resolved to '%s'\n", getpid(),
                    ::basename(_dllib.c_str()), m_dllib.c_str());
            fprintf(stderr, "[omnitrace][dl][pid=%i] %s resolved to '%s'\n", getpid(),
                    ::basename(_userlib.c_str()), m_userlib.c_str());
            OMNITRACE_COMMON_LIBRARY_LOG_END
        }

        auto _search_paths = common::join(':', common::path::dirname(_omnilib),
                                          common::path::dirname(_dllib));
        common::setup_environ(_omnitrace_dl_verbose, _search_paths, _omnilib, _dllib);

        m_omnihandle = open(m_omnilib);
        m_userhandle = open(m_userlib);
        init();
    }

    OMNITRACE_INLINE ~indirect() { dlclose(m_omnihandle); }

    static OMNITRACE_INLINE void* open(const std::string& _lib)
    {
        auto* libhandle = dlopen(_lib.c_str(), _omnitrace_dl_dlopen_flags);

        if(libhandle)
        {
            if(_omnitrace_dl_verbose >= 2)
            {
                OMNITRACE_COMMON_LIBRARY_LOG_START
                fprintf(stderr, "[omnitrace][dl][pid=%i] dlopen(\"%s\", %s) :: success\n",
                        getpid(), _lib.c_str(), _omnitrace_dl_dlopen_descr);
                OMNITRACE_COMMON_LIBRARY_LOG_END
            }
        }
        else
        {
            if(_omnitrace_dl_verbose >= 0)
            {
                perror("dlopen");
                OMNITRACE_COMMON_LIBRARY_LOG_START
                fprintf(stderr, "[omnitrace][dl][pid=%i] dlopen(\"%s\", %s) :: %s\n",
                        getpid(), _lib.c_str(), _omnitrace_dl_dlopen_descr, dlerror());
                OMNITRACE_COMMON_LIBRARY_LOG_END
            }
        }

        dlerror();  // Clear any existing error

        return libhandle;
    }

    OMNITRACE_INLINE void init()
    {
        if(!m_omnihandle) m_omnihandle = open(m_omnilib);

        int _warn_verbose = 0;
        int _info_verbose = 2;
        // Initialize all pointers
        OMNITRACE_DLSYM(omnitrace_init_library_f, m_omnihandle, "omnitrace_init_library");
        OMNITRACE_DLSYM(omnitrace_init_f, m_omnihandle, "omnitrace_init");
        OMNITRACE_DLSYM(omnitrace_finalize_f, m_omnihandle, "omnitrace_finalize");
        OMNITRACE_DLSYM(omnitrace_set_env_f, m_omnihandle, "omnitrace_set_env");
        OMNITRACE_DLSYM(omnitrace_set_mpi_f, m_omnihandle, "omnitrace_set_mpi");
        OMNITRACE_DLSYM(omnitrace_push_trace_f, m_omnihandle, "omnitrace_push_trace");
        OMNITRACE_DLSYM(omnitrace_pop_trace_f, m_omnihandle, "omnitrace_pop_trace");
        OMNITRACE_DLSYM(omnitrace_push_region_f, m_omnihandle, "omnitrace_push_region");
        OMNITRACE_DLSYM(omnitrace_pop_region_f, m_omnihandle, "omnitrace_pop_region");
        OMNITRACE_DLSYM(omnitrace_register_source_f, m_omnihandle,
                        "omnitrace_register_source");
        OMNITRACE_DLSYM(omnitrace_register_coverage_f, m_omnihandle,
                        "omnitrace_register_coverage");

        OMNITRACE_DLSYM(kokkosp_print_help_f, m_omnihandle, "kokkosp_print_help");
        OMNITRACE_DLSYM(kokkosp_parse_args_f, m_omnihandle, "kokkosp_parse_args");
        OMNITRACE_DLSYM(kokkosp_declare_metadata_f, m_omnihandle,
                        "kokkosp_declare_metadata");
        OMNITRACE_DLSYM(kokkosp_request_tool_settings_f, m_omnihandle,
                        "kokkosp_request_tool_settings");
        OMNITRACE_DLSYM(kokkosp_init_library_f, m_omnihandle, "kokkosp_init_library");
        OMNITRACE_DLSYM(kokkosp_finalize_library_f, m_omnihandle,
                        "kokkosp_finalize_library");
        OMNITRACE_DLSYM(kokkosp_begin_parallel_for_f, m_omnihandle,
                        "kokkosp_begin_parallel_for");
        OMNITRACE_DLSYM(kokkosp_end_parallel_for_f, m_omnihandle,
                        "kokkosp_end_parallel_for");
        OMNITRACE_DLSYM(kokkosp_begin_parallel_reduce_f, m_omnihandle,
                        "kokkosp_begin_parallel_reduce");
        OMNITRACE_DLSYM(kokkosp_end_parallel_reduce_f, m_omnihandle,
                        "kokkosp_end_parallel_reduce");
        OMNITRACE_DLSYM(kokkosp_begin_parallel_scan_f, m_omnihandle,
                        "kokkosp_begin_parallel_scan");
        OMNITRACE_DLSYM(kokkosp_end_parallel_scan_f, m_omnihandle,
                        "kokkosp_end_parallel_scan");
        OMNITRACE_DLSYM(kokkosp_begin_fence_f, m_omnihandle, "kokkosp_begin_fence");
        OMNITRACE_DLSYM(kokkosp_end_fence_f, m_omnihandle, "kokkosp_end_fence");
        OMNITRACE_DLSYM(kokkosp_push_profile_region_f, m_omnihandle,
                        "kokkosp_push_profile_region");
        OMNITRACE_DLSYM(kokkosp_pop_profile_region_f, m_omnihandle,
                        "kokkosp_pop_profile_region");
        OMNITRACE_DLSYM(kokkosp_create_profile_section_f, m_omnihandle,
                        "kokkosp_create_profile_section");
        OMNITRACE_DLSYM(kokkosp_destroy_profile_section_f, m_omnihandle,
                        "kokkosp_destroy_profile_section");
        OMNITRACE_DLSYM(kokkosp_start_profile_section_f, m_omnihandle,
                        "kokkosp_start_profile_section");
        OMNITRACE_DLSYM(kokkosp_stop_profile_section_f, m_omnihandle,
                        "kokkosp_stop_profile_section");
        OMNITRACE_DLSYM(kokkosp_allocate_data_f, m_omnihandle, "kokkosp_allocate_data");
        OMNITRACE_DLSYM(kokkosp_deallocate_data_f, m_omnihandle,
                        "kokkosp_deallocate_data");
        OMNITRACE_DLSYM(kokkosp_begin_deep_copy_f, m_omnihandle,
                        "kokkosp_begin_deep_copy");
        OMNITRACE_DLSYM(kokkosp_end_deep_copy_f, m_omnihandle, "kokkosp_end_deep_copy");
        OMNITRACE_DLSYM(kokkosp_profile_event_f, m_omnihandle, "kokkosp_profile_event");
        OMNITRACE_DLSYM(kokkosp_dual_view_sync_f, m_omnihandle, "kokkosp_dual_view_sync");
        OMNITRACE_DLSYM(kokkosp_dual_view_modify_f, m_omnihandle,
                        "kokkosp_dual_view_modify");

#if OMNITRACE_USE_ROCTRACER > 0
        OMNITRACE_DLSYM(hsa_on_load_f, m_omnihandle, "OnLoad");
        OMNITRACE_DLSYM(hsa_on_unload_f, m_omnihandle, "OnUnload");
#endif

#if OMNITRACE_USE_OMPT == 0
        _warn_verbose = 5;
#else
        OMNITRACE_DLSYM(ompt_start_tool_f, m_omnihandle, "ompt_start_tool");
#endif

        if(!m_userhandle) m_userhandle = open(m_userlib);
        _warn_verbose = 0;
        OMNITRACE_DLSYM(omnitrace_user_configure_f, m_userhandle,
                        "omnitrace_user_configure");

        if(omnitrace_user_configure_f)
        {
            (*omnitrace_user_configure_f)(
                OMNITRACE_USER_START_STOP,
                reinterpret_cast<void*>(&omnitrace_user_start_trace_dl),
                reinterpret_cast<void*>(&omnitrace_user_stop_trace_dl));
            (*omnitrace_user_configure_f)(
                OMNITRACE_USER_START_STOP_THREAD,
                reinterpret_cast<void*>(&omnitrace_user_start_thread_trace_dl),
                reinterpret_cast<void*>(&omnitrace_user_stop_thread_trace_dl));
            (*omnitrace_user_configure_f)(
                OMNITRACE_USER_REGION,
                reinterpret_cast<void*>(&omnitrace_user_push_region_dl),
                reinterpret_cast<void*>(&omnitrace_user_pop_region_dl));
        }
    }

public:
    // omnitrace functions
    void (*omnitrace_init_library_f)(void)                                  = nullptr;
    void (*omnitrace_init_f)(const char*, bool, const char*)                = nullptr;
    void (*omnitrace_finalize_f)(void)                                      = nullptr;
    void (*omnitrace_set_env_f)(const char*, const char*)                   = nullptr;
    void (*omnitrace_set_mpi_f)(bool, bool)                                 = nullptr;
    void (*omnitrace_register_source_f)(const char*, const char*, size_t, size_t,
                                        const char*)                        = nullptr;
    void (*omnitrace_register_coverage_f)(const char*, const char*, size_t) = nullptr;
    void (*omnitrace_push_trace_f)(const char*)                             = nullptr;
    void (*omnitrace_pop_trace_f)(const char*)                              = nullptr;
    int (*omnitrace_push_region_f)(const char*)                             = nullptr;
    int (*omnitrace_pop_region_f)(const char*)                              = nullptr;
    int (*omnitrace_user_configure_f)(int, void*, void*)                    = nullptr;

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

    // HSA functions
#if OMNITRACE_USE_ROCTRACER > 0
    bool (*hsa_on_load_f)(HsaApiTable*, uint64_t, uint64_t, const char* const*) = nullptr;
    void (*hsa_on_unload_f)()                                                   = nullptr;
#endif

    // OpenMP functions
#if defined(OMNITRACE_USE_OMPT) && OMNITRACE_USE_OMPT > 0
    ompt_start_tool_result_t* (*ompt_start_tool_f)(unsigned int, const char*);
#endif

private:
    void*       m_omnihandle = nullptr;
    void*       m_userhandle = nullptr;
    std::string m_omnilib    = {};
    std::string m_dllib      = {};
    std::string m_userlib    = {};
};

inline indirect&
get_indirect() OMNITRACE_HIDDEN_API;

indirect&
get_indirect()
{
    static auto  _libomni = get_env("OMNITRACE_LIBRARY", "libomnitrace.so");
    static auto  _libuser = get_env("OMNITRACE_USER_LIBRARY", "libomnitrace-user.so");
    static auto  _libdlib = get_env("OMNITRACE_DL_LIBRARY", "libomnitrace-dl.so");
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
    static auto* _v = new std::atomic<bool>{ get_env("OMNITRACE_INIT_ENABLED", true) };
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

// ensure finalization is called
bool _omnitrace_dl_fini = (std::atexit([]() {
                               if(get_active()) omnitrace_finalize();
                           }),
                           true);
}  // namespace
}  // namespace dl
}  // namespace omnitrace

//--------------------------------------------------------------------------------------//

#define OMNITRACE_DL_INVOKE(...)                                                         \
    ::omnitrace::common::invoke(__FUNCTION__, ::omnitrace::dl::_omnitrace_dl_verbose,    \
                                (::omnitrace::dl::get_thread_status() = false),          \
                                __VA_ARGS__)

#define OMNITRACE_DL_IGNORE(...)                                                         \
    ::omnitrace::common::ignore(__FUNCTION__, ::omnitrace::dl::_omnitrace_dl_verbose,    \
                                __VA_ARGS__)

#define OMNITRACE_DL_INVOKE_STATUS(STATUS, ...)                                          \
    ::omnitrace::common::invoke(__FUNCTION__, ::omnitrace::dl::_omnitrace_dl_verbose,    \
                                STATUS, __VA_ARGS__)

#define OMNITRACE_DL_LOG(LEVEL, ...)                                                     \
    if(::omnitrace::dl::_omnitrace_dl_verbose >= LEVEL)                                  \
    {                                                                                    \
        fflush(stderr);                                                                  \
        OMNITRACE_COMMON_LIBRARY_LOG_START                                               \
        fprintf(stderr, "[omnitrace][" OMNITRACE_COMMON_LIBRARY_NAME "] " __VA_ARGS__);  \
        OMNITRACE_COMMON_LIBRARY_LOG_END                                                 \
        fflush(stderr);                                                                  \
    }

using omnitrace::get_indirect;
namespace dl = omnitrace::dl;

extern "C"
{
    void omnitrace_init_library(void)
    {
        OMNITRACE_DL_INVOKE(get_indirect().omnitrace_init_library_f);
    }

    void omnitrace_init(const char* a, bool b, const char* c)
    {
        if(dl::get_inited() && dl::get_finied())
        {
            OMNITRACE_DL_LOG(
                2, "%s(%s) ignored :: already initialized and finalized\n", __FUNCTION__,
                ::omnitrace::join(::omnitrace::QuoteStrings{}, ", ", a, b, c).c_str());
            return;
        }
        else if(dl::get_inited() && dl::get_active())
        {
            OMNITRACE_DL_LOG(
                2, "%s(%s) ignored :: already initialized and active\n", __FUNCTION__,
                ::omnitrace::join(::omnitrace::QuoteStrings{}, ", ", a, b, c).c_str());
            return;
        }

        bool _invoked = false;
        OMNITRACE_DL_INVOKE_STATUS(_invoked, get_indirect().omnitrace_init_f, a, b, c);
        if(_invoked)
        {
            dl::get_active()          = true;
            dl::get_inited()          = true;
            dl::_omnitrace_dl_verbose = dl::get_omnitrace_dl_env();
        }
    }

    void omnitrace_finalize(void)
    {
        if(dl::get_inited() && dl::get_finied())
        {
            OMNITRACE_DL_LOG(2, "%s() ignored :: already initialized and finalized\n",
                             __FUNCTION__);
            return;
        }
        else if(dl::get_finied() && !dl::get_active())
        {
            OMNITRACE_DL_LOG(2, "%s() ignored :: already finalized but not active\n",
                             __FUNCTION__);
            return;
        }

        bool _invoked = false;
        OMNITRACE_DL_INVOKE_STATUS(_invoked, get_indirect().omnitrace_finalize_f);
        if(_invoked)
        {
            dl::get_active() = false;
            dl::get_finied() = true;
        }
    }

    void omnitrace_push_trace(const char* name)
    {
        if(!dl::get_active()) return;
        if(dl::get_thread_enabled())
        {
            OMNITRACE_DL_INVOKE(get_indirect().omnitrace_push_trace_f, name);
        }
        else
        {
            ++dl::get_thread_count();
        }
    }

    void omnitrace_pop_trace(const char* name)
    {
        if(!dl::get_active()) return;
        if(dl::get_thread_enabled())
        {
            OMNITRACE_DL_INVOKE(get_indirect().omnitrace_pop_trace_f, name);
        }
        else
        {
            if(dl::get_thread_count()-- == 0) omnitrace_user_start_thread_trace_dl();
        }
    }

    void omnitrace_push_region(const char* name)
    {
        if(!dl::get_active()) return;
        if(dl::get_thread_enabled())
        {
            OMNITRACE_DL_INVOKE(get_indirect().omnitrace_push_region_f, name);
        }
        else
        {
            ++dl::get_thread_count();
        }
    }

    void omnitrace_pop_region(const char* name)
    {
        if(!dl::get_active()) return;
        if(dl::get_thread_enabled())
        {
            OMNITRACE_DL_INVOKE(get_indirect().omnitrace_pop_region_f, name);
        }
        else
        {
            if(dl::get_thread_count()-- == 0) omnitrace_user_start_thread_trace_dl();
        }
    }

    void omnitrace_set_env(const char* a, const char* b)
    {
        if(dl::get_inited() && dl::get_active())
        {
            OMNITRACE_DL_IGNORE(2, "already initialized and active", a, b);
            return;
        }
        setenv(a, b, 0);
        OMNITRACE_DL_INVOKE(get_indirect().omnitrace_set_env_f, a, b);
    }

    void omnitrace_set_mpi(bool a, bool b)
    {
        if(dl::get_inited() && dl::get_active())
        {
            OMNITRACE_DL_IGNORE(2, "already initialized and active", a, b);
            return;
        }
        OMNITRACE_DL_INVOKE(get_indirect().omnitrace_set_mpi_f, a, b);
    }

    void omnitrace_register_source(const char* file, const char* func, size_t line,
                                   size_t address, const char* source)
    {
        OMNITRACE_DL_LOG(3, "%s(\"%s\", \"%s\", %zu, %zu, \"%s\")\n", __FUNCTION__, file,
                         func, line, address, source);
        OMNITRACE_DL_INVOKE(get_indirect().omnitrace_register_source_f, file, func, line,
                            address, source);
    }

    void omnitrace_register_coverage(const char* file, const char* func, size_t address)
    {
        OMNITRACE_DL_INVOKE(get_indirect().omnitrace_register_coverage_f, file, func,
                            address);
    }

    int omnitrace_user_start_trace_dl(void)
    {
        dl::get_enabled().store(true);
        return omnitrace_user_start_thread_trace_dl();
    }

    int omnitrace_user_stop_trace_dl(void)
    {
        dl::get_enabled().store(false);
        return omnitrace_user_stop_thread_trace_dl();
    }

    int omnitrace_user_start_thread_trace_dl(void)
    {
        dl::get_thread_enabled() = true;
        return 0;
    }

    int omnitrace_user_stop_thread_trace_dl(void)
    {
        dl::get_thread_enabled() = false;
        return 0;
    }

    int omnitrace_user_push_region_dl(const char* name)
    {
        if(!dl::get_active()) return 0;
        return OMNITRACE_DL_INVOKE(get_indirect().omnitrace_push_region_f, name);
    }

    int omnitrace_user_pop_region_dl(const char* name)
    {
        if(!dl::get_active()) return 0;
        return OMNITRACE_DL_INVOKE(get_indirect().omnitrace_pop_region_f, name);
    }

    //----------------------------------------------------------------------------------//
    //
    //      KokkosP
    //
    //----------------------------------------------------------------------------------//

    void kokkosp_print_help(char* argv0)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_print_help_f, argv0);
    }

    void kokkosp_parse_args(int argc, char** argv)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_parse_args_f, argc, argv);
    }

    void kokkosp_declare_metadata(const char* key, const char* value)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_declare_metadata_f, key, value);
    }

    void kokkosp_request_tool_settings(const uint32_t             version,
                                       Kokkos_Tools_ToolSettings* settings)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_request_tool_settings_f,
                                   version, settings);
    }

    void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                              const uint32_t devInfoCount, void* deviceInfo)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_init_library_f, loadSeq,
                                   interfaceVer, devInfoCount, deviceInfo);
    }

    void kokkosp_finalize_library()
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_finalize_library_f);
    }

    void kokkosp_begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_begin_parallel_for_f, name,
                                   devid, kernid);
    }

    void kokkosp_end_parallel_for(uint64_t kernid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_end_parallel_for_f, kernid);
    }

    void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_begin_parallel_reduce_f, name,
                                   devid, kernid);
    }

    void kokkosp_end_parallel_reduce(uint64_t kernid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_end_parallel_reduce_f, kernid);
    }

    void kokkosp_begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_begin_parallel_scan_f, name,
                                   devid, kernid);
    }

    void kokkosp_end_parallel_scan(uint64_t kernid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_end_parallel_scan_f, kernid);
    }

    void kokkosp_begin_fence(const char* name, uint32_t devid, uint64_t* kernid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_begin_fence_f, name, devid,
                                   kernid);
    }

    void kokkosp_end_fence(uint64_t kernid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_end_fence_f, kernid);
    }

    void kokkosp_push_profile_region(const char* name)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_push_profile_region_f, name);
    }

    void kokkosp_pop_profile_region()
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_pop_profile_region_f);
    }

    void kokkosp_create_profile_section(const char* name, uint32_t* secid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_create_profile_section_f, name,
                                   secid);
    }

    void kokkosp_destroy_profile_section(uint32_t secid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_destroy_profile_section_f,
                                   secid);
    }

    void kokkosp_start_profile_section(uint32_t secid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_start_profile_section_f, secid);
    }

    void kokkosp_stop_profile_section(uint32_t secid)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_stop_profile_section_f, secid);
    }

    void kokkosp_allocate_data(const SpaceHandle space, const char* label,
                               const void* const ptr, const uint64_t size)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_allocate_data_f, space, label,
                                   ptr, size);
    }

    void kokkosp_deallocate_data(const SpaceHandle space, const char* label,
                                 const void* const ptr, const uint64_t size)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_deallocate_data_f, space, label,
                                   ptr, size);
    }

    void kokkosp_begin_deep_copy(SpaceHandle dst_handle, const char* dst_name,
                                 const void* dst_ptr, SpaceHandle src_handle,
                                 const char* src_name, const void* src_ptr, uint64_t size)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_begin_deep_copy_f, dst_handle,
                                   dst_name, dst_ptr, src_handle, src_name, src_ptr,
                                   size);
    }

    void kokkosp_end_deep_copy()
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_end_deep_copy_f);
    }

    void kokkosp_profile_event(const char* name)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_profile_event_f, name);
    }

    void kokkosp_dual_view_sync(const char* label, const void* const data, bool is_device)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_dual_view_sync_f, label, data,
                                   is_device);
    }

    void kokkosp_dual_view_modify(const char* label, const void* const data,
                                  bool is_device)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().kokkosp_dual_view_modify_f, label, data,
                                   is_device);
    }

    //----------------------------------------------------------------------------------//
    //
    //      HSA
    //
    //----------------------------------------------------------------------------------//

#if OMNITRACE_USE_ROCTRACER > 0
    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names)
    {
        return OMNITRACE_DL_INVOKE(get_indirect().hsa_on_load_f, table, runtime_version,
                                   failed_tool_count, failed_tool_names);
    }

    void OnUnload() { return OMNITRACE_DL_INVOKE(get_indirect().hsa_on_unload_f); }
#endif

    //----------------------------------------------------------------------------------//
    //
    //      OMPT
    //
    //----------------------------------------------------------------------------------//
#if OMNITRACE_USE_OMPT > 0
    ompt_start_tool_result_t* ompt_start_tool(unsigned int omp_version,
                                              const char*  runtime_version)
    {
        if(!omnitrace::common::get_env("OMNITRACE_USE_OMPT", true)) return nullptr;
        return OMNITRACE_DL_INVOKE(get_indirect().ompt_start_tool_f, omp_version,
                                   runtime_version);
    }
#endif
}
