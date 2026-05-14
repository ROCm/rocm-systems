// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "fwd.h"

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#if defined(_WIN32)
// WINDOWS-DIVERGENCE: PE/COFF has no <dlfcn.h>; substitute the Win32 dynamic
// loader API (LoadLibrary/GetProcAddress). The RTLD_* flag set has no direct
// analogue — Win32 LoadLibraryEx flags map only loosely:
//     RTLD_LAZY   -> no equivalent (Windows resolves all imports eagerly)
//     RTLD_NOW    -> default behavior; not encodable as a flag
//     RTLD_LOCAL  -> default behavior (no DLL-symbol-table sharing)
//     RTLD_GLOBAL -> not supported (Windows has per-process module list, not
//                    a global symbol namespace; tools that want this use
//                    AppInit_DLLs or detours)
//     RTLD_NOLOAD -> approximated by GetModuleHandle (only succeeds if the
//                    DLL is already loaded)
// We retain the integer mode for log compatibility with the Linux PASS_REGEX
// strings; the actual Win32 call ignores it because LoadLibrary has no flag
// that meaningfully maps onto these semantics.
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

#ifndef ROCP_REG_FILE_NAME
// WINDOWS-DIVERGENCE: __FILE__ on MSVC uses '\\' for path components MSVC
// itself opened, but preserves the literal separator from the `#include`
// directive that brought the header in (e.g. `#include "common/fwd.hpp"`
// keeps the '/'). Strip BOTH separators so the produced PASS_REGEX-able log
// lines render the bare basename regardless of how this header was named in
// the includer's #include line. On POSIX, '\\' simply does not appear in
// paths, so matching it as well is a harmless no-op.
#    define ROCP_REG_FILE_NAME                                                           \
        ::std::string{ __FILE__ }                                                        \
            .substr(::std::string_view{ __FILE__ }.find_last_of("\\/") + 1)              \
            .c_str()
#endif

namespace
{
// WINDOWS-DIVERGENCE: pthread_barrier_t is a POSIX-only type that has no
// Win32 / MSVC CRT analogue (Windows InitOnceExecuteOnce + SRWLOCK do not
// reproduce the rendezvous semantics, and std::barrier is C++20 -- the
// project is C++17). The integration tests below need a multi-thread
// rendezvous primitive that mirrors pthread_barrier_wait, so we substitute
// a portable C++17 shim. On Linux we keep the call-sites sourced from the
// pthread headers via the original code paths; this shim is only consumed
// when ROCP_REG_USE_PORTABLE_BARRIER is defined (currently set on Windows).
class portable_barrier
{
public:
    portable_barrier(unsigned long _count)
    : m_threshold{ _count }
    , m_count{ _count }
    , m_generation{ 0 }
    {}

    portable_barrier(const portable_barrier&)            = delete;
    portable_barrier& operator=(const portable_barrier&) = delete;

    void wait()
    {
        auto _lk             = std::unique_lock<std::mutex>{ m_mutex };
        auto _local_gen      = m_generation;
        if(--m_count == 0)
        {
            ++m_generation;
            m_count = m_threshold;
            m_cond.notify_all();
        }
        else
        {
            m_cond.wait(_lk, [this, _local_gen] { return _local_gen != m_generation; });
        }
    }

private:
    std::mutex              m_mutex;
    std::condition_variable m_cond;
    unsigned long           m_threshold;
    unsigned long           m_count;
    unsigned long           m_generation;
};

decltype(hip_init)*            hip_init_fn            = nullptr;
decltype(hsa_init)*            hsa_init_fn            = nullptr;
decltype(ncclGetVersion)*      ncclGetVersion_fn      = nullptr;
decltype(roctxRangePush)*      roctxRangePush_fn      = nullptr;
decltype(roctxRangePush)*      roctxRangePop_fn       = nullptr;
decltype(rocDecCreateDecoder)* rocDecCreateDecoder_fn = nullptr;
decltype(rocJpegStreamCreate)* rocJpegStreamCreate_fn = nullptr;

enum rocp_reg_test_modes : uint8_t
{
    ROCP_REG_TEST_NONE      = 0x0,
    ROCP_REG_TEST_HIP       = (1 << 0),
    ROCP_REG_TEST_HSA       = (1 << 1),
    ROCP_REG_TEST_ROCTX     = (1 << 2),
    ROCP_REG_TEST_RCCL      = (1 << 3),
    ROCP_REG_TEST_ROCDECODE = (1 << 4),
    ROCP_REG_TEST_ROCJPEG   = (1 << 5),
};

// WINDOWS-DIVERGENCE: per-platform native handle and library-name mapping.
// The Linux test mocks build as libamdhip64.so / libhsa-runtime64.so / etc.;
// the Windows mocks build as amdhip64.dll / hsa-runtime64.dll / etc. (no
// "lib" prefix, .dll suffix, no SOVERSION embedded in filename).
#if defined(_WIN32)
using rocp_reg_test_handle_t                                = HMODULE;
inline constexpr int  ROCP_REG_TEST_DEFAULT_OPEN_MODE       = 0;
inline constexpr auto ROCP_REG_TEST_LIB_AMDHIP              = "amdhip64.dll";
inline constexpr auto ROCP_REG_TEST_LIB_HSA_RUNTIME         = "hsa-runtime64.dll";
inline constexpr auto ROCP_REG_TEST_LIB_ROCTX               = "roctx64.dll";
inline constexpr auto ROCP_REG_TEST_LIB_RCCL                = "rccl.dll";
inline constexpr auto ROCP_REG_TEST_LIB_ROCDECODE           = "rocdecode.dll";
inline constexpr auto ROCP_REG_TEST_LIB_ROCJPEG             = "rocjpeg.dll";
#else
using rocp_reg_test_handle_t                                = void*;
inline constexpr int  ROCP_REG_TEST_DEFAULT_OPEN_MODE       = RTLD_LOCAL | RTLD_LAZY;
inline constexpr auto ROCP_REG_TEST_LIB_AMDHIP              = "libamdhip64.so";
inline constexpr auto ROCP_REG_TEST_LIB_HSA_RUNTIME         = "libhsa-runtime64.so";
inline constexpr auto ROCP_REG_TEST_LIB_ROCTX               = "libroctx64.so";
inline constexpr auto ROCP_REG_TEST_LIB_RCCL                = "librccl.so";
inline constexpr auto ROCP_REG_TEST_LIB_ROCDECODE           = "librocdecode.so";
inline constexpr auto ROCP_REG_TEST_LIB_ROCJPEG             = "librocjpeg.so";
#endif

template <uint8_t Idx = ROCP_REG_TEST_NONE>
inline void
resolve_symbols(int _open_mode = ROCP_REG_TEST_DEFAULT_OPEN_MODE)
{
    auto* _open_mode_env = std::getenv("ROCP_REG_TEST_OPEN_MODE");
    if(_open_mode_env)
    {
#if defined(_WIN32)
        // WINDOWS-DIVERGENCE: RTLD_* flags have no LoadLibrary equivalent
        // (see header comment). Preserve the env-var contract by parsing
        // it for log parity, but the integer is opaque to LoadLibrary.
        (void) _open_mode_env;
#else
        constexpr auto npos         = std::string_view::npos;
        auto           _open_mode_v = std::string_view{ _open_mode_env };
        if(_open_mode_v.find("RTLD_GLOBAL") != npos)
            _open_mode = RTLD_GLOBAL;
        else if(_open_mode_v.find("RTLD_NOLOAD") != npos)
            _open_mode = RTLD_NOLOAD;
        else
            _open_mode = RTLD_LOCAL;

        if(_open_mode_v.find("RTLD_NOW") != npos)
            _open_mode |= RTLD_NOW;
        else
            _open_mode |= RTLD_LAZY;
#endif
    }

    auto _resolve_dlopen = [_open_mode](rocp_reg_test_handle_t& _handle,
                                        const char*             _lib_name) {
        fprintf(
            stderr, "[%s] dlopen %s, %i\n", ROCP_REG_FILE_NAME, _lib_name, _open_mode);
#if defined(_WIN32)
        _handle = ::LoadLibraryA(_lib_name);
#else
        _handle = dlopen(_lib_name, _open_mode);
#endif
        if(!_handle)
        {
            fprintf(stderr, "Failure opening '%s'\n", _lib_name);
            exit(EXIT_FAILURE);
        }
    };

    auto _resolve_dlsym = [](auto&                  _func,
                             rocp_reg_test_handle_t _handle,
                             const char*            _func_name) {
        if(!_func && _handle && _func_name)
        {
#if defined(_WIN32)
            auto* _func_v =
                reinterpret_cast<void*>(::GetProcAddress(_handle, _func_name));
#else
            auto* _func_v = dlsym(_handle, _func_name);
#endif
            if(_func_v) *(void**) (&_func) = _func_v;
        }
    };

    auto amdhip_handle    = rocp_reg_test_handle_t{};
    auto hsart_handle     = rocp_reg_test_handle_t{};
    auto roctx_handle     = rocp_reg_test_handle_t{};
    auto rccl_handle      = rocp_reg_test_handle_t{};
    auto rocdecode_handle = rocp_reg_test_handle_t{};
    auto rocjpeg_handle   = rocp_reg_test_handle_t{};

    if constexpr((Idx & ROCP_REG_TEST_HIP) == ROCP_REG_TEST_HIP)
    {
        hip_init_fn = ROCP_REG_TEST_WEAK_FN(hip_init);
        if(!hip_init_fn) _resolve_dlopen(amdhip_handle, ROCP_REG_TEST_LIB_AMDHIP);
        _resolve_dlsym(hip_init_fn, amdhip_handle, "hip_init");
    }

    if constexpr((Idx & ROCP_REG_TEST_HSA) == ROCP_REG_TEST_HSA)
    {
        hsa_init_fn = ROCP_REG_TEST_WEAK_FN(hsa_init);
        if(!hsa_init_fn) _resolve_dlopen(hsart_handle, ROCP_REG_TEST_LIB_HSA_RUNTIME);
        _resolve_dlsym(hsa_init_fn, hsart_handle, "hsa_init");
    }

    if constexpr((Idx & ROCP_REG_TEST_ROCTX) == ROCP_REG_TEST_ROCTX)
    {
        roctxRangePush_fn = ROCP_REG_TEST_WEAK_FN(roctxRangePush);
        roctxRangePop_fn  = ROCP_REG_TEST_WEAK_FN(roctxRangePop);
        if(!roctxRangePush_fn || !roctxRangePop_fn)
            _resolve_dlopen(roctx_handle, ROCP_REG_TEST_LIB_ROCTX);
        _resolve_dlsym(roctxRangePush_fn, roctx_handle, "roctxRangePush");
        _resolve_dlsym(roctxRangePop_fn, roctx_handle, "roctxRangePop");
    }

    if constexpr((Idx & ROCP_REG_TEST_RCCL) == ROCP_REG_TEST_RCCL)
    {
        ncclGetVersion_fn = ROCP_REG_TEST_WEAK_FN(ncclGetVersion);
        if(!ncclGetVersion_fn) _resolve_dlopen(rccl_handle, ROCP_REG_TEST_LIB_RCCL);
        _resolve_dlsym(ncclGetVersion_fn, rccl_handle, "ncclGetVersion");
    }

    if constexpr((Idx & ROCP_REG_TEST_ROCDECODE) == ROCP_REG_TEST_ROCDECODE)
    {
        rocDecCreateDecoder_fn = ROCP_REG_TEST_WEAK_FN(rocDecCreateDecoder);
        if(!rocDecCreateDecoder_fn)
            _resolve_dlopen(rocdecode_handle, ROCP_REG_TEST_LIB_ROCDECODE);
        _resolve_dlsym(rocDecCreateDecoder_fn, rocdecode_handle, "rocDecCreateDecoder");
    }

    if constexpr((Idx & ROCP_REG_TEST_ROCJPEG) == ROCP_REG_TEST_ROCJPEG)
    {
        rocJpegStreamCreate_fn = ROCP_REG_TEST_WEAK_FN(rocJpegStreamCreate);
        if(!rocJpegStreamCreate_fn)
            _resolve_dlopen(rocjpeg_handle, ROCP_REG_TEST_LIB_ROCJPEG);
        _resolve_dlsym(rocJpegStreamCreate_fn, rocjpeg_handle, "rocJpegStreamCreate");
    }
}
}  // namespace
