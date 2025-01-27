#pragma once

#include "fwd.h"

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef ROCP_REG_FILE_NAME
#    define ROCP_REG_FILE_NAME                                                           \
        ::std::string{ __FILE__ }                                                        \
            .substr(::std::string_view{ __FILE__ }.find_last_of('/') + 1)                \
            .c_str()
#endif

namespace
{
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

template <uint8_t Idx = ROCP_REG_TEST_NONE>
inline void
resolve_symbols(int _open_mode = RTLD_LOCAL | RTLD_LAZY)
{
    auto* _open_mode_env = std::getenv("ROCP_REG_TEST_OPEN_MODE");
    if(_open_mode_env)
    {
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
    }

    auto _resolve_dlopen = [_open_mode](void*& _handle, const char* _lib_name) {
        fprintf(
            stderr, "[%s] dlopen %s, %i\n", ROCP_REG_FILE_NAME, _lib_name, _open_mode);
        _handle = dlopen(_lib_name, _open_mode);
        if(!_handle)
        {
            fprintf(stderr, "Failure opening '%s'\n", _lib_name);
            exit(EXIT_FAILURE);
        }
    };

    auto _resolve_dlsym = [](auto& _func, void* _handle, const char* _func_name) {
        if(!_func && _handle && _func_name)
        {
            auto* _func_v = dlsym(_handle, _func_name);
            if(_func_v) *(void**) (&_func) = _func_v;
        }
    };

    void* amdhip_handle    = nullptr;
    void* hsart_handle     = nullptr;
    void* roctx_handle     = nullptr;
    void* rccl_handle      = nullptr;
    void* rocdecode_handle = nullptr;
    void* rocjpeg_handle   = nullptr;

    if constexpr((Idx & ROCP_REG_TEST_HIP) == ROCP_REG_TEST_HIP)
    {
        hip_init_fn = hip_init;
        if(!hip_init_fn) _resolve_dlopen(amdhip_handle, "libamdhip64.so");
        _resolve_dlsym(hip_init_fn, amdhip_handle, "hip_init");
    }

    if constexpr((Idx & ROCP_REG_TEST_HSA) == ROCP_REG_TEST_HSA)
    {
        hsa_init_fn = hsa_init;
        if(!hsa_init_fn) _resolve_dlopen(hsart_handle, "libhsa-runtime64.so");
        _resolve_dlsym(hsa_init_fn, hsart_handle, "hsa_init");
    }

    if constexpr((Idx & ROCP_REG_TEST_ROCTX) == ROCP_REG_TEST_ROCTX)
    {
        roctxRangePush_fn = roctxRangePush;
        roctxRangePop_fn  = roctxRangePop;
        if(!roctxRangePush_fn || !roctxRangePop_fn)
            _resolve_dlopen(roctx_handle, "libroctx64.so");
        _resolve_dlsym(roctxRangePush_fn, roctx_handle, "roctxRangePush");
        _resolve_dlsym(roctxRangePop_fn, roctx_handle, "roctxRangePop");
    }

    if constexpr((Idx & ROCP_REG_TEST_RCCL) == ROCP_REG_TEST_RCCL)
    {
        ncclGetVersion_fn = ncclGetVersion;
        if(!ncclGetVersion_fn) _resolve_dlopen(rccl_handle, "librccl.so");
        _resolve_dlsym(ncclGetVersion_fn, rccl_handle, "ncclGetVersion");
    }

    if constexpr((Idx & ROCP_REG_TEST_ROCDECODE) == ROCP_REG_TEST_ROCDECODE)
    {
        rocDecCreateDecoder_fn = rocDecCreateDecoder;
        if(!rocDecCreateDecoder_fn) _resolve_dlopen(rocdecode_handle, "librocdecode.so");
        _resolve_dlsym(rocDecCreateDecoder_fn, rocdecode_handle, "rocDecCreateDecoder");
    }

    if constexpr((Idx & ROCP_REG_TEST_ROCJPEG) == ROCP_REG_TEST_ROCJPEG)
    {
        rocJpegStreamCreate_fn = rocJpegStreamCreate;
        if(!rocJpegStreamCreate_fn) _resolve_dlopen(rocjpeg_handle, "librocjpeg.so");
        _resolve_dlsym(rocJpegStreamCreate_fn, rocjpeg_handle, "rocJpegStreamCreate");
    }
}
}  // namespace
