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


#include <rocprofiler-register/rocprofiler-register.h>
#include <amdhip/amdhip.hpp>
#include <hsa-runtime/hsa-runtime.hpp>
#include <rccl/rccl.hpp>
#include <rocdecode/rocdecode.hpp>
#include <rocjpeg/rocjpeg.hpp>
#include <roctx/roctx.hpp>

// WINDOWS-DIVERGENCE: <dlfcn.h> -> Win32 dynamic loader (LoadLibraryA /
// GetProcAddress / GetModuleHandleA). <pthread.h> is unused in this mock.
#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#    include <pthread.h>
#endif
#include <cinttypes>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#ifndef ROCP_REG_FILE_NAME
// WINDOWS-DIVERGENCE: __FILE__ uses '\\' separators on MSVC by default;
// strip either separator so the produced log lines match on both platforms.
#    if defined(_WIN32)
#        define ROCP_REG_FILE_NAME_SEP '\\'
#    else
#        define ROCP_REG_FILE_NAME_SEP '/'
#    endif
#    define ROCP_REG_FILE_NAME                                                           \
        ::std::string{ __FILE__ }                                                        \
            .substr(::std::string_view{ __FILE__ }.find_last_of(                         \
                        ROCP_REG_FILE_NAME_SEP) +                                        \
                    1)                                                                   \
            .c_str()
#endif

// WINDOWS-DIVERGENCE: rocprofiler-register installs as
// "rocprofiler-register.dll" on Windows (no SOVERSION in filename, no "lib"
// prefix); on Linux it is "librocprofiler-register.so".
#if defined(_WIN32)
#    define ROCP_REG_HOST_LIBRARY_NAME "rocprofiler-register.dll"
#else
#    define ROCP_REG_HOST_LIBRARY_NAME "librocprofiler-register.so"
#endif

// WINDOWS-DIVERGENCE: prefer C99 __func__ over MSVC's __FUNCTION__ inside
// this namespace. MSVC's __FUNCTION__ expands to 'rocprofiler::hip_init'
// (namespace-qualified) whereas GCC's expands to just 'hip_init'. __func__
// yields the unqualified identifier on both compilers and keeps Linux's
// PASS_REGEX log strings matching unchanged on Windows.
namespace rocprofiler
{
void
hip_init()
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __func__);
}

void
hsa_init()
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __func__);
}

ncclResult_t
ncclGetVersion(int*)
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __func__);
    return {};
}

rocDecStatus
rocDecCreateDecoder(rocDecDecoderHandle*, RocDecoderCreateInfo*)
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __func__);
    return {};
}

RocJpegStatus
rocJpegStreamCreate(RocJpegStreamHandle* jpeg_stream_handle)
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __func__);
    return {};
}

void
roctx_range_push(const char* name)
{
    printf("[%s][push] %s\n", ROCP_REG_FILE_NAME, name);
}

void
roctx_range_pop(const char* name)
{
    printf("[%s][pop] %s\n", ROCP_REG_FILE_NAME, name);
}

using reginfo_vec_t = std::vector<rocprofiler_register_registration_info_t>;

bool
check_registration_info(const char*          name,
                        uint64_t             lib_version,
                        uint64_t             num_tables,
                        const reginfo_vec_t& infovec)
{
    for(const auto& itr : infovec)
    {
        if(std::string_view{ name } == std::string_view{ itr.common_name })
        {
            return std::tie(lib_version, num_tables) ==
                   std::tie(itr.lib_version, itr.api_table_length);
        }
    }

    return false;
}
}  // namespace rocprofiler

// WINDOWS-DIVERGENCE: GCC visibility attribute -> __declspec(dllexport).
#if defined(_WIN32)
#    define ROCP_REG_TEST_MOCK_EXPORT __declspec(dllexport)
#else
#    define ROCP_REG_TEST_MOCK_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
ROCP_REG_TEST_MOCK_EXPORT int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t);

int
rocprofiler_set_api_table(const char* name,
                          uint64_t    lib_version,
                          uint64_t    lib_instance,
                          void**      tables,
                          uint64_t    num_tables)
{
    // WINDOWS-DIVERGENCE: %lu is 32-bit on Windows but uint64_t is 64-bit;
    // use PRIu64 for cross-platform correctness.
    printf("[%s] %s :: %" PRIu64 " :: %" PRIu64 " :: %" PRIu64 "\n",
           ROCP_REG_FILE_NAME,
           name,
           lib_version,
           lib_instance,
           num_tables);

    auto* _tool_libs = std::getenv("ROCP_TOOL_LIBRARIES");
    if(_tool_libs)
    {
        // WINDOWS-DIVERGENCE: dlopen(RTLD_GLOBAL | RTLD_LAZY) -> LoadLibraryA.
        // RTLD_GLOBAL has no Win32 analogue (Windows has no global symbol
        // namespace); LoadLibraryA always loads with eager binding, so
        // RTLD_LAZY is also a no-op semantically.
#if defined(_WIN32)
        auto* _handle = ::LoadLibraryA(_tool_libs);
#else
        auto* _handle = dlopen(_tool_libs, RTLD_GLOBAL | RTLD_LAZY);
#endif
        if(!_handle)
            throw std::runtime_error{ std::string{ "error opening tool library " } +
                                      _tool_libs };
#if defined(_WIN32)
        auto* _sym = reinterpret_cast<void*>(
            ::GetProcAddress(_handle, "rocprofiler_configure"));
#else
        auto* _sym = dlsym(_handle, "rocprofiler_configure");
#endif
        if(!_sym)
            throw std::runtime_error{ std::string{ "tool library " } +
                                      std::string{ _tool_libs } +
                                      " did not contain rocprofiler_configure symbol" };
    }

    auto registration_info = ::rocprofiler::reginfo_vec_t{};
    {
        // WINDOWS-DIVERGENCE: dlopen(RTLD_NOLOAD) returns a handle only if
        // the library is already loaded into the process. The Win32
        // equivalent is GetModuleHandleA, which never increments the refcount
        // and only succeeds when the module is already present.
#if defined(_WIN32)
        auto* _handle = ::GetModuleHandleA(ROCP_REG_HOST_LIBRARY_NAME);
#else
        auto* _handle =
            dlopen(ROCP_REG_HOST_LIBRARY_NAME, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
#endif
        if(!_handle)
            throw std::runtime_error{ std::string{ "error opening " } +
                                      std::string{ ROCP_REG_HOST_LIBRARY_NAME } +
                                      " library " };
#if defined(_WIN32)
        auto* _sym = reinterpret_cast<void*>(::GetProcAddress(
            _handle, "rocprofiler_register_iterate_registration_info"));
#else
        auto* _sym = dlsym(_handle, "rocprofiler_register_iterate_registration_info");
#endif
        if(!_sym)
            throw std::runtime_error{
                std::string{ ROCP_REG_HOST_LIBRARY_NAME } +
                " did not contain "
                "rocprofiler_register_iterate_registration_info symbol"
            };

        auto _func = [](rocprofiler_register_registration_info_t* _info,
                        void*                                     _vdata) -> int {
            auto* _vec = static_cast<::rocprofiler::reginfo_vec_t*>(_vdata);
            _vec->emplace_back(*_info);
            return 0;
        };

        auto iterate_registration_info =
            reinterpret_cast<decltype(&rocprofiler_register_iterate_registration_info)>(
                _sym);
        iterate_registration_info(_func, &registration_info);
    }

    using hip_table_t       = hip::HipApiTable;
    using hsa_table_t       = hsa::HsaApiTable;
    using roctx_table_t     = roctx::ROCTxApiTable;
    using rccl_table_t      = rccl::rcclApiFuncTable;
    using rocdecode_table_t = rocdecode::rocdecodeApiFuncTable;
    using rocjpeg_table_t   = rocjpeg::rocjpegApiFuncTable;

    auto* _wrap_v = std::getenv("ROCP_REG_TEST_WRAP");
    bool  _wrap   = (_wrap_v != nullptr && std::stoi(_wrap_v) != 0);

    if(_wrap)
    {
        if(num_tables != 1)
            throw std::runtime_error{ std::string{ "unexpected number of tables: " } +
                                      std::to_string(num_tables) };

        if(tables == nullptr) throw std::runtime_error{ "nullptr to tables" };
        if(tables[0] == nullptr) throw std::runtime_error{ "nullptr to tables[0]" };

        if(std::string_view{ name } == "hip")
        {
            hip_table_t* _table = static_cast<hip_table_t*>(tables[0]);
            _table->hip_init_fn = &::rocprofiler::hip_init;
        }
        else if(std::string_view{ name } == "hsa")
        {
            hsa_table_t* _table = static_cast<hsa_table_t*>(tables[0]);
            _table->hsa_init_fn = &::rocprofiler::hsa_init;
        }
        else if(std::string_view{ name } == "roctx")
        {
            roctx_table_t* _table     = static_cast<roctx_table_t*>(tables[0]);
            _table->roctxRangePush_fn = &::rocprofiler::roctx_range_push;
            _table->roctxRangePop_fn  = &::rocprofiler::roctx_range_pop;
        }
        else if(std::string_view{ name } == "rccl")
        {
            rccl_table_t* _table      = static_cast<rccl_table_t*>(tables[0]);
            _table->ncclGetVersion_fn = &::rocprofiler::ncclGetVersion;
        }
        else if(std::string_view{ name } == "rocdecode")
        {
            rocdecode_table_t* _table      = static_cast<rocdecode_table_t*>(tables[0]);
            _table->rocDecCreateDecoder_fn = &rocprofiler::rocDecCreateDecoder;
        }
        else if(std::string_view{ name } == "rocjpeg")
        {
            rocjpeg_table_t* _table        = static_cast<rocjpeg_table_t*>(tables[0]);
            _table->rocJpegStreamCreate_fn = &rocprofiler::rocJpegStreamCreate;
        }
    }

    if(!::rocprofiler::check_registration_info(
           name, lib_version, num_tables, registration_info))
    {
        auto ss = std::stringstream{};
        ss << "no matching registration info for " << name << " "
           << " version " << lib_version << " (# tables = " << num_tables << ")";
        throw std::runtime_error{ ss.str() };
    }

    return 0;
}
}

// WINDOWS-DIVERGENCE: __attribute__((constructor)) (GCC) has no MSVC
// equivalent. The portable C++ way to run code at DLL load is a static
// initializer. We declare a namespace-scope variable whose initializer has
// the same observable effects as the GCC constructor (sets the flag,
// prints the line). This runs during DLL initialization on both platforms,
// before any exported function is called by a client.
bool rocprofiler_test_lib_link = false;

namespace
{
struct rocp_ctor_t
{
    rocp_ctor_t()
    {
        rocprofiler_test_lib_link = true;
        printf("[%s] %s\n", ROCP_REG_FILE_NAME, "rocp_ctor");
    }
};
const auto rocp_ctor_instance = rocp_ctor_t{};
}  // namespace
