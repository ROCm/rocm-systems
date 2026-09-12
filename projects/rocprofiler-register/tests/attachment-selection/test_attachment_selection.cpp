// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-register/rocprofiler-register.h>
#include <hsa-runtime/hsa-runtime.hpp>

#include "common/defines.hpp"

#include <dlfcn.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string_view>

struct rocprofiler_client_id_t;
struct rocprofiler_tool_configure_result_t;

#if !defined(ROCPROFILER_REGISTER_TEST_NO_AMBIENT_CONFIGURE)
// Model a dormant framework-owned client symbol. Attachment should take
// precedence unless startup profiling was explicitly requested.
extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
    ROCPROFILER_REGISTER_TEST_PUBLIC_API;

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    return nullptr;
}
#endif

namespace
{
bool
verify_configure_symbol()
{
    if(dlsym(RTLD_DEFAULT, "rocprofiler_configure") != nullptr) return true;
    std::cerr << "Test FAILED: ambient rocprofiler_configure is not discoverable\n";
    return false;
}

bool
verify_preload_dependency(const char* wrapper_path, const char* dependency_path)
{
    auto* handle = dlopen(wrapper_path, RTLD_LAZY | RTLD_NOLOAD);
    if(!handle)
    {
        std::cerr << "Test FAILED: preload wrapper is not loaded\n";
        return false;
    }

    auto* symbol = dlsym(handle, "rocprofiler_configure");
    auto  info   = Dl_info{};
    auto  result =
        (symbol != nullptr && dladdr(symbol, &info) != 0 && info.dli_fname != nullptr);
    if(result)
    {
        auto ec = std::error_code{};
        result  = std::filesystem::equivalent(std::filesystem::path{ info.dli_fname },
                                             std::filesystem::path{ dependency_path },
                                             ec);
    }
    dlclose(handle);

    if(!result)
        std::cerr << "Test FAILED: preload wrapper did not resolve configure from its "
                     "dependency\n";
    return result;
}

bool
verify_runtime_attach()
{
    using attach_func_t = rocprofiler_register_error_code_t (*)(const char*, const char*);

    auto* symbol = dlsym(RTLD_DEFAULT, "rocprofiler_register_attach");
    if(symbol == nullptr)
    {
        std::cerr << "Test FAILED: rocprofiler_register_attach is not discoverable\n";
        return false;
    }

    auto attach = reinterpret_cast<attach_func_t>(symbol);
    auto status = attach(nullptr, "libgeneric-tool.so");
    if(status != ROCP_REG_SUCCESS)
    {
        std::cerr << "Test FAILED: runtime attachment returned " << status << '\n';
        return false;
    }

    std::cout << "Test PASSED: runtime attachment loaded the SDK with FORCE_LOAD=0\n";
    return true;
}
}  // namespace

int
main(int argc, char** argv)
{
    auto runtime_attach =
        (argc == 2 && std::string_view{ argv[1] } == "--runtime-attach");
    if(!verify_configure_symbol()) return 1;
    if(argc == 3 && !verify_preload_dependency(argv[1], argv[2])) return 1;
    if(argc != 1 && argc != 3 && !runtime_attach) return 1;

    hsa_init();
    if(runtime_attach && !verify_runtime_attach()) return 1;
    return 0;
}
