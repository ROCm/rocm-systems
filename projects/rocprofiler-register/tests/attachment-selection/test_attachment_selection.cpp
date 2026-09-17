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
#include <iostream>
#include <string_view>

struct rocprofiler_client_id_t;
struct rocprofiler_tool_configure_result_t;

#if !defined(ROCPROFILER_REGISTER_TEST_NO_AMBIENT_CONFIGURE)
// Model a dormant framework-owned client symbol. Its normal startup discovery
// must not prevent the independent attachment capability from being initialized.
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
#if defined(ROCPROFILER_REGISTER_TEST_NO_AMBIENT_CONFIGURE)
    return true;
#else
    if(dlsym(RTLD_DEFAULT, "rocprofiler_configure") != nullptr) return true;
    std::cerr << "Test FAILED: ambient rocprofiler_configure is not discoverable\n";
    return false;
#endif
}

bool
verify_attachment_capability()
{
    using initialized_func_t = int (*)();

    auto* symbol = dlsym(RTLD_DEFAULT, "rocprofiler_test_attachment_initialized");
    if(symbol == nullptr)
    {
        std::cerr << "Test FAILED: attachment library was not loaded\n";
        return false;
    }

    auto initialized = reinterpret_cast<initialized_func_t>(symbol);
    if(initialized() == 1) return true;

    std::cerr << "Test FAILED: attachment capability was not initialized\n";
    return false;
}

bool
verify_sdk_loaded()
{
    if(dlsym(RTLD_DEFAULT, "rocprofiler_set_api_table") != nullptr) return true;
    std::cerr << "Test FAILED: rocprofiler-sdk was not loaded\n";
    return false;
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

    std::cout << "Test fixture: runtime attachment loaded the SDK\n";
    return true;
}
}  // namespace

int
main(int argc, char** argv)
{
    auto runtime_attach =
        (argc == 2 && std::string_view{ argv[1] } == "--runtime-attach");
    if(!verify_configure_symbol()) return 1;
    if(argc != 1 && !runtime_attach) return 1;

    hsa_init();
    if(!verify_attachment_capability()) return 1;
    if(runtime_attach && !verify_runtime_attach()) return 1;
    if(!verify_sdk_loaded()) return 1;

    std::cout << "Test PASSED: attachment coexists with startup initialization\n";
    return 0;
}
