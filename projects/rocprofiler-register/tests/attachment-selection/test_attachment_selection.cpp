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

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

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
std::vector<char>
make_environment_buffer(std::string_view name, std::string_view value)
{
    auto pair_count = uint32_t{ 1 };
    auto buffer     = std::vector<char>(sizeof(pair_count));
    std::memcpy(buffer.data(), &pair_count, sizeof(pair_count));
    buffer.insert(buffer.end(), name.begin(), name.end());
    buffer.emplace_back('\0');
    buffer.insert(buffer.end(), value.begin(), value.end());
    buffer.emplace_back('\0');
    return buffer;
}

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
    using detach_func_t = rocprofiler_register_error_code_t (*)();

    auto* attach_symbol = dlsym(RTLD_DEFAULT, "rocprofiler_register_attach");
    auto* detach_symbol = dlsym(RTLD_DEFAULT, "rocprofiler_register_detach");
    if(attach_symbol == nullptr || detach_symbol == nullptr)
    {
        std::cerr << "Test FAILED: runtime attachment functions are not discoverable\n";
        return false;
    }

    auto attach = reinterpret_cast<attach_func_t>(attach_symbol);
    auto detach = reinterpret_cast<detach_func_t>(detach_symbol);

    setenv("ROCPROFILER_REGISTER_TEST_ATTACH_DELAY_MS", "50", 1);
    auto concurrent_status =
        std::array<rocprofiler_register_error_code_t, 2>{ ROCP_REG_ERROR_CODE_END,
                                                          ROCP_REG_ERROR_CODE_END };
    auto first_attach = std::thread{ [&]() {
        concurrent_status[0]= attach(nullptr, "libgeneric-tool.so");
    } };
    auto second_attach = std::thread{ [&]() {
        concurrent_status[1]= attach(nullptr, "libgeneric-tool.so");
    } };
    first_attach.join();
    second_attach.join();
    unsetenv("ROCPROFILER_REGISTER_TEST_ATTACH_DELAY_MS");

    using max_concurrent_attach_calls_t = int (*)();
    auto* max_concurrent_symbol =
        dlsym(RTLD_DEFAULT, "rocprofiler_test_max_concurrent_attach_calls");
    if(max_concurrent_symbol == nullptr)
    {
        std::cerr << "Test FAILED: attach concurrency query is not discoverable\n";
        return false;
    }
    auto max_concurrent_attach_calls =
        reinterpret_cast<max_concurrent_attach_calls_t>(max_concurrent_symbol);
    auto observed_max_concurrent_attach_calls = max_concurrent_attach_calls();
    if(concurrent_status[0] != ROCP_REG_SUCCESS ||
       concurrent_status[1] != ROCP_REG_SUCCESS ||
       observed_max_concurrent_attach_calls != 1 || detach() != ROCP_REG_SUCCESS)
    {
        std::cerr << "Test FAILED: runtime attachment calls were not serialized\n";
        return false;
    }

    auto status = attach(nullptr, "libgeneric-tool.so");
    if(status != ROCP_REG_SUCCESS || detach() != ROCP_REG_SUCCESS)
    {
        std::cerr << "Test FAILED: initial runtime attachment lifecycle failed\n";
        return false;
    }

    constexpr auto failure_env = "ROCPROFILER_REGISTER_TEST_ATTACH_FAILURE";
    setenv(failure_env, "0", 1);
    auto failure_buffer        = make_environment_buffer(failure_env, "1");
    status                     = attach(failure_buffer.data(), "libgeneric-tool.so");
    const auto* restored_value = std::getenv(failure_env);
    if(status != ROCP_REG_ROCPROFILER_ERROR || restored_value == nullptr ||
       std::string_view{ restored_value } != "0")
    {
        std::cerr << "Test FAILED: failed runtime attachment did not restore the "
                     "target environment\n";
        return false;
    }

    status = attach(nullptr, "libgeneric-tool.so");
    unsetenv(failure_env);
    if(status != ROCP_REG_SUCCESS || detach() != ROCP_REG_SUCCESS)
    {
        std::cerr << "Test FAILED: runtime attachment lifecycle did not recover after "
                     "rollback\n";
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
