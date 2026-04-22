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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <rocprofiler-register/rocprofiler-register.h>

#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef ROCP_REG_FILE_NAME
#    define ROCP_REG_FILE_NAME                                                           \
        ::std::string{ __FILE__ }                                                        \
            .substr(::std::string_view{ __FILE__ }.find_last_of('/') + 1)                \
            .c_str()
#endif

extern "C" {
rocprofiler_register_error_code_t
rocprofiler_register_attach(const char* environment_buffer, const char* tool_lib_path);
}

namespace
{
struct callback_state
{
    int                                         count = 0;
    rocprofiler_register_tool_activation_mode_t mode  = ROCP_REG_TOOL_ACTIVATION_NONE;
};

const char*
mode_string(rocprofiler_register_tool_activation_mode_t mode)
{
    switch(mode)
    {
        case ROCP_REG_TOOL_ACTIVATION_STARTUP: return "startup";
        case ROCP_REG_TOOL_ACTIVATION_ATTACH: return "attach";
        case ROCP_REG_TOOL_ACTIVATION_NONE: return "none";
    }

    return "unknown";
}

void
require(bool cond, const char* msg)
{
    if(!cond) throw std::runtime_error{ msg };
}

bool
env_enabled(const char* name)
{
    auto* itr = std::getenv(name);
    return (itr != nullptr && std::string_view{ itr } == "1");
}

void
tool_activation_cb(rocprofiler_register_tool_activation_mode_t mode, void* data)
{
    auto* state = static_cast<callback_state*>(data);
    ++state->count;
    state->mode = mode;
    printf("[%s] callback %s\n", ROCP_REG_FILE_NAME, mode_string(mode));
}

void
register_runtime_callback(callback_state& state)
{
    auto status = rocprofiler_register_runtime_tool_activation_callback(
        "test-runtime", &tool_activation_cb, &state);
    require(status == ROCP_REG_SUCCESS, "failed to register runtime activation callback");
}

void
activate_startup_tool()
{
    static int dummy_table = 7;
    void*      table       = static_cast<void*>(&dummy_table);
    auto       reg_id      = rocprofiler_register_library_indentifier_t{};
    auto       status      = rocprofiler_register_library_api_table(
        "hsa", nullptr, 20100, &table, 1, &reg_id);

    require(status == ROCP_REG_SUCCESS, "startup activation did not succeed");
    printf("[%s] startup identifier %lu\n", ROCP_REG_FILE_NAME, reg_id.handle);
}

void
register_attach_placeholder()
{
    static int dummy_table = 11;
    void*      table       = static_cast<void*>(&dummy_table);
    auto       reg_id      = rocprofiler_register_library_indentifier_t{};
    auto       status      = rocprofiler_register_library_api_table(
        "rocattach", nullptr, 1, &table, 1, &reg_id);

    require(status == ROCP_REG_SUCCESS || status == ROCP_REG_NO_TOOLS,
            "rocattach placeholder registration failed");
}

void
activate_attach_tool()
{
    register_attach_placeholder();
    auto status = rocprofiler_register_attach(nullptr, "libgeneric-tool.so");
    require(status == ROCP_REG_SUCCESS, "late attach activation did not succeed");
}
}  // namespace

int
main()
{
    auto state                    = callback_state{};
    auto register_after_activate  = env_enabled("ROCP_REG_TEST_TOOL_ACTIVE_REGISTER_AFTER");
    auto late_attach_mode         = env_enabled("ROCP_REG_TEST_TOOL_ACTIVE_ATTACH");
    auto expected_activation_mode = late_attach_mode ? ROCP_REG_TOOL_ACTIVATION_ATTACH
                                                     : ROCP_REG_TOOL_ACTIVATION_STARTUP;

    if(!register_after_activate)
    {
        register_runtime_callback(state);
        register_runtime_callback(state);
        require(state.count == 0, "callback fired before tool activation");
    }

    if(late_attach_mode)
        activate_attach_tool();
    else
        activate_startup_tool();

    if(register_after_activate)
    {
        register_runtime_callback(state);
        register_runtime_callback(state);
    }

    require(state.count == 1, "callback was not invoked exactly once");
    require(state.mode == expected_activation_mode, "callback mode did not match");

    printf("[%s] callback_count %d\n", ROCP_REG_FILE_NAME, state.count);
    printf("[%s] success %s %s\n",
           ROCP_REG_FILE_NAME,
           mode_string(expected_activation_mode),
           register_after_activate ? "after" : "before");
    return 0;
}
