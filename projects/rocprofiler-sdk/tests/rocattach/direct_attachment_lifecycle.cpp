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

#include <rocprofiler-sdk/registration.h>

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>

extern "C" rocprofiler_status_t
rocprofiler_load_attachment_tool(const char*);

extern "C" rocprofiler_status_t
rocprofiler_attach();

extern "C" rocprofiler_status_t
rocprofiler_detach();

namespace
{
bool
load_register()
{
    auto* handle = dlopen("librocprofiler-register.so", RTLD_GLOBAL | RTLD_LAZY);
    if(handle != nullptr) return true;

    std::cerr << "Direct attachment lifecycle test FAILED: could not load "
                 "rocprofiler-register: "
              << dlerror() << '\n';
    return false;
}

bool
configure_anytime_client(const char* tool_path)
{
    auto* handle = dlopen(tool_path, RTLD_LOCAL | RTLD_LAZY);
    if(handle == nullptr)
    {
        std::cerr << "Direct attachment lifecycle test FAILED: could not load "
                     "anytime tool: "
                  << dlerror() << '\n';
        return false;
    }

    auto configure         = rocprofiler_configure_func_t{};
    *(void**) (&configure) = dlsym(handle, "rocprofiler_configure");
    if(configure == nullptr || rocprofiler_force_configure(configure) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Direct attachment lifecycle test FAILED: anytime "
                     "configuration failed\n";
        return false;
    }

    setenv("ROCP_TOOL_LIBRARIES", tool_path, 1);
    setenv("ROCPROFILER_TEST_EXPECT_CLIENT_PRIORITY", "1", 1);
    return true;
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 2 && argc != 3)
    {
        std::cerr << "usage: direct-attachment-lifecycle [ANYTIME_TOOL] "
                     "ATTACHMENT_TOOL\n";
        return 1;
    }

    if(!load_register()) return 1;

    const auto* attachment_tool = argv[argc - 1];
    setenv("ROCPROFILER_TEST_EXPECT_ATTACH_COUNT", "1", 1);
    setenv("ROCPROFILER_TEST_EXPECT_CLIENT_PRIORITY", "0", 1);

    if(argc == 3 && !configure_anytime_client(argv[1])) return 1;

    if(rocprofiler_load_attachment_tool(attachment_tool) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "Direct attachment lifecycle test FAILED: attachment tool "
                     "load failed\n";
        return 1;
    }

    using context_active_t  = int (*)();
    auto* attachment_handle = dlopen(attachment_tool, RTLD_LOCAL | RTLD_LAZY | RTLD_NOLOAD);
    auto  context_active    = context_active_t{};
    if(attachment_handle != nullptr)
        *(void**) (&context_active) =
            dlsym(attachment_handle, "rocprofiler_test_callbackless_context_active");

    auto check_context = [context_active](int expected, const char* phase) {
        if(context_active == nullptr || context_active() == expected) return true;
        std::cerr << "Direct attachment lifecycle test FAILED: callback-less "
                     "context had the wrong state "
                  << phase << '\n';
        return false;
    };

    if(!check_context(0, "before attach") || rocprofiler_attach() != ROCPROFILER_STATUS_SUCCESS ||
       !check_context(1, "after attach") || rocprofiler_detach() != ROCPROFILER_STATUS_SUCCESS ||
       !check_context(0, "after detach"))
    {
        std::cerr << "Direct attachment lifecycle test FAILED: attachment "
                     "lifecycle call failed\n";
        return 1;
    }

    std::cout << "Direct attachment lifecycle test completed\n";
    return 0;
}
