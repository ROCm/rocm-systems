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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Attach, detach and attach again in one process with a tool (reattach-after-detach-tool) that
// finalizes itself from tool_detach. The finalize request issued during detach must not
// unregister the client, otherwise the next attach finds no clients.

#include <rocprofiler-sdk/rocprofiler.h>

#include <dlfcn.h>

#include <iostream>

extern "C" {
// Exported by rocprofiler-sdk and used by rocprofiler-register for runtime attachment.
rocprofiler_status_t
rocprofiler_attach(void);

rocprofiler_status_t
rocprofiler_detach(void);
}

namespace
{
template <typename FuncT>
FuncT
tool_symbol(void* handle, const char* name)
{
    auto func         = FuncT{};
    *(void**) (&func) = dlsym(handle, name);
    return func;
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " <tool library>\n";
        return 2;
    }

    // ROCP_TOOL_LIBRARIES loaded the tool when the SDK initialized; do not load it again
    auto* tool = dlopen(argv[1], RTLD_LAZY | RTLD_NOLOAD);
    if(!tool)
    {
        std::cerr << "Tool library was not loaded by rocprofiler-sdk\n";
        return 2;
    }

    auto initialized = tool_symbol<int (*)()>(tool, "reattach_tool_initialized");
    auto counts =
        tool_symbol<void (*)(unsigned*, unsigned*, unsigned*)>(tool, "reattach_tool_counts");
    auto finalize = tool_symbol<void (*)()>(tool, "reattach_tool_finalize");
    if(!initialized || !counts || !finalize || initialized() == 0)
    {
        std::cerr << "Tool was not configured\n";
        return 2;
    }

    constexpr unsigned sessions = 3;
    for(unsigned session = 1; session <= sessions; ++session)
    {
        auto attach_status = rocprofiler_attach();
        if(attach_status != ROCPROFILER_STATUS_SUCCESS)
        {
            std::cerr << "Attach " << session << " failed with status " << attach_status << " ("
                      << rocprofiler_get_status_name(attach_status) << ")\n";
            return 1;
        }

        auto detach_status = rocprofiler_detach();
        if(detach_status != ROCPROFILER_STATUS_SUCCESS)
        {
            std::cerr << "Detach " << session << " failed with status " << detach_status << " ("
                      << rocprofiler_get_status_name(detach_status) << ")\n";
            return 1;
        }
    }

    unsigned attach_calls = 0, detach_calls = 0, fini_calls = 0;
    counts(&attach_calls, &detach_calls, &fini_calls);
    if(attach_calls != sessions || detach_calls != sessions || fini_calls != 0)
    {
        std::cerr << "Unexpected callback counts: attach=" << attach_calls
                  << " detach=" << detach_calls << " fini=" << fini_calls << "\n";
        return 1;
    }

    // Outside of a detach, the same request finalizes the client as usual.
    finalize();
    counts(&attach_calls, &detach_calls, &fini_calls);
    if(fini_calls != 1)
    {
        std::cerr << "Client finalize outside detach did not finalize the tool\n";
        return 1;
    }

    std::cout << "Reattach after detach succeeded\n";
    return 0;
}
