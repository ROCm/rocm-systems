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

// Attach, detach and attach again in one process with tools (tests/tools/reattach-tool.cpp) that
// finalize themselves from tool_detach. The finalize requests issued during a detach must not
// unregister the clients, otherwise the next attach finds no clients, and finalization at process
// exit must not run underneath a tool callback.
//
//   reattach-after-detach [<mode>] <tool library>...
//
// Every tool library must also be listed in ROCP_TOOL_LIBRARIES: rocprofiler-sdk loads them when
// it initializes (ROCPROFILER_LIBRARY_CTOR=1). Without a mode, the driver runs three
// attach/detach sessions with every tool (one tool exercises the single-client finalize() path,
// two tools the per-client path) and then has the first tool finalize itself outside of a
// detach, which must finalize it. Modes:
//   --finalize-on-helper-thread  tool_detach makes its finalize request from a helper thread that
//                                it joins
//   --exit-during-detach         the process exits while tool_detach runs on another thread;
//                                finalization must wait for it and then finalize the tool
//   --exit-from-detach           the tool calls exit() from tool_detach; rocprofiler-sdk must
//                                still finalize the tool at exit
//   --exit-from-finalize         after the sessions, the first tool calls exit() from its
//                                finalize; the process must exit without deadlocking and finalize
//                                the other tools
//   --concurrent-finalize        after the sessions, two threads request the first tool's
//                                finalization at once (two tools, so the per-client path is
//                                taken); the second request must return only once the tool's
//                                finalize callback has completed

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <dlfcn.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

extern "C" {
// Exported by rocprofiler-sdk and used by rocprofiler-register for runtime attachment.
rocprofiler_status_t
rocprofiler_attach(void);

rocprofiler_status_t
rocprofiler_detach(void);
}

namespace
{
struct tool_t
{
    const char* path                                           = nullptr;
    int (*initialized)()                                       = nullptr;
    void (*counts)(unsigned*, unsigned*, unsigned*, unsigned*) = nullptr;
    void (*finalize)()                                         = nullptr;
    void (*set_mode)(unsigned, unsigned, int, int, int)        = nullptr;
    int (*in_callback)()                                       = nullptr;

    struct counts_t
    {
        unsigned attach = 0;
        unsigned detach = 0;
        unsigned fini   = 0;
        unsigned failed = 0;
    };

    counts_t get_counts() const
    {
        auto _v = counts_t{};
        counts(&_v.attach, &_v.detach, &_v.fini, &_v.failed);
        return _v;
    }
};

template <typename FuncT>
bool
load_symbol(void* handle, const char* name, FuncT& func)
{
    *reinterpret_cast<void**>(&func) = dlsym(handle, name);
    return func != nullptr;
}

bool
load_tool(const char* path, tool_t& tool)
{
    // ROCP_TOOL_LIBRARIES loaded the tool when the SDK initialized; do not load it again
    auto* handle = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if(!handle)
    {
        std::cerr << "Tool library " << path << " was not loaded by rocprofiler-sdk\n";
        return false;
    }

    tool.path = path;
    if(!load_symbol(handle, "reattach_tool_initialized", tool.initialized) ||
       !load_symbol(handle, "reattach_tool_counts", tool.counts) ||
       !load_symbol(handle, "reattach_tool_finalize", tool.finalize) ||
       !load_symbol(handle, "reattach_tool_set_mode", tool.set_mode) ||
       !load_symbol(handle, "reattach_tool_in_callback", tool.in_callback) ||
       tool.initialized() == 0)
    {
        std::cerr << "Tool " << path << " was not configured\n";
        return false;
    }
    return true;
}

int
report_failure(const char* what, unsigned session, rocprofiler_status_t status)
{
    std::cerr << what << " " << session << " failed with status " << status << " ("
              << rocprofiler_get_status_name(status) << ")\n";
    return 1;
}

int
exit_during_detach(const tool_t& tool)
{
    tool.set_mode(500, 0, 0, 0, 0);  // tool_detach takes 500 ms

    if(auto status = rocprofiler_attach(); status != ROCPROFILER_STATUS_SUCCESS)
        return report_failure("Attach", 1, status);

    std::thread{[]() { rocprofiler_detach(); }}.detach();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while(tool.in_callback() == 0)
    {
        if(std::chrono::steady_clock::now() > deadline)
        {
            std::cerr << "tool_detach never started\n";
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }

    // exit while tool_detach is still running on the other thread
    std::exit(EXIT_SUCCESS);
}

int
concurrent_finalize(const tool_t& tool)
{
    constexpr auto delay = std::chrono::milliseconds{500};
    tool.set_mode(0, delay.count(), 0, 0, 0);  // tool_fini takes 500 ms

    auto first = std::thread{[&tool]() { tool.finalize(); }};

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while(tool.in_callback() == 0)
    {
        if(std::chrono::steady_clock::now() > deadline)
        {
            std::cerr << "tool_fini never started\n";
            first.join();
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }

    // a second request for the same client while its finalize callback runs on another thread
    const auto start = std::chrono::steady_clock::now();
    tool.finalize();
    const auto waited = std::chrono::steady_clock::now() - start;
    first.join();

    if(auto c = tool.get_counts(); c.fini != 1 || c.failed != 0)
    {
        std::cerr << "Unexpected callback counts after concurrent finalization: fini=" << c.fini
                  << " failed=" << c.failed << "\n";
        return 1;
    }
    if(waited < delay / 2)
    {
        std::cerr << "Concurrent finalization request returned after "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(waited).count()
                  << " ms, before the finalize callback completed\n";
        return 1;
    }

    std::cout << "Concurrent finalization request waited for the finalize callback\n";
    return 0;
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto mode        = std::string_view{(argc > 1) ? argv[1] : ""};
    const bool has_mode    = mode.substr(0, 2) == "--";
    const bool helper_mode = mode == "--finalize-on-helper-thread";
    const bool exit_during = mode == "--exit-during-detach";
    const bool exit_detach = mode == "--exit-from-detach";
    const bool exit_fini   = mode == "--exit-from-finalize";
    const bool concurrent  = mode == "--concurrent-finalize";
    const int  first_tool  = (has_mode) ? 2 : 1;
    if(argc <= first_tool ||
       (has_mode && !helper_mode && !exit_during && !exit_detach && !exit_fini && !concurrent))
    {
        std::cerr << "usage: " << argv[0]
                  << " [--finalize-on-helper-thread|--exit-during-detach|--exit-from-detach|"
                     "--exit-from-finalize|--concurrent-finalize] <tool library>...\n";
        return 2;
    }

    auto tools = std::vector<tool_t>{};
    for(int i = first_tool; i < argc; ++i)
    {
        if(!load_tool(argv[i], tools.emplace_back())) return 2;
    }

    if(exit_during) return exit_during_detach(tools.front());
    tools.front().set_mode(0, 0, exit_detach ? 1 : 0, exit_fini ? 1 : 0, helper_mode ? 1 : 0);

    constexpr unsigned sessions = 3;
    for(unsigned session = 1; session <= sessions; ++session)
    {
        if(auto status = rocprofiler_attach(); status != ROCPROFILER_STATUS_SUCCESS)
            return report_failure("Attach", session, status);
        if(auto status = rocprofiler_detach(); status != ROCPROFILER_STATUS_SUCCESS)
            return report_failure("Detach", session, status);

        // in --exit-from-detach mode the tool called exit() from this detach
        if(exit_detach)
        {
            std::cerr << "tool_detach did not call exit()\n";
            return 1;
        }

        int finalized = -1;
        rocprofiler_is_finalized(&finalized);
        if(finalized != 0)
        {
            std::cerr << "rocprofiler-sdk reports finalization status " << finalized
                      << " after detach " << session << "\n";
            return 1;
        }
    }

    for(const auto& tool : tools)
    {
        auto c = tool.get_counts();
        if(c.attach != sessions || c.detach != sessions || c.fini != 0 || c.failed != 0)
        {
            std::cerr << "Unexpected callback counts for " << tool.path << ": attach=" << c.attach
                      << " detach=" << c.detach << " fini=" << c.fini << " failed=" << c.failed
                      << "\n";
            return 1;
        }
    }

    if(concurrent) return concurrent_finalize(tools.front());

    // Outside of a detach, the same request finalizes the client as usual: with one client this
    // finalizes rocprofiler-sdk, with several only the requesting client. In --exit-from-finalize
    // mode the tool calls exit() from its finalize instead of returning, and the other tools must
    // be finalized by the exit.
    tools.front().finalize();
    if(exit_fini)
    {
        std::cerr << "tool_fini did not call exit()\n";
        return 1;
    }

    for(size_t i = 0; i < tools.size(); ++i)
    {
        auto           c        = tools.at(i).get_counts();
        const unsigned expected = (i == 0) ? 1 : 0;
        if(c.fini != expected)
        {
            std::cerr << "Client finalize outside detach: " << tools.at(i).path << " was finalized "
                      << c.fini << " time(s), expected " << expected << "\n";
            return 1;
        }
    }

    std::cout << "Reattach after detach succeeded with " << tools.size() << " client(s)\n";
    return 0;
}
