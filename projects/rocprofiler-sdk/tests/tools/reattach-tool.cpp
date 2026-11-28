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

// Tool library for the reattach tests (tests/anytime-tool-config/reattach-after-detach and
// tests/rocattach/reattach_self_finalizing_test). Like rocprofiler-systems, it finalizes itself
// from tool_detach by calling the client finalize function it received in initialize. It starts
// its context on every attach, checks the lifecycle it observes (printing
// "reattach-tool[<name>]: FAIL: ..." on a violation) and exposes hooks to the in-process test
// driver. It is built once per client name (REATTACH_TOOL_NAME) so that several clients can be
// registered in one process.

#include <rocprofiler-sdk/experimental/registration.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <thread>

#if !defined(REATTACH_TOOL_NAME)
#    define REATTACH_TOOL_NAME "reattach-tool"
#endif

namespace
{
rocprofiler_client_finalize_t          client_finalizer = nullptr;
std::optional<rocprofiler_client_id_t> client_id        = {};
rocprofiler_context_id_t               context_id       = {0};
std::atomic<unsigned>                  attach_calls{0};
std::atomic<unsigned>                  detach_calls{0};
std::atomic<unsigned>                  fini_calls{0};
std::atomic<unsigned>                  failures{0};
std::atomic<bool>                      in_detach{false};
std::atomic<bool>                      in_fini{false};
std::atomic<bool>                      statics_destroyed{false};

// test driver options (reattach_tool_set_mode)
std::atomic<unsigned> detach_delay_msec{0};
std::atomic<unsigned> fini_delay_msec{0};
std::atomic<bool>     exit_in_detach{false};
std::atomic<bool>     exit_in_fini{false};
std::atomic<bool>     finalize_on_helper_thread{false};

// Stands in for the static objects of a real tool, which its tool_detach uses. At process exit
// they are destroyed before rocprofiler-sdk finalizes, so an exit racing with a tool_detach on
// another thread must wait for the tool_detach first. tool_detach reports it if it finds them
// destroyed; tool_fini reports an exit that did not wait, whether or not tool_detach got that far.
struct static_object_guard
{
    ~static_object_guard() { statics_destroyed.store(true); }
} tool_static_object = {};

void
report(const char* msg)
{
    std::printf("reattach-tool[%s]: %s\n", REATTACH_TOOL_NAME, msg);
    std::fflush(stdout);
}

void
fail(const char* msg)
{
    ++failures;
    std::printf("reattach-tool[%s]: FAIL: %s (attach=%u detach=%u fini=%u)\n",
                REATTACH_TOOL_NAME,
                msg,
                attach_calls.load(),
                detach_calls.load(),
                fini_calls.load());
    std::fflush(stdout);
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void*)
{
    client_finalizer = fini_func;
    if(rocprofiler_create_context(&context_id) != ROCPROFILER_STATUS_SUCCESS)
        fail("rocprofiler_create_context failed");
    return 0;
}

void
tool_fini(void*)
{
    in_fini.store(true);
    if(++fini_calls != 1) fail("tool_fini called more than once");
    if(attach_calls.load() != detach_calls.load()) fail("tool_fini ran during an attach session");
    if(in_detach.load()) fail("tool_fini ran while tool_detach was still running");

    std::printf("reattach-tool[%s]: tool_fini after %u session(s)\n",
                REATTACH_TOOL_NAME,
                attach_calls.load());
    std::fflush(stdout);

    // optionally simulate a finalize that takes a while (e.g. writing output)
    if(auto delay = fini_delay_msec.load(); delay > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds{delay});

    in_fini.store(false);
    if(exit_in_fini.load())
    {
        report("calling exit() from tool_fini");
        std::exit(EXIT_SUCCESS);
    }
}

int
tool_attach(rocprofiler_client_detach_t,
            rocprofiler_context_id_t* context_ids,
            uint64_t                  context_ids_length,
            void*)
{
    std::printf("reattach-tool[%s]: tool_attach session %u\n", REATTACH_TOOL_NAME, ++attach_calls);
    std::fflush(stdout);

    if(fini_calls.load() != 0) fail("tool_attach called after tool_fini");
    if(attach_calls.load() != detach_calls.load() + 1) fail("tool_attach called while attached");

    // a reattached session must get its context back and be able to start it
    bool found = false;
    for(uint64_t i = 0; i < context_ids_length; ++i)
        if(context_ids[i].handle == context_id.handle) found = true;
    if(!found) fail("tool_attach did not receive the tool's context");

    int finalized = -1;
    rocprofiler_is_finalized(&finalized);
    if(finalized != 0) fail("rocprofiler-sdk reports finalizing or finalized during tool_attach");

    if(rocprofiler_start_context(context_id) != ROCPROFILER_STATUS_SUCCESS)
        fail("rocprofiler_start_context failed in tool_attach");

    return 0;
}

void
tool_detach(void*)
{
    std::printf(
        "reattach-tool[%s]: tool_detach session %u\n", REATTACH_TOOL_NAME, attach_calls.load());
    std::fflush(stdout);
    ++detach_calls;
    in_detach.store(true);

    // the tool tears itself down at the end of each attach session, like rocprofiler-systems,
    // optionally from a helper thread that tool_detach waits for
    if(client_finalizer && client_id)
    {
        if(finalize_on_helper_thread.load())
            std::thread{[]() { client_finalizer(*client_id); }}.join();
        else
            client_finalizer(*client_id);
    }
    if(fini_calls.load() != 0) fail("the finalize request from tool_detach finalized the tool");

    // a detach is not a finalization of rocprofiler-sdk
    int finalized = -1;
    rocprofiler_is_finalized(&finalized);
    if(finalized != 0) fail("rocprofiler-sdk reports finalizing or finalized during tool_detach");

    // optionally simulate a tool_detach that takes a while (e.g. writing output)
    if(auto delay = detach_delay_msec.load(); delay > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds{delay});

    if(statics_destroyed.load())
        fail("the tool's static objects were destroyed during tool_detach");

    in_detach.store(false);
    if(exit_in_detach.load())
    {
        report("calling exit() from tool_detach");
        std::exit(EXIT_SUCCESS);
    }
}
}  // namespace

extern "C" {
rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* identifier)
{
    identifier->name = REATTACH_TOOL_NAME;
    client_id.emplace(*identifier);
    static auto result = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini, nullptr};
    return &result;
}

rocprofiler_tool_configure_attach_result_t*
rocprofiler_configure_attach(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    static auto result = rocprofiler_tool_configure_attach_result_t{
        sizeof(rocprofiler_tool_configure_attach_result_t), tool_attach, tool_detach, nullptr};
    return &result;
}

// hooks for the in-process test driver (tests/anytime-tool-config/reattach-after-detach.cpp)

__attribute__((visibility("default"))) int
reattach_tool_initialized()
{
    return (client_finalizer && client_id && context_id.handle != 0) ? 1 : 0;
}

__attribute__((visibility("default"))) void
reattach_tool_counts(unsigned* attach, unsigned* detach, unsigned* fini, unsigned* failed)
{
    *attach = attach_calls.load();
    *detach = detach_calls.load();
    *fini   = fini_calls.load();
    *failed = failures.load();
}

// the tool's finalize request, made outside of a detach
__attribute__((visibility("default"))) void
reattach_tool_finalize()
{
    if(client_finalizer && client_id) client_finalizer(*client_id);
}

__attribute__((visibility("default"))) void
reattach_tool_set_mode(unsigned detach_delay_ms,
                       unsigned fini_delay_ms,
                       int      exit_from_detach,
                       int      exit_from_fini,
                       int      finalize_on_helper)
{
    detach_delay_msec.store(detach_delay_ms);
    fini_delay_msec.store(fini_delay_ms);
    exit_in_detach.store(exit_from_detach != 0);
    exit_in_fini.store(exit_from_fini != 0);
    finalize_on_helper_thread.store(finalize_on_helper != 0);
}

// whether tool_detach or tool_fini is running
__attribute__((visibility("default"))) int
reattach_tool_in_callback()
{
    return (in_detach.load() || in_fini.load()) ? 1 : 0;
}
}
