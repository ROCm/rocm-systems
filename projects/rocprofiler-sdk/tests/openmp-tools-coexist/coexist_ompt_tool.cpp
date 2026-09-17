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

// A mock "second" OMPT tool (standing in for TAU / Score-P / a custom tool)
// used to exercise rocprofiler-sdk's OMPT keep-or-defer role decision -- the
// EITHER/OR hand-off, NOT simultaneous hosting. It exposes a strong
// ompt_start_tool and records whether the OpenMP runtime actually made it the
// OMPT tool (initialize() called) and a few callback counts. On teardown it
// writes a JSON summary to $COEXIST_MOCK_OUTPUT for pytest.
//
// COEXIST_MOCK_MODE selects ompt_start_tool behavior:
//   own   (default) - act as a normal OMPT tool: return our own result. The SDK
//                     uses this when it DEFERS (no SDK client wants OMPT).
//   defer           - return rocprofiler_ompt_start_tool() (the SDK's own
//                     result), i.e. hand the OMPT tool role BACK to the SDK.
//                     Used (preloaded first, so the runtime binds us) to prove
//                     the defer-back path arms the SDK's OMPT.

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE 1  // for RTLD_DEFAULT
#endif

#include <rocprofiler-sdk/ompt/omp-tools.h>

#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace
{
ompt_set_callback_t set_callback = nullptr;

std::atomic<bool>     initialized{false};
std::atomic<bool>     finalized{false};
std::atomic<uint64_t> n_thread_begin{0};
std::atomic<uint64_t> n_device_initialize{0};
std::atomic<uint64_t> n_target_submit_emi{0};

void
on_thread_begin(ompt_thread_t, ompt_data_t*)
{
    ++n_thread_begin;
}

void
on_device_initialize(int, const char*, ompt_device_t*, ompt_function_lookup_t, const char*)
{
    ++n_device_initialize;
}

void
on_target_submit_emi(ompt_scope_endpoint_t, ompt_data_t*, ompt_id_t*, unsigned int)
{
    ++n_target_submit_emi;
}

void
write_summary()
{
    static std::once_flag _once;
    std::call_once(_once, []() {
        const char* path = ::getenv("COEXIST_MOCK_OUTPUT");
        FILE*       os   = path ? ::fopen(path, "w") : stderr;
        if(os == nullptr) os = stderr;

        auto b = [](const std::atomic<bool>& v) { return v.load() ? "true" : "false"; };

        fprintf(os,
                "{\n"
                "  \"coexist-mock-tool\": {\n"
                "    \"initialized\": %s,\n"
                "    \"finalized\": %s,\n"
                "    \"events\": {\n"
                "      \"thread_begin\": %llu,\n"
                "      \"device_initialize\": %llu,\n"
                "      \"target_submit_emi\": %llu\n"
                "    }\n"
                "  }\n"
                "}\n",
                b(initialized),
                b(finalized),
                static_cast<unsigned long long>(n_thread_begin.load()),
                static_cast<unsigned long long>(n_device_initialize.load()),
                static_cast<unsigned long long>(n_target_submit_emi.load()));
        if(os != stderr) ::fclose(os);
    });
}

int
my_initialize(ompt_function_lookup_t lookup, int, ompt_data_t*)
{
    initialized.store(true);
    set_callback = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if(set_callback)
    {
        set_callback(ompt_callback_thread_begin,
                     reinterpret_cast<ompt_callback_t>(on_thread_begin));
        set_callback(ompt_callback_device_initialize,
                     reinterpret_cast<ompt_callback_t>(on_device_initialize));
        set_callback(ompt_callback_target_submit_emi,
                     reinterpret_cast<ompt_callback_t>(on_target_submit_emi));
    }
    return 1;  // nonzero => we use OMPT
}

void
my_finalize(ompt_data_t*)
{
    finalized.store(true);
    write_summary();
}

// Fallback for the defer case, where my_finalize never fires because we were
// never made the tool: write the summary (initialized=false) at teardown.
// write_summary() is idempotent, so whichever fires first wins.
struct final_summary_writer
{
    ~final_summary_writer() { write_summary(); }
};
final_summary_writer g_final_summary_writer{};
}  // namespace

extern "C" __attribute__((visibility("default"))) ompt_start_tool_result_t*
ompt_start_tool(unsigned int omp_version, const char* runtime_version)
{
    const char* mode = ::getenv("COEXIST_MOCK_MODE");
    if(mode == nullptr) mode = "own";

    if(strcmp(mode, "defer") == 0)
    {
        using start_tool_t = ompt_start_tool_result_t* (*) (unsigned int, const char*);
        auto* sdk =
            reinterpret_cast<start_tool_t>(::dlsym(RTLD_DEFAULT, "rocprofiler_ompt_start_tool"));
        return sdk ? sdk(omp_version, runtime_version) : nullptr;
    }

    static ompt_start_tool_result_t result = {&my_initialize, &my_finalize, {0}};
    return &result;
}
