// rocprofv3-qi: queue-interception-only kernel-trace tool.
//
// Demonstrates that hsa_amd_queue_intercept_create + intercept_register works
// on Windows (WSL DXG) for kernel-dispatch tracing, without falling back to
// the doorbell-store hook used by rocprofv3-min. Substitutes hsa_queue_create
// with the AMD intercept extension and registers a packet writer that does
// per-dispatch signal swap + async handler. No HIP API trace, no memory copy
// trace.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "../include/rocprofv3_qi_registration.h"
#include "trace_buffer.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct HsaApiTable;

namespace rocprofv3_min {
    void install_hsa_wrappers(HsaApiTable*);

    bool g_kernel_trace_enabled      = true;
    bool g_hip_trace_enabled         = false;
    bool g_memory_copy_trace_enabled = false;

    static std::string g_output_dir;
    static std::atomic<bool> g_flushed{false};

    static void load_env() {
        const char* d = std::getenv("ROCPROFV3_OUTPUT_DIR");
        g_output_dir = d ? d : std::string(".");
    }

    static void flush_once() {
        bool expected = false;
        if (!g_flushed.compare_exchange_strong(expected, true)) return;
        TraceBuffers::instance().flush(g_output_dir, GetCurrentProcessId());
        std::fprintf(stderr, "[rocprofv3-qi] flushed traces to %s\n", g_output_dir.c_str());
    }

    static int tool_init(rocprofv3_qi_client_finalize_t /*finalize_func*/, void* /*tool_data*/) {
        std::fprintf(stderr, "[rocprofv3-qi] tool_init: queue-interception kernel-trace\n");
        std::atexit(&flush_once);
        return 0;
    }
}

extern "C" __declspec(dllexport)
rocprofv3_qi_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/, const char* /*runtime_version*/,
                      uint32_t /*priority*/, void* /*client_id*/)
{
    rocprofv3_min::load_env();
    static rocprofv3_qi_tool_configure_result_t cfg = {
        sizeof(rocprofv3_qi_tool_configure_result_t),
        &rocprofv3_min::tool_init,
        nullptr,
        nullptr
    };
    std::fprintf(stderr, "[rocprofv3-qi] rocprofiler_configure called\n");
    return &cfg;
}

extern "C" __declspec(dllexport)
int rocprofiler_set_api_table(const char* name, uint64_t /*lib_version*/,
                              uint64_t /*lib_instance*/, void** tables,
                              uint64_t num_tables)
{
    if (!name || !tables || num_tables == 0) return 0;
    std::fprintf(stderr, "[rocprofv3-qi] rocprofiler_set_api_table: name=%s tables=%llu\n",
                 name, (unsigned long long)num_tables);

    if (std::strcmp(name, "hsa") == 0) {
        rocprofv3_min::install_hsa_wrappers(static_cast<HsaApiTable*>(tables[0]));
    }
    return 0;
}

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    switch (reason) {
        case DLL_PROCESS_ATTACH:
            rocprofv3_min::load_env();
            break;
        case DLL_PROCESS_DETACH:
            rocprofv3_min::flush_once();
            break;
    }
    return TRUE;
}
