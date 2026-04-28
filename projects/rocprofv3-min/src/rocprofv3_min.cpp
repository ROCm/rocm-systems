// rocprofv3-min: minimal rocprofiler-sdk-shaped DLL.
//
// Loaded by rocprofiler-register.dll when the broker scans for
// configure-able tools. Implements just enough of the rocprofiler-sdk
// registration ABI to:
//   1. Be discovered (rocprofiler_configure)
//   2. Receive HIP/HSA dispatch tables (rocprofiler_set_api_table)
//   3. Wrap a curated subset of HIP and HSA entries with timing shims
//   4. Flush three CSV files at process exit

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "../include/rocprofv3_min_registration.h"
#include "trace_buffer.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Forward decls of headers from rocr / clr -- we forward-declare the table
// types so this TU doesn't pull every transitive header.
struct HsaApiTable;
struct HipDispatchTable;
struct HipCompilerDispatchTable;

namespace rocprofv3_min {
    void install_hip_wrappers(HipDispatchTable*);
    void install_hip_compiler_wrappers(HipCompilerDispatchTable*);
    void install_hsa_wrappers(HsaApiTable*);

    bool g_hip_trace_enabled         = false;
    bool g_kernel_trace_enabled      = false;
    bool g_memory_copy_trace_enabled = false;

    static std::string g_output_dir;
    static std::atomic<bool> g_flushed{false};

    static bool env_truthy(const char* name) {
        const char* v = std::getenv(name);
        if (!v) return false;
        if (*v == '\0' || *v == '0') return false;
        return true;
    }

    static void load_env() {
        g_hip_trace_enabled         = env_truthy("ROCPROFV3_HIP_TRACE");
        g_kernel_trace_enabled      = env_truthy("ROCPROFV3_KERNEL_TRACE");
        g_memory_copy_trace_enabled = env_truthy("ROCPROFV3_MEMORY_COPY_TRACE");
        const char* d = std::getenv("ROCPROFV3_OUTPUT_DIR");
        g_output_dir = d ? d : std::string(".");
    }

    static void flush_once() {
        bool expected = false;
        if (!g_flushed.compare_exchange_strong(expected, true)) return;
        TraceBuffers::instance().flush(g_output_dir, GetCurrentProcessId());
        std::fprintf(stderr, "[rocprofv3-min] flushed traces to %s\n", g_output_dir.c_str());
    }

    static int tool_init(rocprofv3_min_client_finalize_t /*finalize_func*/, void* /*tool_data*/) {
        std::fprintf(stderr, "[rocprofv3-min] tool_init: hip=%d kernel=%d memcopy=%d\n",
                     (int)g_hip_trace_enabled, (int)g_kernel_trace_enabled,
                     (int)g_memory_copy_trace_enabled);
        std::atexit(&flush_once);
        return 0;
    }
}

// ---- Public exports ----

extern "C" __declspec(dllexport)
rocprofv3_min_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/, const char* /*runtime_version*/,
                      uint32_t /*priority*/, void* /*client_id*/)
{
    rocprofv3_min::load_env();
    static rocprofv3_min_tool_configure_result_t cfg = {
        sizeof(rocprofv3_min_tool_configure_result_t),
        &rocprofv3_min::tool_init,
        nullptr,
        nullptr
    };
    std::fprintf(stderr, "[rocprofv3-min] rocprofiler_configure called\n");
    return &cfg;
}

extern "C" __declspec(dllexport)
int rocprofiler_set_api_table(const char* name, uint64_t /*lib_version*/,
                              uint64_t /*lib_instance*/, void** tables,
                              uint64_t num_tables)
{
    if (!name || !tables || num_tables == 0) return 0;
    std::fprintf(stderr, "[rocprofv3-min] rocprofiler_set_api_table: name=%s tables=%llu\n",
                 name, (unsigned long long)num_tables);

    if (std::strcmp(name, "hip") == 0) {
        if (rocprofv3_min::g_hip_trace_enabled) {
            rocprofv3_min::install_hip_wrappers(static_cast<HipDispatchTable*>(tables[0]));
        }
    } else if (std::strcmp(name, "hip_compiler") == 0) {
        if (rocprofv3_min::g_hip_trace_enabled) {
            rocprofv3_min::install_hip_compiler_wrappers(
                static_cast<HipCompilerDispatchTable*>(tables[0]));
        }
    } else if (std::strcmp(name, "hsa") == 0) {
        if (rocprofv3_min::g_kernel_trace_enabled || rocprofv3_min::g_memory_copy_trace_enabled) {
            rocprofv3_min::install_hsa_wrappers(static_cast<HsaApiTable*>(tables[0]));
        }
    } else if (std::strcmp(name, "hip_tools") == 0) {
        // No tools-table interception in this minimal build.
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
