// Tool configuration library for late-start dynamic loading test
// This library contains the rocprofiler_configure function and tool callbacks
// It gets loaded dynamically AFTER HIP has already initialized

#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/callback_tracing.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

// External counter (defined in main_dynamic.cpp)
extern std::atomic<uint64_t> g_hip_api_calls;

namespace
{
// Tool callback for HIP API tracing
void
hip_api_callback(rocprofiler_callback_tracing_record_t record,
                 rocprofiler_user_data_t*              user_data,
                 void*                                 callback_data)
{
    (void) user_data;
    (void) callback_data;

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        g_hip_api_calls.fetch_add(1, std::memory_order_relaxed);
    }
}

// Note: Kernel dispatch tracing is NOT supported in late-start scenarios
// because it requires queue interception at queue creation time.
// Queues that already exist before profiling starts cannot be intercepted retroactively.

// Tool initialization callback - THIS is where context creation happens!
int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    (void) fini_func;
    (void) tool_data;

    printf("  Tool initialization callback invoked\n");

    // Create a profiling context
    rocprofiler_context_id_t context_id = {.handle = 0};  // MUST initialize to zero!
    rocprofiler_status_t status = rocprofiler_create_context(&context_id);

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "  Failed to create context: %d\n", status);
        return -1;
    }

    printf("  Created profiling context\n");

    // Configure HIP API callback tracing
    status = rocprofiler_configure_callback_tracing_service(
        context_id,
        ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
        nullptr,    // operations (nullptr = all operations)
        0,          // operations count
        hip_api_callback,
        nullptr);   // callback data

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "  Failed to configure HIP API tracing: %d\n", status);
        return -1;
    }

    printf("  Configured HIP API callback tracing\n");

    // NOTE: Kernel dispatch tracing is intentionally NOT configured here
    // Late-start does not support kernel dispatch tracing because it requires
    // queue interception at queue creation time. Existing queues cannot be
    // retroactively intercepted.

    // Start the context
    status = rocprofiler_start_context(context_id);

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "  Failed to start context: %d\n", status);
        return -1;
    }

    printf("  Started profiling context\n");

    return 0;
}

// Tool finalization callback
void
tool_fini(void* tool_data)
{
    (void) tool_data;
    printf("  Tool finalization callback invoked\n");
}

}  // namespace

// rocprofiler_configure function
// This gets called by rocprofiler_force_configure() when we trigger late-start
// It should just return the configuration structure - actual context setup happens in tool_init
extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* client_id)
{
    (void) version;
    (void) runtime_version;
    (void) priority;
    (void) client_id;

    printf("  rocprofiler_configure() called - registering tool callbacks\n");

    // Allocate result structure (SDK will free this)
    auto* result = (rocprofiler_tool_configure_result_t*) malloc(
        sizeof(rocprofiler_tool_configure_result_t));

    result->size       = sizeof(rocprofiler_tool_configure_result_t);
    result->initialize = tool_init;     // Will be called to create context
    result->finalize   = tool_fini;     // Will be called on shutdown
    result->tool_data  = nullptr;

    printf("  Registered initialization and finalization callbacks\n");

    return result;
}
