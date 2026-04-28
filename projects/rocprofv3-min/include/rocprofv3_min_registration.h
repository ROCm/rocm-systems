// Minimal subset of rocprofiler-sdk registration ABI sufficient for
// rocprofiler-register.dll to enumerate this library and call into it.
// Lifted (kept binary-compatible) from rocprofiler-sdk/registration.h.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocprofv3_min_client_id_t
{
    size_t         size;
    const char*    name;
    const uint32_t handle;
} rocprofv3_min_client_id_t;

typedef void (*rocprofv3_min_client_finalize_t)(rocprofv3_min_client_id_t);

typedef int (*rocprofv3_min_tool_initialize_t)(rocprofv3_min_client_finalize_t finalize_func,
                                               void*                            tool_data);

typedef void (*rocprofv3_min_tool_finalize_t)(void* tool_data);

typedef struct rocprofv3_min_tool_configure_result_t
{
    size_t                          size;
    rocprofv3_min_tool_initialize_t initialize;
    rocprofv3_min_tool_finalize_t   finalize;
    void*                           tool_data;
} rocprofv3_min_tool_configure_result_t;

#ifdef __cplusplus
}
#endif
