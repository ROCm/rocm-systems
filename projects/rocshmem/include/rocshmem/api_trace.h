#pragma once

#include <stddef.h>
#include <stdint.h>

// Ensure HIP platform is defined
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime_api.h>
#include "rocshmem/rocshmem_common.hpp"
using rocshmem_team_t = rocshmem::rocshmem_team_t;

#ifdef __cplusplus
extern "C" {
#endif

// should only be increased if fundamental changes to dispatch table(s)
#define ROCSHMEM_API_TRACE_VERSION_MAJOR 0

// should be increased every time new members are added to existing dispatch tables
#define ROCSHMEM_API_TRACE_VERSION_PATCH 7

// =============================================================================
// HOST STREAM OPERATIONS (HOST_DISPATCH)
// =============================================================================
typedef void (*barrier_all_on_stream_fn_t)(hipStream_t stream);

typedef void (*quiet_on_stream_fn_t)(hipStream_t);

typedef void (*sync_all_on_stream_fn_t)(hipStream_t);

typedef void (*alltoallmem_on_stream_fn_t)(rocshmem_team_t team, void *dest,
                                           const void *source, size_t size,
                                           hipStream_t stream);

typedef void (*broadcastmem_on_stream_fn_t)(rocshmem_team_t team, void *dest,
                                            const void *source, size_t nelems,
                                            int pe_root, hipStream_t stream);

typedef void (*getmem_on_stream_fn_t)(void *dest, const void *source, size_t nelems,
                                      int pe, hipStream_t stream);

typedef void (*putmem_on_stream_fn_t)(void *dest, const void *source, size_t nelems,
                                      int pe, hipStream_t stream);

typedef void (*putmem_signal_on_stream_fn_t)(void *dest, const void *source,
                                             size_t nelems, uint64_t *sig_addr,
                                             uint64_t signal, int sig_op, int pe,
                                             hipStream_t stream);

typedef void (*signal_wait_until_on_stream_fn_t)(uint64_t *sig_addr, int cmp,
                                                 uint64_t cmp_value,
                                                 hipStream_t stream);

typedef struct rocshmemApiFuncTable
{
    // DO NOT REORDER - ADD NEW FUNCTIONS AT BOTTOM ONLY
    uint64_t                                  size;

    // Host Stream Operations (9 functions)
    barrier_all_on_stream_fn_t                barrier_all_on_stream_fn;
    quiet_on_stream_fn_t                      quiet_on_stream_fn;
    sync_all_on_stream_fn_t                   sync_all_on_stream_fn;
    alltoallmem_on_stream_fn_t                alltoallmem_on_stream_fn;
    broadcastmem_on_stream_fn_t               broadcastmem_on_stream_fn;
    getmem_on_stream_fn_t                     getmem_on_stream_fn;
    putmem_on_stream_fn_t                     putmem_on_stream_fn;
    putmem_signal_on_stream_fn_t              putmem_signal_on_stream_fn;
    signal_wait_until_on_stream_fn_t          signal_wait_until_on_stream_fn;

} rocshmemApiFuncTable;

// Function table accessor
const rocshmemApiFuncTable* RocshmemGetFunctionTable();

#ifdef __cplusplus
}
#endif
