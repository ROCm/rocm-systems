/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_INCLUDE_ROCSHMEM_API_TRACE_H
#define LIBRARY_INCLUDE_ROCSHMEM_API_TRACE_H

#include <stddef.h>
#include <stdint.h>

#include <hip/hip_runtime_api.h>

// rocshmem_team_t mirrors the definition in rocshmem_common.hpp's namespace.
typedef uint64_t *rocshmem_team_t;

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

#endif  // LIBRARY_INCLUDE_ROCSHMEM_API_TRACE_H
