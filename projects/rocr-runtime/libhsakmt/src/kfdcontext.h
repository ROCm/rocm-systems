/*
 * Copyright © 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _KFDCONTEXT_H_
#define _KFDCONTEXT_H_

#include <stdint.h>

struct hsa_kfd_topology_context;
struct hsa_kfd_queue_context;
struct hsa_kfd_fmm_context;
struct hsa_kfd_event_context;
struct hsa_kfd_debug_context;
struct hsa_kfd_perf_context;

/*
 * HsaKFDContext
 *
 * Represents the execution context for a connection to the Kernel Fusion Driver (KFD).
 *
 * This structure encapsulates all state required to manage a KFD session, including:
 *   - The file descriptor associated with the open KFD device
 *   - Related resources tied to this file descriptor
 *
 * Multiple HsaKFDContext instances can coexist simultaneously, each maintaining its own
 * independent set of resources. These contexts are fully isolated from one another and
 * must not have their resources mixed. For example, memory resources created in
 * context A cannot be used in context B directly. If resources need to be shared between
 * contexts, they must be explicitly exported and imported using the appropriate APIs.
 */
typedef struct _HsaKFDContext
{
    /* File descriptor for the KFD device */
    int fd;

    /* Topology context for managing system topology information */
    struct hsa_kfd_topology_context *topology_context;

    /* Queue context for managing user queues */
    struct hsa_kfd_queue_context *queue_context;

    /* Memory management context for managing memory */
    struct hsa_kfd_fmm_context *fmm_context;

    /* Event context for managing events */
    struct hsa_kfd_event_context *event_context;

    /* Debug context for managing debug operations */
    struct hsa_kfd_debug_context *debug_context;

    /* perf context for managing perf operations */
    struct hsa_kfd_perf_context *perf_context;
} HsaKFDContext;

// Initialize a pre-allocated HsaKFDContext with the given file descriptor
void hsakmt_kfdcontext_init_context(int fd, HsaKFDContext *ctx);
// Release all resources associated with the given KFD context
void hsakmt_kfdcontext_clear_context(HsaKFDContext *ctx);

struct hsa_kfd_topology_context *hsakmt_kfdcontext_get_topology_context(HsaKFDContext *ctx);
struct hsa_kfd_fmm_context *hsakmt_kfdcontext_get_fmm_context(HsaKFDContext *ctx);
struct hsa_kfd_queue_context *hsakmt_kfdcontext_get_queue_context(HsaKFDContext *ctx);
struct hsa_kfd_event_context *hsakmt_kfdcontext_get_event_context(HsaKFDContext *ctx);
struct hsa_kfd_debug_context *hsakmt_kfdcontext_get_debug_context(HsaKFDContext *ctx);
struct hsa_kfd_perf_context *hsakmt_kfdcontext_get_perf_context(HsaKFDContext *ctx);
#endif /* _KFDCONTEXT_H_ */
