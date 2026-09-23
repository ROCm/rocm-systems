// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#define TORCH_TRACE_COLLECTOR_ABI_REVISION 2U

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Returns the plain-C interface revision expected by the Python loader. This
 * revision covers the exported function contract; install() separately gates
 * private PyTorch layouts using the loaded libtorch_cpu and libc10 build
 * identities.
 * @return TORCH_TRACE_COLLECTOR_ABI_REVISION.
 */
__attribute__((visibility("default"))) uint32_t torch_trace_collector_abi_revision(void);

/**
 * Installs one process-global PyTorch RecordFunction callback.
 *
 * This operation is thread-safe and idempotent. It verifies that the loaded
 * PyTorch runtime is explicitly compatible before registering the callback.
 * The caller must keep the collector loaded for the rest of the process.
 *
 * @return Zero on success and non-zero on failure.
 */
__attribute__((visibility("default"))) int torch_trace_collector_install(void);

/**
 * Publishes the current Python launcher thread ID to PyTorch autograd workers.
 * Calls may be nested, but are accepted only after install succeeds. Every
 * successful push must be paired with a pop call on the same thread.
 * @param launcher_tid Native operating-system thread ID of the caller.
 * @return Zero on success and non-zero on failure.
 */
__attribute__((visibility("default"))) int torch_trace_collector_push_launcher_tid(uint64_t launcher_tid);

/**
 * Removes the most recently published launcher thread ID on this thread.
 * @return Zero on success and non-zero when no matching value exists or the
 * operation fails.
 */
__attribute__((visibility("default"))) int torch_trace_collector_pop_launcher_tid(void);

#ifdef __cplusplus
}
#endif
