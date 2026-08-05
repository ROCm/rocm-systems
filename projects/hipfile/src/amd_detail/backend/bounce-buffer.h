/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>

namespace hipFile {

/// @internal
/// @brief Reserve the calling thread's bounce buffer
///
/// Synchronous fallback IO stages data through a host buffer. The buffer is
/// pinned (page-locked) with hipHostMalloc so that the HIP runtime can DMA
/// directly to and from it, instead of staging pageable memory through its own
/// internal pinned buffers.
///
/// Pinning memory is expensive relative to an IO, so the buffer is cached in
/// thread local storage and reused by subsequent operations on the same
/// thread. The buffer only ever grows, and is released when the thread exits or
/// when releaseThreadBounceBuffer() is called.
///
/// @param size Minimum size of the buffer in bytes
///
/// @return A pointer to at least @p size bytes of host memory, owned by the
///         calling thread. The contents are undefined.
///
/// @throws Hip::RuntimeError If the buffer could not be pinned
void *getThreadBounceBuffer(size_t size);

/// @internal
/// @brief Release the calling thread's bounce buffer, if it has one
///
/// The buffer is released when the thread exits, so calling this is only
/// necessary to return the memory early. Must not be called while the buffer is
/// in use.
void releaseThreadBounceBuffer() noexcept;

/// @internal
/// @brief Get the size of the calling thread's bounce buffer
/// @return The size of the buffer in bytes, or 0 if the thread has no buffer
size_t getThreadBounceBufferSize() noexcept;

}
