/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SYMMETRIC_BUFFER_HELPERS_HPP
#define SYMMETRIC_BUFFER_HELPERS_HPP

#include "nccl.h"
#include <cstddef>

/**
 * @file SymmetricBufferHelpers.hpp
 * @brief RAII helpers for ncclMemAlloc + ncclCommWindowRegister lifecycle.
 *
 * Provides reusable utilities for allocating cuMem VMM-backed buffers and
 * registering them as NCCL symmetric windows.  This combination is required
 * for RCCL features that dispatch through the CE (Copy-Engine) collective
 * path: ceCollTaskAppend in enqueue.cc only triggers CE when
 * ncclDevrFindWindow returns non-NULL sendWin and recvWin, which requires
 * the buffers to be registered with NCCL_WIN_COLL_SYMMETRIC.
 *
 * Allocation and registration pattern (mirrors rccl-tests/src/common.cu):
 *   1. ncclMemAlloc(&ptr, bytes)
 *        – allocates cuMem/VMM-backed device memory (requires ncclCuMemEnable() == 1)
 *   2. ncclCommWindowRegister(comm, ptr, bytes, &win, NCCL_WIN_COLL_SYMMETRIC)
 *        – registers the buffer in the symmetric window table so that
 *          ncclDevrFindWindow can locate it during collective dispatch
 *   3. collective call (CE dispatch proceeds with non-NULL sendWin/recvWin)
 *   4. ncclCommWindowDeregister(comm, win) + ncclMemFree(ptr) on cleanup
 *
 * Prerequisites:
 *   - ncclCuMemEnable() must return 1 (ROCm >= 7.12 or 7.0.2.x with the
 *     CUMEM_DRIVER_CAPABLE backport; controlled by NCCL_CUMEM_ENABLE env var)
 *   - comm->symmetricSupport must be true (hardware supports VMM RDMA)
 *
 * If either prerequisite is absent, ncclMemAlloc falls back to hipMalloc and
 * ncclCommWindowRegister fails (cuMemGetAddressRange only works on VMM memory).
 * Callers should treat a non-ncclSuccess return from ncclSymBufAlloc as a
 * test-skip condition rather than a hard failure.
 *
 * NOTE: All functions and types live in namespace RCCLTestHelpers to match
 *       DeviceBufferHelpers.hpp and HostBufferHelpers.hpp conventions.
 */

namespace RCCLTestHelpers
{

// ============================================================================
// SymBuf – RAII owner of one ncclMemAlloc + ncclCommWindowRegister pair
// ============================================================================

/**
 * @brief RAII wrapper that owns a cuMem-allocated device buffer registered as
 *        a NCCL symmetric window.
 *
 * Lifetime rules:
 *   - Construct with default constructor; populate via ncclSymBufAlloc().
 *   - Destructor calls ncclCommWindowDeregister then ncclMemFree automatically.
 *   - Non-copyable; move-constructible for container storage.
 *
 * Usage:
 * @code
 * RCCLTestHelpers::SymBuf sendBuf;
 * ASSERT_EQ(ncclSuccess, RCCLTestHelpers::ncclSymBufAlloc(comm, bytes, sendBuf));
 * // use sendBuf.ptr as the device pointer ...
 * // sendBuf auto-destructs at end of scope
 * @endcode
 */
struct SymBuf
{
    void*        ptr  = nullptr; ///< cuMem-allocated device pointer
    ncclWindow_t win  = nullptr; ///< symmetric window handle
    ncclComm_t   comm = nullptr; ///< owning communicator (not ref-counted)

    SymBuf()                         = default;
    SymBuf(const SymBuf&)            = delete;
    SymBuf& operator=(const SymBuf&) = delete;

    SymBuf(SymBuf&& o) noexcept : ptr(o.ptr), win(o.win), comm(o.comm)
    {
        o.ptr = nullptr;
        o.win = nullptr;
        o.comm = nullptr;
    }

    ~SymBuf()
    {
        release();
    }

    /**
     * @brief Explicitly release the window and buffer before end of scope.
     *        Safe to call multiple times (idempotent after first call).
     */
    void release()
    {
        if(win && comm)
        {
            ncclCommWindowDeregister(comm, win);
            win = nullptr;
        }
        if(ptr)
        {
            ncclMemFree(ptr);
            ptr = nullptr;
        }
        comm = nullptr;
    }

    /// @return true if the buffer has been successfully allocated and registered.
    bool valid() const { return ptr != nullptr && win != nullptr; }
};

// ============================================================================
// ncclSymBufAlloc – allocate + register in one call
// ============================================================================

/**
 * @brief Allocate a cuMem VMM buffer and register it as a symmetric window.
 *
 * Combines ncclMemAlloc + ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC) into
 * a single call.  On any failure the SymBuf is left in a clean empty state and
 * any partially acquired resources are released before returning.
 *
 * Returns ncclSuccess on success.  Typical non-success results:
 *   - ncclInternalError / ncclSystemError if cuMem is not enabled or VMM RDMA
 *     is unsupported on this hardware — treat as a test-skip condition.
 *   - ncclInvalidArgument if bytes == 0 or comm is null.
 *
 * @param comm   NCCL communicator that owns the window registration.
 * @param bytes  Allocation size in bytes.
 * @param[out] sb SymBuf to populate.  Must be in the default-constructed state.
 * @return ncclResult_t
 */
inline ncclResult_t ncclSymBufAlloc(ncclComm_t comm, size_t bytes, SymBuf& sb)
{
    sb.comm = comm;

    ncclResult_t ret = ncclMemAlloc(&sb.ptr, bytes);
    if(ret != ncclSuccess)
    {
        sb.ptr  = nullptr;
        sb.comm = nullptr;
        return ret;
    }

    ret = ncclCommWindowRegister(comm, sb.ptr, bytes, &sb.win, NCCL_WIN_COLL_SYMMETRIC);
    if(ret != ncclSuccess)
    {
        ncclMemFree(sb.ptr);
        sb.ptr  = nullptr;
        sb.win  = nullptr;
        sb.comm = nullptr;
    }
    return ret;
}

} // namespace RCCLTestHelpers

#endif // SYMMETRIC_BUFFER_HELPERS_HPP
