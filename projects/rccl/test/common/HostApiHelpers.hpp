/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file HostApiHelpers.hpp
 * @brief Helper utilities for testing RCCL's one-sided RMA (Host API)
 *
 * Provides:
 * - allocFineGrainBuffer / freeFineGrainBuffer via ncclMemAlloc / ncclMemFree
 * - NcclWindowGuard: RAII wrapper for ncclCommWindowRegister / ncclCommWindowDeregister
 * - fillPatternBytes / verifyPatternBytes for byte-level test patterns on CPU-accessible
 *   (fine-grain) memory
 *
 */

#ifndef HOST_API_HELPERS_HPP
#define HOST_API_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED

#include "rccl/rccl.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace RCCLHostApiHelpers
{

// ============================================================================
// Fine-grain buffer allocation
// ============================================================================

/**
 * @brief Allocate fine-grain (CPU-accessible) memory via ncclMemAlloc.
 *
 * On ROCm the returned pointer is host-accessible without hipMemcpy.
 *
 * @param ptr   Out: receives allocated pointer.
 * @param size  Requested size in bytes.
 * @return ncclResult_t from ncclMemAlloc.
 */
inline ncclResult_t allocFineGrainBuffer(void** ptr, size_t size)
{
    return ncclMemAlloc(ptr, size);
}

/**
 * @brief Free fine-grain memory previously allocated with allocFineGrainBuffer.
 *
 * @param ptr Pointer to free (no-op if nullptr).
 * @return ncclResult_t from ncclMemFree, or ncclSuccess if ptr was nullptr.
 */
inline ncclResult_t freeFineGrainBuffer(void* ptr)
{
    if(!ptr)
        return ncclSuccess;
    return ncclMemFree(ptr);
}

// ============================================================================
// RAII guard for ncclWindow_t
// ============================================================================

/**
 * @struct NcclWindowGuard
 * @brief RAII wrapper for ncclCommWindowRegister / ncclCommWindowDeregister.
 *
 * Constructor calls ncclCommWindowRegister. Destructor calls
 * ncclCommWindowDeregister only if win_ is non-null.
 *
 * Usage:
 * @code
 *   void* buf = nullptr;
 *   allocFineGrainBuffer(&buf, kSize);
 *   ncclWindow_t win = nullptr;
 *   NcclWindowGuard wg(comm, buf, kSize, &win, NCCL_WIN_DEFAULT);
 *   if (win == nullptr) { GTEST_SKIP() << "Windows not supported"; }
 *   // ... use win ...
 *   // Destructor calls ncclCommWindowDeregister automatically.
 * @endcode
 */
struct NcclWindowGuard
{
    ncclComm_t   comm_  = nullptr;
    ncclWindow_t win_   = nullptr;
    ncclResult_t initResult_ = ncclSuccess;

    NcclWindowGuard() = default;

    /**
     * @brief Register a memory window.
     *
     * @param comm     RCCL communicator (collective across all ranks).
     * @param buff     Buffer to register (fine-grain, CPU-accessible).
     * @param size     Size of the buffer in bytes.
     * @param winOut   Out: receives the window handle.
     * @param winFlags Window flags (use NCCL_WIN_DEFAULT = 0x00 for proxy GIN path).
     */
    NcclWindowGuard(ncclComm_t    comm,
                    void*         buff,
                    size_t        size,
                    ncclWindow_t* winOut,
                    int           winFlags)
        : comm_(comm)
    {
        initResult_ = ncclCommWindowRegister(comm_, buff, size, &win_, winFlags);
        if(winOut)
            *winOut = win_;
    }

    ~NcclWindowGuard()
    {
        if(win_ && comm_)
        {
            ncclResult_t res = ncclCommWindowDeregister(comm_, win_);
            if(res != ncclSuccess)
            {
                fprintf(stderr,
                        "WARNING: ncclCommWindowDeregister failed in destructor: %s\n",
                        ncclGetErrorString(res));
            }
            win_ = nullptr;
        }
    }

    // Non-copyable, non-movable (owns the handle)
    NcclWindowGuard(const NcclWindowGuard&)            = delete;
    NcclWindowGuard& operator=(const NcclWindowGuard&) = delete;
    NcclWindowGuard(NcclWindowGuard&&)                 = delete;
    NcclWindowGuard& operator=(NcclWindowGuard&&)      = delete;

    ncclResult_t initResult() const { return initResult_; }
    ncclWindow_t win()        const { return win_; }
    bool         valid()      const { return win_ != nullptr; }
};

// ============================================================================
// Pattern fill / verify (hipMemcpy staging — works on device and fine-grain)
// ============================================================================

/** Fill buf with pattern (seed + i) % 256 via hipMemcpy staging. */
inline void FillBuf(void* buf, size_t size, int seed)
{
    std::vector<uint8_t> tmp(size);
    for(size_t i = 0; i < size; ++i)
        tmp[i] = static_cast<uint8_t>((seed + i) % 256);
    ASSERT_EQ(hipMemcpy(buf, tmp.data(), size, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

/** Return true iff buf matches (seed + i) % 256 for all i. */
inline bool VerifyBuf(const void* buf, size_t size, int seed)
{
    std::vector<uint8_t> staging(size);
    if(hipMemcpy(staging.data(), buf, size, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipDeviceSynchronize() != hipSuccess)
        return false;
    for(size_t i = 0; i < size; ++i)
    {
        if(staging[i] != static_cast<uint8_t>((seed + i) % 256))
        {
            fprintf(stderr,
                    "VerifyBuf: mismatch at byte %zu: expected %u got %u (seed=%d)\n",
                    i,
                    static_cast<unsigned>((seed + i) % 256),
                    static_cast<unsigned>(staging[i]),
                    seed);
            return false;
        }
    }
    return true;
}

/** Fill buf with a constant sentinel byte via hipMemset. */
inline void FillSentinel(void* buf, size_t size, uint8_t value)
{
    ASSERT_EQ(hipMemset(buf, value, size), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

/** Return true iff every byte in buf equals value. */
inline bool AllSentinel(const void* buf, size_t size, uint8_t value)
{
    std::vector<uint8_t> staging(size);
    if(hipMemcpy(staging.data(), buf, size, hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    if (hipDeviceSynchronize() != hipSuccess)
        return false;
    for(size_t i = 0; i < size; ++i)
    {
        if(staging[i] != value)
        {
            fprintf(stderr,
                    "AllSentinel: mismatch at byte %zu: expected 0x%02x got 0x%02x\n",
                    i,
                    static_cast<unsigned>(value),
                    static_cast<unsigned>(staging[i]));
            return false;
        }
    }
    return true;
}

} // namespace RCCLHostApiHelpers

#endif // MPI_TESTS_ENABLED

#endif // HOST_API_HELPERS_HPP
