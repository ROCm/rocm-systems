/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2014-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/
/**
 ***********************************************************************************************************************
 * @file  palMemTracker.h
 * @brief PAL utility collection MemTracker class declaration.
 ***********************************************************************************************************************
 */

#pragma once

#if PAL_MEMTRACK

#include "palIntrusiveList.h"
#include "palMutex.h"

namespace Util
{

// Forward declarations
struct AllocInfo;
struct FreeInfo;
struct MemTrackerElem;
enum   SystemAllocType : uint32;

/// @internal
///
/// An alloc-less list used by the MemTracker to keep track of all allocations.
typedef IntrusiveList<MemTrackerElem> MemTrackerList;

/// @internal
///
/// Specifies whether a particular memory block was allocated with PAL_MALLOC/PAL_CALLOC, PAL_NEW, or PAL_NEW_ARRAY.
/// Used to verify correct matching with PAL_FREE, PAL_DELETE, and PAL_DELETE_ARRAY.
enum class MemBlkType : uint32
{
    Malloc = 0,
    New,
    NewArray,
};

/// @internal
///
/// Internal structure used by MemTracker to store information on each allocation.
struct MemTrackerElem
{
    size_t          size;       ///< Size of allocation request.
    MemBlkType      blockType;  ///< Memory block type (malloc, new, new array).
    const char*     pFilename;  ///< File that requested allocation.
    uint32          lineNumber; ///< Line number that requested allocation.
    void*           pClientMem; ///< Starting "client usable" data address.
    void*           pOrigMem;   ///< Original address of the allocation returned from our underlying allocator.
    size_t          allocNum;   ///< The number of the memory allocation. 1 based.
    MemTrackerList* pList;      ///< The list this struct is in. It helps check which MemTracker owns this struct.
};

/**
 ***********************************************************************************************************************
 * @brief Class responsible for tracking allocations and frees to notify the developer of memory leaks.
 *
 * Tracking is enabled/disabled via the PAL_MEMTRACK define.
 ***********************************************************************************************************************
 */
template <typename Allocator>
class MemTracker
{
public:
    /// Constructor.
    ///
    /// @param [in] pAllocator The allocator that will allocate memory if required.
    MemTracker(Allocator*const pAllocator);
    ~MemTracker();

    /// Performs any non-safe initialization that cannot be done in the constructor.
    ///
    /// @returns Result::Success if initialization is successful, otherwise an appropriate error.
    Result Init();

    /// Allocates a block of memory and track it using the memory tracker.
    ///
    /// @param [in] allocInfo Contains information about the requested allocation.
    ///
    /// @returns Pointer to the allocated memory, nullptr if the allocation failed.
    void* Alloc(
        const AllocInfo& allocInfo);

    /// Frees a block of memory.
    ///
    /// @param [in] freeInfo Contains information about the requested free.
    void Free(
        const FreeInfo& freeInfo);

private:
    void* AddMemElement(
        void*       pMem,
        size_t      bytes,
        size_t      align,
        MemBlkType  blockType,
        const char* pFilename,
        uint32      lineNumber);

    void* RemoveMemElement(void* pMem, MemBlkType blockType);

    void MemoryReport();
    void FreeLeakedMemory();

    // Sentinel patterns used to detect memory underrun.
    static constexpr uint32 UnderrunSentinel = 0xDEADBEEF;
    // Sentinel patterns used to detect memory overrun.
    static constexpr uint32 OverrunSentinel  = 0xCAFEBABE;

    // Size of markers for underruns/overruns.  Setting this to 0 disables this feature.
    static constexpr size_t MarkerSizeUints = PAL_CACHE_LINE_BYTES / sizeof(uint32);

    // Size of underrun/overrun markers in bytes.
    static constexpr size_t MarkerSizeBytes = MarkerSizeUints * sizeof(uint32);

    MemTrackerList     m_trackerList;      // The list of active allocations.
    Mutex              m_mutex;            // Serializes access to list of active allocations.

    const size_t       m_markerSizeUints;  // Member variable copy of MarkerSizeUints.  Only used to prevent compiler
                                           //  warnings when MarkerSizeUints is 0.
    const size_t       m_markerSizeBytes;  // Member variable copy of MarkerSizeBytes.  Only used to prevent compiler
                                           //  warnings when MarkerSizeBytes is 0.

    Allocator*const    m_pAllocator;       // Allocator for performing the actual allocations.

    size_t             m_nextAllocNum;     // The allocation number that the next allocated block will receive.
    const size_t       m_breakOnAllocNum;  // The allocation number to trigger a debug break on.

    PAL_DISALLOW_COPY_AND_ASSIGN(MemTracker);
};

} // Util

#endif
