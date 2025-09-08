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
 * @file  palMemTrackerImpl.h
 * @brief PAL utility collection MemTracker class implementations.
 ***********************************************************************************************************************
 */

#pragma once

#if PAL_MEMTRACK

#include "palIntrusiveListImpl.h"
#include "palMemTracker.h"
#include "palSysMemory.h"

#include <cstring>

namespace Util
{

/// Table to convert a blockType to a string. Used by the logging routines.
constexpr const char* MemBlkTypeStr[] =
{
    "Malloc",       ///< MemBlkType::Malloc
    "New",          ///< MemBlkType::New
    "NewArray",     ///< MemBlkType::NewArray
};

// =====================================================================================================================
template <typename Allocator>
MemTracker<Allocator>::MemTracker(
    Allocator*const pAllocator)
    :
    m_markerSizeUints(MarkerSizeUints),
    m_markerSizeBytes(MarkerSizeBytes),
    m_pAllocator(pAllocator),
    m_nextAllocNum(1),
    m_breakOnAllocNum(0)
{
}

// =====================================================================================================================
template <typename Allocator>
MemTracker<Allocator>::~MemTracker()
{
    // Clean-up leaked memory if needed
    if (m_trackerList.IsEmpty() == false)
    {
        // If the list isn't empty, we have a leak.  The leak could either be caused by an internal PAL leak,
        // a client leak, or even the application not destroying API objects.
        PAL_ALERT_ALWAYS();

        // Dump out a list of unfreed blocks.
        MemoryReport();

        FreeLeakedMemory();
    }
}

// =====================================================================================================================
template <typename Allocator>
Result MemTracker<Allocator>::Init()
{
    return Result::Success;
}

// =====================================================================================================================
// Adds the newly allocated memory block to the list of blocks for tracking.
//
// The tracking information includes things like filename, line numbers, and type of block.  Also, given a pointer,
// adds the Underrun/Overrun markers to the memory allocated, and return a pointer to the actual client usable memory.
//
// See MemTracker::Alloc() which is used to allocate memory that is being tracked.
template <typename Allocator>
void* MemTracker<Allocator>::AddMemElement(
    void*       pMem,        // [in,out] Original pointer allocated by MemTracker::Alloc.
    size_t      bytes,       // Client requested allocation size in bytes.
    size_t      align,       // The max of the client-requested alignment or the internal alignment, in bytes.
    MemBlkType  blockType,   // Block type based on calling allocation routine.
    const char* pFilename,   // Client filename that is requesting the memory.
    uint32      lineNumber)  // Line number in client file that is requesting the memory.
{
    // Our internal data is all relative to the client pointer so find that first. See Alloc for more details.
    //   (align1)(MemTrackerList::Node)(MemTrackerElem)(underflow tracker)(client allocation)(align2)(overflow tracker)
    constexpr size_t InternalSize = sizeof(MemTrackerList::Node) + sizeof(MemTrackerElem);

    void*const pClientMem = VoidPtrAlign(VoidPtrInc(pMem, m_markerSizeBytes + InternalSize), align);
    uint32*    pUnderrun  = static_cast<uint32*>(VoidPtrDec(pClientMem, m_markerSizeBytes));
    uint32*    pOverrun   = static_cast<uint32*>(VoidPtrInc(pClientMem, Pow2Align(bytes, sizeof(uint32))));

    auto*const pNewElement = static_cast<MemTrackerElem*>(VoidPtrDec(pUnderrun, sizeof(MemTrackerElem)));
    void*const pNewNodeMem = VoidPtrDec(pNewElement, sizeof(MemTrackerList::Node));
    auto*const pNewNode    = PAL_PLACEMENT_NEW(pNewNodeMem) MemTrackerList::Node(pNewElement);

    // Mark the memory with the underrun/overrun marker.
    for (uint32 markerUints = 0; markerUints < m_markerSizeUints; ++markerUints)
    {
        *pUnderrun++ = UnderrunSentinel;
        *pOverrun++  = OverrunSentinel;
    }

    pNewElement->size       = bytes;
    pNewElement->pFilename  = pFilename;
    pNewElement->lineNumber = lineNumber;
    pNewElement->blockType  = blockType;
    pNewElement->pClientMem = pClientMem;
    pNewElement->pOrigMem   = pMem;
    pNewElement->pList      = &m_trackerList;

    MutexAuto lock(&m_mutex);

    // Trigger an assert if we're about to allocate the break-on-allocation number.
    if (m_nextAllocNum == m_breakOnAllocNum)
    {
        PAL_ASSERT_ALWAYS();
    }

    pNewElement->allocNum = m_nextAllocNum;
    ++m_nextAllocNum;

    m_trackerList.PushFront(pNewNode);

    return pClientMem;
}

// =====================================================================================================================
// Removes an allocated block from the list of blocks used for tracking.
//
// The routine checks for invalid frees (and duplicate frees). Also, the routine is able to detect mismatched alloc/free
// usage based on the blockType.  The routine is called with the pointer to the client usable memory and returns the
// pointer to the allocated memory.
//
// See MemTracker::Free() which is used to free memory that is being tracked.
template <typename Allocator>
void* MemTracker<Allocator>::RemoveMemElement(
    void*       pClientMem,  // Pointer to client usable memory.
    MemBlkType  blockType)   // Block type based on calling deallocation routine.
{
    void* pOrigPtr = nullptr;

    // Recall that this is our internal memory layout. See Alloc for more details.
    //   (align1)(MemTrackerList::Node)(MemTrackerElem)(underflow tracker)(client allocation)(align2)(overflow tracker)
    uint32*    pUnderrun    = static_cast<uint32*>(VoidPtrDec(pClientMem, m_markerSizeBytes));
    auto*const pCurrent     = static_cast<MemTrackerElem*>(VoidPtrDec(pUnderrun, sizeof(MemTrackerElem)));
    auto*const pCurrentNode = static_cast<MemTrackerList::Node*>(VoidPtrDec(pCurrent, sizeof(MemTrackerList::Node)));
    uint32*    pOverrun     = static_cast<uint32*>(VoidPtrInc(pClientMem, Pow2Align(pCurrent->size, sizeof(uint32))));

    // We should not be trying to free something twice or trying to free something which has not been allocated
    // by this MemTracker. We can verify both of these things by checking that the tracker's pList is equal to the
    // MemTracker's list.
    if (pCurrent->pList != &m_trackerList)
    {
        // A free was attempted on an unrecognized pointer.
        PAL_DPERROR("Invalid Free Attempted with ptr = : (%#x)", pClientMem);
    }
    else if (pCurrent->blockType != blockType)
    {
        // We have a mismatch in the alloc/free pair, e.g. PAL_NEW with PAL_FREE etc.  return early here without freeing
        // the memory so it shows up as a leak.
        PAL_DPERROR("Trying to Free %s as %s.",
                    MemBlkTypeStr[static_cast<uint32>(pCurrent->blockType)],
                    MemBlkTypeStr[static_cast<uint32>(blockType)]);
    }
    else
    {
        // We should check for memory corruption due to overflow or underflow before continuing because any underflow
        // might indicate that our internal state is corrupted. This could lead to a crash in the code below.
        for (uint32 markerUints = 0; markerUints < m_markerSizeUints; ++markerUints)
        {
            PAL_ASSERT(*pUnderrun++ == UnderrunSentinel);
            PAL_ASSERT(*pOverrun++  == OverrunSentinel);
        }

        // Remove our tracker from the list and set it's pList to null to detect a double-free in the future.
        MutexAuto lock(&m_mutex);

        m_trackerList.Erase(pCurrentNode);

        pCurrent->pList = nullptr;
        pOrigPtr        = pCurrent->pOrigMem;
    }

    // Return a pointer to the actual allocated block.
    return pOrigPtr;
}

// =====================================================================================================================
// Allocates a block of memory and tracks it using the memory tracker.
template <typename Allocator>
void* MemTracker<Allocator>::Alloc(
    const AllocInfo& allocInfo)
{
    // Allocating zero bytes of memory results in undefined behavior.
    PAL_ASSERT(allocInfo.bytes > 0);

    void* pMem = nullptr;

    // We want to allocate extra memory from the caller's allocator, in this layout:
    //   (align1)(MemTrackerList::Node)(MemTrackerElem)(underflow tracker)(client allocation)(align2)(overflow tracker)
    // Here's why we need each of those sections:
    //   1. align1 is zero or more bytes needed to align the client allocation and our internal data.
    //   2. The MemTrackerList::Node object, which is used to link this allocation into m_trackerList.
    //   3. The MemTrackerElem struct contains bookkeeping data we need to report memory errors.
    //   4. The underflow and overflow trackers detect out of bounds writes. They are optional.
    //   5. The client allocation, which is actually returned to the caller.
    //   6. align2 is zero or more bytes needed to DWORD-align the overflow tracker.
    constexpr size_t InternalAlignment = Max(alignof(MemTrackerList::Node), alignof(MemTrackerElem));
    const size_t     paddedAlignBytes  = Max(allocInfo.alignment, InternalAlignment);
    const size_t     paddedSizeBytes   = (paddedAlignBytes +                            // 1
                                          sizeof(MemTrackerList::Node) +                // 2
                                          sizeof(MemTrackerElem) +                      // 3
                                          m_markerSizeBytes +                           // 4.a
                                          Pow2Align(allocInfo.bytes, sizeof(uint32)) +  // 5 & 6
                                          m_markerSizeBytes);                           // 4.b

    const AllocInfo memTrackerInfo(paddedSizeBytes, paddedAlignBytes, allocInfo.zeroMem, allocInfo.allocType,
                                   allocInfo.blockType, allocInfo.pFilename, allocInfo.lineNumber);

    pMem = m_pAllocator->Alloc(memTrackerInfo);

    if (pMem != nullptr)
    {
        // Don't bother adding a failed allocation to the Memtrack list.
        pMem = AddMemElement(pMem,
                             allocInfo.bytes,
                             paddedAlignBytes,
                             allocInfo.blockType,
                             allocInfo.pFilename,
                             allocInfo.lineNumber);
    }

    return pMem;
}

// =====================================================================================================================
// Frees a block of memory.  The routine is called with the pointer to the client usable memory.
//
// See MemTracker::RemoveMemElement() which is used to validate the free.
template <typename Allocator>
void MemTracker<Allocator>::Free(
    const FreeInfo& freeInfo)
{
    // Don't want to call RemoveMemElement if the ptr is null.
    if (freeInfo.pClientMem != nullptr)
    {
        void* pMem = RemoveMemElement(freeInfo.pClientMem, freeInfo.blockType);

        // If this free call is valid (RemoveMemElement doesn't return nullptr), release the memory.
        if (pMem != nullptr)
        {
            m_pAllocator->Free(FreeInfo(pMem, freeInfo.blockType));
        }
    }
}

// =====================================================================================================================
// Frees all memory that has not been explicitly freed (in other words, memory that has leaked).  This function is only
// expected to be called when the memory tracker is being destroyed.
template <typename Allocator>
void MemTracker<Allocator>::FreeLeakedMemory()
{
    for (MemTrackerList::Iter iter = m_trackerList.Begin(); iter.IsValid(); )
    {
        MemTrackerElem*const pCurrent = iter.Get();

        // Free will release the memory for tracking and the actual element. This will invalidate our list iterator
        // unless we advance the iterator first.
        iter.Next();

        Free(FreeInfo(pCurrent->pClientMem, pCurrent->blockType));
    }
}

// =====================================================================================================================
// Outputs information about leaked memory by traversing the memory tracker list.
template <typename Allocator>
void MemTracker<Allocator>::MemoryReport()
{
    // When this env var is set to non-zero, don't report leaks.
    // Useful for crashing apps that don't give us a chance to clean up.
    const char* pToggle = getenv("AMDPAL_NO_LEAK_REPORT");

    if ((pToggle == nullptr) || (atoi(pToggle) == 0))
    {
        PAL_DPWARN("================ List of Leaked Blocks ================");

        for (MemTrackerList::Iter iter = m_trackerList.Begin(); iter.IsValid(); iter.Next())
        {
            MemTrackerElem*const pCurrent = iter.Get();

            PAL_DPWARN(
                "ClientMem = 0x%p, AllocSize = %8d, MemBlkType = %s, File = %-15s, LineNumber = %8d, AllocNum = %8d",
                pCurrent->pClientMem,
                pCurrent->size,
                MemBlkTypeStr[static_cast<uint32>(pCurrent->blockType)],
                pCurrent->pFilename,
                pCurrent->lineNumber,
                pCurrent->allocNum);
        }

        PAL_DPWARN("================ End of List ===========================");
    }
}

} // Util

#endif
