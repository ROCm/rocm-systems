/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
 * @file  palBuddyAllocator.h
 * @brief PAL utility BuddyAllocator class declaration.
 ***********************************************************************************************************************
 */

#pragma once

#include "palUtil.h"
#include "palHashSet.h"
#include "palHashMap.h"
#include "palMutex.h"

namespace Util
{

/**
 ***********************************************************************************************************************
 * @brief  Buddy Allocator
 *
 * Responsible for managing small GPU memory requests by allocating a large base allocation and dividing it into
 * appropriately sized suballocation blocks.
 ***********************************************************************************************************************
 */
template <typename Allocator>
class BuddyAllocator
{
public:
    /// Constructor.
    ///
    /// @param [in]  pAllocator     The allocator that will allocate memory if required.
    /// @param [in]  baseAllocSize  The size of the base allocation this buddy allocator suballocates.
    /// @param [in]  minAllocSize   The size of the smallest block this buddy allocator can allocate.
    BuddyAllocator(
        Allocator* pAllocator,
        gpusize    baseAllocSize,
        gpusize    minAllocSize);
    ~BuddyAllocator();

    /// Initializes the buddy allocator.
    ///
    /// @returns Success if the buddy allocator has been successfully initialized.
    Result Init();

    /// Suballocates a block from the base allocation that this buddy allocator manages.  Expects @ref ClaimGpuMemory to
    /// be called directly before it.  If a memory manager with multiple buddyAllocators is used, use pattern should
    /// be: Iterate through buddyAllocators calling ClaimGpuMemory, if one returns @ref Success break out of the loop,
    /// then call Allocate on that buddyAllocator.  If none return @ref Success, then a new buddyAllocator needs to be
    /// created.  The purpose of splitting up buddyAllocator selection and Allocation is to reduce lock contention in
    /// multithreaded memory managers.
    ///
    /// @param [in]  size           The size of the requested suballocation.
    /// @param [in]  alignment      The alignment requirements of the requested suballocation.
    /// @param [out] pOffset        The offset the suballocated block starts within the base allocation.
    ///
    /// @returns Success if the allocation succeeded, @ref ErrorOutOfMemory if there isn't enough system memory to
    ///          fulfill the request, or @ref ErrorOutOfGpuMemory if there isn't a large enough block free in the
    ///          base allocation to fulfill the request.
    ///
    /// @warning Unless @ref ClaimGpuMemory is called before every single call, the results of @ref Allocate will
    ///          be invalid.  If @ref ClaimGpuMemory returns @ref Success, then @ref ErrorOutOfGpuMemory will never be
    ///          returned.
    Result Allocate(
        gpusize  size,
        gpusize  alignment,
        gpusize* pOffset);

    /// Frees a previously allocated suballocation.
    ///
    /// @param [in]  offset         The offset the suballocated block starts within the base allocation.
    /// @param [in]  size           Optional parameter specifying the size of the original allocation.
    /// @param [in]  alignment      Optional parameter specifying the alignment of the original allocation.
    void Free(
        gpusize offset,
        gpusize size = 0,
        gpusize alignment = 0);

    /// Tells whether the base allocation is completely free. If the returned value is true then the caller is safe
    /// to deallocate the base allocation.
    bool IsEmpty() const
    {
        return (m_numSuballocations == 0);
    }

    /// Returns the size of the largest allocation that can be suballocated with this buddy allocator.
    gpusize MaximumAllocationSize() const;

    /// Claims (doesn't allocate) some memory, used to quickly determine if a pool of memory has availible memory.
    /// Doesn't affect internal state unless Result::Success is returned
    ///
    /// @param [in]  size           The size of the requested suballocation.
    /// @param [in]  alignment      The alignment requirements of the requested suballocation.
    ///
    /// @returns Success if there is enough memory in this buddyAllocator to allocate the requested size of memory,
    ///          @ref ErrorOutOfGpuMemory if there is not enough memory
    ///
    /// @warning Unless this is called to test availible memory before every call to Allocate, then the results will not
    ///          be valid.
    Result ClaimGpuMemory(
        gpusize size,
        gpusize alignment);

    /// Checks if @ref ClaimGpuMemory can actually claim memory, can be used to find the best fit pool.  This function
    /// does NOT acquire a lock on the structures ClaimGpuMemory uses, and does NOT claim or allocate the memory.
    ///
    /// @param [in]  size           The size of the requested suballocation.
    /// @param [in]  alignment      The alignment requirements of the requested suballocation.
    /// @param [out] pKval          The highest kval that will need to be split will be stored here.
    ///
    /// @returns Success if there is enough memory in this buddyAllocator to allocate the requested size of memory,
    ///          @ref ErrorOutOfGpuMemory if there is not enough memory
    ///
    Result CheckIfOpenMemory(
        gpusize size,
        gpusize alignment,
        uint32* pKval);

private:
    typedef Util::HashSet<gpusize, Allocator, JenkinsHashFunc> FreeSet;
    typedef Util::HashMap<gpusize, uint32, Allocator, JenkinsHashFunc> UsedMap;

    Result GetNextFreeBlock(
        uint32   kval,
        gpusize* pOffset);

    Result FreeBlock(gpusize offset);

    static constexpr gpusize KvalToSize(uint32 kVal) { return (1ull << kVal); }

    static uint32 SizeToKval(gpusize size) { return Log2(size); }

    Allocator* const    m_pAllocator;

    const uint32        m_baseAllocKval;
    const uint32        m_minKval;

    // Array of hashSets of blocks that are free at each level
    FreeSet*            m_pFreeBlockSets;

    // Hashmap of blocks that are used, key=offset, value=level (kval)
    UsedMap*            m_pUsedBlockMap;
    // List of the free memory at each level
    uint32*             m_pNumFreeList;
    // The highest Kval that has at least 1 free block (used in ClaimGpuMemory)
    uint32              m_highestFreeKval;

    uint32              m_numSuballocations;

    // mutex on altering the numFreeList
    Util::Mutex         m_numFreeMutex;
    // mutex on the used block map
    Util::Mutex         m_usedBlockMapMutex;
    // array of mutexes, one for each freeBlockSet
    Util::Mutex*        m_pFreeSetMutexes;
    // mutex on the freeing.  Serialize freeing blocks and don't allow allocating blocks while one is freeing.  Based on
    // testing, applications typically don't try to free and allocate memory at the same time, and almost all of the
    // memory freeing is done at the end of the application.
    Util::RWLock        m_freeLock;

    // Set to true if ClaimGpuMemory is ever called on this buddyAllocator.  This signals to free to not merge blocks
    // if m_pNumFreeList[kval - m_minKval] = 0
    bool                m_usedClaim;

    // HashSet and HashMap utility functions
    Result  InsertToFreeSet(gpusize offset, uint32 kval);
    bool    GetKvalUsed(gpusize offset, uint32* pKval);
    Result  SetKvalUsed(gpusize offset, uint32 kval);
    Result  PopFromFreeSet(gpusize* pOffset, uint32 kval);
    bool    IsOffsetFree(gpusize offset, uint32 kval);
    Result  RemoveOffsetFromFreeSet(gpusize offset, uint32 kval);
    Result  RemoveOffsetFromUsedMap(gpusize offset);

    PAL_DISALLOW_COPY_AND_ASSIGN(BuddyAllocator);
    PAL_DISALLOW_DEFAULT_CTOR(BuddyAllocator);
};

} // Util
