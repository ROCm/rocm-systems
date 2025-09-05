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
 * @file  palBuddyAllocatorImpl.h
 * @brief PAL utility BuddyAllocator class implementation.
 ***********************************************************************************************************************
 */

#pragma once

#include "palBuddyAllocator.h"
#include "palHashMapImpl.h"
#include "palHashSetImpl.h"
#include "palInlineFuncs.h"
#include "palSysMemory.h"

namespace Util
{

// =====================================================================================================================
template <typename Allocator>
BuddyAllocator<Allocator>::BuddyAllocator(
    Allocator* pAllocator,
    gpusize    baseAllocSize,
    gpusize    minAllocSize)
    :
    m_pAllocator(pAllocator),
    m_baseAllocKval(SizeToKval(baseAllocSize)),
    m_minKval(SizeToKval(minAllocSize)),
    m_pFreeBlockSets(nullptr),
    m_pUsedBlockMap(nullptr),
    m_pNumFreeList(nullptr),
    m_numSuballocations(0),
    m_pFreeSetMutexes(nullptr),
    m_usedClaim(false)
{
    // Allocator must be non-null
    PAL_ASSERT(m_pAllocator != nullptr);

    // Base allocation size must be POT
    PAL_ASSERT(KvalToSize(m_baseAllocKval) == baseAllocSize);

    // Minimum allocation size must be POT
    PAL_ASSERT(KvalToSize(m_minKval) == minAllocSize);
}

// =====================================================================================================================
template <typename Allocator>
BuddyAllocator<Allocator>::~BuddyAllocator()
{
    // lock this here to ensure no other thread was doing anything with the buddyAllocator when the destructor is called
    RWLockAuto<RWLock::ReadWrite> freeLock(&m_freeLock);
    if (m_pFreeBlockSets != nullptr)
    {
        const uint32 numKvals = m_baseAllocKval - m_minKval;
        for (uint32 i = 0; i < numKvals; ++i)
        {
            // Call the destructor
            m_pFreeBlockSets[i].~HashSet();
        }

        // Free the block list array
        PAL_SAFE_FREE(m_pFreeBlockSets, m_pAllocator);
    }
    if (m_pUsedBlockMap != nullptr)
    {
        PAL_SAFE_DELETE(m_pUsedBlockMap, m_pAllocator);
    }
    if (m_pNumFreeList != nullptr)
    {
        PAL_SAFE_DELETE_ARRAY(m_pNumFreeList, m_pAllocator);
    }
    if (m_pFreeSetMutexes != nullptr)
    {
        PAL_SAFE_DELETE_ARRAY(m_pFreeSetMutexes, m_pAllocator);
    }
}

// =====================================================================================================================
// Gets maximum allocation size supported by this buddy allocator.
template <typename Allocator>
gpusize BuddyAllocator<Allocator>::MaximumAllocationSize() const
{
    // NOTE: Report one less than our base allocation k-value because there's no sense in suballocating a memory
    // request which is larger than half a chunk
    return KvalToSize(m_baseAllocKval - 1);
}

// =====================================================================================================================
// Initializes the buddy allocator.
template <typename Allocator>
Result BuddyAllocator<Allocator>::Init()
{
    PAL_ASSERT(m_pFreeBlockSets == nullptr);
    PAL_ASSERT(m_pUsedBlockMap == nullptr);
    PAL_ASSERT(m_pNumFreeList == nullptr);
    PAL_ASSERT(m_pFreeSetMutexes == nullptr);

    // start out with success and take it away if something fails.
    Result result = Result::Success;

    const uint32 numKvals = m_baseAllocKval - m_minKval;

    // one hashSet per kval
    m_pFreeBlockSets = static_cast<FreeSet*>(PAL_MALLOC(sizeof(FreeSet) * numKvals,
                                                        m_pAllocator,
                                                        AllocInternal));
    // Initialize the hashSets.
    if (m_pFreeBlockSets != nullptr)
    {
        for (uint32 i = 0; i < numKvals; ++i)
        {
            // max number of entries at a level is: 2^distFromTop
            const uint32 maxEntriesKval = 1 << (m_baseAllocKval - (i + m_minKval));
            // 32 is a suitable max, however its the higher kvals won't even need 32 buckets.
            const uint32 bucketsNeeded = Min(maxEntriesKval / (PAL_CACHE_LINE_BYTES) + 1, 32u);

            PAL_PLACEMENT_NEW(&m_pFreeBlockSets[i]) FreeSet(bucketsNeeded, m_pAllocator);
            result = m_pFreeBlockSets[i].Init();

            // if we failed the Init of the hashSet, delete the ones we did create, and free the array.  This avoids
            // having to keep track of the hashSets we did initialize in the destructor by just destroying it here.
            if (result != Result::Success)
            {
                for (uint32 j = 0; j <= i; j++)
                {
                    m_pFreeBlockSets[j].~HashSet();
                }
                PAL_SAFE_FREE(m_pFreeBlockSets, m_pAllocator);
                break;
            }
        }
    }
    else
    {
        result = Result::ErrorOutOfMemory;
    }

    if (result == Result::Success)
    {
        m_pNumFreeList = static_cast<uint32*>(PAL_NEW_ARRAY(uint32, numKvals, m_pAllocator, AllocInternal));
        if (m_pNumFreeList == nullptr)
        {
            result = Result::ErrorOutOfMemory;
        }
    }

    if (result == Result::Success)
    {
        m_pFreeSetMutexes = static_cast<Mutex*>(PAL_NEW_ARRAY(Mutex, numKvals, m_pAllocator, AllocInternal));
        if (m_pFreeSetMutexes == nullptr)
        {
            result = Result::ErrorOutOfMemory;
        }
    }

    const uint32 maxUsedEntries = 1 << (m_baseAllocKval - m_minKval);
    const uint32 usedBucketsNeeded = maxUsedEntries / (PAL_CACHE_LINE_BYTES * 8) + 1;

    if (result == Result::Success)
    {
        // one hashMap for getting the kval a used block is at
        m_pUsedBlockMap = static_cast<UsedMap*>(PAL_NEW(UsedMap, m_pAllocator, AllocInternal)
                                                (usedBucketsNeeded, m_pAllocator));
        if (m_pUsedBlockMap != nullptr)
        {
            result = m_pUsedBlockMap->Init();
        }
        else
        {
            result = Result::ErrorOutOfMemory;
        }
    }

    // if we successfully allocated all the memory we need, create the first two free blocks.
    if (result == Result::Success)
    {
        memset(m_pNumFreeList, 0, sizeof(uint32) * numKvals);
        // We need to create the first two largest-size blocks and add them to the last block list
        const uint32   blockKval = (m_baseAllocKval - 1);
        const gpusize  blockSize = KvalToSize(blockKval);
        FreeSet* pTopFreeSet = &m_pFreeBlockSets[blockKval - m_minKval];

        // mark both of these as free blocks
        result = pTopFreeSet->Insert(0);
        if (result == Result::Success)
        {
            // even though this will never be reached, to pass the asserts, this needs to be
            // as this kval
            result = m_pUsedBlockMap->Insert(0, blockKval + 1);
        }
        if (result == Result::Success)
        {
            result = pTopFreeSet->Insert(blockSize);
        }
        m_pNumFreeList[blockKval - m_minKval] = 2;
        m_highestFreeKval = blockKval;
    }
    PAL_ALERT(result != Result::Success);
    return result;
}

// =====================================================================================================================
// Suballocates a block from the base allocation that this buddy allocator manages. If no free space is found then an
// appropriate error is returned.
// In order for m_pNumFreeList bookkeeping to be correct, ClaimGpuMemory MUST be called directly before this call to
// Allocate.  The buddyAllocator will still work without this, but the results of ClaimGpuMemory will not be correct.
// unless it is called before every call to Allocate.
template <typename Allocator>
Result BuddyAllocator<Allocator>::Allocate(
    gpusize  size,
    gpusize  alignment,
    gpusize* pOffset)
{
    PAL_ASSERT(m_pFreeBlockSets != nullptr);
    PAL_ASSERT(m_pUsedBlockMap != nullptr);
    PAL_ASSERT(m_pNumFreeList != nullptr);
    PAL_ASSERT(m_pFreeSetMutexes != nullptr);
    PAL_ASSERT(pOffset != nullptr);
    PAL_ASSERT(size <= MaximumAllocationSize());

    // Pad the requested allocation size to the nearest POT of the size and alignment
    const uint32 kval = Max(SizeToKval(Pow2Pad(Max(size, alignment))), m_minKval);

    RWLockAuto<RWLock::ReadOnly> freeLock(&m_freeLock);
    Result result = GetNextFreeBlock(kval, pOffset);
    // mark this kval as used here.
    if (result == Result::Success)
    {
        result = SetKvalUsed(*pOffset, kval);
    }

    if (result == Result::Success)
    {
        // Increment the number of suballocations this buddy allocator manages
        AtomicIncrement(&m_numSuballocations);
    }
    return result;
}

// =====================================================================================================================
// Gets the next free block by recursively dividing larger blocks until a suitible sized block is created.
template <typename Allocator>
Result BuddyAllocator<Allocator>::GetNextFreeBlock(
    uint32   kval,
    gpusize* pOffset)
{
    Result result = Result::ErrorOutOfGpuMemory;
    if (kval < m_baseAllocKval)
    {
        // this lock can not get any more fine grained
        MutexAuto freeSetLock(&(m_pFreeSetMutexes[kval - m_minKval]));
        result = PopFromFreeSet(pOffset, kval);

        if (result == Result::ErrorOutOfGpuMemory)
        {   // we didn't find a block at this kval, search the next level up
            result = GetNextFreeBlock(kval + 1, pOffset);

            if (result == Result::Success)
            {
                // insert our buddy to the free set
                gpusize buddyOffset = *pOffset + KvalToSize(kval);
                result = InsertToFreeSet(buddyOffset, kval);
                PAL_ASSERT(result == Result::Success);
            }
        }
        else
        {
            // only two valid options are ErrorOutOfGpuMemory and Success, other result means the hashing failed.
            PAL_ASSERT(result == Result::Success);
        }
    }
    PAL_ALERT_MSG(result != Result::Success,
                  "This should only fail if ClaimGpuMemory() is not called before this call to Allocate().");
    return result;
}

// =====================================================================================================================
// Frees the memory at the given offset, if it's buddy is also free, merges the two and recursively calls this again.
// This doesn't need any internal locks because Free accquires an exclusive lock on the entire allocator (freeLock), and
// the lock on the m_pNumFreeList.  These locks could potentially be more fine grained, however freeing and allocating
// don't typically happen at the same time, and Freeing is already much faster than allocating.
template <typename Allocator>
Result BuddyAllocator<Allocator>::FreeBlock(
    gpusize offset)
{
    Result result = Result::ErrorUnknown;
    uint32 usedKval;
    bool offsetUsed = GetKvalUsed(offset, &usedKval);
    PAL_ASSERT(offsetUsed);
    PAL_ASSERT(usedKval >= m_minKval && usedKval < m_baseAllocKval);

    gpusize buddyOffset = offset ^ KvalToSize(usedKval);
    gpusize offsetUp = Min(offset, buddyOffset);

    // we don't want merge if we are on the top level.  We also don't want to merge if a call to claim was made that
    // claimed the buddy we are about to free.
    if (IsOffsetFree(buddyOffset, usedKval) && (usedKval < m_baseAllocKval -1) &&
       ((m_pNumFreeList[usedKval - m_minKval] > 0) || (m_usedClaim == false)))
    {   // We can combine the two blocks and mark the one in the level above as free
        // And do this recursively
        result = RemoveOffsetFromFreeSet(buddyOffset, usedKval);
        if (result == Result::Success)
        {
            // even though the block is going to be freed, need to set the kval as used
            // so that on the recursive call it will be found and freed again.
            PAL_ASSERT_MSG((m_pNumFreeList[usedKval - m_minKval] != 0) || (m_usedClaim == false),
                          "This should only fail if ClaimGpuMemory() is not called before this call to Allocate().");
            m_pNumFreeList[usedKval - m_minKval] -= 1;
            result = SetKvalUsed(offsetUp, usedKval + 1);
        }
        // if this offset isn't the one that will be set as free in the next level up, we just need to remove it.
        if ((result == Result::Success) && (offset != offsetUp))
        {
            result = RemoveOffsetFromUsedMap(offset);
        }
        if (result == Result::Success)
        {
            result = FreeBlock(offsetUp);
        }
    }
    else
    {   // We mark this block as free in this level
        result = InsertToFreeSet(offset, usedKval);

        if (result == Result::Success)
        {
            m_pNumFreeList[usedKval - m_minKval] += 1;
            m_highestFreeKval = Util::Max(usedKval, m_highestFreeKval);
            if (offsetUp == offset)
            {   // if on the same offset as level up, move where the used block is
                result = SetKvalUsed(offsetUp, usedKval + 1);
            }
            else
            {   // if at the top of this offset, remove is from used map
                result = RemoveOffsetFromUsedMap(offset);
            }
        }
    }
    return result;
}
// =====================================================================================================================
// Frees a suballocated block making it available for future re-use.
template <typename Allocator>
void BuddyAllocator<Allocator>::Free(
    gpusize offset,
    gpusize size,
    gpusize alignment)
{
    RWLockAuto<RWLock::ReadWrite> freeLock(&m_freeLock);
    MutexAuto numFreeMutex(&m_numFreeMutex);

    PAL_ASSERT(m_pFreeBlockSets != nullptr);
    PAL_ASSERT(m_pUsedBlockMap != nullptr);
    PAL_ASSERT(m_pNumFreeList != nullptr);
    PAL_ASSERT(m_pFreeSetMutexes != nullptr);

    Result result = FreeBlock(offset);

    // Freeing should always succeed unless something went wrong with the allocation scheme
    PAL_ASSERT(result == Result::Success);

    // Decrement the number of suballocations this buddy allocator manages
    AtomicDecrement(&m_numSuballocations);
}

// =====================================================================================================================
// Claims the memory that will be used when Allocate is called.
// Returns ErrorOutOfGpuMemory if this buddyAllocator has no free blocks, otherwise returns Success.
template <typename Allocator>
Result BuddyAllocator<Allocator>::ClaimGpuMemory(
    gpusize size,
    gpusize alignment)
{
    // Set this to true as soon as the first call to claim is done to signal to Free that claim is being used.
    m_usedClaim = true;

    PAL_ASSERT(m_pNumFreeList != nullptr);
    // Pad the requested allocation size to the nearest POT of the size and alignment
    uint32 kval = Max(SizeToKval(Pow2Pad(Max(size, alignment))), m_minKval);
    PAL_ASSERT(kval >= m_minKval && kval < m_baseAllocKval);

    Result result = Result::ErrorOutOfGpuMemory;

    // Do this check twice to avoid taking the lock at all if we have no chance of Claiming the memory.  This will stop
    // this thread from locking on this, as well as other threads from waiting longer for no reason.
    if (kval <= m_highestFreeKval)
    {
        MutexAuto numFreeLock(&m_numFreeMutex);
        if (kval <= m_highestFreeKval)
        {
            PAL_ASSERT(m_pNumFreeList[m_highestFreeKval - m_minKval] != 0);
            result = Result::Success;
            // First we add one to each level for every buddy we'll insert
            while (m_pNumFreeList[kval - m_minKval] == 0)
            {
                m_pNumFreeList[kval - m_minKval] += 1;
                kval++;
            }

            PAL_ASSERT(kval <= m_highestFreeKval);
            PAL_ASSERT_MSG(m_pNumFreeList[kval - m_minKval] > 0,
                "This should only fail if ClaimGpuMemory() is not called before every call to Allocate().");
            // Then we subtract one for the block we will use or split to the lower level
            m_pNumFreeList[kval - m_minKval] -= 1;

            PAL_ASSERT(m_highestFreeKval >= m_minKval);
            while (m_pNumFreeList[m_highestFreeKval - m_minKval] == 0)
            {
                m_highestFreeKval--;
                // in this case, there will be no more space left on the entire buddyAllocator
                if (m_highestFreeKval < m_minKval)
                {
                    break;
                }
            }
        }
    }
    return result;
}

// =====================================================================================================================
// Used to search through pools before claiming memory to find the one that will fragment the least.  pKval will have
// be the highest level needed to be split up for this pool, so the pool with the lowest value will be best.  Can NOT
// guarantee the memory will still be availible by the time this thread calls ClaimGpuMemory.
template <typename Allocator>
Result BuddyAllocator<Allocator>::CheckIfOpenMemory(
    gpusize size,
    gpusize alignment,
    uint32* pKval)
{
    PAL_ASSERT(m_pNumFreeList != nullptr);
    // Pad the requested allocation size to the nearest POT of the size and alignment
    const uint32 kval = Max(SizeToKval(Pow2Pad(Max(size, alignment))), m_minKval);
    PAL_ASSERT(kval >= m_minKval && kval < m_baseAllocKval);

    Result result = Result::ErrorOutOfGpuMemory;
    if ((kval <= m_highestFreeKval))
    {
        result = Result::Success;
        if (pKval != nullptr)
        {
            uint32 topKval = kval;
            for (; topKval < m_baseAllocKval; topKval++)
            {
                if (m_pNumFreeList[topKval - m_minKval] != 0)
                {
                    *pKval = topKval;
                    break;
                }
            }
        }
    }
    return result;
}

// Hashset helper functions.
// =====================================================================================================================
template <typename Allocator>
Result BuddyAllocator<Allocator>::InsertToFreeSet(
    gpusize offset,
    uint32 kval)
{
    FreeSet* pFreeSet = &m_pFreeBlockSets[kval - m_minKval];
    PAL_ASSERT(pFreeSet->Contains(offset) == false);
    Result result = pFreeSet->Insert(offset);

    return result;
}

// =====================================================================================================================
template <typename Allocator>
bool BuddyAllocator<Allocator>::GetKvalUsed(
    gpusize offset,
    uint32* pKval)
{
    bool isUsed;
    MutexAuto usedBlockMapLock(&m_usedBlockMapMutex);
    uint32* usedKval = m_pUsedBlockMap->FindKey(offset);
    if (usedKval == nullptr)
    {
        isUsed = false;
    }
    else
    {
        isUsed = true;
        if (pKval != nullptr)
        {
            *pKval = *usedKval;
        }
    }
    return isUsed;
}

// =====================================================================================================================
template <typename Allocator>
Result BuddyAllocator<Allocator>::SetKvalUsed(
    gpusize offset,
    uint32 kval)
{
    uint32* pKval;
    bool existed;
    MutexAuto usedBlockMapLock(&m_usedBlockMapMutex);
    Result result = m_pUsedBlockMap->FindAllocate(offset, &existed, &pKval);
    if (result == Result::Success)
    {
        *pKval = kval;
    }
    PAL_ASSERT(result == Result::Success);
    return result;
}

// =====================================================================================================================
// If there are free blocks at this level, removes one, if not, returns Result::ErrorOutOfGpuMemory
template <typename Allocator>
Result BuddyAllocator<Allocator>::PopFromFreeSet(
    gpusize* pOffset,
    uint32 kval)
{
    Result result = Result::ErrorUnknown;

    FreeSet* pFreeSet = &m_pFreeBlockSets[kval - m_minKval];
    PAL_ASSERT(pFreeSet != nullptr);

    auto freeSetIt = pFreeSet->Begin();
    if (freeSetIt.Get() != nullptr)
    {
        *pOffset = freeSetIt.Get()->key;
        bool eraseRes = pFreeSet->Erase(*pOffset);
        if (eraseRes)
        {
            result = Result::Success;
        }
        else
        {
            // we got the offset from the iterator, no reason for it to fail.
            PAL_ASSERT_ALWAYS();
        }
    }
    else
    {
        result = Result::ErrorOutOfGpuMemory;
    }
    return result;
}

// =====================================================================================================================
template <typename Allocator>
bool BuddyAllocator<Allocator>::IsOffsetFree(
    gpusize offset,
    uint32 kval)
{
    bool isIn = m_pFreeBlockSets[kval - m_minKval].Contains(offset);

    return isIn;
}

// =====================================================================================================================
template <typename Allocator>
Result BuddyAllocator<Allocator>::RemoveOffsetFromFreeSet(
    gpusize offset,
    uint32 kval)
{
    FreeSet* pFreeSet = &m_pFreeBlockSets[kval - m_minKval];
    bool eraseRes = pFreeSet->Erase(offset);
    return (eraseRes) ? Result::Success : Result::ErrorInvalidValue;
}

// =====================================================================================================================
template <typename Allocator>
Result BuddyAllocator<Allocator>::RemoveOffsetFromUsedMap(
    gpusize offset)
{
    Result result = Result::Success;
    MutexAuto usedBlockMapLock(&m_usedBlockMapMutex);
    bool removeRes = m_pUsedBlockMap->Erase(offset);

    if (removeRes == false)
    {
        result = Result::ErrorInvalidValue;
    }
    return result;
}
} // Pal
