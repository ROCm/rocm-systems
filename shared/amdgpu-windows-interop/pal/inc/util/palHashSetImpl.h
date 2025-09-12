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
 * @file  palHashSetImpl.h
 * @brief PAL utility collection HashSet class implementation.
 ***********************************************************************************************************************
 */

#pragma once

#include "palHashBaseImpl.h"
#include "palHashSet.h"

namespace Util
{

// =====================================================================================================================
// Inserts a key if it doesn't already exist.
template<typename Key,
         typename Allocator,
         template<typename> class HashFunc,
         template<typename> class EqualFunc,
         typename AllocFunc,
         size_t GroupSize>
Result HashSet<Key, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>::Insert(
    const Key& key)
{
    Key* pKey = const_cast<Key*>(&key);
    bool existed;
    const Result result = FindAllocate(&pKey, &existed);
    if (existed == false)
    {
        *pKey = key;
    }
    return result;
}

// =====================================================================================================================
// Finds a given entry; if no entry was found, allocate it.
template<typename Key,
         typename Allocator,
         template<typename> class HashFunc,
         template<typename> class EqualFunc,
         typename AllocFunc,
         size_t GroupSize>
Result HashSet<Key, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>::FindAllocate(
    Key** ppKey,
    bool* pExisted)
{
    PAL_ASSERT(ppKey != nullptr);
    PAL_ASSERT(pExisted != nullptr);

    Result result = Result::ErrorOutOfMemory;

    // Get the bucket base address.
    Entry* pGroup = this->InitAndFindBucket(**ppKey);

    Entry* pMatchingEntry = nullptr;

    while (pGroup != nullptr)
    {
        const uint32 numEntries = this->GetGroupFooterNumEntries(pGroup);
        // Search this entry group.
        uint32 i = 0;
        for (; i < numEntries; i++)
        {
            if (this->m_equalFunc(pGroup[i].key, **ppKey))
            {
                // We've found the entry.
                pMatchingEntry = &(pGroup[i]);
                *pExisted = true;
                break;
            }
        }

        if ((pMatchingEntry == nullptr) && (i < Base::EntriesInGroup))
        {
            // We've reached the end of the bucket and the entry was not found. Allocate this entry for the key.
            *pExisted = false;
            *ppKey = &pGroup[i].key;
            pMatchingEntry = &(pGroup[i]);
            this->m_numEntries++;
            this->SetGroupFooterNumEntries(pGroup, numEntries + 1);
        }

        if (pMatchingEntry != nullptr)
        {
            result = Result::Success;
            break;
        }

        // Chain to the next entry group.
        pGroup = this->AllocateNextGroup(pGroup);
    }

    PAL_ASSERT(result == Result::Success);

    return result;
}

// =====================================================================================================================
// Searches for the specified key to see if it exists.
template<typename Key,
         typename Allocator,
         template<typename> class HashFunc,
         template<typename> class EqualFunc,
         typename AllocFunc,
         size_t GroupSize>
bool HashSet<Key, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>::Contains(
    const Key& key
    ) const
{
    // Get the bucket base address.
    Entry* pGroup = this->FindBucket(key);
    Entry* pMatchingEntry = nullptr;

    while (pGroup != nullptr)
    {
        const uint32 numEntries = this->GetGroupFooterNumEntries(pGroup);

        // Search this entry group.
        uint32 i = 0;
        for (; i < numEntries; i++)
        {
            if (this->m_equalFunc(pGroup[i].key, key))
            {
                // We've found the entry.
                pMatchingEntry = &(pGroup[i]);
                break;
            }
        }

        if ((pMatchingEntry != nullptr) || (i < Base::EntriesInGroup))
        {
            break;
        }

        // Chain to the next entry group.
        pGroup = this->GetNextGroup(pGroup);
    }

    return (pMatchingEntry != nullptr);
}

// =====================================================================================================================
// Removes an entry with the specified key.
template<typename Key,
         typename Allocator,
         template<typename> class HashFunc,
         template<typename> class EqualFunc,
         typename AllocFunc,
         size_t GroupSize>
bool HashSet<Key, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>::Erase(
    const Key& key)
{
    // Get the bucket base address.
    Entry* pGroup = this->FindBucket(key);

    Entry* pFoundEntry = nullptr;
    Entry* pLastEntry = nullptr;

    Entry* pLastEntryGroup = nullptr;

    // Find the entry to delete.
    while ((pGroup != nullptr))
    {
        const uint32 numEntries = this->GetGroupFooterNumEntries(pGroup);

        // Search this entry
        uint32 i = 0;
        for (; i < numEntries; i++)
        {
            if (this->m_equalFunc(pGroup[i].key, key) == true)
            {
                // We shouldn't find the same key twice.
                PAL_ASSERT(pFoundEntry == nullptr);

                pFoundEntry = &(pGroup[i]);
            }

            // keep track of last entry of all groups in bucket
            pLastEntry = &(pGroup[i]);
            pLastEntryGroup = pGroup;
        }

        // Chain to the next entry group
        pGroup = this->GetNextGroup(pGroup);
    }

    // Copy the last entry's data into the entry that we are removing and invalidate the last entry as it now appears
    // earlier in the list.  This also handles the case where the entry to be removed is the last entry.
    if (pFoundEntry != nullptr)
    {
        PAL_ASSERT(pLastEntry != nullptr);

        pFoundEntry->key = pLastEntry->key;
        memset(pLastEntry, 0, sizeof(Entry));

        PAL_ASSERT(this->m_numEntries > 0);
        this->m_numEntries--;
        const uint32 numEntries = this->GetGroupFooterNumEntries(pLastEntryGroup);
        this->SetGroupFooterNumEntries(pLastEntryGroup, numEntries - 1);
    }

    return (pFoundEntry != nullptr);
}

} // Util
