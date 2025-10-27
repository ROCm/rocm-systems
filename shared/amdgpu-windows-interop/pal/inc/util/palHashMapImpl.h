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
 * @file  palHashMapImpl.h
 * @brief PAL utility collection HashMap class implementation.
 ***********************************************************************************************************************
 */

#pragma once

#include "palHashBaseImpl.h"
#include "palHashMap.h"

namespace Util
{

// =====================================================================================================================
// Gets a pointer to the value that matches the key.  If the key is not present, a pointer to empty space for the value
// is returned.
template<typename Key,
         typename Value,
         typename Allocator,
         template<typename> class HashFunc,
         template<typename> class EqualFunc,
         typename AllocFunc,
         size_t GroupSize>
Result HashMap<Key, Value, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>::FindAllocate(
    const Key& key,       // Key to search for.
    bool*      pExisted,  // [out] True if a matching key was found.
    Value**    ppValue)   // [out] Pointer to the value entry of the hash map's entry for the specified key.
{
    PAL_ASSERT(pExisted != nullptr);
    PAL_ASSERT(ppValue != nullptr);

    Result result = Result::ErrorOutOfMemory;

    // Get the bucket base address....
    Entry* pGroup = this->InitAndFindBucket(key);

    *pExisted = false;
    *ppValue  = nullptr;

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
                *pExisted = true;
                break;
            }
        }

        // We've reached the end of the allocated buckets and the entry was not found.
        // Allocate this entry for the key.
        if ((pMatchingEntry == nullptr) && (i < Base::EntriesInGroup))
        {
            pGroup[i].key = key;
            pMatchingEntry = &(pGroup[i]);
            this->m_numEntries++;
            this->SetGroupFooterNumEntries(pGroup, numEntries + 1);
        }

        if (pMatchingEntry != nullptr)
        {
            *ppValue = &(pMatchingEntry->value);
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
// Gets a pointer to the value that matches the key.  Returns null if no entry is present matching the specified key.
template<typename Key,
         typename Value,
         typename Allocator,
         template<typename> class HashFunc,
         template<typename> class EqualFunc,
         typename AllocFunc,
         size_t GroupSize>
Value* HashMap<Key, Value, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>::FindKey(
    const Key& key
    ) const
{
    // Get the bucket base address.
    Entry* pGroup = this->FindBucket(key);
    Entry* pMatchingEntry = nullptr;

    while (pGroup != nullptr)
    {
        const uint32 numEntries = this->GetGroupFooterNumEntries(pGroup);

        // Search this entry group
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

    return (pMatchingEntry != nullptr) ? &(pMatchingEntry->value) : nullptr;
}

// =====================================================================================================================
// Inserts a key/value pair entry if it doesn't already exist.
template<typename Key,
         typename Value,
         typename Allocator,
         template<typename> class HashFunc,
         template<typename> class EqualFunc,
         typename AllocFunc,
         size_t GroupSize>
Result HashMap<Key, Value, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>::Insert(
    const Key&   key,
    const Value& value)
{
    bool   existed = true;
    Value* pValue  = nullptr;

    Result result = FindAllocate(key, &existed, &pValue);

    // Add the new value if it did not exist already. If FindAllocate returns Success, pValue != nullptr.
    if ((result == Result::Success) && (existed == false))
    {
        *pValue = value;
    }

    PAL_ASSERT(result == Result::Success);

    return result;
}

// =====================================================================================================================
// Removes an entry with the specified key.
template<typename Key,
         typename Value,
         typename Allocator,
         template<typename> class HashFunc,
         template<typename> class EqualFunc,
         typename AllocFunc,
         size_t GroupSize>
bool HashMap<Key, Value, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>::Erase(
    const Key& key)
{
    // Get the bucket base address.
    Entry* pGroup = this->FindBucket(key);

    Entry* pFoundEntry = nullptr;
    Entry* pLastEntry = nullptr;
    Entry* pLastEntryGroup = nullptr;

    // Find the entry to delete
    while (pGroup != nullptr)
    {
        const uint32 numEntries = this->GetGroupFooterNumEntries(pGroup);

        // Search each group
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

        // Chain to the next entry group.
        pGroup = this->GetNextGroup(pGroup);
    }

    // Copy the last entry's data into the entry that we are removing and invalidate the last entry as it now appears
    // earlier in the list.  This also handles the case where the entry to be removed is the last entry.
    if (pFoundEntry != nullptr)
    {
        PAL_ASSERT(pLastEntry != nullptr);

        pFoundEntry->key   = pLastEntry->key;
        pFoundEntry->value = pLastEntry->value;

        memset(pLastEntry, 0, sizeof(Entry));

        PAL_ASSERT(this->m_numEntries > 0);
        this->m_numEntries--;
        const uint32 numEntries = this->GetGroupFooterNumEntries(pLastEntryGroup);
        this->SetGroupFooterNumEntries(pLastEntryGroup, numEntries - 1);
    }

    return (pFoundEntry != nullptr);
}

} // Util
