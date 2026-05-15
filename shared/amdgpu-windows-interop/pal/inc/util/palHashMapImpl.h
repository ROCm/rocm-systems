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

    Entry* pEntry = nullptr;
    Result result = Base::FindAllocateEntry(key, pExisted, &pEntry);
    if (result == Result::Success)
    {
        *ppValue = &pEntry->value;
    }

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
    Entry* pEntry = Base::FindEntry(key);
    return (pEntry != nullptr) ? &pEntry->value : nullptr;
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
    Entry* pEntry  = nullptr;

    Result result = Base::FindAllocateEntry(key, &existed, &pEntry);

    // Add the new value if it did not exist already. If FindAllocate returns Success, pValue != nullptr.
    if ((result == Result::Success) && (existed == false))
    {
        pEntry->value = value;
    }

    PAL_ASSERT(result == Result::Success);

    return result;
}

} // Util
