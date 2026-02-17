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
    Entry* pEntry = nullptr;
    bool   existed = false;
    return Base::FindAllocateEntry(key, &existed, &pEntry);
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

    static_assert(offsetof(Entry, key) == 0);
    return Base::FindAllocateEntry(**ppKey, pExisted, reinterpret_cast<Entry**>(ppKey));
}

} // Util
