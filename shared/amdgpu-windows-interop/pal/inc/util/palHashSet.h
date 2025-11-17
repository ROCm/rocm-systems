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
 * @file  palHashSet.h
 * @brief PAL utility collection HashSet class declaration.
 ***********************************************************************************************************************
 */

#pragma once

#include "palHashBase.h"

namespace Util
{

/// Encapsulates one entry of a hash set.
template<typename Key>
struct HashSetEntry
{
    Key key;  ///< Hash set entry key.
};

/**
 ***********************************************************************************************************************
 * @brief Templated hash set container.
 *
 * This is meant for storing elements of an arbitrary (but uniform) key type. Supported operations:
 *
 * - Searching
 * - Insertion
 * - Deletion
 * - Iteration
 *
 * HashFunc is a functor for hashing keys.  Built-in choices for HashFunc are:
 *
 * - DefaultHashFunc: Good choice when the key is a pointer.
 * - JenkinsHashFunc: Good choice when the key is arbitrary binary data.
 * - StringJenkinsHashFunc: Good choice when the key is a C-style string.
 *
 * EqualFunc is a functor for comparing keys.  Built-in choices for EqualFunc are:
 *
 * - DefaultEqualFunc: Determines keys are equal by bitwise comparison.
 * - StringEqualFunc: Treats keys as a char* and compares them as C-style strings.
 *
 * @warning This class is not thread-safe for Insert, Erase, or iteration!
 * @warning Init() must be called before using this container. Begin() and Reset() can be safely called before
 *          initialization and Begin() will always return an iterator that points to null.
 *
 * For more details please refer to @ref HashBase.
 ***********************************************************************************************************************
 */
template<typename Key,
         typename Allocator,
         template<typename> class HashFunc = DefaultHashFunc,
         template<typename> class EqualFunc = DefaultEqualFunc,
         typename AllocFunc = HashAllocator<Allocator>,
         size_t GroupSize = PAL_CACHE_LINE_BYTES * 2>
class HashSet : public HashBase<Key,
                                HashSetEntry<Key>,
                                Allocator,
                                HashFunc<Key>,
                                EqualFunc<Key>,
                                AllocFunc,
                                GroupSize>
{
public:
    /// Convenience typedef for a templated entry of this hash set.
    typedef HashSetEntry<Key> Entry;

    /// @internal Constructor
    ///
    /// @param [in] numBuckets Number of buckets to allocate for this hash container.  The initial hash container will
    ///                        take (buckets * GroupSize) bytes.
    /// @param [in] pAllocator Pointer to an allocator that will create system memory requested by this hash container.
    explicit HashSet(uint32 numBuckets, Allocator*const pAllocator) : Base::HashBase(numBuckets, pAllocator) {}
    virtual ~HashSet() { }

    /// Finds a given entry; if no entry was found, allocate it.
    ///
    /// @param [in]  ppKey    Key to search for.
    /// @param [out] pExisted True if an entry for the specified key existed before this call was made.
    ///                       False indicates that a new entry was allocated as a result of this call.
    ///
    /// @returns @ref Success if the operation completed successfully
    ///          @ref ErrorOutOfMemory if the operation failed because an internal memory allocation failed.
    Result FindAllocate(Key** ppKey, bool* pExisted);

    /// Returns true if the specified key exists in the set.
    ///
    /// @param [in] key Key to search for.
    ///
    /// @returns True if the specified key exists in the set.
    bool Contains(const Key& key) const;

    /// Inserts an entry.
    ///
    /// No action will be taken if an entry matching this key already exists in the set.
    ///
    /// @param [in] key New entry to insert.
    ///
    /// @returns @ref Success if the operation completed successfully, or @ref ErrorOutOfMemory if the operation failed
    ///          because an internal memory allocation failed.
    Result Insert(const Key& key);

    /// Removes an entry that matches the specified key.
    ///
    /// @param [in] key Key of the entry to erase.
    ///
    /// @returns True if the erase completed successfully, false if an entry for this key did not exist.
    bool Erase(const Key& key);

private:
    // Typedef for the specialized 'HashBase' object we're inheriting from so we can use properly qualified names when
    // accessing members of HashBase.
    typedef HashBase<Key, HashSetEntry<Key>, Allocator, HashFunc<Key>, EqualFunc<Key>, AllocFunc, GroupSize> Base;

    PAL_DISALLOW_DEFAULT_CTOR(HashSet);
    PAL_DISALLOW_COPY_AND_ASSIGN(HashSet);
};

} // Util
