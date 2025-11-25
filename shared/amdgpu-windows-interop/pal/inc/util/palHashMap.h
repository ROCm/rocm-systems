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
 * @file  palHashMap.h
 * @brief PAL utility collection HashMap class declaration.
 ***********************************************************************************************************************
 */

#pragma once

#include "palHashBase.h"

namespace Util
{

/// Encapsulates one key/value pair in a hash map.
template<typename Key, typename Value>
struct HashMapEntry
{
    Key   key;    ///< Hash map entry key.
    Value value;  ///< Hash map entry value.
};

/**
 ***********************************************************************************************************************
 * @brief Templated hash map container.
 *
 * This container is meant for storing elements of an arbitrary (but uniform) key/value type.  Supported operations:
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
 * @warning This class is not thread-safe for Insert, FindAllocate, Erase, or iteration!
 * @warning Init() must be called before using this container. Begin() and Reset() can be safely called before
 *          initialization and Begin() will always return an iterator that points to null.
 *
 * For more details please refer to @ref HashBase.
 ***********************************************************************************************************************
 */
template<typename Key,
         typename Value,
         typename Allocator,
         template<typename> class HashFunc  = DefaultHashFunc,
         template<typename> class EqualFunc = DefaultEqualFunc,
         typename AllocFunc = HashAllocator<Allocator>,
         size_t GroupSize = PAL_CACHE_LINE_BYTES * 2>
class HashMap : public HashBase<Key, HashMapEntry<Key, Value>, Allocator, HashFunc<Key>, EqualFunc<Key>, AllocFunc, GroupSize>
{
public:
    /// Convenience typedef for a templated entry of this hash map.
    typedef HashMapEntry<Key, Value> Entry;

    /// @internal Constructor
    ///
    /// @param [in] numBuckets Number of buckets to allocate for this hash container.  The initial hash container will
    ///                        take (buckets * GroupSize) bytes.
    /// @param [in] pAllocator Pointer to an allocator that will create system memory requested by this hash container.
    explicit HashMap(uint32 numBuckets, Allocator*const pAllocator): Base::HashBase(numBuckets, pAllocator) { }
    virtual ~HashMap() { }

    /// Finds a given entry; if no entry was found, allocate it.
    ///
    /// @param [in]  key      Key to search for.
    /// @param [out] pExisted True if an entry for the specified key existed before this call was made.  False indicates
    ///                       that a new entry was allocated as a result of this call.
    /// @param [out] ppValue  Readable/writeable value in the hash map corresponding to the specified key.
    ///
    /// @returns @ref Success if the operation completed successfully, or @ref ErrorOutOfMemory if the operation failed
    ///          because an internal memory allocation failed.
    Result FindAllocate(const Key& key, bool* pExisted, Value** ppValue);

    /// Gets a pointer to the value that matches the specified key.
    ///
    /// @param [in] key Key to search for.
    ///
    /// @returns A pointer to the value that matches the specified key or null if an entry for the key does not exist.
    Value* FindKey(const Key& key) const;

    /// Inserts a key/value pair entry if the key doesn't already exist in the hash map.
    ///
    /// @warning No action will be taken if an entry matching this key already exists, even if the specified value
    ///          differs from the current value stored in the entry matching the specified key.
    ///
    /// @param [in] key   Key of the new entry to insert.
    /// @param [in] value Value of the new entry to insert.
    ///
    /// @returns @ref Success if the operation completed successfully, or @ref ErrorOutOfMemory if the operation failed
    ///          because an internal memory allocation failed.
    Result Insert(const Key& key, const Value& value);

    /// Removes an entry that matches the specified key.
    ///
    /// @param [in] key Key of the entry to erase.
    ///
    /// @returns True if the erase completed successfully, false if an entry for this key did not exist.
    bool Erase(const Key& key);

private:
    // Typedef for the specialized 'HashBase' object we're inheriting from so we can use properly qualified names when
    // accessing members of HashBase.
    typedef HashBase<Key, HashMapEntry<Key, Value>, Allocator, HashFunc<Key>, EqualFunc<Key>, AllocFunc, GroupSize> Base;

    PAL_DISALLOW_DEFAULT_CTOR(HashMap);
    PAL_DISALLOW_COPY_AND_ASSIGN(HashMap);
};

} // Util
