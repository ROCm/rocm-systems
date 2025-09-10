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
 * @file  palHashBase.h
 * @brief PAL utility collection shared structures and class declarations used by the HashMap and HashSet containers.
 ***********************************************************************************************************************
 */

#pragma once

#include "palSysMemory.h"

namespace Util
{

/// Entry at the end of a group allocation which contains the size and a pointer to the next group.
template <typename Entry>
struct GroupFooter
{
    Entry* pNextGroup;
    uint32 numEntries;
};

// Forward declarations.
template<typename Key,
         typename Entry,
         typename Allocator,
         typename HashFunc,
         typename EqualFunc,
         typename AllocFunc,
         size_t   GroupSize> class HashBase;

/// Default hash functor.
///
/// Just directly returns bits 31-6 of the key's first dword.  This is a decent hash if the key is a pointer.
template<typename Key>
struct DefaultHashFunc
{
    /// Shifts the key to the right and use the resulting bits as a uint hash.
    ///
    /// @param [in] pVoidKey Pointer to the key to be hashed.  If the key is a pointer, which is the best use case for
    ///                      this hash function, then this is really a pointer to a pointer.
    /// @param [in] keyLen   Ignored.
    ///
    /// @returns 32-bit uint hash value.
    uint32 operator()(const void* pVoidKey, uint32 keyLen) const;

    static constexpr uint32 ShiftNum = 6;  ///< Right shift bit number

    /// Makes sure the hashing result always contain at least minNumBits bits.
    void Init(uint32 minNumBits) const
    {
        PAL_ASSERT((Min(sizeof(Key), sizeof(uint32)) * 8) >= (minNumBits + ShiftNum));
        PAL_ALERT_MSG(sizeof(Key) > sizeof(void*), "Usage of DefaultHashFunc for non-pointer types!");
    }
};

/// Jenkins hash functor.
///
/// Compute hash value according to the Jenkins algorithm.  A description of the algorithm is found here:
/// http://burtleburtle.net/bob/hash/doobs.html
/// By Bob Jenkins, 1996. bob_jenkins@compuserve.com. You may use this
/// code any way you wish, private, educational, or commercial. It's free.
/// See http://ourworld.compuserve.com/homepages/bob_jenkins/evahash.htm
/// Use for hash table lookup, or anything where one collision in 2^^32 is
/// acceptable. Do NOT use for cryptographic purposes.
template<typename Key>
struct JenkinsHashFunc
{
    /// Hashes the specified key value via the Jenkins hash algorithm.
    ///
    /// @param [in] pVoidKey Pointer to the key to be hashed.
    /// @param [in] keyLen   Amount of data at pVoidKey to hash, in bytes.
    ///
    /// @returns 32-bit uint hash value.
    uint32 operator()(const void* pVoidKey, uint32 keyLen) const;

    /// No init job. Defined to be compatible with default hash func.
    void Init(uint32) const { }
};

/// Jenkins hash functor for C-style strings.
///
/// Compute hash value according to the Jenkins algorithm.  A description of the algorithm is found here:
/// http://burtleburtle.net/bob/hash/doobs.html
/// By Bob Jenkins, 1996. bob_jenkins@compuserve.com. You may use this
/// code any way you wish, private, educational, or commercial. It's free.
/// See http://ourworld.compuserve.com/homepages/bob_jenkins/evahash.htm
/// Use for hash table lookup, or anything where one collision in 2^^32 is
/// acceptable. Do NOT use for cryptographic purposes.
///
/// @note This hash function is for char* keys only, since the regular JenkinsHashFunc will attempt to do a hash on the
/// address of the pointer, as opposed to the actual string.
template<typename Key>
struct StringJenkinsHashFunc : JenkinsHashFunc<Key>
{
    /// Hashes the specified C-style string key via the Jenkins hash algorithm.
    ///
    /// @param [in] pVoidKey Pointer to the key string (i.e., this is a char**) to be hashed.
    /// @param [in] keyLen   Amount of data at pVoidKey to hash, in bytes.  Should always be sizeof(char*).
    ///
    /// @returns 32-bit uint hash value.
    uint32 operator()(const void* pVoidKey, uint32 keyLen) const;
};

/// Generic compare functor for types with arbitrary size.
///
/// Used by @ref HashBase to prevent defining compare functions for each type.
template<typename Key>
struct DefaultEqualFunc
{
    /// Returns true if key1 and key2 are equal (have identical memory contents).
    bool operator()(const Key& key1, const Key& key2) const
    {
        return (memcmp(&key1, &key2, sizeof(Key)) == 0);
    }
};

/// String compare functor for use with C-style strings.  memcmp doesn't work well for strings, so this uses strcmp.
template<typename Key>
struct StringEqualFunc
{
    /// Returns true if the strings in key1 and key2 are equal.
    bool operator()(const Key& key1, const Key& key2) const;
};

/**
 ***********************************************************************************************************************
 * @brief  Fixed-size, growable, and lazy-free memory pool allocator.
 *
 * Memory is divided into blocks and stored in a fixed-sized structure array.  One blocks is made of fixed-sized groups.
 * Blocks grows exponentially, that is, each block has twice the number of groups than the previous one.
 *
 * @warning This class is not thread-safe!
 ***********************************************************************************************************************
 */
template <typename Allocator>
class HashAllocator
{
public:
    /// Constructor.
    ///
    /// @param [in] groupSize  Fixed allocation size.  Allocate() will only be able to create allocations of this size.
    /// @param [in] alignment  Required alignment of the allocation in bytes.
    /// @param [in] pAllocator Pointer to an allocator that will create system memory requested by this hash container.
    HashAllocator(size_t groupSize, uint32 alignment, Allocator*const pAllocator);

    ~HashAllocator();

    /// Allocates a new block of memory.
    ///
    /// No size parameter, the size of allocation is fixed to the groupSize parameter specified in the constructor.
    ///
    /// @returns A pointer to the allocate memory, or null if the allocation failed.
    void* Allocate();

    /// Recycles all allocated memory.  Memory isn't actually freed, but becomes available for reuse.
    void Reset();

    /// Allocates memory using allocator callbacks.
    ///
    /// @note In order for this AllocFunc to be classified as an Allocator itself, we must define an
    ///       Alloc(const AllocInfo&) function.
    ///
    /// @param [in] allocInfo   Structure containing information about memory allocation.
    ///
    /// @returns Pointer to memory allocated.
    void* Alloc(
        const AllocInfo& allocInfo)
        { return m_pAllocator->Alloc(allocInfo); }

    /// Frees memory using allocator callbacks.
    ///
    /// @note In order for this AllocFunc to be classified as an Allocator itself, we must define a
    ///       Free(const FreeInfo&) function.
    ///
    /// @param [in] freeInfo    Structure containing information about memory needing to be freed.
    void Free(
        const FreeInfo& freeInfo)
        { return m_pAllocator->Free(freeInfo); }

    /// Returns true if the allocator is in the 'reset' state and has no active allocations.
    ///
    /// @note There may still be reused but clean allocations present.
    bool IsClean() const { return m_curBlock == -1; }

private:
    struct MemBlock
    {
        void*  pMemory;    // Pointer to the memory allocated for this block.
        uint32 numGroups;  // Number of groups in the block.
        uint32 curGroup;   // Current group index to be allocated.
    };

    // For the i-th block, it will hold Pow(2,i) groups, the whole array could have 4G groups.
    static constexpr int32 NumBlocks = 32; // Number of blocks.

    MemBlock        m_blocks[NumBlocks];  // Memory blocks holding exponentially growing memory.
    const size_t    m_groupSize;          // Fixed-group-size for each group in one block.
    uint32          m_alignment;          // Required alignment of the allocation in bytes.
    int32           m_curBlock;           // Current block index memory is being allocated from.  -1 indicates the
                                          // allocator has just been created and hasn't created any blocks yet.
    Allocator*const m_pAllocator;         // Allocator for this hash allocation function.
};

// =====================================================================================================================
template<typename Allocator>
HashAllocator<Allocator>::HashAllocator(
    size_t          groupSize,   // Fixed allocation size.  Allocate() will always create allocations of this size.
    uint32          alignment,   // Required alignment of the allocation in bytes.
    Allocator*const pAllocator)  // Allocator for this hash allocation function.
    :
    m_groupSize(groupSize),
    m_alignment(alignment),
    m_curBlock(-1),
    m_pAllocator(pAllocator)
{
    for (int32 i = 0; i < NumBlocks; i++)
    {
        m_blocks[i].pMemory = nullptr;
        m_blocks[i].curGroup = 0;
        m_blocks[i].numGroups = (1 << i);
    }
}

// =====================================================================================================================
template<typename Allocator>
HashAllocator<Allocator>::~HashAllocator()
{
    for (int32 i = 0; i < NumBlocks; i++)
    {
        if (m_blocks[i].pMemory == nullptr)
        {
            break;
        }
        else
        {
            PAL_SAFE_FREE(m_blocks[i].pMemory, m_pAllocator);
        }
    }
}

/**
 ***********************************************************************************************************************
 * @brief  Iterator for traversal of elements in a Hash container.
 *
 * Backward iterating is not supported since there is no "footer" or "header" for a hash container.
 ***********************************************************************************************************************
 */
template<
    typename Key,
    typename Entry,
    typename Allocator,
    typename HashFunc,
    typename EqualFunc,
    typename AllocFunc,
    size_t GroupSize>
class HashIterator
{
public:
    /// Convenience typedef for the associated container for this templated iterator.
    typedef HashBase<Key, Entry, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize> Container;

    ~HashIterator() { }

    /// Returns a pointer to current entry.  Will return null if the iterator has been advanced off the end of the
    /// container.
    Entry* Get() const { return m_pCurrentEntry; }

    /// Advances the iterator to the next position (move forward).
    void Next();

    /// Resets the iterator to its starting point.
    void Reset();

private:
    HashIterator(const Container* pHashContainer, uint32 startBucket);

    const Container* const m_pContainer;     // Hash container that we're iterating over.
    const uint32           m_startBucket;    // Bucket where we start iterating.
    uint32                 m_currentBucket;  // Current bucket we're iterating.
    Entry*                 m_pCurrentGroup;  // Current group we're iterating (belongs to the current bucket).
    Entry*                 m_pCurrentEntry;  // Current entry we're at now (belongs to the current group).
    uint32                 m_indexInGroup;   // Index of current entry in the group.
    PAL_DISALLOW_DEFAULT_CTOR(HashIterator);

    // Although this is a transgression of coding standards, it means that Container does not need to have a public
    // interface specifically to implement this class. The added encapsulation this provides is worthwhile.
    friend class HashBase<Key, Entry, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>;
};

/**
 ***********************************************************************************************************************
 * @brief Templated base class for HashMap and HashSet, supporting the ability to store, find, and remove entries.
 *
 * The hash container has a fixed number of buckets.  These buckets contain a growable number of entry groups.  Each
 * entry group contains a fixed number of entries and a pointer to the next entry group in the bucket.
 *
 * The following restrictions are made in order to tune it to the desired usage:
 *
 * - The item must be significantly smaller than a cache line.
 * - The key must be POD-style type.
 *
 * This class aims to be very efficient when looking up the key and storing small attached items is the primary concern.
 * It's therefore not desired to have the key associated with a pointer to the attached data, because the attached data
 * may be of similar or smaller size than the pointer anyway, it would also introduce much unnecessary memory
 * management, and it would imply a minimum of two cache misses in the typical lookup case.
 *
 * The idea is that these entry groups can be exactly the size of a cache line, so an entry group can be scanned with
 * only a single cache miss.  This extends the load factor that the hash-map can manage before performance begins to
 * degrade.  For the very small items that we expect, this should be a significant advantage; we expect one cache miss
 * pretty much always, so packing the items together would not be a significant gain, and the cost in memory usage is
 * (relatively) small.
 *
 * The initial hash container will use up about (buckets * GroupSize) bytes.
 ***********************************************************************************************************************
 */
template<
    typename Key,
    typename Entry,
    typename Allocator,
    typename HashFunc,
    typename EqualFunc,
    typename AllocFunc,
    size_t GroupSize>
class HashBase
{
public:
    /// Convenience typedef for iterators of this templated HashBase.
    typedef HashIterator<Key, Entry, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize> Iterator;

    /// Initializes the hash container. This no longer needs to be called by a client of this API; instead
    /// subclasses call InitAndFindBucket() instead of FindBucket() in any method that might insert a
    /// new entry.
    ///
    /// @returns @ref Success if the initialization completed successfully, or ErrorOutOfMemory if the operation failed
    ///          due to an internal failure to allocate system memory.
    Result Init();

    /// Returns number of entries in the container.
    uint32 GetNumEntries() const { return m_numEntries; }

    /// Returns an iterator pointing to the first entry.
    Iterator Begin() const;

    /// Empty the hash container.
    void Reset();

protected:
    /// @internal Constructor
    ///
    /// @param [in] numBuckets Number of buckets to allocate for this hash container.  The initial hash container will
    ///                        take (buckets * GroupSize) bytes.
    /// @param [in] pAllocator The allocator that will allocate memory if required.
    explicit HashBase(uint32 numBuckets, Allocator*const pAllocator);
    virtual ~HashBase() { PAL_SAFE_FREE(m_pMemory, &m_allocator); }

    /// @internal Ensures that the hash table has been allocated, then finds the bucket that matches
    /// the specified key
    ///
    /// @param [in] key Key to find matching bucket for.
    ///
    /// @returns Pointer to the bucket corresponding to the specified key.
    Entry* InitAndFindBucket(const Key& key);

    /// @internal Finds the bucket that matches the specified key. A subclass should use this only if it
    /// is searching for an entry. If it might want to insert a new entry, it should use InitAndFindBucket()
    /// instead.
    ///
    /// @param [in] key Key to find matching bucket for.
    ///
    /// @returns Pointer to the bucket corresponding to the specified key.
    Entry* FindBucket(const Key& key) const;

    /// @internal Returns pointer to the next group of the specified group.
    ///
    /// @param [in] pGroup Current group to find next group for.
    ///
    /// @returns Pointer to the next group.
    static Entry* GetNextGroup(Entry* pGroup);

    /// @internal Helper function which returns a pointer to the footer
    ///
    /// @param [in] pGroup Group which contains the footer we want.
    ///
    /// @returns Pointer to the footer
    static GroupFooter<Entry>* GetGroupFooter(Entry* pGroup);

    /// @internal Helper function which returns the number of entries from the footer
    ///
    /// @param [in] pGroup Group which contains the footer we want.
    ///
    /// @returns The number of entries from the footer
    static uint32 GetGroupFooterNumEntries(Entry* pGroup);

    /// @internal Helper function which sets the number of entries in the footer
    ///
    /// @param [in] pGroup Group which contains the footer we want.
    /// @param [in] numEntries Number of entries that should be set.
    static void SetGroupFooterNumEntries(Entry* pGroup, uint32 numEntries);

    /// @internal Helper function which returns the next group from the footer
    ///
    /// @param [in] pGroup Group which contains the footer we want.
    ///
    /// @returns Pointer to the next group from the footer
    static Entry* GetGroupFooterNextGroup(Entry* pGroup);

    /// @internal Helper function which sets the pointer to the next group in the footer
    ///
    /// @param [in] pGroup Group which contains the footer we want.
    /// @param [in] pNextGroup The next group that should be written into the footer.
    static void SetGroupFooterNextGroup(Entry* pGroup, Entry* pNextGroup);

    /// @internal Allocates a new group if the footer of the specified group is null.
    ///
    /// @param [in] pGroup Current group to allocate a next group for.
    ///
    /// @returns Pointer to the next group.
    Entry* AllocateNextGroup(Entry* pGroup);

    const HashFunc  m_hashFunc;       ///< @internal Hash functor object.
    const EqualFunc m_equalFunc;      ///< @internal Key compare function object.
    AllocFunc       m_allocator;      ///< @internal Allocator object.

    uint32          m_numBuckets;     ///< @internal Buckets in the hash table; Padded to power of 2.
    uint32          m_numEntries;     ///< @internal Entries in the table.
    size_t          m_memorySize;     ///< @internal Memory allocation size for m_pMemory.
    void*           m_pMemory;        ///< @internal Base address as allocated (before alignment).

    static constexpr size_t EntrySize = sizeof(Entry);             ///< @internal Size (in bytes) of a single entry.

    /// Size (in bytes) of the footer space of a group linking to next group.
    static constexpr size_t GroupFooterSize = sizeof(GroupFooter<Entry>);

    /// Number of entries in a single group.
    static constexpr uint32 EntriesInGroup = ((GroupSize - GroupFooterSize) / EntrySize);

    // There must be at least one entry in each group.
    static_assert((EntriesInGroup >= 1), "Hash container entry is too big.");

private:
    PAL_DISALLOW_DEFAULT_CTOR(HashBase);
    PAL_DISALLOW_COPY_AND_ASSIGN(HashBase);

    // Although this is a transgression of coding standards, it prevents HashIterator requiring a public constructor;
    // constructing a 'bare' HashIterator (i.e. without calling HashSet::GetIterator) can never be a legal operation, so
    // this means that these two classes are much safer to use.
    friend class HashIterator<Key, Entry, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>;
};

// =====================================================================================================================
template<
    typename Key,
    typename Entry,
    typename Allocator,
    typename HashFunc,
    typename EqualFunc,
    typename AllocFunc,
    size_t GroupSize>
HashIterator<Key, Entry, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>::HashIterator(
    const Container*  pContainer,   ///< [retained] The hash container to iterate over
    uint32            startBucket)  ///< The beginning bucket
    :
    m_pContainer(pContainer),
    m_startBucket(startBucket),
    m_currentBucket(m_startBucket),
    m_indexInGroup(0)
{
    if (m_startBucket < m_pContainer->m_numBuckets)
    {
        m_pCurrentGroup = static_cast<Entry*>(VoidPtrInc(m_pContainer->m_pMemory,
                                                         m_startBucket * GroupSize));
    }
    else
    {
        m_pCurrentGroup = nullptr;
    }

    m_pCurrentEntry = m_pCurrentGroup;
}

// =====================================================================================================================
template<
    typename Key,
    typename Entry,
    typename Allocator,
    typename HashFunc,
    typename EqualFunc,
    typename AllocFunc,
    size_t GroupSize>
HashBase<Key, Entry, Allocator, HashFunc, EqualFunc, AllocFunc, GroupSize>::HashBase(
    uint32          numBuckets,
    Allocator*const pAllocator)
    :
    m_hashFunc(),
    m_equalFunc(),
    m_allocator(GroupSize, alignof(Entry), pAllocator),
    m_numBuckets((numBuckets > 0) ? Pow2Pad(numBuckets) : 1), // We always need at least one bucket.
    m_numEntries(0),
    m_memorySize(m_numBuckets * GroupSize),
    m_pMemory(nullptr)
{
}

} // Util
