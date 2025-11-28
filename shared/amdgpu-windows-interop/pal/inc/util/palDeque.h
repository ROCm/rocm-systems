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
 * @file  palDeque.h
 * @brief PAL utility collection Deque and DequeIterator class declarations.
 ***********************************************************************************************************************
 */

#pragma once

#include "palAssert.h"
#include "palSysMemory.h"

namespace Util
{

// Forward declarations.
template<typename T, typename Allocator> class Deque;

/// @internal Private structure used by Deque and its iterators to store chunks of data elements.
struct DequeBlockHeader
{
    DequeBlockHeader* pPrev;   ///< Pointer to the previous block.
    DequeBlockHeader* pNext;   ///< Pointer to the next block.
    void*             pStart;  ///< Pointer to the first element in this block.
    void*             pEnd;    ///< Pointer to the last element in this block.
};

/**
 ***********************************************************************************************************************
 * @brief  Iterator for traversal of elements in a Deque collection.
 *
 * Allows traversal of all elements in a Deque going either forwards or backwards.  If you traverse off either end of
 * the deque, then you must create a new iterator by calling either the Deque's Begin() or End() method.
 ***********************************************************************************************************************
 */
template<typename T, typename Allocator>
class DequeIterator
{
public:
    /// Trivial destructor.
    ~DequeIterator() { }

    /// Returns a pointer to the current element.  Will return null if we've gone past the end.
    T* Get() const { return m_pCurrent; }

    /// Advances the iterator to the next position (move forward).
    void Next();

    /// Advances the iterator to the previous position (move backward).
    void Prev();

    /// Check if the element the iterator references is valid.
    bool IsValid() const { return m_pCurrent != nullptr; }

private:
    DequeIterator(const Deque<T, Allocator>* pDeque, DequeBlockHeader* pHeader, T* pCurrent);

    const Deque<T, Allocator>*const m_pDeque;          // The Deque we're iterating over.
    const DequeBlockHeader*         m_pCurrentHeader;  // The block we're iterating over.
    T*                              m_pCurrent;        // Pointer to the current element.  Null if we've gone past the
                                                       // end.

    PAL_DISALLOW_DEFAULT_CTOR(DequeIterator);

    // Although this is a transgression of coding standards, it means that Deque does not need to have a public
    // interface specifically to implement this class. The added encapsulation this provides is worthwhile.
    friend class Deque<T, Allocator>;
};

/**
 ***********************************************************************************************************************
 * @brief  Simple templated deque container - a double-ended queue.
 *
 * This is meant for storing elements of an arbitrary (but uniform) type. Operations which this class supports are:
 *
 * - Insertion from the front and back.
 * - Deletion from the front and back.
 * - Forwards and reverse iteration
 *
 * @warning This class is not thread-safe for push, pop, or iteration!
 *
 * @note This class is only designed to work with native types and POD-style structures. If it is needed to have a Deque
 *       of complex objects with nontrivial destructors, copy constructors or assign operators, then a specialized
 *       implementation of CleanupElement() will need to be explicitly defined.
 ***********************************************************************************************************************
 */
template<typename T, typename Allocator>
class Deque
{
public:
    /// Constructor.
    ///
    /// @param [in] pAllocator The allocator that will allocate memory if required.
    Deque(Allocator*const pAllocator, size_t numElementsPerBlock = 256);
    ~Deque();

    /// Returns the number of elements in the deque.
    size_t NumElements() const { return m_numElements; }

    /// Returns an iterator pointing to the first element in the deque.
    ///
    /// @returns An iterator pointing at the front end of the deque.
    DequeIterator<T, Allocator> Begin() const { return DequeIterator<T, Allocator>(this, m_pFrontHeader, m_pFront); }

    /// Returns an iterator pointing to the last element in the deque.
    ///
    /// This is somewhat different from std::deque.End() which returns a pointer to the theoretical object _past_ the
    /// end of the deque.
    ///
    /// @returns An iterator pointing at the back end of the deque.
    DequeIterator<T, Allocator> End() const { return DequeIterator<T, Allocator>(this, m_pBackHeader, m_pBack); }

    ///@{
    /// Returns the element at the location specified.
    ///
    /// @warning Calling this function with an out-of-bounds index will cause an access violation!
    ///
    /// @param [in] index Integer location of the element needed.
    ///
    /// @returns The element at location specified by index by reference
    T& At(uint32 index);

    const T& At(uint32 index) const;

    T& operator[](uint32 index);
    const T& operator[](uint32 index) const;
    ///@}

    /// Returns the object at the front of the deque.
    ///
    /// @warning This will cause an access violation if called on an empty deque!
    ///
    /// @returns Reference to the item stored at the front end of the deque.
    T& Front() const
    {
        PAL_ASSERT(m_numElements != 0);
        return *m_pFront;
    }

    /// Returns the object at the tail of the deque.
    ///
    /// @warning This will cause an access violation if called on an empty deque!
    ///
    /// @returns Reference to the item stored at the back end of the deque.
    T& Back() const
    {
        PAL_ASSERT(m_numElements != 0);
        return *m_pBack;
    }

    /// Pushes a copy of the specified item onto the front of the deque.
    ///
    /// @param [in] data Item to be added to the front of the deque.
    ///
    /// @returns @ref Success if the item was successfully added to the deque or @ref ErrorOutOfMemory if the operation
    ///          failed because of an internal failure to allocate system memory.
    Result PushFront(const T& data);

    /// Emplaces a newly constructed item onto the front of the deque.
    ///
    /// @param [in] args arguments used to construct the new item.
    ///
    /// @returns @ref Success if the item was successfully added to the deque or @ref ErrorOutOfMemory if the operation
    ///          failed because of an internal failure to allocate system memory.
    template<typename... Args>
    Result EmplaceFront(Args&&... args);

    /// Pushes a copy of the specified item onto the back of the deque.
    ///
    /// @param [in] data Item to be added to the back of the deque.
    ///
    /// @returns @ref Success if the item was successfully added to the deque or @ref ErrorOutOfMemory if the operation
    ///          failed because of an internal failure to allocate system memory.
    Result PushBack(const T& data);

    /// Emplaces a newly constructed item onto the back of the deque.
    ///
    /// @param [in] args arguments used to construct the new item.
    ///
    /// @returns @ref Success if the item was successfully added to the deque or @ref ErrorOutOfMemory if the operation
    ///          failed because of an internal failure to allocate system memory.
    template<typename... Args>
    Result EmplaceBack(Args&&... args);

    /// Pops the first item off the front of the deque, returning the popped value.
    ///
    /// @param [out] pOut Item popped off the front of the deque.
    ///
    /// @returns @ref Success if the item was successfully popped from the deque or @ref ErrorUnavailable if the deque
    ///          is empty.
    Result PopFront(T* pOut);

    /// Pops the first item off the back of the deque, returning the popped value.
    ///
    /// @param [out] pOut Item popped off the back of the deque.
    ///
    /// @returns @ref Success if the item was successfully popped from the deque or @ref ErrorUnavailable if the deque
    ///          is empty.
    Result PopBack(T* pOut);

private:
    Result AllocateFront(T**);
    Result AllocateBack(T**);
    DequeBlockHeader* AllocateNewBlock();
    void FreeUnusedBlock(DequeBlockHeader* pHeader);

    // A helper function to avoid duplication in const and non-const versions of At().
    T& InternalAt(uint32 index) const;

    size_t            m_numElements;         // Number of elements
    const size_t      m_numElementsPerBlock; // Block granularity when we need to alloc a new one

    DequeBlockHeader* m_pFrontHeader;        // First block of data elements,  null for empty deques.
    DequeBlockHeader* m_pBackHeader;         // Last block of data elements, null for empty deques/

    T*                m_pFront;              // First data element, null for empty deques.
    T*                m_pBack;               // Last data element, null for empty deques.

    DequeBlockHeader* m_pLazyFreeHeader;     // Cached pointer to the most-recently freed block.

    Allocator*const   m_pAllocator;          // Pointer to the allocator for this deque.

    PAL_DISALLOW_COPY_AND_ASSIGN(Deque);

    // Although this is a transgression of coding standards, it prevents DequeIterator requiring a public constructor;
    // constructing a 'bare' DequeIterator (i.e. without calling Deque::GetIterator) can never be a legal operation, so
    // this means that these two classes are much safer to use.
    friend class DequeIterator<T, Allocator>;
};

// =====================================================================================================================
template<typename T, typename Allocator>
Deque<T, Allocator>::Deque(
    Allocator*const pAllocator,
    size_t          numElementsPerBlock)
    :
    m_numElements(0),
    m_numElementsPerBlock(numElementsPerBlock),
    m_pFrontHeader(nullptr),
    m_pBackHeader(nullptr),
    m_pFront(nullptr),
    m_pBack(nullptr),
    m_pLazyFreeHeader(nullptr),
    m_pAllocator(pAllocator)
{
}

// =====================================================================================================================
// Frees all of the blocks this object allocated over its lifetime.
template<typename T, typename Allocator>
Deque<T, Allocator>::~Deque()
{
    if (!std::is_trivial<T>::value)
    {
        while (m_pFrontHeader != nullptr)
        {
            // Explicitly destroy the removed value since it's non-trivial and advance.
            // We must destroy all of them in the current block before freeing it.
            m_pFront->~T();
            ++m_pFront;
            --m_numElements;

            if ((m_pFront == m_pFrontHeader->pEnd) || (m_numElements == 0))
            {
                // Okay, the front block is now empty. Free it and advance to the next block.
                DequeBlockHeader* pBlockToFree = m_pFrontHeader;
                m_pFrontHeader = m_pFrontHeader->pNext;
                PAL_SAFE_FREE(pBlockToFree, m_pAllocator);

                if (m_pFrontHeader != nullptr)
                {
                    // Fixup to the new block.
                    m_pFront = static_cast<T*>(m_pFrontHeader->pStart);
                }
            }
        }
    }
    else
    {
        // Elements are trivial so skip iterating through elements and free each block.
        while (m_pFrontHeader != nullptr)
        {
            DequeBlockHeader* pBlockToFree = m_pFrontHeader;
            m_pFrontHeader = m_pFrontHeader->pNext;
            PAL_SAFE_FREE(pBlockToFree, m_pAllocator);
        }
    }

    if (m_pLazyFreeHeader != nullptr)
    {
        PAL_SAFE_FREE(m_pLazyFreeHeader, m_pAllocator);
    }
}

} // Util
