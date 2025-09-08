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
 * @file  palIntrusiveList.h
 * @brief PAL utility collection IntrusiveList and IntrusiveListIterator class declarations.
 ***********************************************************************************************************************
 */

#pragma once

#include "palUtil.h"
#include "palAssert.h"

namespace Util
{

// Forward declarations.
template<typename T> class IntrusiveList;
template<typename T> class IntrusiveListIterator;

/**
 ***********************************************************************************************************************
 * @brief  Encapsulates one node of an intrusive double-linked-list.
 *
 * A node is associated with one data pointer at construction. The data pointer cannot be changed and must be non-null.
 *
 * Note that InList() allows intrusive list users to verify if a given value has been stored in a list without iterating
 * over the list provided that each node object has been designated for a particular list.
 ***********************************************************************************************************************
 */
template<typename T>
class IntrusiveListNode
{
public:
    /// @param [in,out] pData  Address of the data element which contains this intrusive node.
    explicit IntrusiveListNode(T* pData);

    /// Returns true if this node is present in an intrusive list.
    bool InList() const;

private:
    // This special constructor is provided for IntrusiveList's sentinel node which must have a null data pointer.
    IntrusiveListNode();

    T*const               m_pData; // The data object that contains this node.
    IntrusiveListNode<T>* m_pPrev; // Previous node in the list or null if this node is not in a list.
    IntrusiveListNode<T>* m_pNext; // Next node in the list or null if this node is not in a list.

    PAL_DISALLOW_COPY_AND_ASSIGN(IntrusiveListNode);

    // Although this is a transgression of coding standards, it prevents IntrusiveListNode from requiring public
    // accessor functions.  The added encapsulation this provides is worthwhile.
    friend class IntrusiveList<T>;
    friend class IntrusiveListIterator<T>;
};

/**
 ***********************************************************************************************************************
 * @brief  Iterator for traversal of elements in a List collection.
 *
 * Allows traversal of all elements in a List going either forwards or backwards.
 ***********************************************************************************************************************
 */
template<typename T>
class IntrusiveListIterator
{
public:
    ~IntrusiveListIterator() { }

    /// Returns true unless the iterator has advanced past the end of the list.
    bool IsValid() const { return m_pCurrent != m_pSentinel; }

    /// Returns a pointer to the current element.  Returns null if the iterator is invalid.
    T* Get() const { return m_pCurrent->m_pData; }

    /// Advances the iterator to the previous position (move backward).
    void Prev() { m_pCurrent = m_pCurrent->m_pPrev; }

    /// Advances the iterator to the next position (move forward).
    void Next() { m_pCurrent = m_pCurrent->m_pNext; }

    /// Moves the iterator back to the start of the list.
    void Restart() { m_pCurrent = m_pSentinel->m_pNext; }

private:
    IntrusiveListIterator(const IntrusiveListNode<T>*const pSentinel, IntrusiveListNode<T>* pStart);

    const IntrusiveListNode<T>*const m_pSentinel; // We need the sentinel to locate the list boundaries.
    IntrusiveListNode<T>*            m_pCurrent;  // Pointer to the current node.

    PAL_DISALLOW_DEFAULT_CTOR(IntrusiveListIterator);

    // Although this is a transgression of coding standards, it means that List does not need to have a public interface
    // specifically to implement this class.  The added encapsulation this provides is worthwhile.
    friend class IntrusiveList<T>;
};

/**
 ***********************************************************************************************************************
 * @brief Templated, doubly-linked, intrusive, list container.
 *
 * This is meant for storing non-null pointers to elements of an arbitrary type using externally managed nodes.
 * Operations which this class supports are:
 *
 * - Insertion at any point
 * - Deletion at any point
 * - Forwards and reverse iteration
 *
 * @warning This class is not thread-safe.
 ***********************************************************************************************************************
 */
template<typename T>
class IntrusiveList
{
public:
    /// A convenient shorthand for IntrusiveListNode.
    typedef IntrusiveListNode<T> Node;

    /// A convenient shorthand for IntrusiveListIterator.
    typedef IntrusiveListIterator<T> Iter;

    IntrusiveList();

    /// Returns the number of elements in the list, not counting the sentinel.
    size_t NumElements() const { return m_numElements; }

    /// Returns true if the list is empty.
    bool IsEmpty() const { return m_sentinel.m_pNext == &m_sentinel; }

    /// Returns an iterator pointing to the first element in the list.  If the list is empty, the iterator starts out
    /// pointing at the permanent sentinel node.
    ///
    /// @returns An iterator pointing at the front end of the list.
    Iter Begin() const { return Iter(&m_sentinel, m_sentinel.m_pNext); }

    /// Returns an iterator pointing to the last element in the list.  If the list is empty, the iterator starts out
    /// pointing at the permanent sentinel node.
    ///
    /// @returns An iterator pointing at the back end of the list.
    Iter End() const { return Iter(&m_sentinel, m_sentinel.m_pPrev); }

    /// Returns the data pointer at the front of the list.
    ///
    /// @returns The data pointer at the front of the list or null if the list is empty.
    T* Front() const { return m_sentinel.m_pNext->m_pData; }

    /// Returns the data pointer at the back of the list.
    ///
    /// @returns The data pointer at the back of the list or null if the list is empty.
    T* Back() const { return m_sentinel.m_pPrev->m_pData; }

    /// Pushes the specified node onto the front of the list.
    ///
    /// @param [in] pNode Externally-owned list node to link into the list.
    void PushFront(Node* pNode) { InsertBefore(m_sentinel.m_pNext, pNode); }

    /// Pushes the specified node onto the back of the list.
    ///
    /// @param [in] pNode Externally-owned list node to link into the list.
    void PushBack(Node* pNode) { InsertBefore(&m_sentinel, pNode); }

    /// Pushes the contents of pSource onto the front of this list.  The ordering of pSource is preserved, meaning that
    /// the front of pSource will be the new front of this list.  Note that pSource will be left entirely empty.
    ///
    /// It is illegal to call this function with an empty pSource.
    ///
    /// @param [in] pSource The contents of pSource will be pushed in-order onto the front of this list.
    void PushFrontList(IntrusiveList<T>* pSource);

    /// Pushes the contents of pSource onto the back of this list.  The ordering of pSource is preserved, meaning that
    /// the end of pSource will be the new end of this list.  Note that pSource will be left entirely empty.
    ///
    /// It is illegal to call this function with an empty pSource.
    ///
    /// @param [in] pSource The contents of pSource will be pushed in-order onto the back of this list.
    void PushBackList(IntrusiveList<T>* pSource);

    /// Inserts the specified node before a particular node in a list.
    ///
    /// If the iterator has advanced off the end of the list (i.e., the iterator is invalid), the added node will be the
    /// new tail node.
    ///
    /// @param [in] iter  Identifies a node where the insertion should take place.  The iterator will point to the same
    ///                   spot in the list after insertion.
    /// @param [in] pNode Externally-owned list node to link into the list.
    void InsertBefore(const Iter& iter, Node* pNode);

    /// Removes the node at the specified position from the list.
    ///
    /// It is illegal to call this function with an iterator that has already advanced off the end of the list.
    ///
    /// @param [in,out] pIter Iterator identifying the node to be removed.  After the node is removed, this iterator
    ///                       will be advanced to the next node.  If this call removes the final remaining node in the
    ///                       list then the iterator will point at the sentinel and will be invalid.
    void Erase(Iter* pIter);

    /// Removes the node at the specified position from the list. It is illegal to call this function with a Node that
    /// not in this list.
    ///
    /// @param [in] pNode Node to be removed.
    void Erase(Node* pNode);

    /// Removes all nodes from the list.
    void EraseAll();

    /// Truncates the list without touching the elements
    void InvalidateList()
    {
        m_sentinel.m_pNext = &m_sentinel;
        m_sentinel.m_pPrev = &m_sentinel;
        m_numElements = 0;
    }

private:
    void InsertBefore(Node* pBeforeMe, Node* pNode);
    void Unlink(Node* pNode);

    Node   m_sentinel;    // Ties the head to the tail and signifies the boundary of the list.
    size_t m_numElements; // Number of elements.

    PAL_DISALLOW_COPY_AND_ASSIGN(IntrusiveList);
};

// =====================================================================================================================
// This is the public node constructor; it must be given a non-null data pointer.
template<typename T>
IntrusiveListNode<T>::IntrusiveListNode(
    T* pData)
    :
    m_pData(pData),
    m_pPrev(nullptr),
    m_pNext(nullptr)
{
    PAL_ASSERT(pData != nullptr);
}

// =====================================================================================================================
// This is the private node constructor which is used exclusively for sentinel nodes.
template<typename T>
IntrusiveListNode<T>::IntrusiveListNode()
    :
    m_pData(nullptr),
    m_pPrev(nullptr),
    m_pNext(nullptr)
{
}

// =====================================================================================================================
// Returns true if this node is present in an intrusive list.
template<typename T>
bool IntrusiveListNode<T>::InList() const
{
    // The node pointers should always be null or non-null together.
    PAL_DEBUG_BUILD_ONLY_ASSERT((m_pPrev == nullptr) == (m_pNext == nullptr));

    return (m_pNext != nullptr);
}

// =====================================================================================================================
template<typename T>
IntrusiveListIterator<T>::IntrusiveListIterator(
    const IntrusiveListNode<T>*const pSentinel,
    IntrusiveListNode<T>*            pStart)
    :
    m_pSentinel(pSentinel),
    m_pCurrent(pStart)
{
}

} // Util
