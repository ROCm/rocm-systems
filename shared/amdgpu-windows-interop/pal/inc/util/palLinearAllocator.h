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
 * @file  palLinearAllocator.h
 * @brief * @brief PAL utility allocator LinearAllocator class.
 ***********************************************************************************************************************
 */

#pragma once

#include "palIntrusiveList.h"
#include "palSysMemory.h"

namespace Util
{

/**
 ***********************************************************************************************************************
 * @brief A linear allocator that allocates virtual memory.
 *
 * To improve performance, a linear allocator can be used in performance-critical areas to avoid unnecessary heap
 * allocations. The VirtualLinearAllocator will instead reserve a specified amount of virtual address space and will
 * incrementally back it with real memory as necessary.
 *
 * As clients reach a steady state, allocations from this allocator will become "free," essentially just costing a
 * pointer increment.
 *
 * This allocator can be used with any of the memory management macros. @see Allocators for more information about the
 * Allocation pattern.
 ***********************************************************************************************************************
 */
class VirtualLinearAllocator
{
public:
    /// Constructor.
    ///
    /// @param [in] size Maximum size, in bytes, of virtual memory that this allocator should reserve.
    ///                  Does not need to be aligned to page size.
    VirtualLinearAllocator(size_t size) :
        m_pStart(nullptr),
        m_pCurrent(nullptr),
        m_size(size),
        m_pageSize(0) {}

    /// Destructor.
    virtual ~VirtualLinearAllocator()
    {
        if (m_pStart != nullptr)
        {
            // Free all of the pages.
            Result result = VirtualRelease(m_pStart, m_size);
            PAL_ASSERT(result == Result::_Success);
        }
    }

    /// Initializes the linear allocator by reserving the requested number of pages.
    ///
    /// @returns Result::Success if memory reservation and committing of the first page is successful.
    Result Init()
    {
        m_pageSize = VirtualPageSize();
        m_size     = Pow2Align(m_size, m_pageSize);

        Result result = VirtualReserve(m_size, &m_pStart);

        if (result == Result::_Success)
        {
            result = VirtualCommit(m_pStart, m_pageSize);
        }

        if (result == Result::_Success)
        {
            m_pCurrent         = m_pStart;
            m_pCommittedToPage = VoidPtrInc(m_pCurrent, m_pageSize);
        }

        return result;
    }

    /// Allocates a block of memory.
    ///
    /// @param [in] allocInfo Contains information about the requested allocation.
    ///
    /// @returns Pointer to the allocated memory, nullptr if the allocation failed.
    void*  Alloc(const AllocInfo& allocInfo)
    {
        void* pAlignedCurrent = VoidPtrAlign(m_pCurrent, allocInfo.alignment);
        void* pNextCurrent    = VoidPtrInc(pAlignedCurrent, allocInfo.bytes);
        void* pAlignedEnd     = VoidPtrAlign(pNextCurrent, m_pageSize);

        if (allocInfo.bytes > Remaining())
        {
            pAlignedCurrent = nullptr;
        }
        else if (pAlignedEnd > m_pCommittedToPage)
        {
            const size_t commitBytes = VoidPtrDiff(pAlignedEnd, m_pCommittedToPage);

            const Result result = VirtualCommit(m_pCommittedToPage, commitBytes);

            if (result == Result::_Success)
            {
                m_pCommittedToPage = VoidPtrInc(m_pCommittedToPage, commitBytes);
                m_pCurrent         = pNextCurrent;
            }
            else
            {
                // Return nullptr if allocation fails.
                pAlignedCurrent = nullptr;
            }
        }
        else
        {
            m_pCurrent = pNextCurrent;
        }

        return pAlignedCurrent;
    }

    /// Frees a block of memory.
    ///
    /// @param [in] freeInfo Contains information about the requested free.
    void   Free(const FreeInfo& freeInfo) {}

    /// Rewinds the current pointer to the specified location to reuse already allocated memory.
    ///
    /// @param pStart   Where to reset the m_pCurrent to.
    /// @param decommit If true, pages that are rewound are freed/decommitted.
    void   Rewind(void* pStart, bool decommit)
    {
        PAL_ASSERT((m_pStart <= pStart) && (pStart <= m_pCurrent));

        if (pStart != m_pCurrent)
        {
            if (decommit)
            {
                void*        pStartPage   = VoidPtrAlign(VoidPtrInc(pStart, 1), m_pageSize);
                void*        pCurrentPage = VoidPtrAlign(m_pCurrent, m_pageSize);
                const size_t numPages     = VoidPtrDiff(pCurrentPage, pStartPage) / m_pageSize;

                if (numPages > 0)
                {
                    Result result = VirtualDecommit(pStartPage, m_pageSize * numPages);
                    PAL_ASSERT(result == Result::_Success);

                    m_pCommittedToPage = pStartPage;
                }
            }
#if DEBUG
            else
            {
                void*        pStartPage   = VoidPtrAlign(VoidPtrInc(pStart, 1), m_pageSize);
                void*        pCurrentPage = VoidPtrAlign(m_pCurrent, m_pageSize);
                const size_t numDwords    = VoidPtrDiff(pCurrentPage, pStartPage) / sizeof(uint32);
                uint32*      pNewCurrent  = static_cast<uint32*>(pStartPage);

                for (size_t dword = 0; dword < numDwords; dword++)
                {
                    pNewCurrent[dword] = 0xDEADBEEF;
                }
            }
#endif

            m_pCurrent = pStart;
        }
    }

    /// Returns the current pointer to backing memory.
    ///
    /// @returns Current pointer to backing memory.
    void* Current() { return m_pCurrent; }

    /// Returns the starting pointer to backing memory.
    ///
    /// @returns Pointer to the start of backing memory.
    void* Start() { return m_pStart; }

    /// Returns the number of bytes that have been allocated.
    ///
    /// @returns Number of bytes allocated through this allocator.
    size_t BytesAllocated() { return VoidPtrDiff(m_pCurrent, m_pStart); }

    /// Compute remaining unallocated space in the allocator; once this space is exhausted allocations will fail.
    ///
    /// @returns The size of the remaining unallocated space in bytes.
    size_t Remaining() const { return m_size - VoidPtrDiff(m_pCurrent, m_pStart); }

private:
    void*  m_pStart;            ///< Pointer to where the backing allocation starts.
    void*  m_pCurrent;          ///< Pointer to the current position of backing memory.
    void*  m_pCommittedToPage;  ///< Pointer to the end of the last committed page.

    size_t m_size;              ///< Size of the allocation.
    size_t m_pageSize;          ///< OS' defined page size.

    PAL_DISALLOW_DEFAULT_CTOR(VirtualLinearAllocator);
    PAL_DISALLOW_COPY_AND_ASSIGN(VirtualLinearAllocator);
};

/**
 ***********************************************************************************************************************
 * @brief A "resource acquisition is initialization" (RAII) wrapper for the LinearAllocator classes.
 *
 * The RAII paradigm allows critical sections to be automatically acquired during this class' constructor, and
 * automatically released when a stack-allocated wrapper object goes out-of-scope.  As such, it only makes sense to use
 * this class for stack-allocated objects.
 *
 * This object will ensure that anything allocated the object is allocated on the stack and when it goes out of scope
 * will be properly "rewound" by the allocator.  See the below example.
 *
 *
 *     {
 *         [Current pointer = 0x10]
 *         LinearAllocatorAuto allocator(pPtrToAllocator);
 *         Allocations occur ...
 *         [Current pointer = 0x80]
 *     }
 *     [Current pointer rewinds = 0x10]
 ***********************************************************************************************************************
 */
template <class LinearAllocator>
class LinearAllocatorAuto
{
public:
    /// Tracks the current start pointer.
    ///
    /// @param pAllocator The allocator to wrap.
    /// @param decommit   Whether to decommit any pages of memory allocated when this goes out of scope.
    LinearAllocatorAuto(LinearAllocator* pAllocator, bool decommit)
        :
        m_pAllocator(pAllocator),
#if PAL_MEMTRACK
        m_memTracker(pAllocator),
#endif
        m_pStart(nullptr),
        m_decommit(decommit)
    {
        PAL_ASSERT(pAllocator != nullptr);
        m_pStart = m_pAllocator->Current();

#if PAL_MEMTRACK
        Result result = m_memTracker.Init();
        PAL_ASSERT(result == Result::_Success);
#endif
    }

    /// Rewinds any allocations made when this goes out of scope.
    ~LinearAllocatorAuto()
    {
        m_pAllocator->Rewind(m_pStart, m_decommit);
    }

    /// Allocates a block of memory.
    ///
    /// @param [in] allocInfo Contains information about the requested allocation.
    ///
    /// @returns Pointer to the allocated memory, nullptr if the allocation failed.
    void* Alloc(const AllocInfo& allocInfo)
    {
        void* pMemory = nullptr;
#if PAL_MEMTRACK
        pMemory = m_memTracker.Alloc(allocInfo);
#else
        pMemory = m_pAllocator->Alloc(allocInfo);
#endif

        return pMemory;
    }

    /// Frees a block of memory.
    ///
    /// @param [in] freeInfo Contains information about the requested free.
    void  Free(const FreeInfo& freeInfo)
    {
#if PAL_MEMTRACK
        m_memTracker.Free(freeInfo);
#else
        m_pAllocator->Free(freeInfo);
#endif
    }

private:
    LinearAllocator*const       m_pAllocator;  ///< The LinearAllocator which this object wraps.

#if PAL_MEMTRACK
    MemTracker<LinearAllocator> m_memTracker;  ///< Memory tracker for this LinearAllocatorAuto.
#endif

    void*                       m_pStart;      ///< Where the LinearAllocator started when wrapped by this.
    const bool                  m_decommit;    ///< Whether to decommit any pages of memory allocated on destruction.

    PAL_DISALLOW_DEFAULT_CTOR(LinearAllocatorAuto);
    PAL_DISALLOW_COPY_AND_ASSIGN(LinearAllocatorAuto);
};

/**
 ***********************************************************************************************************************
 * @brief A simple extension of VirtualLinearAllocator that contains an IntrusiveListNode pointing at itself.
 *        This makes it very easy to create and manage IntrusiveLists of VirtualLinearAllocators.
 ***********************************************************************************************************************
 */
class VirtualLinearAllocatorWithNode : public VirtualLinearAllocator
{
public:
    /// Constructor.
    VirtualLinearAllocatorWithNode(size_t size) : VirtualLinearAllocator(size), m_node(this) {}

    /// Destructor.
    virtual ~VirtualLinearAllocatorWithNode() {}

    /// Gets this linear allocator's associated IntrusiveListNode.
    ///
    /// @returns Pointer to this allocator's associated IntrusiveListNode.
    IntrusiveListNode<VirtualLinearAllocatorWithNode>* GetNode() { return &m_node; }

private:
    IntrusiveListNode<VirtualLinearAllocatorWithNode> m_node;

    PAL_DISALLOW_DEFAULT_CTOR(VirtualLinearAllocatorWithNode);
    PAL_DISALLOW_COPY_AND_ASSIGN(VirtualLinearAllocatorWithNode);
};

} // Util
