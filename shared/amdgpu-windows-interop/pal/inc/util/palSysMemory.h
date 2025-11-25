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
 * @file  palSysMemory.h
 * @brief PAL utility collection system memory management macros.
 ***********************************************************************************************************************
 */

#pragma once

#include "palAssert.h"
#include "palInlineFuncs.h"
#include "palMemTracker.h"
#include <type_traits>
#include <cstddef>

// Forward declarations
namespace Util { struct AllocInfo; }
namespace Util { struct FreeInfo;  }
namespace Util { enum SystemAllocType : uint32; }

#if !defined(__GNUC__) || (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ > 8))
/// Default malloc alignment. Usually equal to 16 bytes for x64 targets.
#define PAL_DEFAULT_MEM_ALIGN alignof(std::max_align_t)
#else
// GCC versions prior to 4.9 break C++11 compatibility by putting max_align_t in the global namespace.
/// Default malloc alignment. Usually equal to 16 bytes for x64 targets.
#define PAL_DEFAULT_MEM_ALIGN alignof(::max_align_t)
#endif

namespace Util
{

/// Informs that @p p is aligned to at least @p Alignment.
template<size_t Alignment, typename T>
constexpr T* AssumeAligned(T* p)
{
    return static_cast<T*>(__builtin_assume_aligned(p, Alignment));
}

} // Util

#if PAL_MEMTRACK

/// @internal Malloc allocation method with extra memory leak tracking arguments.
#define PAL_MALLOC_BASE(_size, _align, _allocator, _allocType, _memBlkType) \
    _allocator->Alloc(::Util::AllocInfo(_size, _align, false, _allocType, _memBlkType, __FILE__, __LINE__))

/// @internal Calloc allocation method with extra memory leak tracking arguments.
#define PAL_CALLOC_BASE(_size, _align, _allocator, _allocType, _memBlkType) \
    _allocator->Alloc(::Util::AllocInfo(_size, _align, true, _allocType, _memBlkType, __FILE__, __LINE__))

/// @internal Free method with extra memory leak tracking arguments.
#define PAL_FREE_BASE(_ptr, _allocator, _memBlkType) \
 _allocator->Free(::Util::FreeInfo(const_cast<void*>(static_cast<const volatile void*>(_ptr)), _memBlkType))

#else

/// @internal Malloc method not wrapped with memory leak tracking.
#define PAL_MALLOC_BASE(_size, _align, _allocator, _allocType, _memBlkType) \
    _allocator->Alloc(::Util::AllocInfo(_size, _align, false, _allocType))

/// @internal Calloc method not wrapped with memory leak tracking.
#define PAL_CALLOC_BASE(_size, _align, _allocator, _allocType, _memBlkType) \
    _allocator->Alloc(::Util::AllocInfo(_size, _align, true, _allocType))

/// @internal Free method not wrapped with memory leak tracking.
#define PAL_FREE_BASE(_ptr, _allocator, _memBlkType) \
    _allocator->Free(::Util::FreeInfo(const_cast<void*>(static_cast<const volatile void*>(_ptr))))

#endif

/// Allocates heap memory in place of malloc().
///
/// This macro is used internally by PAL, and will potentially result in a callback to the client for actual allocation.
/// The client is also free to use this macro in order to take advantage of PAL's memory leak tracking.
#define PAL_MALLOC_ALIGNED(_size, _align, _allocator, _allocType) \
    PAL_MALLOC_BASE((_size), (_align), (_allocator), (_allocType), ::Util::MemBlkType::Malloc)

/// Same as @ref PAL_MALLOC_ALIGNED with alignment set to the alignment of the largest native scalar type.
#define PAL_MALLOC(_size, _allocator, _allocType) \
    PAL_MALLOC_ALIGNED(_size, PAL_DEFAULT_MEM_ALIGN, _allocator, _allocType)

/// Allocates zero-initialized heap memory in place of calloc().  See @ref PAL_MALLOC_ALIGNED.
#define PAL_CALLOC_ALIGNED(_size, _align, _allocator, _allocType) \
    PAL_CALLOC_BASE((_size), (_align), (_allocator), (_allocType), ::Util::MemBlkType::Malloc)

/// Same as @ref PAL_CALLOC_ALIGNED with alignment set to the alignment of the largest native scalar type.
#define PAL_CALLOC(_size, _allocator, _allocType) \
    PAL_CALLOC_ALIGNED(_size, PAL_DEFAULT_MEM_ALIGN, _allocator, _allocType)

/// Frees heap memory allocated with the @ref PAL_MALLOC* or @ref PAL_CALLOC* macros.
#define PAL_FREE(_ptr, _allocator) PAL_FREE_BASE((_ptr), (_allocator), ::Util::MemBlkType::Malloc)

/// Safe free macro.  Pointer is set to null after the free.
#define PAL_SAFE_FREE(_ptr, _allocator) { PAL_FREE((_ptr), (_allocator)); (_ptr) = nullptr; }

/// @internal
///
/// This type only exists to force a unique override for placement new.  We need to override placement new in order to
/// call the constructor in the PAL_NEW and PAL_NEW_ARRAY implementations, but we do not want to overload global
/// placement new or include \<new\> since either could interfere with the client.  Adding a dummy parameter allows us
/// to define a PAL-only placement new implementation.
namespace Util
{
struct Dummy
{
    explicit Dummy() { }  ///< Explicit default constructor prevents this from being instantiated via unqualified "{}".
};
}

/// @internal
///
/// PAL-internal placement new override.  The Dummy is used to ensure there won't be a conflict if a client tries to
/// override global placement new.
///
/// @param [in] size    Size of the memory allocation.
/// @param [in] pObjMem Memory where object will be constructed.
/// @param [in] dummy   Unused.
extern void* PAL_CDECL operator new(
    size_t        size,
    void*         pObjMem,
    Util::Dummy   dummy) noexcept;

/// @internal
///
/// Silences compiler warnings about not have a matching delete for the placement new override above.  Will never be
/// called.
///
/// @param [in] pObj    Unused.
/// @param [in] pObjMem Unused.
/// @param [in] dummy   Unused.
extern void PAL_CDECL operator delete(
    void*        pObj,
    void*        pObjMem,
    Util::Dummy  dummy) noexcept;

/// Placement new macro.
#define PAL_PLACEMENT_NEW(_ptr) new((_ptr), ::Util::Dummy{})

/// Allocates heap memory and calls constructor for an object of the specified type.
///
/// This macro is used internally by PAL, and will potentially result in a callback to the client for actual allocation.
/// The client is also free to use this macro.
///
/// Instead of calling "MyClass* pMyClass = new MyClass(arg1, arg2)", call
/// "MyClass* pMyClass = PAL_NEW(MyClass, AllocInternal)(arg1, arg2)".
#define PAL_NEW(_className, _allocator, _allocType) \
    PAL_PLACEMENT_NEW( \
        PAL_MALLOC_BASE(sizeof(_className), alignof(_className), (_allocator), (_allocType), ::Util::MemBlkType::New)) \
        _className

/// Calls destructor and frees heap memory for the object allocated with PAL_NEW*.
#define PAL_DELETE(_ptr, _allocator)                                    \
{                                                                       \
    /* we want to evaluate the expression (_allocator) before calling
       the destructor because the destructor might have side effects */ \
    auto _allocator_ = (_allocator);                                    \
    ::Util::Destructor(_ptr);                                           \
    PAL_FREE_BASE((_ptr), _allocator_, ::Util::MemBlkType::New);        \
}

/// Calls destructor and frees heap memory for "this".  Use this macro to delete an object without a public destructor.
#define PAL_DELETE_THIS(_className, _allocator)                         \
{                                                                       \
    /* we want to evaluate the expression (_allocator) before calling
       the destructor because the destructor might have side effects */ \
    auto _allocator_ = (_allocator);                                    \
    this->~_className();                                                \
    PAL_FREE_BASE(this, _allocator_, ::Util::MemBlkType::New);          \
}

/// Safe delete macro.  Pointer is set to null after the delete.
#define PAL_SAFE_DELETE(_ptr, _allocator) { PAL_DELETE(_ptr, _allocator); (_ptr) = nullptr; }

/// Allocates an array of the specified object type.
///
/// For non-POD types, the default constructor will be called.  Default constructor is not available for POD types
/// (i.e., PAL_NEW_ARRAY(int, 3, AllocInternal)() won't work.
#if PAL_MEMTRACK
#define PAL_NEW_ARRAY(_className, _arrayCnt, _allocator, _allocType) \
    ::Util::NewArray<_className>((_arrayCnt), (_allocator), (_allocType), __FILE__, __LINE__)
#else
#define PAL_NEW_ARRAY(_className, _arrayCnt, _allocator, _allocType) \
    ::Util::NewArray<_className>((_arrayCnt), (_allocator), (_allocType))
#endif

/// Destroys an array of the specified object type.
///
/// For non-POD types, the destructor will be called.
#define PAL_DELETE_ARRAY(_ptr, _allocator) ::Util::DeleteArray(_ptr, _allocator)

/// Safe delete array macro.  Pointer is set to null after the delete.
#define PAL_SAFE_DELETE_ARRAY(_ptr, _allocator) { PAL_DELETE_ARRAY(_ptr, _allocator); (_ptr) = nullptr; }

namespace Util
{
/// Specifies the usage of a system memory allocation made via a client allocation callback.
///
/// The selected type gives the client an idea of the expected lifetime of the allocation, perhaps allowing intelligent
/// selection of sub-allocation pool, etc.
///
/// @note This is a weak uint32 enum where all PAL values set the top bit.  The client is free to use PAL's memory
///       utilities for their own allocations with their own uint32 enum using the range 0 to 0x7FFFFFFF.  The client's
///       allocation callback can then separately handle any memory allocation category, whether allocated by PAL or
///       themselves.
///
/// @see AllocCallbacks
/// @see AllocFunc
enum SystemAllocType : uint32
{
    /// Indicates an allocation will be attached to a client-created PAL object and will not be freed until the client
    /// frees the associated object.  This type will be specified when allocation callbacks are made during a PAL create
    /// call (e.g., IDevice::CreateGraphicsPipeline()).
    AllocObject         = 0x80000000,

    /// Indicates an allocation is for internal PAL use.  The client should assume such allocations have a long
    /// lifetime, and may not be freed until IPlatform::Destroy() is called.
    AllocInternal       = 0x80000001,

    /// Indicates an allocation is for internal PAL use and that the lifetime of the allocation will be short.
    /// Typically this will be specified for heap allocations that will be freed before control is returned to the
    /// client.
    AllocInternalTemp   = 0x80000002,

    /// Indicates an allocation was requested by the shader compiler.
    AllocInternalShader = 0x80000003
};

/// Function pointer type defining a callback for client-controlled system memory allocation.
///
/// @see AllocCallbacks
///
/// @ingroup LibInit
///
/// @param [in] pClientData Pointer to client-defined data.  The pClientData value specified in the pAllocCb parameter
///                         to CreatePlatform() will be passed back to the client on every allocation callback.
/// @param [in] size        Size of the requested allocation in bytes. Must be non-zero.
/// @param [in] alignment   Required alignment of the requested allocation in bytes. Must be a power of two.
/// @param [in] allocType   Hint to client about expected allocation usage and lifetime.  See @ref SystemAllocType.
///
/// @returns Pointer to system memory with the specified size and alignment.  nullptr means that the allocation failed.
typedef void* (PAL_STDCALL *AllocFunc)(
    void*           pClientData,
    size_t          size,
    size_t          alignment,
    SystemAllocType allocType);

/// Function pointer type defining a callback for client-controlled system memory deallocation.
///
/// @see AllocCallbacks
///
/// @ingroup LibInit
///
/// @param [in] pClientData Pointer to client-defined data.  The pClientData value specified in the pAllocCb parameter
///                         to CreatePlatform() will be passed back to the client on every free callback.
/// @param [in] pMem        System memory pointer to be freed.  The specified pointer must have been allocated by an
///                         @ref AllocFunc callback.
typedef void (PAL_STDCALL *FreeFunc)(
    void* pClientData,
    void* pMem);

/// Specifies client-provided system allocation callbacks.  Used as a parameter to Pal::CreatePlatform().
///
/// @ingroup LibInit
struct AllocCallbacks
{
    void*     pClientData;  ///< Opaque pointer to data of client's choosing.  This pointer will be passed back to
                            ///  every @ref AllocFunc and @ref FreeFunc call made by PAL.
    AllocFunc pfnAlloc;     ///< System memory allocation callback.  @see AllocFunc.
    FreeFunc  pfnFree;      ///< System memory deallocation callback.  @see FreeFunc.
};

/// Information about requested allocation.
///
/// Contains necessary information (size, alignment, etc.) to allocate new system memory.
///
/// @note If memory leak tracking is enabled, additional parameters are available from this structure. The allocator
///       does not need to use any of this information, but can if desired.
///
/// @see Allocators
struct AllocInfo
{
    /// Constructor.
    AllocInfo(
        size_t          bytes,      ///< [in] Number of bytes to allocate.
        size_t          alignment,  ///< [in] Required alignment of the requested allocation in bytes.
        bool            zeroMem,    ///< [in] True for calloc, false for malloc.
        SystemAllocType allocType   ///< [in] Hint on type of allocation and lifetime for client callbacks.
#if PAL_MEMTRACK
        , MemBlkType    blockType,  ///< [in] Type of allocation (malloc, new, or new array).
        const char*     pFilename,  ///< [in] Source filename that requested the memory allocation.
        uint32          lineNumber  ///< [in] Line number in the source file that requested the memory allocation.
#endif
        )
        :
        bytes(bytes),
        alignment(alignment),
        zeroMem(zeroMem),
        allocType(allocType)
#if PAL_MEMTRACK
        , blockType(blockType),
        pFilename(pFilename),
        lineNumber(lineNumber)
#endif
    {}

    size_t                bytes;        ///< Number of bytes to allocate.
    const size_t          alignment;    ///< Required alignment of the requested allocation in bytes.
    const bool            zeroMem;      ///< True for calloc, false for malloc.
    const SystemAllocType allocType;    ///< Hint on type of allocation and lifetime for client callbacks.
#if PAL_MEMTRACK
    const MemBlkType      blockType;    ///< Type of allocation (malloc, new, or new array).
    const char*           pFilename;    ///< Source filename that requested the memory allocation.
    const uint32          lineNumber;   ///< Line number in the source file that requested the memory allocation.
#endif
};

/// Information about freeing a specified allocation.
///
/// Contains necessary information about memory that needs to be freed.
///
/// @note If memory leak tracking is enabled, additional parameters are available from this structure. The allocator
///       does not need to use any of this information, but can if desired.
///
/// @see Allocators
struct FreeInfo
{
    /// Constructor.
    FreeInfo(
        void*        pClientMem ///< [in] Pointer to memory allocation.
#if PAL_MEMTRACK
        , MemBlkType blockType  ///< [in] Type of free (free, delete, or delete array).
#endif
        )
        :
        pClientMem(pClientMem)
#if PAL_MEMTRACK
        , blockType(blockType)
#endif
    {}

    void*            pClientMem;    ///< Pointer to memory allocation.
#if PAL_MEMTRACK
    const MemBlkType blockType;     ///< Type of free (free, delete, or delete array).
#endif
};

/**
 ***********************************************************************************************************************
 * @brief Wraps a AllocCallbacks struct into a class compatible with PAL's Allocator concept.
 ***********************************************************************************************************************
 */
class ForwardAllocator
{
public:
    /// Constructor.
    ForwardAllocator(const AllocCallbacks& callbacks) : m_callbacks(callbacks) { }

    /// Allocates memory using the provided pfnAlloc callback.
    ///
    /// @param [in] allocInfo Contains information about the requested allocation.
    ///
    /// @returns Pointer to the allocated memory, nullptr if the allocation failed.
    void* Alloc(const AllocInfo& allocInfo)
    {
        // Allocating zero bytes of memory results in undefined behavior.
        PAL_ASSERT(allocInfo.bytes > 0);

        void* pMem = m_callbacks.pfnAlloc(m_callbacks.pClientData,
                                          allocInfo.bytes,
                                          allocInfo.alignment,
                                          allocInfo.allocType);

        if ((pMem != nullptr) && allocInfo.zeroMem)
        {
            memset(pMem, 0, allocInfo.bytes);
        }

        return pMem;
    }

    /// Frees memory using the provided pfnFree callback.
    ///
    /// @param [in] freeInfo Contains information about the requested free.
    void Free(const FreeInfo& freeInfo)
    {
        if (freeInfo.pClientMem != nullptr)
        {
            m_callbacks.pfnFree(m_callbacks.pClientData, freeInfo.pClientMem);
        }
    }

private:
    const AllocCallbacks m_callbacks;
};

/**
************************************************************************************************************************
* @brief A wrapper for Trackable (using MemTracker) memory allocator that wraps ForwardAllocator.
************************************************************************************************************************
*/
#if PAL_MEMTRACK
class ForwardAllocatorTracked
{
public:
    /// Constructor
    ForwardAllocatorTracked(const AllocCallbacks& callbacks)
        :
    m_allocator(callbacks),
    m_memTracker(&m_allocator)
    {
    }

    /// Allocates a block of memory.
    ///
    /// @param [in] allocInfo Contains information about the requested allocation.
    ///
    /// @returns Pointer to the allocated memory, nullptr if the allocation failed.
    void* Alloc(const AllocInfo& allocInfo)
    {
        return m_memTracker.Alloc(allocInfo);
    }

    /// Frees a block of memory.
    ///
    /// @param [in] freeInfo Contains information about the requested free.
    void  Free(const FreeInfo& freeInfo)
    {
        m_memTracker.Free(freeInfo);
    }

private:
    Util::ForwardAllocator       m_allocator;  ///< The ForwardAllocator which this object wraps.
    MemTracker<ForwardAllocator> m_memTracker; ///< Memory tracker for this ForwardAllocator.
};
#else
using ForwardAllocatorTracked = ForwardAllocator;
#endif

/**
************************************************************************************************************************
* @brief A wrapper representing an allocator const-pointer.  Can be implicitly constructed from any Allocator pointer.
*
* IndirectAllocator is a type-erasure replacement for `Allocator*const pAllocator`, to abstract around Allocator types.
* This allows classes to not need fully template on `typename Allocator`, at the cost of more pointer-indirection.
*
* Const-correctness should be treated as `Allocator*const pAllocator` - the pointed-to Allocator may be mutable.
************************************************************************************************************************
*/
class IndirectAllocator
{
public:
    /// Implicit conversion from any Allocator pointer.
    template <typename Allocator>
    IndirectAllocator(Allocator*const pAllocator)
        :
        m_pAllocator(pAllocator),
        m_pfnAlloc(&DispatchAlloc<Allocator>),
        m_pfnFree(&DispatchFree<Allocator>)
    { }

    /// Constructor specialization for a pointer to another IndirectAllocator, which acts like a copy constructor.
    IndirectAllocator(const IndirectAllocator*const pAllocator) : IndirectAllocator(*pAllocator) { }

    /// Allocates memory.
    ///
    /// @param [in] allocInfo Contains information about the requested allocation.
    ///
    /// @returns Pointer to the allocated memory, nullptr if the allocation failed.
    void* Alloc(const AllocInfo& allocInfo) const { return m_pfnAlloc(m_pAllocator, allocInfo); }

    /// Frees memory.
    ///
    /// @param [in] freeInfo Contains information about the requested free.
    void  Free(const FreeInfo& freeInfo) const { return m_pfnFree(m_pAllocator, freeInfo); }

    /// Returns true if the allocator == nullptr.  Used in place of `pAllocator == nullptr`.
    constexpr bool operator==(std::nullptr_t) const { return m_pAllocator == nullptr; }

private:
    /// @internal Allocation dispatch function. This is what the non-template @ref m_pfnAlloc callback references.
    template <typename Allocator>
    static void* DispatchAlloc(void*const pAllocator, const AllocInfo& allocInfo)
    {
        auto*const pTypedAllocator = static_cast<Allocator*const>(pAllocator);
        return pTypedAllocator->Alloc(allocInfo);
    }

    /// @internal Free dispatch function. This is what the non-template @ref m_pfnFree callback references.
    template <typename Allocator>
    static void  DispatchFree(void*const pAllocator, const FreeInfo& freeInfo)
    {
        auto*const pTypedAllocator = static_cast<Allocator*const>(pAllocator);
        pTypedAllocator->Free(freeInfo);
    }

    using DispatchAllocCb = void* (*)(void*const, const AllocInfo&);
    using DispatchFreeCb  = void  (*)(void*const, const FreeInfo&);

    void*const m_pAllocator;

    const DispatchAllocCb m_pfnAlloc;
    const DispatchFreeCb  m_pfnFree;
};

/**
 ***********************************************************************************************************************
 * @brief A generic allocator class that allocate and free memory for general purpose use.
 ***********************************************************************************************************************
 */
class GenericAllocator
{
public:
    /// Allocates memory.
    ///
    /// @param [in] allocInfo Contains information about the requested allocation.
    ///
    /// @returns Pointer to the allocated memory, nullptr if the allocation failed.
    static void* Alloc(const AllocInfo& allocInfo);

    /// Frees memory.
    ///
    /// @param [in] freeInfo Contains information about the requested free.
    static void Free(const FreeInfo& freeInfo);
};

/**
************************************************************************************************************************
* @brief A wrapper for Trackable (using MemTracker) memory allocator that wraps GenericAllocator.
************************************************************************************************************************
*/
#if PAL_MEMTRACK
class GenericAllocatorTracked
{
public:
    /// Constructor
    GenericAllocatorTracked()
        :
        m_memTracker(&m_allocator)
    {
        Result result = m_memTracker.Init();
        PAL_ASSERT(result == Result::_Success);
    }

    /// Allocates a block of memory.
    ///
    /// @param [in] allocInfo Contains information about the requested allocation.
    ///
    /// @returns Pointer to the allocated memory, nullptr if the allocation failed.
    void* Alloc(const AllocInfo& allocInfo)
    {
        void* pMemory = m_memTracker.Alloc(allocInfo);
        return pMemory;
    }

    /// Frees a block of memory.
    ///
    /// @param [in] freeInfo Contains information about the requested free.
    void  Free(const FreeInfo& freeInfo)
    {
        m_memTracker.Free(freeInfo);
    }

private:
    GenericAllocator             m_allocator;  ///< The GenericAllocator which this object wraps.

    MemTracker<GenericAllocator> m_memTracker; ///< Memory tracker for this GenericAllocator.
};
#else
using GenericAllocatorTracked = GenericAllocator;
#endif

/// Returns the OS-specific page size.
///
/// @note    All virtual reservations/commits/decommits/releases must be aligned to the value returned by this function.
///
/// @return  The OS-specific size, in bytes, of a page.
extern size_t VirtualPageSize();

/// Reserves the specified amount of virtual address space.
///
/// @param [in]  sizeInBytes Size in bytes of the requested reservation. Must be aligned to the page size returned from
///                          @ref Util::VirtualPageSize();
/// @param [out] ppOut       Pointer to reserved memory. Must not be null.
/// @param [in]  pMem        Pointer to the starting virtual address to reserve.
///                          Subject to failure "ErrorOutOfMemory" if any pages in that range have already been reserved
///                          elsewhere by the process.
/// @param [in]  alignment   Optional parameter specifying the alignment of the reserved cpu VA.
///
/// @returns Success if reservation is successful.
///          Otherwise:
///             - ErrorOutOfMemory if memory reservation failed.
///             - ErrorInvalidValue if sizeInBytes is zero.
///             - ErrorInvalidPointer if ppOut is null.
extern Result VirtualReserve(size_t sizeInBytes, void** ppOut, void* pMem = nullptr, size_t alignment = 1);

/// Commits the specified amount of virtual address space, requesting backing memory from the OS.
///
/// @param [in]  pMem         Pointer to the start of reserved memory to commit. Must be aligned to the page size
///                           returned from @ref Util::VirtualPageSize();
/// @param [in]  sizeInBytes  Size in bytes of the requested commit. Must be aligned to the page size returned from
///                           @ref Util::VirtualPageSize();
/// @param [in]  isExecutable Indicate if the committed memory is used for execution on gpu(For instance: PM4, kernel).
///                           Used only on Windows.
///
/// @returns Success if committing is successful.
///          Otherwise:
///             - ErrorOutOfMemory if memory committing failed.
///             - ErrorInvalidValue if sizeInBytes is zero.
///             - ErrorInvalidPointer if pMem is null.
extern Result VirtualCommit(void* pMem, size_t sizeInBytes, bool isExecutable = false);

/// Decommits the specified amount of virtual address space, freeing the backing memory back to the OS.
///
/// @param [in]  pMem        Pointer to the start of committed memory. Must be aligned to the page size returned from
///                          @ref Util::VirtualPageSize();
/// @param [in]  sizeInBytes Size in bytes of how much to decommit. Must be aligned to the page size returned from
///                          @ref Util::VirtualPageSize();
///
/// @returns Success if decommitting is successful.
///          Otherwise:
///             - ErrorInvalidValue if sizeInBytes is zero.
///             - ErrorInvalidPointer if pMem is null.
extern Result VirtualDecommit(void* pMem, size_t sizeInBytes);

/// Releases the specified amount of virtual address space, both freeing the backing memory and virtual address space
/// back to the OS.
///
/// @param [in]  pMem        Pointer to the start of reserved memory. Must be aligned to the page size returned from
///                          @ref Util::VirtualPageSize();
/// @param [in]  sizeInBytes Size in bytes of how much to release. Must be aligned to the page size returned from
///                          @ref Util::VirtualPageSize();
///
/// @returns Success if decommitting is successful.
///          Otherwise:
///             - ErrorInvalidValue if sizeInBytes is zero
///             - ErrorInvalidPointer if pMem is null.
extern Result VirtualRelease(void* pMem, size_t sizeInBytes);

/// @internal
///
/// OS-specific implementation to install default allocation callbacks in the specified structure.  Expected to be
/// called during CreatePlatform if the client doesn't specify their own allocation callbacks.
///
/// @param [in,out] pAllocCb Allocation callback structure to be updated with the OS-specific default callbacks.
///
/// @returns Success if successful, otherwise an appropriate error code.
extern Result OsInitDefaultAllocCallbacks(AllocCallbacks* pAllocCb);

/// @internal Internal template implementation for calling a destructor from PAL_DELETE or PAL_DELETE_ARRAY.
///
/// @param [in] p Object to be destructed.
template<typename T>
void Destructor(T* p)
{
    if ((p != nullptr) && !std::is_trivial<T>::value)
    {
        p->~T();
    }
}

/// @internal
///
/// Internal template implementation of PAL_NEW_ARRAY.
///
/// For non-POD types, allocate extra memory and store the array count for use by the destructor.  An entire extra cache
/// line is allocated in order to prevent misaligning the actual array data.  In practice, we rarely allocate an array
/// of non-POD objects, so this is likely not an issue.
///
/// @param [in] arrayCnt   Number of entries in the array.
/// @param [in] pAllocator The allocator that will allocate the memory for the array.
/// @param [in] allocType  Hint to client on the lifetime/type of allocation.
/// @param [in] pFilename  Source filename that requested the new array.
/// @param [in] lineNumber Line number in the source file that requested the new array.
///
/// @returns Pointer to the allocated array, nullptr if the allocation failed.
template<typename T, typename Allocator>
T* NewArray(
    size_t          arrayCnt,
    Allocator*      pAllocator,
    SystemAllocType allocType
#if PAL_MEMTRACK
    ,
    const char*     pFilename,
    uint32          lineNumber
#endif
    )
{
    size_t align      = alignof(T);
    size_t allocSize  = sizeof(T) * arrayCnt;
    size_t headerSize = 0;

    if (!std::is_trivial<T>::value)
    {
        align      = Max(align, alignof(size_t));
        headerSize = Max(align, sizeof(size_t));
        allocSize += headerSize;
    }

#if PAL_MEMTRACK
    const Util::AllocInfo info(allocSize, align, false, allocType, MemBlkType::NewArray, pFilename, lineNumber);
#else
    const Util::AllocInfo info(allocSize, align, false, allocType);
#endif

    T* pRet = static_cast<T*>(pAllocator->Alloc(info));

    if ((!std::is_trivial<T>::value) && (pRet != nullptr))
    {
        pRet = static_cast<T*>(Util::VoidPtrInc(static_cast<void*>(pRet), headerSize));

        size_t* pArrayCnt = static_cast<size_t*>(Util::VoidPtrDec(static_cast<void*>(pRet), sizeof(size_t)));
        *pArrayCnt = arrayCnt;

        T* pCurObj = static_cast<T*>(pRet);
        for (uint32 i = 0; i < arrayCnt; i++)
        {
            PAL_PLACEMENT_NEW(pCurObj) T;
            pCurObj++;
        }
    }

    return pRet;
}

/// @internal
///
/// Internal template implementation for PAL_DELETE_ARRAY.
///
/// For non-POD types, find the array count stored before the client pointer, and call the destructor on each object in
/// the array.
///
/// @param [in] p          Pointer to the memory to be deleted.
/// @param [in] pAllocator The allocator that will free the memory for the array.
template<typename T, typename Allocator>
void DeleteArray(T* p, Allocator* pAllocator)
{
    if ((p != nullptr) && !std::is_trivial<T>::value)
    {
        const size_t  headerSize = Max(Max(alignof(T), alignof(size_t)), sizeof(size_t));
        const size_t* pArrayCnt  = const_cast<const size_t*>(reinterpret_cast<const volatile size_t*>(p)) - 1;

        for (uint32 i = 0; i < *pArrayCnt; i++)
        {
            Destructor(p + i);
        }

        p = const_cast<T*>(reinterpret_cast<const T*>((Util::VoidPtrDec(pArrayCnt + 1, headerSize))));
    }

#if PAL_MEMTRACK
    const Util::FreeInfo info(const_cast<void*>(static_cast<const volatile void*>(p)), MemBlkType::NewArray);
#else
    const Util::FreeInfo info(const_cast<void*>(static_cast<const volatile void*>(p)));
#endif

    pAllocator->Free(info);
}

constexpr size_t FastMemCpyMaxSmallSize = 64;

typedef void* (PAL_CDECL *FastMemCpySmallFunc)(void* pDst, const void* pSrc, size_t count);

extern const FastMemCpySmallFunc FastMemCpySmallFuncTable[];

/// A version of memcpy that has fewer branches for small copies.  It computes an index into
/// a table based on the size requested then jumps to a branchless memcpy for that size.
/// Note that The compiler will NOT inline this if the count is known at compile time. The
/// regular memcpy() will be inlined and should be used.  Note further that it is NOT always
/// obvious that the count is not known. Consider the case:
///   void SetData(UINT count, UINT* pData)
///   {
///     ...
///     memcpy(pBuf, pData, count*sizeof(UINT));
///     ...
///   }
/// and SetData is called like this
///   SetData(4, buf);
/// In such a case, although the memcpy itself has an unknown size, if the SetData function
/// inlines and is used with a fixed count, the memcpy count is actually known. Such cases need
/// to be carefully managed.

PAL_FORCE_INLINE void* FastMemCpy(void* pDst, const void* pSrc, size_t count)
{
    // The last entry in the table handles all entries larger than 64 bytes, so clamping the size
    // to 64 calls the correct routine.
    const size_t index = Min(count, FastMemCpyMaxSmallSize + 1);
    return (FastMemCpySmallFuncTable[index])(pDst, pSrc, count);
}

/// Get the default allocation callback.
///
/// @param [out] pAllocCb Pointer to the allocation callback structure. Must not be null.
void PAL_STDCALL GetDefaultAllocCb(Util::AllocCallbacks* pAllocCb);

} // Util
