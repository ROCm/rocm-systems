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
 * @file  palGpuMemoryBindable.h
 * @brief Defines the Platform Abstraction Library (PAL) IGpuMemoryBindable interface and related types.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palDestroyable.h"

namespace Pal
{

// Forward declarations.
class IGpuMemory;

/// Reports required properties of a GPU memory object bound to a specific object.  The client must query these
/// properties via IGpuMemoryBindable::GetGpuMemoryRequirements() and bind an @ref IGpuMemory object matching these
/// requirements to the @ref IGpuMemoryBindable object using IGpuMemoryBindable::BindGpuMemory().
struct GpuMemoryRequirements
{
    union
    {
        struct
        {
            uint32 cpuAccess : 1;  ///< CPU access is required. If set, the client must not set cpuInvisible in
                                   ///  GpuMemoryCreateFlags and must provide CPU visible heaps or CPU visible heap
                                   ///  access mode. If not set, it's strongly recommended to set cpuInvisible.
            uint32 reserved : 31;  ///< Reserved for future use.
        };
        uint32 u32All;             ///< Flags packed as 32-bit uint.
    } flags;                       ///< Flags specifying required GPU memory properties.

    gpusize size;                  ///< Amount of GPU memory required, in bytes.
    gpusize alignment;             ///< Required GPU memory virtual address alignment, in bytes.
    uint32  heapCount;             ///< Number of valid entries in heaps[].
    GpuHeap heaps[GpuHeapCount];   ///< List of allowed heaps for the GPU memory in order of predicted performance.
};

/**
 ***********************************************************************************************************************
 * @interface IGpuMemoryBindable
 * @brief     Interface inherited by objects that may require GPU memory be bound to them.
 *
 * In the future, PAL may discover a need to allocate GPU memory for a class that currently doesn't require it.  In that
 * situation, that class will be updated to inherit from IGpuMemoryBindable.  This change would break backward
 * compatibility and would result in the major interface version being incremented.
 ***********************************************************************************************************************
 */
class IGpuMemoryBindable : public IDestroyable
{
public:
    /// Queries the GPU memory properties required by this object.  The client should query properties with this method,
    /// create/sub-allocate a memory range matching the requirements, then bind the memory to the object via
    /// @ref BindGpuMemory().
    ///
    /// @note Not all objects may actually need GPU memory, and in that case the memory properties will reflect a 0 size
    ///       and alignment.
    ///
    /// @param [out] pGpuMemReqs Required properties of GPU memory to be bound to this object.  Includes properties like
    ///                          size, alignment, and allowed heaps.
    virtual void GetGpuMemoryRequirements(
        GpuMemoryRequirements* pGpuMemReqs) const = 0;

    /// Binds GPU memory to this object according to the requirements queried via GetGpuMemoryRequirements().
    ///
    /// Binding memory to objects other than images automatically initializes the object memory as necessary. Image
    /// objects used as color or depth-stencil targets have to be explicitly initialized in command buffers using a
    /// ICmdBuffer::CmdReleaseThenAcquire() command to transition them out of the LayoutUninitializedTarget usage.
    ///
    /// Binding memory to an object automatically unbinds any previously bound memory. There is no need to bind null to
    /// an object to explicitly unbind a previously bound allocation before binding a new allocation.
    ///
    /// This call is invalid on objects that have no memory requirements, even if binding null.
    ///
    /// @param [in] pGpuMemory GPU memory to be bound.  If null, the previous binding will be released.
    /// @param [in] offset     Offset into the GPU memory where the object's memory range should begin.  This allows
    ///                        sub-allocating many object's GPU memory from the same IGpuMemory object.
    ///
    /// @returns Success if the specified GPU memory was successfully bound to the object.  Otherwise, one of the
    ///          following errors may be returned:
    ///          + ErrorUnavailable if binding a non-image to a virtual allocation.
    ///          + ErrorInvalidAlignment if the offset does not match the alignment requirements of the object.
    ///          + ErrorInvalidMemorySize if the object's required memory size does not fit completely within the given
    ///            memory object at the specified offset.
    virtual Result BindGpuMemory(
        IGpuMemory* pGpuMemory,
        gpusize     offset) = 0;

    /// Returns the GPU memory object and offset that this object is bound to or nullptr and 0 if not bound.
    ///
    /// @param [out] ppGpuMemory Returns the GPU memory object to the address specified in this pointer.
    ///                          Returns nullptr if this object is not bound to any GPU memory.
    /// @param [out] pOffset     Returns the GPU memory offset to the address specified in this pointer.
    ///                          Returns 0 if this object is not bound to any GPU memory.
    ///
    /// @returns Success if the GPU memory and offset was successfully returned. Otherwise, one of the following errors
    ///          may be returned:
    ///          + ErrorGpuMemoryNotBound if this object is not bound to any GPU memory.
    ///          + ErrorInvalidPointer if either ppGpuMemory or pOffset is nullptr.
    ///          + ErrorUnavailable if binding is not supported in the derived class
    virtual Result GetGpuMemory(
        IGpuMemory** ppGpuMemory,
        gpusize* pOffset) const = 0;

protected:
    /// @internal Destructor.  Prevent use of delete operator on this interface.  Client must destroy objects by
    /// explicitly calling IDestroyable::Destroy() and is responsible for freeing the system memory allocated for the
    /// object on their own.
    virtual ~IGpuMemoryBindable() { }
};

} // Pal
