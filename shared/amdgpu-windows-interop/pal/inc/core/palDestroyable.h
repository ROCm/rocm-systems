/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
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
 * @file  palDestroyable.h
 * @brief Defines the Platform Abstraction Library (PAL) IDestroyable interface.
 ***********************************************************************************************************************
 */

#pragma once

namespace Pal
{

/**
 ***********************************************************************************************************************
 * @interface IDestroyable
 * @brief     Interface inherited by objects that must be explicitly destroyed by the client.
 *
 * This includes all objects except:
 *
 * + @ref IColorTargetView, @ref IDepthStencilView - These classes are treated as SRDs by the DX12 runtime.  Therefore,
 *   PAL guarantees that no action needs to be taken at Destroy() - the client should just free the memory backing these
 *   classes.
 * + @ref IDevice - These objects are created during IPlatform::EnumerateDevices() and are automatically destroyed
 *   along with the Platform object.
 * + @ref IPrivateScreen - These objects are created as during IPlatform::EnumerateDevices() based on
 *   which screens are attached to each device.  They are automatically destroyed along with the Platform object.
 ***********************************************************************************************************************
 */
class IDestroyable
{
public:
    /// Frees all internal resources associated with this object. This method does not free the system memory associated
    /// with the object (as specified in pPlacementAddr during creation); the client is responsible for freeing that
    /// memory because they allocated it.
    ///
    /// @warning  Calling this method leaves all references to this object in an invalid state, including references
    ///           within a pending or executing ICmdBuffer. Active GPU workloads or calling any PAL methods that use
    ///           any of these invalid references results in undefined behavior.
    virtual void Destroy() = 0;

protected:
    /// @internal Destructor.  Prevent use of delete operator on this interface.  Client must destroy objects by
    /// explicitly calling IDestroyable::Destroy() and is responsible for freeing the system memory allocated for the
    /// object on their own.
    virtual ~IDestroyable() { }
};

/**
 ***********************************************************************************************************************
 * @page PAL Object Reference Model
 *
 * ### The Platform and Device Hierarchy
 *
 * For the purposes of this page, a PAL object is a class instance which descends from the PAL @ref IPlatform object.
 * For example this includes @ref IDevice objects which are directly created by an IPlatform as well as @ref IGpuMemory
 * objects which are created by IDevice. In this way we can imagine creating a "family tree" of PAL objects which starts
 * with an IPlatform object and branches out to one IDevice object for each supported device. Each of those IDevice
 * nodes then branches further into many leaf nodes covering the myriad of PAL state objects in a typical application.
 *
 * Generally speaking, PAL objects that were created by one IDevice object may not be referenced by any PAL objects
 * that were created by some other IDevice object. This is a core PAL design assumption and breaking it will result
 * in undefined behavior. When cross-device communication is desired, the remote IDevice must create a new object
 * which imports the remote object into its side of the "family tree" (e.g. @ref IDevice::OpenSharedGpuMemory).
 *
 * All PAL objects which inherit from @ref IDestroyable **must** be destroyed via @ref IDestroyable::Destroy before
 * their parent IDevice object or grandparent IPlatform object can be destroyed. This is critical as many of PAL's
 * Destroy implementations interact with the object's parent IDevice.
 *
 * Note that there are PAL objects which must respect the above IDevice separation rules but do not follow the Destroy
 * rules. Typically these objects act more as "plain old data" than a typical C++ class. The @ref IColorTargetView and
 * @ref IDepthStencilView objects are good examples of this. They do not inherit from IDestroyable because they are
 * treated as SRDs by the DX12 runtime. PAL is not permitted to hold references to these objects and the client is only
 * responsible for freeing their backing memory when the client is done with them.
 *
 * ### References Between Sibling IDestroyable PAL Objects
 *
 * Many IDestroyable PAL objects created by the same IDevice will hold references to each other. For example, an
 * @ref IImage holds a reference to the @ref IGpuMemory that was most recently bound via
 * @ref IGpuMemoryBindable::BindGpuMemory.
 *
 * In general, these references are not "refcounted". When the client calls Destroy on some PAL object, all existing
 * references to that object become implicitly invalid. PAL has no way to detect if a reference is valid so the client
 * must not do anything that invokes any invalid references. For example, destroying the IGpuMemory bound to an IImage
 * renders that IImage's memory binding invalid. If the client then calls @ref IImage::CopyMemoryToImage undefined
 * behavior occurs. If instead the client calls BindGpuMemory first, either to explicitly remove the old binding or
 * replace it with a new valid IGpuMemory object, then CopyMemoryToImage would either fail with ErrorInvalidPointer or
 * succeed, respectively.
 *
 * All parallel threads, both on the CPU and on the GPU, must be considered here. It is illegal to destroy a PAL object
 * that any processor is currently referencing or may reference in the future. For example, calling
 * @ref IQueue::SignalQueueSemaphore on some @ref IQueueSemaphore gives that IQueue a reference to the semaphore which
 * lasts until the SignalQueueSemaphore operation completes. The client must allow that IQueue to complete all queued
 * operation up to and including that SignalQueueSemaphore before the IQueueSemaphore object can be destroyed.
 *
 * Similarly, all @ref ICmdBuffer objects submitted to the GPU via @ref IQueue::Submit must be permitted to fully
 * execute before any of those ICmdBuffer objects **or any other objects referenced by those ICmdBuffer objects**
 * can be destroyed. This is a common source of application and client bugs so it bears repeating:
 *
 * @warning  It is illegal to destroy any PAL objects before all command buffer submissions referencing those objects
 *           have completed execution.
 *
 * Note that it is safe to destroy PAL objects before explicitly clearing references to those objects. In fact, it is
 * perfectly fine to never clear invalid references to destroyed objects if those references are never invoked. This
 * gives the client general freedom to implement their process cleanup routines as they see fit, as long as they still
 * respect the rule that IDevice objects and IPlatform objects must be destroyed last.
 *
 * ### References Stored by ICmdBuffer Objects
 *
 * The general rules described in the previous section apply to @ref ICmdBuffer objects and any PAL objects recorded to
 * those command buffers. However, the way they work can be a bit unintuitive as there is no way to unbind just one
 * object from a command recording. If a "Cmd" method takes a reference to a PAL object, the client must imagine that
 * this reference has been directly baked into the GPU's packet stream.
 *
 * For example, calling @ref ICmdBuffer::CmdBindPipeline with a reference to @ref IPipeline "objA" immediately followed
 * by another CmdBindPipeline call with a reference to "objB" does **not** remove the prior reference to "objA". Both
 * objects are now referenced by the command buffer!
 *
 * There are three ways to get an ICmdBuffer to release all of its references:
 *   1. Destroy the command buffer object.
 *   2. Explicitly reset the command buffer via @ref ICmdBuffer::Reset.
 *   3. Implicitly reset the command buffer via @ref ICmdBuffer::Begin.
 *
 * But as mentioned in the previous section, if the client is only concerned with destroying all objects then the order
 * of destruction does not matter. For example, if an IPipeline was bound to an ICmdBuffer it is safe to destroy the
 * IPipeline before the ICmdBuffer.
 *
 * Additionally, every ICmdBuffer object holds an implicit reference on its associated @ref ICmdAllocator object if that
 * ICmdAllocator was created with autoMemoryReuse enabled. In other words, it is only safe to destroy an ICmdAllocator
 * before its ICmdBuffers if autoMemoryReuse is disabled.
 *
 ***********************************************************************************************************************
 */

} // Pal
