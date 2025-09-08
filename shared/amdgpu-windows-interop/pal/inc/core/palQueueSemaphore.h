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
 * @file  palQueueSemaphore.h
 * @brief Defines the Platform Abstraction Library (PAL) IQueueSemaphore interface and related types.
 ***********************************************************************************************************************
 */

#pragma once

#include "pal.h"
#include "palDestroyable.h"
#include <chrono>

#if defined(_WIN32)
struct _SECURITY_ATTRIBUTES;
#endif

namespace Pal
{

// Forward declarations.
class IQueueSemaphore;

/// Specifies properties for @ref IQueueSemaphore creation.  Input structure to IDevice::CreateQueueSemaphore().
struct QueueSemaphoreCreateInfo
{
    union
    {
        struct
        {
            /// This queue semaphore may be opened for use by a different device.
            /// For DX12 native fence, the flag needs to be consistent with D3DDDI_SYNCHRONIZATIONOBJECT_FLAGS.Shared
            /// given by DX runtime.
            uint32 shareable              :  1;
            /// This queue semaphore can only be shared through Nt handle.
            /// For DX12 native fence, the flag needs to be consistent with
            /// D3DDDI_SYNCHRONIZATIONOBJECT_FLAGS.NtSecuritySharing given by DX runtime.
            uint32 sharedViaNtHandle      :  1;
            uint32 externalOpened         :  1;  ///< Semaphore was created by other APIs
            /// This queue semaphore is a timeline semaphore. Timeline semaphores have a 64-bit unsigned integer payload
            /// which gets monotonically increased with each Signal operation. A wait on a timeline semaphore blocks the
            /// waiter until the specified payload value has been signaled.
            /// For DX12 native fence, runtime determines initialCount. Therefore, timeline flag has to be set.
            uint32 timeline               :  1;
            /// Do not signal the queue semaphore to max if the device is lost.
            /// For DX12 native fence, the flag needs to be consistent with
            /// D3DDDI_SYNCHRONIZATIONOBJECT_FLAGS.NoSignalMaxValueOnTdr given by DX runtime.
            uint32 noSignalOnDeviceLost   :  1;
            /// For native fence only. If it's 0x0, the native fence type is D3DDDI_NATIVEFENCE_TYPE_DEFAULT.
            /// If it's 0x1, native fence type is D3DDDI_NATIVEFENCE_TYPE_INTRA_GPU.
            /// For DX12, the value is determined by runtime. DXCP needs to set it by reading D3DDDI_NATIVEFENCEINFO.
            uint32 gpuOnly                :  1;
            /// This queue semaphore will be a monitored fence if this flag set, even if OS supports native fence.
            uint32 forceUseMonitoredFence :  1;
            uint32 reserved               : 25;  ///< Reserved for future use.
        };
        uint32 u32All;              ///< Flags packed as 32-bit uint.
    } flags;                        ///< Queue semaphore creation flags.

    uint32 maxCount;                ///< The maximum signal count; once reached, further signals are dropped.  Must be
                                    ///  non-zero and no more than maxSemaphoreCount in @ref DeviceProperties.  For
                                    ///  example, a value of one would request a binary semaphore.
                                    ///  NOTE: maxCount does not apply to timeline semaphores.

    uint64 initialCount;            ///< Initial value for timeline semaphores. (or)
                                    ///  Initial count value for counting semaphores.
                                    ///  Must not be larger than maxCount for counting semaphores.
                                    ///  For DX12 native fence, DXCP needs to pass InitialFenceValue from
                                    ///  D3DDDI_NATIVEFENCEINFO.

};

/// Specifies parameters for opening a queue semaphore for use on another device.  Input structure to
/// IDevice::OpenSharedQueueSemaphore().
struct QueueSemaphoreOpenInfo
{
    /// Shared queue semaphore object from another device to be opened.
    IQueueSemaphore* pSharedQueueSemaphore;
};

/// Specifies parameters for opening a queue semaphore created by other APIs such as D3D.
struct ExternalQueueSemaphoreOpenInfo
{
    union
    {
        struct
        {
            uint32 crossProcess       :  1;   ///< This semaphore is created in another process.
            uint32 sharedViaNtHandle  :  1;   ///< The shared semaphore handle is NT handle.
            uint32 isReference        :  1;   ///< If set, then the opened semaphore will reference the same sync
                                              ///< object in the kernel.  Otherwise, the object is copied to the
                                              ///< new Semaphore.
            /// This queue semaphore is a timeline semaphore. Timeline semaphores have a 64-bit unsigned integer payload
            /// which gets monotonically increased with each Signal operation. A wait on a timeline semaphore blocks the
            /// waiter until the specified payload value has been signaled.
            uint32 timeline          :  1;
            uint32 reserved          : 28;  ///< Reserved for future use.
        };
        uint32 u32All;                  ///< Flags packed as 32-bit uint.
    } flags;                            ///< External queue semaphore open flags.

    OsExternalHandle externalSemaphore; ///< External shared semaphore handle.

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION < 882
#if defined(__unix__) && PAL_KMT_BUILD
    uint64           syncFdSignalValue; ///< Signal timeline value when importing the state of a sync file
#endif
#endif
};

/// Specifies parameters for exporting a queue semaphore. Input structure to IQueueSemaphore::ExportExternalHandle().
struct QueueSemaphoreExportInfo
{
    union
    {
        struct
        {
            uint32 isReference        :  1;   ///< If set, then the semaphore exporting a handle that reference the
                                              ///< same sync object in the kernel.  Otherwise, the object is copied
                                              ///< to the new Semaphore.
            uint32 reserved           : 31;   ///< Resevered for future use.
        };
        uint32 u32All;                        ///< Flags packed as 32-bit uint.
    } flags;                                  ///< External queue semaphore export flags.

#if PAL_KMT_BUILD
    const _SECURITY_ATTRIBUTES* pSecurityAttributes; ///< It specifies the security descriptor and the inheritable
                                                     ///  attribute.
    const wchar_t*              pNtObjectName;       ///< A name to NT handle, if the object is exported as a NT
                                                     ///  handle with a name, and then the handle can be acquired
                                                     ///  via this name.
    uint32                      accessFlags;         ///< Desried access rights of GPU memory.
#if defined(__unix__)
    uint64                      syncFdWaitValue;     ///< Wait timeline value when exporting the state of a sync file
#endif
#endif
};

/**
 ***********************************************************************************************************************
 * @interface IQueueSemaphore
 * @brief     Semaphore object used to synchronize GPU work performed by multiple, parallel queues.
 *
 * These semaphores are used by calling IQueue::SignalQueueSemaphore() and IQueue::WaitQueueSemaphore().
 *
 * @see IDevice::CreateQueueSemaphore()
 * @see IDevice::OpenSharedQueueSemaphore()
 ***********************************************************************************************************************
 */
class IQueueSemaphore : public IDestroyable
{
public:
    /// An IQueue::WaitQueueSemaphore operation may need to be sent down to the OS after the corresponding
    /// IQueue::SignalQueueSemaphore operation due to GPU scheduler limitations. This method checks if any queues have
    /// batched-up commands waiting for a SignalQueueSemaphore operation to appear.
    ///
    /// @returns True if one or more queues have some number of commands batched-up waiting for other queues to signal
    ///          this semaphore. False otherwise.
    virtual bool HasStalledQueues() = 0;

    /// Query timeline Semaphore payload
    ///
    /// @param [out] pValue           returned payload from querying
    ///
    /// @returns Success if the timeline semaphore is queried successful.  Otherwise, one of the following errors may
    ///          be returned:
    ///          + ErrorInvalidValue if an unexpected conversion error occurs.
    ///          + ErrorInvalidObjectType if semaphore is non-timeline type.
    virtual Result QuerySemaphoreValue(
        uint64*                  pValue) = 0;

    /// Wait on timeline Semaphore points, to be clarified, this is a CPU wait.
    ///
    /// @param    [in]  value            Indicate which point to be waited.
    /// @param    [in]  timeout          the max waiting time, timeout is the timeout period in units of nanoseconds.
    ///
    /// @returns Success if the timeline semaphore point is waited successful.  Otherwise, one of the following errors
    ///          may be returned:
    ///          + ErrorInvalidValue if an unexpected conversion error occurs.
    ///          + ErrorInvalidObjectType if semaphore is non-timeline type.
    virtual Result WaitSemaphoreValue(
        uint64                   value,
        std::chrono::nanoseconds timeout) = 0;

    /// Signal on timeline Semaphore points, to be clarified, this is a CPU signal.
    ///
    /// @param    [in]  value            Indicate which point to be signaled.
    ///
    /// @returns Success if the timeline semaphore point is signaled successful.  Otherwise, one of the following errors
    ///          may be returned:
    ///          + ErrorInvalidValue if an unexpected conversion error occurs.
    ///          + ErrorInvalidObjectType if semaphore is non-timeline type.
    virtual Result SignalSemaphoreValue(
        uint64                   value) = 0;

#if PAL_KMT_BUILD || PAL_AMDGPU_BUILD
    /// Returns an OS-specific handle which can be used to refer to this semaphore object across processes. This will
    /// return a null or invalid handle if the object was not created with the external create flag set.
    ///
    /// @param  [in] exportInfo    Information describing how the Semamphore handle should be exported.
    /// @note This function is only available for Linux builds.
    ///
    /// @returns An OS-specific handle which can be used to access the semaphore object across processes.
    virtual OsExternalHandle ExportExternalHandle(
        const QueueSemaphoreExportInfo& exportInfo) const = 0;
#endif

#if defined(_WIN32)
    /// Returns an OS-specific handle which can be used by another device to access the semaphore object.
    ///
    /// @returns An OS-specific handle which can be used by another device to access the semaphore object.
    virtual OsExternalHandle ExportKmtHandle() const = 0;
#endif

    /// Returns the value of the associated arbitrary client data pointer.
    /// Can be used to associate arbitrary data with a particular PAL object.
    ///
    /// @returns Pointer to client data.
    void* GetClientData() const
    {
        return m_pClientData;
    }

    /// Sets the value of the associated arbitrary client data pointer.
    /// Can be used to associate arbitrary data with a particular PAL object.
    ///
    /// @param  [in]    pClientData     A pointer to arbitrary client data.
    void SetClientData(
        void* pClientData)
    {
        m_pClientData = pClientData;
    }

protected:
    /// @internal Constructor. Prevent use of new operator on this interface. Client must create objects by explicitly
    /// called the proper create method.
    IQueueSemaphore() : m_pClientData(nullptr) {}

    /// @internal Destructor.  Prevent use of delete operator on this interface.  Client must destroy objects by
    /// explicitly calling IDestroyable::Destroy() and is responsible for freeing the system memory allocated for the
    /// object on their own.
    virtual ~IQueueSemaphore() { }

private:
    /// @internal Client data pointer. This can have an arbitrary value and can be returned by calling GetClientData()
    /// and set via SetClientData().
    /// For non-top-layer objects, this will point to the layer above the current object.
    void* m_pClientData;
};

} // Pal
