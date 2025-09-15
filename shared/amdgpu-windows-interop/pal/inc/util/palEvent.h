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
 * @file  palEvent.h
 * @brief PAL utility collection Event class declaration.
 ***********************************************************************************************************************
 */

#pragma once

#include "palTime.h"
#include "palUtil.h"

namespace Util
{

/// Specifies the flags for event.
struct EventCreateFlags
{
    union
    {
        struct
        {
            uint32 manualReset       : 1;  ///< If true, the event is created as manual reset.
            uint32 initiallySignaled : 1;  ///< If true, the event is created in signaled state.
#if defined(_WIN32)
            uint32 canBeInherited    : 1;  ///< If true, the event can be inherited by child process, it's
                                           ///  Windows-specific.
            uint32 reserved          : 29; ///< Reserved for future use.
#else
            uint32 semaphore         : 1;  ///< If true, provide semaphore-like semantics for reads from the file
                                           ///  descriptor.
            uint32 nonBlocking       : 1;  ///< If true, set the O_NONBLOCK file status flag on the new file descriptor.
            uint32 closeOnExecute    : 1;  ///< If true, set the close-on-exec flag for the new file descriptor.
            uint32 reserved          : 27; ///< Reserved for future use.
#endif
        };
        uint32 u32All;                      ///< Flags packed as 32-bit uint.
    };
};

/**
 ***********************************************************************************************************************
 * @brief Synchronization primitive that can either be in the _set_ or _reset_ state.
 *
 * Threads can call WaitForEvents() to block waiting for an Event object to be _set_.  This is useful for fine-grain
 * synchronization between threads.
 *
 * Event objects start out in the _reset_ state.
 ***********************************************************************************************************************
 */
class Event
{
public:
    Event();
    ~Event();
    /// Initializes the event object.  Clients must call this before using the Event object.
    ///
    /// @param flags                Event creation flags.
    /// @param pName                Specified the event's name, it's Windows-specific, Windows uses this name to
    ///                             uniquely identify fence objects across processes.
    /// @returns Success if the event was successfully initialized, otherwise an appropriate error code.
    Result Init(
        const EventCreateFlags& flags
#if defined(_WIN32)
        ,
        const wchar_t* pName = nullptr
#endif
    );
    /// Changes the event state to _set_
    ///
    /// @returns Success unless the Event has not been initialized yet (@ref ErrorUnavailable) or an unexpected internal
    ///          error occured when calling the OS (ErrorUnknown).
    Result Set() const;

    /// Changes the event state to _reset_.
    ///
    /// @returns Success unless the Event has not been initialized yet (ErrorUnavailable) or an unexpected
    ///         internal error occured when calling the OS (ErrorUnknown).
    Result Reset() const;

    /// Waits for the event to enter the _set_ state before returning control to the caller.  The event will change to
    /// the _reset_ state if manualReset was false on initialization.
    ///
    /// @param [in] timeout Max time to wait, in seconds.  If zero, this call will poll the event without blocking.
    ///
    /// @returns Success if the wait completed successfully or Timeout if the wait did not complete but the operation
    ///          timed out.  Otherwise, one of the following errors may be returned:
    ///          + ErrorInvalidValue will be returned if the timeout is negative.
    ///          + ErrorUnknown may be returned if an unexpected internal occurs when calling the OS.
    Result Wait(fseconds timeout) const;

#if defined(_WIN32)
    /// On Windows, a handle to an OS event primitive is a HANDLE, which is just a void*.
    typedef void* EventHandle;
#else
    /// On Linux, a handle to an OS event primitive is a file descriptor, which is just an int.
    typedef int32 EventHandle;
#endif

    /// Returns a handle to the actual OS event primitive associated with this object.
    EventHandle GetHandle() const { return m_hEvent; }

    /// Open event handle.
    Result Open(EventHandle handle, bool isReference);

    /// Constant EventHandle value which represents an invalid event object.
    static const EventHandle InvalidEvent;

private:
    EventHandle m_hEvent;      // OS-specific event handle.
    bool        m_isReference; // If true, the event is a global sharing object handle (not a duplicate) which is
                               // imported from external, so it can't be closed in the currect destructor, and can only
                               // be closed by the creater.

    PAL_DISALLOW_COPY_AND_ASSIGN(Event);
};
} // Util
