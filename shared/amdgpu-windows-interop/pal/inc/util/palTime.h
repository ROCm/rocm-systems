/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
 * @file  palTime.h
 * @brief PAL time-related utility collection.
 ***********************************************************************************************************************
 */

#pragma once

#include <chrono>

namespace Util
{

/// Specifies a class that implements a timestamp.
class Timestamp
{
public:
    /// Creates a new timestamp object that records the time it was created.
    Timestamp();

    /// Returns the timestamp as a C-string.
    const char* CStr() const { return m_data; }

private:
    char m_data[64];
};

#if PAL_CLIENT_INTERFACE_MAJOR_VERSION >= 873
/// Seconds stored as a float instead of an integer.
using fseconds      = std::chrono::duration<float>;
/// Milliseconds stored as a float instead of an integer.
using fmilliseconds = std::chrono::duration<float, std::milli>;
/// Microseconds stored as a float instead of an integer.
using fmicroseconds = std::chrono::duration<float, std::micro>;
/// Nanoseconds stored as a float instead of an integer.
using fnanoseconds  = std::chrono::duration<float, std::nano>;

/// A time_point who's epoch is January 1st 1970 and uses seconds for the duration.
/// C++20 guarantees us that system_clock's epoch is always January 1st 1970 on all platforms.
/// system_clock's internal duration is still implementation defined.
/// On Windows it's hundreds-of-nanoseconds and on Linux it's seconds.
/// However time_point has it's own duration type.
/// As long as we go through the time_point to interpret the duration then everything should be in terms of seconds.
using SecondsSinceEpoch = std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;

/// Like std::chrono::duration_cast, but it preserves the special 'infinite' value used in timeouts.
template<class DestDuration, class Rep, class Period>
constexpr DestDuration TimeoutCast(
    const std::chrono::duration<Rep, Period>& d)
{
    if (d == (std::chrono::duration<Rep, Period>::max)())
    {
        return (DestDuration::max)();
    }
    else
    {
        return std::chrono::duration_cast<DestDuration, Rep, Period>(d);
    }
}
#endif

} // Util
