// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

namespace rocprofiler
{
/// Depth rather than a flag: buffer delivery can happen from inside record emission.
inline thread_local int record_producer_depth = 0;  // NOLINT(misc-definitions-in-headers)

/// True while the calling thread is writing records or delivering a buffer callback. A drain
/// waits on these threads, so one entered from here waits on the caller's own work.
inline bool
on_record_producer_thread()
{
    return record_producer_depth > 0;
}

/// Marks the calling thread as writing records for the lifetime of the object. Nests.
struct scoped_record_producer
{
    scoped_record_producer() { ++record_producer_depth; }
    ~scoped_record_producer() { --record_producer_depth; }

    scoped_record_producer(const scoped_record_producer&) = delete;
    scoped_record_producer(scoped_record_producer&&)      = delete;
    scoped_record_producer& operator=(const scoped_record_producer&) = delete;
    scoped_record_producer& operator=(scoped_record_producer&&) = delete;
};
}  // namespace rocprofiler
