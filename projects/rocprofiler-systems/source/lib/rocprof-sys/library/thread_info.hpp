// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "core/utility.hpp"

#include <pthread.h>
#include <thread>
#include <timemory/backends/threading.hpp>

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

namespace rocprofsys
{
//  InternalTID:  zero-based, process-local thread-ID from atomic increment
//                from user-created threads and rocprof-sys-created threads.
//                This value may vary based on threads created by different
//                backends.
//
//  SystemTID:    system thread-ID. Should be same value as what is seen
//                in debugger, etc.
//
//  SequentTID:   zero-based, process-local thread-ID based on the sequence of
//                user-created threads which are created in-between the
//                initialization and finalization of rocprof-sys.
//                In theory, rocprof-sys will never increment this value
//                because of a thread explicitly by rocprof-sys or
//                by other of the dependent libraries. Most commonly
//                used for indexing into rocprof-sys's thread-local data.
//
//  NativeHandle: value of static_cast<int64_t>(pthread_self())
//
enum ThreadIdType : int
{
    InternalTID = 0,
    SystemTID   = 1,  // system thread id
    SequentTID  = 2,
    PthreadID   = 3,
    StlThreadID = 4,
};

struct thread_index_data
{
    using stl_tid_t    = std::thread::id;
    using native_tid_t = pthread_t;

    // the lookup value is always incremented for each thread
    // the system value is the tid provided by the operating system
    // the internal value is the value which the user expects
    int64_t      internal_value = utility::get_thread_index();
    int64_t      system_value   = tim::threading::get_sys_tid();
    int64_t      sequent_value  = tim::threading::get_id();
    native_tid_t pthread_value  = ::pthread_self();
    stl_tid_t    stl_value      = std::this_thread::get_id();

    std::string as_string() const;
};

int64_t grow_data(int64_t);

struct thread_info
{
    using index_data_t    = std::optional<thread_index_data>;
    using lifetime_data_t = std::pair<uint64_t, uint64_t>;
    using native_handle_t = std::thread::native_handle_type;

    ~thread_info()                  = default;
    thread_info(const thread_info&) = delete;
    thread_info(thread_info&&)      = default;

    thread_info& operator=(const thread_info&) = delete;
    thread_info& operator=(thread_info&&)      = default;

    static void set_start(uint64_t, bool _force = false);
    static void set_stop(uint64_t);

    uint64_t get_start() const;
    uint64_t get_stop() const;

    bool            is_valid_time(uint64_t _ts) const;
    bool            is_valid_lifetime(uint64_t _beg, uint64_t _end) const;
    bool            is_valid_lifetime(lifetime_data_t) const;
    lifetime_data_t get_valid_lifetime(lifetime_data_t) const;

    std::string as_string() const;

    static bool                              exists();
    static size_t                            get_peak_num_threads();
    static const std::optional<thread_info>& init(bool _offset = false);
    static const std::optional<thread_info>& get();
    static const std::optional<thread_info>& get(native_handle_t&);
    static const std::optional<thread_info>& get(native_handle_t&&);
    static const std::optional<thread_info>& get(std::thread::id);
    static const std::optional<thread_info>& get(int64_t _tid, ThreadIdType _type);
    // note: get(native_handle_t) overloaded to & and && to prevent implicit conversion

    bool            is_offset    = false;
    const int64_t*  causal_count = nullptr;
    index_data_t    index_data   = {};
    lifetime_data_t lifetime     = { 0, 0 };

    friend std::ostream& operator<<(std::ostream& _os, const thread_info& _v)
    {
        return (_os << _v.as_string());
    }

private:
    thread_info() = default;
};
}  // namespace rocprofsys

namespace std
{
inline std::string
to_string(const rocprofsys::thread_info& _info)
{
    return _info.as_string();
}
}  // namespace std
