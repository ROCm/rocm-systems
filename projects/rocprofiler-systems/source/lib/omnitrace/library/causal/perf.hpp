// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/containers/c_array.hpp"
#include "library/defines.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <linux/perf_event.h>
#include <sys/types.h>

#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#    include <sys/syscall.h>
#    define gettid() syscall(SYS_gettid)
#endif

// Workaround for missing hw_breakpoint.h include file:
//   This include file just defines constants used to configure watchpoint registers.
//   This will be constant across x86 systems.
enum
{
    HW_BREAKPOINT_X = 4
};

namespace omnitrace
{
namespace causal
{
namespace perf
{
/// An enum class with all the available sampling data
enum class sample : uint64_t
{
    ip        = PERF_SAMPLE_IP,
    pid_tid   = PERF_SAMPLE_TID,
    time      = PERF_SAMPLE_TIME,
    addr      = PERF_SAMPLE_ADDR,
    id        = PERF_SAMPLE_ID,
    stream_id = PERF_SAMPLE_STREAM_ID,
    cpu       = PERF_SAMPLE_CPU,
    period    = PERF_SAMPLE_PERIOD,

#if defined(PREF_SAMPLE_READ)
    read = PERF_SAMPLE_READ,
#else
    read         = 0,
#endif

    callchain = PERF_SAMPLE_CALLCHAIN,
    raw       = PERF_SAMPLE_RAW,

#if defined(PERF_SAMPLE_BRANCH_STACK)
    branch_stack = PERF_SAMPLE_BRANCH_STACK,
#else
    branch_stack = 0,
#endif

#if defined(PERF_SAMPLE_REGS_USER)
    regs = PERF_SAMPLE_REGS_USER,
#else
    regs         = 0,
#endif

#if defined(PERF_SAMPLE_STACK_USER)
    stack = PERF_SAMPLE_STACK_USER,
#else
    stack        = 0,
#endif

    last = PERF_SAMPLE_MAX
};

struct perf_event
{
    enum class record_type;
    struct record;
    struct sample_record;

    /// Default constructor
    perf_event() = default;
    /// Move constructor
    perf_event(perf_event&& other) noexcept;

    /// Close the perf event file and unmap the ring buffer
    ~perf_event();

    /// Move assignment is supported
    perf_event& operator=(perf_event&& other) noexcept;

    perf_event(const perf_event&) = delete;
    perf_event& operator=(const perf_event&) = delete;

    /// Open a perf_event file using the given options structure
    bool open(struct perf_event_attr& pe, pid_t pid = 0, int cpu = -1);
    bool open(double, uint32_t, pid_t pid = 0, int cpu = -1);

    /// Read event count
    uint64_t get_count() const;

    /// Start counting events and collecting samples
    void start() const;

    /// Stop counting events
    void stop() const;

    /// Close the perf_event file and unmap the ring buffer
    void close();

    /// Configure the perf_event file to deliver a signal when samples are ready to be
    /// processed
    void set_ready_signal(int sig) const;

    /// Check if this perf_event was configured to collect a type of sample data
    inline bool is_sampling(sample s) const
    {
        return (m_sample_type & static_cast<uint64_t>(s)) != 0u;
    }

    /// Get the configuration for this perf_event's read format
    inline uint64_t get_read_format() const { return m_read_format; }

    /// An enum to distinguish types of records in the mmapped ring buffer
    enum class record_type
    {
        mmap       = PERF_RECORD_MMAP,
        lost       = PERF_RECORD_LOST,
        comm       = PERF_RECORD_COMM,
        exit       = PERF_RECORD_EXIT,
        throttle   = PERF_RECORD_THROTTLE,
        unthrottle = PERF_RECORD_UNTHROTTLE,
        fork       = PERF_RECORD_FORK,
        read       = PERF_RECORD_READ,
        sample     = PERF_RECORD_SAMPLE,

#if defined(PERF_RECORD_MMAP2)
        mmap2 = PERF_RECORD_MMAP2
#else
        mmap2 = 0
#endif
    };

    class iterator;

    /// A generic record type
    struct record
    {
        friend class perf_event::iterator;

        record_type get_type() const { return static_cast<record_type>(m_header->type); }

        inline bool is_mmap() const { return get_type() == record_type::mmap; }
        inline bool is_lost() const { return get_type() == record_type::lost; }
        inline bool is_comm() const { return get_type() == record_type::comm; }
        inline bool is_exit() const { return get_type() == record_type::exit; }
        inline bool is_throttle() const { return get_type() == record_type::throttle; }
        inline bool is_unthrottle() const
        {
            return get_type() == record_type::unthrottle;
        }
        inline bool is_fork() const { return get_type() == record_type::fork; }
        inline bool is_read() const { return get_type() == record_type::read; }
        inline bool is_sample() const { return get_type() == record_type::sample; }
        inline bool is_mmap2() const { return get_type() == record_type::mmap2; }

        uint64_t                     get_ip() const;
        uint64_t                     get_pid() const;
        uint64_t                     get_tid() const;
        uint64_t                     get_time() const;
        uint32_t                     get_cpu() const;
        container::c_array<uint64_t> get_callchain() const;

    private:
        record(const perf_event& source, struct perf_event_header* header)
        : m_source(source)
        , m_header(header)
        {}

        template <sample SampleT, typename Tp = void*>
        Tp locate_field() const;

        const perf_event&         m_source;
        struct perf_event_header* m_header;
    };

    class iterator
    {
    public:
        iterator(perf_event& source, struct perf_event_mmap_page* mapping);
        ~iterator();

        void   next();
        record get();
        bool   has_data() const;

        iterator& operator++();
        record    operator*() { return get(); }
        bool      operator!=(const iterator& other) const;

    private:
        perf_event&                  m_source;
        size_t                       m_index   = 0;
        size_t                       m_head    = 0;
        struct perf_event_mmap_page* m_mapping = nullptr;

        // Buffer to hold the current record. Just a hack until records play nice with the
        // ring buffer
        uint8_t _buf[4096];
    };

    /// Get an iterator to the beginning of the memory mapped ring buffer
    iterator begin() { return iterator(*this, m_mapping); }

    // Get an iterator to the end of the memory mapped ring buffer
    iterator end() { return iterator(*this, nullptr); }

private:
    // Copy data out of the mmap ring buffer
    static void copy_from_ring_buffer(struct perf_event_mmap_page* mapping,
                                      ptrdiff_t index, void* dest, size_t bytes);

    /// File descriptor for the perf event
    long m_fd = -1;

    /// Memory mapped perf event region
    struct perf_event_mmap_page* m_mapping = nullptr;

    /// The sample type from this perf_event's configuration
    uint64_t m_sample_type = 0;
    /// The read format from this perf event's configuration
    uint64_t m_read_format = 0;
};
}  // namespace perf
}  // namespace causal
}  // namespace omnitrace
