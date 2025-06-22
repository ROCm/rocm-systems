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

#include "core/containers/c_array.hpp"
#include "core/defines.hpp"
#include "core/locking.hpp"
#include "core/perf.hpp"

#include <timemory/backends/papi.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <linux/perf_event.h>
#include <regex>
#include <set>
#include <string>
#include <sys/types.h>

namespace rocprofsys
{
namespace perf
{
struct perf_event
{
    static constexpr uint32_t max_batch_size = 32;

    struct record;
    struct sample_record;
    class iterator;

    /// Default constructor
    perf_event() = default;
    /// Move constructor
    perf_event(perf_event&& other) noexcept;

    /// Close the perf event file and unmap the ring buffer
    ~perf_event();

    /// Move assignment is supported
    perf_event& operator=(perf_event&& other) noexcept;

    perf_event(const perf_event&)            = delete;
    perf_event& operator=(const perf_event&) = delete;

    /// Open a perf_event file using the given options structure
    std::optional<std::string> open(struct perf_event_attr& pe, pid_t pid = 0,
                                    int cpu = -1);
    std::optional<std::string> open(double, uint32_t = 0, pid_t pid = 0, int cpu = -1);

    /// Return file descriptor
    long get_fileno() const;

    /// Read event count
    uint64_t get_count() const;

    /// Get the batch size
    uint32_t get_batch_size() const { return m_batch_size; }

    /// Start counting events and collecting samples
    bool start() const;

    /// Stop counting events
    bool stop() const;

    /// Check if counting events and collecting samples
    bool is_open() const;

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

    /// A generic record type
    struct record
    {
        friend class perf_event::iterator;

        record()                             = default;
        ~record()                            = default;
        record(const record&)                = default;
        record(record&&) noexcept            = default;
        record& operator=(const record&)     = default;
        record& operator=(record&&) noexcept = default;

        bool is_valid() const { return (m_source != nullptr && m_header != nullptr); }
        operator bool() const { return is_valid(); }

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
        uint64_t                     get_period() const;
        uint32_t                     get_cpu() const;
        container::c_array<uint64_t> get_callchain() const;

    private:
        record(const perf_event* source, struct perf_event_header* header)
        : m_source(source)
        , m_header(header)
        {}

        template <sample SampleT, typename Tp = void*>
        Tp locate_field() const;

        const perf_event*         m_source = nullptr;
        struct perf_event_header* m_header = nullptr;
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

    /// Get an iterator to the end of the memory mapped ring buffer
    iterator end() { return iterator(*this, nullptr); }

private:
    // Copy data out of the mmap ring buffer
    static void copy_from_ring_buffer(struct perf_event_mmap_page* mapping,
                                      ptrdiff_t index, void* dest, size_t bytes);

    uint32_t m_batch_size = 10;

    /// File descriptor for the perf event
    long m_fd = -1;

    /// Memory mapped perf event region
    struct perf_event_mmap_page* m_mapping = nullptr;

    /// The sample type from this perf_event's configuration
    uint64_t m_sample_type = 0;
    /// The read format from this perf event's configuration
    uint64_t m_read_format = 0;
};

/// provides thread-local instance of perf_event
std::unique_ptr<perf_event>&
get_instance(int64_t _tid);
}  // namespace perf
}  // namespace rocprofsys
