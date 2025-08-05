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

#include "library/perf.hpp"
#include "core/debug.hpp"
#include "core/locking.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "library/thread_data.hpp"

#include <timemory/log/logger.hpp>
#include <timemory/log/macros.hpp>
#include <timemory/units.hpp>

#include <asm/unistd.h>
#include <ctime>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <mutex>
#include <poll.h>
#include <regex>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#if !defined(ROCPROFSYS_RETURN_ERROR_MSG)
#    define ROCPROFSYS_RETURN_ERROR_MSG(COND, ...)                                       \
        if((COND))                                                                       \
        {                                                                                \
            auto _msg_ss = std::stringstream{};                                          \
            _msg_ss << __VA_ARGS__;                                                      \
            return std::optional<std::string>{ _msg_ss.str() };                          \
        }
#endif

#if !defined(ROCPROFSYS_FATAL)
#    define ROCPROFSYS_FATAL TIMEMORY_FATAL
#endif

#if !defined(ROCPROFSYS_ASSERT)
#    define ROCPROFSYS_ASSERT(COND) (COND) ? ::tim::log::base() : TIMEMORY_FATAL
#endif

namespace rocprofsys
{
namespace perf
{
namespace
{
struct SizeParams
{
    const size_t num_pages = 2;
    const size_t page      = units::get_page_size();
    const size_t data      = num_pages * page;
    const size_t mmap      = data + page;
};
const SizeParams sizes = {};
}  // namespace

long
perf_event_open(struct perf_event_attr* hw_event, pid_t _pid, int _cpu, int group_fd,
                unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, _pid, _cpu, group_fd, flags);
}

/// Move constructor
perf_event::perf_event(perf_event&& rhs) noexcept
{
    // Release resources if the current perf_event is initialized and not equal to this
    // one
    if(m_fd != -1 && m_fd != rhs.m_fd)
    {
        ::close(m_fd);
        ROCPROFSYS_VERBOSE(1, "Closed perf event fd %li\n", m_fd);
    }

    if(m_mapping != nullptr && m_mapping != rhs.m_mapping) munmap(m_mapping, sizes.mmap);

    // take rhs perf event's file descriptor and replace it with -1
    m_fd     = rhs.m_fd;
    rhs.m_fd = -1;

    // take rhs perf_event's mapping and replace it with nullptr
    m_mapping     = rhs.m_mapping;
    rhs.m_mapping = nullptr;

    // Copy over the sample type and read format
    m_sample_type = rhs.m_sample_type;
    m_read_format = rhs.m_read_format;
}

/// Close the perf_event file descriptor and unmap the ring buffer
perf_event::~perf_event() { close(); }

/// Move assignment
perf_event&
perf_event::operator=(perf_event&& rhs) noexcept
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
    if(&rhs == this) return *this;

    // Release resources if the current perf_event is initialized and not equal to this
    // one
    if(m_fd != -1 && m_fd != rhs.m_fd) ::close(m_fd);
    if(m_mapping != nullptr && m_mapping != rhs.m_mapping) munmap(m_mapping, sizes.mmap);

    // take rhs perf event's file descriptor and replace it with -1
    m_fd     = rhs.m_fd;
    rhs.m_fd = -1;

    // take rhs perf_event's mapping and replace it with nullptr
    m_mapping     = rhs.m_mapping;
    rhs.m_mapping = nullptr;

    // Copy over the sample type and read format
    m_sample_type = rhs.m_sample_type;
    m_read_format = rhs.m_read_format;

    return *this;
}

// Open a perf_event file and map it (if sampling is enabled)
std::optional<std::string>
perf_event::open(struct perf_event_attr& _pe, pid_t _pid, int _cpu)
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
    m_sample_type = _pe.sample_type;
    m_read_format = _pe.read_format;
    m_batch_size  = _pe.wakeup_events;

    // Set some mandatory fields
    _pe.size     = sizeof(struct perf_event_attr);
    _pe.disabled = 1;

    // Open the file
    m_fd = perf_event_open(&_pe, _pid, _cpu, -1, 0);
    if(m_fd == -1)
    {
        std::string path = "/proc/sys/kernel/perf_event_paranoid";

        auto file = std::ifstream{ path.c_str() };

        ROCPROFSYS_RETURN_ERROR_MSG(!file,
                                    "Failed to open " << path << ": " << strerror(errno));

        int value = 4;
        file >> value;

        ROCPROFSYS_RETURN_ERROR_MSG(file.bad(), "Failed to read from "
                                                    << path << ": " << strerror(errno));

        ROCPROFSYS_RETURN_ERROR_MSG(
            true, "Failed to open perf event. Consider tweaking "
                      << path << " to 2 or less " << "(current value is " << value
                      << "), "
                      << "or run rocprof-sys as a privileged user (with CAP_SYS_ADMIN).");
    }

    // If sampling, map the perf event file
    if(_pe.sample_type != 0 && _pe.sample_period != 0)
    {
        void* ring_buffer =
            mmap(nullptr, sizes.mmap, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);

        ROCPROFSYS_RETURN_ERROR_MSG(
            ring_buffer == MAP_FAILED,
            "Mapping perf_event ring buffer failed. Make sure the current user has "
            "permission to invoke the perf tool, and that the program being profiled "
            "does not use an excessive number of threads (>1000)");

        m_mapping = reinterpret_cast<struct perf_event_mmap_page*>(ring_buffer);
    }

    return std::optional<std::string>{};
}

std::optional<std::string>
perf_event::open(double _freq, uint32_t _batch_size, pid_t _pid, int _cpu)
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
    uint64_t               _period = (1.0 / _freq) * units::sec;
    struct perf_event_attr _pe;

    if(_batch_size > 0)
        m_batch_size = _batch_size;
    else
        _batch_size = m_batch_size;

    memset(&_pe, 0, sizeof(_pe));
    _pe.type           = PERF_TYPE_SOFTWARE;
    _pe.config         = PERF_COUNT_SW_TASK_CLOCK;
    _pe.sample_type    = PERF_SAMPLE_IP | PERF_SAMPLE_CALLCHAIN;
    _pe.sample_period  = _period;
    _pe.wakeup_events  = _batch_size;
    _pe.exclude_idle   = 1;
    _pe.exclude_kernel = 1;
    _pe.disabled       = 1;
    // potential additions
    _pe.inherit                  = 0;
    _pe.exclude_hv               = 1;
    _pe.exclude_callchain_kernel = 1;
    _pe.use_clockid              = 1;
    _pe.clockid                  = CLOCK_REALTIME;
    // _pe.precise_ip               = 0;
    // _pe.exclusive                = 1;
    // _pe.pinned                   = 1;

    return open(_pe, _pid, _cpu);
}

/// Read event count
long
perf_event::get_fileno() const
{
    return m_fd;
}

/// Read event count
uint64_t
perf_event::get_count() const
{
    uint64_t count;
    ROCPROFSYS_REQUIRE(read(m_fd, &count, sizeof(uint64_t)) == sizeof(uint64_t))
        << "Failed to read event count from perf_event file";
    return count;
}

/// Start counting events
bool
perf_event::start() const
{
    if(m_fd != -1)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        ROCPROFSYS_REQUIRE(ioctl(m_fd, PERF_EVENT_IOC_ENABLE, 0) != -1)
            << "Failed to start perf event: " << strerror(errno);
    }
    return (m_fd != -1);
}

/// Stop counting events
bool
perf_event::stop() const
{
    if(m_fd != -1)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        ROCPROFSYS_REQUIRE(ioctl(m_fd, PERF_EVENT_IOC_DISABLE, 0) != -1)
            << "Failed to stop perf event: " << strerror(errno) << " (" << m_fd << ")";
    }
    return (m_fd != -1);
}

bool
perf_event::is_open() const
{
    return (m_fd != -1);
}

void
perf_event::close()
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
    stop();

    if(m_fd != -1)
    {
        ::close(m_fd);
        m_fd = -1;
    }

    if(m_mapping != nullptr)
    {
        munmap(m_mapping, sizes.mmap);
        m_mapping = nullptr;
    }
}

void
perf_event::set_ready_signal(int sig) const
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
    // Set the perf_event file to async
    ROCPROFSYS_REQUIRE(fcntl(m_fd, F_SETFL, fcntl(m_fd, F_GETFL, 0) | O_ASYNC) != -1)
        << "failed to set perf_event file to async mode";

    // Set the notification signal for the perf file
    ROCPROFSYS_REQUIRE(fcntl(m_fd, F_SETSIG, sig) != -1)
        << "failed to set perf_event file signal";

    // Set the current thread as the owner of the file (to target signal delivery)
    ROCPROFSYS_REQUIRE(fcntl(m_fd, F_SETOWN, gettid()) != -1)
        << "failed to set the owner of the perf_event file";
}

void
perf_event::iterator::next()
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    struct perf_event_header _hdr;

    // Copy out the record header
    perf_event::copy_from_ring_buffer(m_mapping, m_index, &_hdr,
                                      sizeof(struct perf_event_header));

    // Advance to the next record
    m_index += _hdr.size;
}

perf_event::iterator::iterator(perf_event& _source, struct perf_event_mmap_page* _mapping)
: m_source{ _source }
, m_mapping{ _mapping }
{
    if(_mapping != nullptr)
    {
        m_index = _mapping->data_tail;
        m_head  = _mapping->data_head;
    }
    else
    {
        m_index = 0;
        m_head  = 0;
    }
}

perf_event::iterator::~iterator()
{
    if(m_mapping != nullptr)
    {
        m_mapping->data_tail = m_index;
    }
}

perf_event::iterator&
perf_event::iterator::operator++()
{
    next();
    return *this;
}

bool
perf_event::iterator::operator!=(const iterator& other) const
{
    return has_data() != other.has_data();
}

perf_event::record
perf_event::iterator::get()
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    // Copy out the record header
    perf_event::copy_from_ring_buffer(m_mapping, m_index, _buf,
                                      sizeof(struct perf_event_header));

    // Get a pointer to the header
    struct perf_event_header* header = reinterpret_cast<struct perf_event_header*>(_buf);

    // Copy out the entire record
    perf_event::copy_from_ring_buffer(m_mapping, m_index, _buf, header->size);

    return perf_event::record(&m_source, header);
}

bool
perf_event::iterator::has_data() const
{
    // If there is no ring buffer, there is no data
    if(m_mapping == nullptr)
    {
        return false;
    }

    // If there isn't enough data in the ring buffer to hold a header, there is no data
    if(m_index + sizeof(struct perf_event_header) >= m_head)
    {
        return false;
    }

    struct perf_event_header _hdr;
    perf_event::copy_from_ring_buffer(m_mapping, m_index, &_hdr,
                                      sizeof(struct perf_event_header));

    // If the first record is larger than the available data, nothing can be read
    if(m_index + _hdr.size > m_head)
    {
        return false;
    }

    return true;
}

void
perf_event::copy_from_ring_buffer(struct perf_event_mmap_page* _mapping, ptrdiff_t _index,
                                  void* _dest, size_t _nbytes)
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    uintptr_t _base    = reinterpret_cast<uintptr_t>(_mapping) + sizes.page;
    size_t    _beg_idx = _index % sizes.data;
    size_t    _end_idx = _beg_idx + _nbytes;

    if(_end_idx <= sizes.data)
    {
        memcpy(_dest, reinterpret_cast<void*>(_base + _beg_idx), _nbytes);
    }
    else
    {
        size_t _chunk_size2 = _end_idx - sizes.data;
        size_t _chunk_size1 = _nbytes - _chunk_size2;

        void* _dest2 =
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(_dest) + _chunk_size1);

        memcpy(_dest, reinterpret_cast<void*>(_base + _beg_idx), _chunk_size1);
        memcpy(_dest2, reinterpret_cast<void*>(_base), _chunk_size2);
    }
}

uint64_t
perf_event::record::get_ip() const
{
    ROCPROFSYS_ASSERT(is_sample() && m_source != nullptr &&
                      m_source->is_sampling(sample::ip))
        << "Record does not have an ip field (" << is_sample() << "|" << m_source << ")";
    return *locate_field<sample::ip, uint64_t*>();
}

uint64_t
perf_event::record::get_pid() const
{
    ROCPROFSYS_ASSERT(is_sample() && m_source != nullptr &&
                      m_source->is_sampling(sample::pid_tid))
        << "Record does not have a `pid` field (" << is_sample() << "|" << m_source
        << ")";
    return locate_field<sample::pid_tid, uint32_t*>()[0];
}

uint64_t
perf_event::record::get_tid() const
{
    ROCPROFSYS_ASSERT(is_sample() && m_source != nullptr &&
                      m_source->is_sampling(sample::pid_tid))
        << "Record does not have a `tid` field (" << is_sample() << "|" << m_source
        << ")";
    return locate_field<sample::pid_tid, uint32_t*>()[1];
}

uint64_t
perf_event::record::get_time() const
{
    ROCPROFSYS_ASSERT(is_sample() && m_source != nullptr &&
                      m_source->is_sampling(sample::time))
        << "Record does not have a 'time' field (" << is_sample() << "|" << m_source
        << ")";
    return *locate_field<sample::time, uint64_t*>();
}

uint64_t
perf_event::record::get_period() const
{
    ROCPROFSYS_ASSERT(is_sample() && m_source != nullptr &&
                      m_source->is_sampling(sample::period))
        << "Record does not have a 'period' field (" << is_sample() << "|" << m_source
        << ")";
    return *locate_field<sample::period, uint64_t*>();
}

uint32_t
perf_event::record::get_cpu() const
{
    ROCPROFSYS_ASSERT(is_sample() && m_source != nullptr &&
                      m_source->is_sampling(sample::cpu))
        << "Record does not have a 'cpu' field (" << is_sample() << "|" << m_source
        << ")";
    return *locate_field<sample::cpu, uint32_t*>();
}

container::c_array<uint64_t>
perf_event::record::get_callchain() const
{
    ROCPROFSYS_ASSERT(is_sample() && m_source != nullptr &&
                      m_source->is_sampling(sample::callchain))
        << "Record does not have a callchain field (" << is_sample() << "|" << m_source
        << ")";

    uint64_t* _base = locate_field<sample::callchain, uint64_t*>();
    uint64_t  _size = *_base;
    // Advance the callchain array pointer past the size
    ++_base;
    return container::wrap_c_array(_base, _size);
}

template <sample SampleT, typename Tp>
Tp
perf_event::record::locate_field() const
{
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    uintptr_t p =
        reinterpret_cast<uintptr_t>(m_header) + sizeof(struct perf_event_header);

    // Walk through the fields in the sample structure. Once the requested field is
    // reached, return. Skip past any unrequested fields that are included in the sample
    // type

    // ip
    if constexpr(SampleT == sample::ip) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::ip)) p += sizeof(uint64_t);

    // pid, tid
    if constexpr(SampleT == sample::pid_tid) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::pid_tid))
        p += sizeof(uint32_t) + sizeof(uint32_t);

    // time
    if constexpr(SampleT == sample::time) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::time)) p += sizeof(uint64_t);

    // addr
    if constexpr(SampleT == sample::addr) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::addr)) p += sizeof(uint64_t);

    // id
    if constexpr(SampleT == sample::id) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::id)) p += sizeof(uint64_t);

    // stream_id
    if constexpr(SampleT == sample::stream_id) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::stream_id))
        p += sizeof(uint64_t);

    // cpu
    if constexpr(SampleT == sample::cpu) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::cpu))
        p += sizeof(uint32_t) + sizeof(uint32_t);

    // period
    if constexpr(SampleT == sample::period) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::period))
        p += sizeof(uint64_t);

    // value
    if constexpr(SampleT == sample::read) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::read))
    {
        uint64_t read_format = m_source->get_read_format();
        if(read_format & PERF_FORMAT_GROUP)
        {
            // Get the number of values in the read format structure
            uint64_t nr = *reinterpret_cast<uint64_t*>(p);
            // The default size of each entry is a u64
            size_t sz = sizeof(uint64_t);
            // If requested, the id will be included with each value
            if(read_format & PERF_FORMAT_ID) sz += sizeof(uint64_t);
            // Skip over the entry count, and each entry
            p += sizeof(uint64_t) + nr * sz;
        }
        else
        {
            // Skip over the value
            p += sizeof(uint64_t);
            // Skip over the id, if included
            if(read_format & PERF_FORMAT_ID) p += sizeof(uint64_t);
        }

        // Skip over the time_enabled field
        if(read_format & PERF_FORMAT_TOTAL_TIME_ENABLED) p += sizeof(uint64_t);
        // Skip over the time_running field
        if(read_format & PERF_FORMAT_TOTAL_TIME_RUNNING) p += sizeof(uint64_t);
    }

    // callchain
    if constexpr(SampleT == sample::callchain) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::callchain))
    {
        uint64_t nr = *reinterpret_cast<uint64_t*>(p);
        p += sizeof(uint64_t) + (nr * sizeof(uint64_t));
    }

    // raw
    if constexpr(SampleT == sample::raw) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::raw))
    {
        uint32_t raw_size = *reinterpret_cast<uint32_t*>(p);
        p += sizeof(uint32_t) + raw_size;
    }

    // branch_stack
    if constexpr(SampleT == sample::branch_stack) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::branch_stack))
        ROCPROFSYS_FATAL << "Branch stack sampling is not supported";

    // regs
    if constexpr(SampleT == sample::regs) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::regs))
        ROCPROFSYS_FATAL << "Register sampling is not supported";

    // stack
    if constexpr(SampleT == sample::stack) return reinterpret_cast<Tp>(p);
    if(m_source != nullptr && m_source->is_sampling(sample::stack))
        ROCPROFSYS_FATAL << "Stack sampling is not supported";

    // end
    if constexpr(SampleT == sample::last) return reinterpret_cast<Tp>(p);

    ROCPROFSYS_FATAL << "Unsupported sample field requested!";

    if constexpr(std::is_pointer<Tp>::value)
        return nullptr;
    else
        return Tp{};
}

namespace
{
inline auto&
get_instances()
{
    using thread_data_t = thread_data<identity<std::unique_ptr<perf_event>>, perf_event>;
    static auto& _v     = thread_data_t::instance(construct_on_init{});
    return _v;
}
}  // namespace

std::unique_ptr<perf_event>&
get_instance(int64_t _tid)
{
    static auto nullInstance = std::unique_ptr<perf_event>{ nullptr };
    auto&       _data        = get_instances();

    // If get_instances() returned an empty object, we have to return a reference to a
    // static null instance, or else we will crash.
    if(_data == nullptr)
    {
        return nullInstance;
    }

    if(static_cast<size_t>(_tid) >= _data->size())
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        _data->resize(_tid + 1);
    }
    return _data->at(_tid);
}
}  // namespace perf
}  // namespace rocprofsys
