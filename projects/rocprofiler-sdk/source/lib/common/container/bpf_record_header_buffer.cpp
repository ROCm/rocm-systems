// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/container/bpf_record_header_buffer.hpp"
#include "lib/common/units.hpp"

#include <sys/mman.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace rocprofiler::common::container
{
bpf_record_header_buffer::bpf_record_header_buffer(size_t num_bytes) { allocate(num_bytes); }

bpf_record_header_buffer::~bpf_record_header_buffer() { destroy_storage(); }

bpf_record_header_buffer::bpf_record_header_buffer(bpf_record_header_buffer&& rhs) noexcept
{
    move_from(std::move(rhs));
}

bpf_record_header_buffer&
bpf_record_header_buffer::operator=(bpf_record_header_buffer&& rhs) noexcept
{
    if(this != &rhs)
    {
        destroy_storage();
        move_from(std::move(rhs));
    }
    return *this;
}

bool
bpf_record_header_buffer::allocate(size_t num_bytes)
{
    if(is_allocated()) return false;

    auto  total_size = page_aligned_size(num_bytes + slot_size(1, alignof(std::max_align_t)));
    auto* ptr =
        mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

    if(ptr == MAP_FAILED)
    {
        m_ptr      = nullptr;
        m_capacity = 0;
        m_init     = false;
        return false;
    }

    m_ptr      = ptr;
    m_capacity = total_size;
    m_init     = true;
    return true;
}

size_t
bpf_record_header_buffer::get_num_record_headers()
{
    return m_record_count.load(std::memory_order_acquire);
}

size_t
bpf_record_header_buffer::clear()
{
    auto _lk = scope_destructor{[&]() { unlock(); }, [&]() { lock(); }};

    auto records = m_record_count.load(std::memory_order_acquire);
    auto bytes   = m_write_count.load(std::memory_order_acquire);
    if(m_ptr && bytes > 0) std::memset(m_ptr, 0, std::min(bytes, m_capacity));

    m_write_count.store(0, std::memory_order_release);
    m_record_count.store(0, std::memory_order_release);
    return records;
}

size_t
bpf_record_header_buffer::reset()
{
    auto records = m_record_count.load(std::memory_order_acquire);
    destroy_storage();
    return records;
}

void
bpf_record_header_buffer::save(std::fstream& fs)
{
    auto _lk = scope_destructor{[&]() { unlock(); }, [&]() { lock(); }};

    auto write_count  = m_write_count.load(std::memory_order_acquire);
    auto record_count = m_record_count.load(std::memory_order_acquire);

    fs.write(reinterpret_cast<char*>(&m_capacity), sizeof(m_capacity));
    fs.write(reinterpret_cast<char*>(&write_count), sizeof(write_count));
    fs.write(reinterpret_cast<char*>(&record_count), sizeof(record_count));
    if(m_ptr && m_capacity > 0) fs.write(reinterpret_cast<char*>(m_ptr), m_capacity);
}

void
bpf_record_header_buffer::load(std::fstream& fs)
{
    auto capacity     = size_t{0};
    auto write_count  = size_t{0};
    auto record_count = size_t{0};

    fs.read(reinterpret_cast<char*>(&capacity), sizeof(capacity));
    fs.read(reinterpret_cast<char*>(&write_count), sizeof(write_count));
    fs.read(reinterpret_cast<char*>(&record_count), sizeof(record_count));

    reset();
    auto  total_size = page_aligned_size(capacity);
    auto* ptr =
        mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if(ptr == MAP_FAILED) throw std::bad_alloc{};

    m_ptr      = ptr;
    m_capacity = total_size;
    m_init     = true;
    fs.read(reinterpret_cast<char*>(m_ptr), m_capacity);
    m_write_count.store(write_count, std::memory_order_release);
    m_record_count.store(record_count, std::memory_order_release);
}

size_t
bpf_record_header_buffer::align_up(size_t value, size_t alignment)
{
    if(alignment == 0) return value;
    auto remainder = value % alignment;
    return (remainder == 0) ? value : (value + alignment - remainder);
}

size_t
bpf_record_header_buffer::page_aligned_size(size_t value)
{
    return align_up(value, units::get_page_size());
}

size_t
bpf_record_header_buffer::slot_size(size_t payload_size, size_t payload_align)
{
    auto header_offset = align_up(sizeof(slot_header), alignof(rocprofiler_record_header_t));
    auto payload_offset =
        align_up(header_offset + sizeof(rocprofiler_record_header_t), payload_align);
    return align_up(payload_offset + payload_size, alignof(std::max_align_t));
}

void*
bpf_record_header_buffer::reserve_slot(size_t                        payload_size,
                                       size_t                        payload_align,
                                       rocprofiler_record_header_t** record)
{
    if(!m_ptr || !record) return nullptr;
    if(payload_size > std::numeric_limits<uint32_t>::max()) return nullptr;

    auto total_size = slot_size(payload_size, payload_align);
    if(total_size > m_capacity) return nullptr;

    auto write_count = size_t{0};
    do
    {
        write_count = m_write_count.load(std::memory_order_acquire);
        if(write_count + total_size > m_capacity) return nullptr;
    } while(!m_write_count.compare_exchange_strong(
        write_count, write_count + total_size, std::memory_order_acq_rel));

    auto* base          = static_cast<std::byte*>(m_ptr) + write_count;
    auto  header_offset = align_up(sizeof(slot_header), alignof(rocprofiler_record_header_t));
    auto  payload_offset =
        align_up(header_offset + sizeof(rocprofiler_record_header_t), payload_align);

    auto* slot           = reinterpret_cast<slot_header*>(base);
    slot->total_size     = static_cast<uint32_t>(total_size);
    slot->header_offset  = static_cast<uint32_t>(header_offset);
    slot->payload_offset = static_cast<uint32_t>(payload_offset);
    slot->payload_size   = static_cast<uint32_t>(payload_size);

    *record = reinterpret_cast<rocprofiler_record_header_t*>(base + header_offset);
    std::memset(*record, 0, sizeof(rocprofiler_record_header_t));

    return base + payload_offset;
}

void
bpf_record_header_buffer::destroy_storage()
{
    if(m_ptr && m_capacity > 0)
    {
        munmap(m_ptr, m_capacity);
    }

    m_requested.store(0, std::memory_order_release);
    m_locked.store(0, std::memory_order_release);
    m_write_count.store(0, std::memory_order_release);
    m_record_count.store(0, std::memory_order_release);
    m_ptr      = nullptr;
    m_capacity = 0;
    m_init     = false;
}

void
bpf_record_header_buffer::move_from(bpf_record_header_buffer&& rhs) noexcept
{
    m_requested.store(rhs.m_requested.load(std::memory_order_acquire), std::memory_order_release);
    m_locked.store(rhs.m_locked.load(std::memory_order_acquire), std::memory_order_release);
    m_write_count.store(rhs.m_write_count.load(std::memory_order_acquire),
                        std::memory_order_release);
    m_record_count.store(rhs.m_record_count.load(std::memory_order_acquire),
                         std::memory_order_release);
    m_ptr      = rhs.m_ptr;
    m_capacity = rhs.m_capacity;
    m_init     = rhs.m_init;

    rhs.m_requested.store(0, std::memory_order_release);
    rhs.m_locked.store(0, std::memory_order_release);
    rhs.m_write_count.store(0, std::memory_order_release);
    rhs.m_record_count.store(0, std::memory_order_release);
    rhs.m_ptr      = nullptr;
    rhs.m_capacity = 0;
    rhs.m_init     = false;
}
}  // namespace rocprofiler::common::container
