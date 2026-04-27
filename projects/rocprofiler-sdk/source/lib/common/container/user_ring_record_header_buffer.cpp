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

#include "lib/common/container/user_ring_record_header_buffer.hpp"
#include "lib/common/units.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string_view>
#include <utility>

namespace rocprofiler::common::container
{
namespace
{
std::atomic<uint64_t>&
next_generation()
{
    static auto _v = std::atomic<uint64_t>{1};
    return _v;
}

struct tls_shard_entry
{
    const user_ring_record_header_buffer* owner      = nullptr;
    uint64_t                              generation = 0;
    void*                                 shard      = nullptr;
};

thread_local auto tls_shards = std::vector<tls_shard_entry>{};
}  // namespace

user_ring_record_header_buffer::user_ring_record_header_buffer()
: m_generation{next_generation().fetch_add(1, std::memory_order_acq_rel)}
{}

user_ring_record_header_buffer::user_ring_record_header_buffer(size_t num_bytes)
: user_ring_record_header_buffer{}
{
    allocate(num_bytes);
}

user_ring_record_header_buffer::~user_ring_record_header_buffer() { destroy_storage(); }

user_ring_record_header_buffer::user_ring_record_header_buffer(
    user_ring_record_header_buffer&& rhs) noexcept
: user_ring_record_header_buffer{}
{
    move_from(std::move(rhs));
}

user_ring_record_header_buffer&
user_ring_record_header_buffer::operator=(user_ring_record_header_buffer&& rhs) noexcept
{
    if(this != &rhs)
    {
        destroy_storage();
        move_from(std::move(rhs));
    }
    return *this;
}

bool
user_ring_record_header_buffer::allocate(size_t num_bytes)
{
    auto _lk = std::lock_guard<std::mutex>{m_mutex};
    if(is_allocated()) return false;

    m_capacity = page_aligned_size(num_bytes + slot_size(1, alignof(std::max_align_t)));
    m_storage  = use_shared_mmap() ? storage_mode::shared_mmap : storage_mode::private_memory;
    m_init     = true;
    return true;
}

size_t
user_ring_record_header_buffer::get_num_record_headers()
{
    return m_record_count.load(std::memory_order_acquire);
}

size_t
user_ring_record_header_buffer::clear()
{
    auto _lk = scope_destructor{[&]() { unlock(); }, [&]() { lock(); }};
    auto records = m_record_count.exchange(0, std::memory_order_acq_rel);

    auto _shards_lk = std::lock_guard<std::mutex>{m_mutex};
    for(auto& itr : m_shards)
    {
        itr->reset();
    }
    return records;
}

size_t
user_ring_record_header_buffer::reset()
{
    auto _lk     = scope_destructor{[&]() { unlock(); }, [&]() { lock(); }};
    auto records = m_record_count.load(std::memory_order_acquire);
    destroy_storage();
    return records;
}

void
user_ring_record_header_buffer::save(std::fstream& fs)
{
    auto _lk = scope_destructor{[&]() { unlock(); }, [&]() { lock(); }};

    auto record_count = m_record_count.load(std::memory_order_acquire);
    fs.write(reinterpret_cast<char*>(&m_capacity), sizeof(m_capacity));
    fs.write(reinterpret_cast<char*>(&record_count), sizeof(record_count));

    auto _shards_lk = std::lock_guard<std::mutex>{m_mutex};
    for(auto& shard_v : m_shards)
    {
        auto*  base       = static_cast<std::byte*>(shard_v->ptr);
        size_t offset     = 0;
        auto   write_size = shard_v->published.load(std::memory_order_acquire);
        while(base && offset < write_size)
        {
            auto* slot = reinterpret_cast<slot_header*>(base + offset);
            if(slot->total_size == 0 || offset + slot->total_size > write_size) break;

            auto* record = reinterpret_cast<rocprofiler_record_header_t*>(
                base + offset + slot->header_offset);
            auto* payload = base + offset + slot->payload_offset;
            auto  hash    = record->hash;

            fs.write(reinterpret_cast<char*>(&hash), sizeof(hash));
            fs.write(reinterpret_cast<char*>(&slot->payload_size), sizeof(slot->payload_size));
            fs.write(reinterpret_cast<char*>(payload), slot->payload_size);
            offset += slot->total_size;
        }
    }
}

void
user_ring_record_header_buffer::load(std::fstream& fs)
{
    auto capacity     = size_t{0};
    auto record_count = size_t{0};

    fs.read(reinterpret_cast<char*>(&capacity), sizeof(capacity));
    fs.read(reinterpret_cast<char*>(&record_count), sizeof(record_count));

    reset();
    allocate(capacity);

    auto* _shard = create_shard_for_load(record_count * slot_size(1, alignof(std::max_align_t)));
    if(!_shard) throw std::bad_alloc{};

    for(size_t i = 0; i < record_count; ++i)
    {
        auto hash         = uint64_t{0};
        auto payload_size = uint32_t{0};

        fs.read(reinterpret_cast<char*>(&hash), sizeof(hash));
        fs.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));

        auto* record = static_cast<rocprofiler_record_header_t*>(nullptr);
        auto* addr   = _shard->reserve_slot(payload_size, alignof(std::max_align_t), &record);
        if(!addr) throw std::bad_alloc{};

        fs.read(reinterpret_cast<char*>(addr), payload_size);
        record->hash    = hash;
        record->payload = addr;
        m_record_count.fetch_add(1, std::memory_order_acq_rel);
    }
}

bool
user_ring_record_header_buffer::shard::allocate(size_t bytes, storage_mode mode)
{
    destroy();
    capacity = user_ring_record_header_buffer::page_aligned_size(bytes);
    storage  = mode;

    if(storage == storage_mode::shared_mmap)
    {
        ptr = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0);
        if(ptr == MAP_FAILED) ptr = nullptr;
    }
    else
    {
        auto alignment = std::align_val_t{static_cast<size_t>(units::get_page_size())};
        ptr            = ::operator new(capacity, alignment, std::nothrow);
    }

    if(!ptr)
    {
        capacity = 0;
        return false;
    }

    std::memset(ptr, 0, capacity);
    reset();
    return true;
}

void
user_ring_record_header_buffer::shard::destroy()
{
    if(ptr)
    {
        if(storage == storage_mode::shared_mmap)
            munmap(ptr, capacity);
        else
            ::operator delete(
                ptr, std::align_val_t{static_cast<size_t>(units::get_page_size())});
    }

    ptr      = nullptr;
    capacity = 0;
    reset();
}

void
user_ring_record_header_buffer::shard::reset()
{
    if(ptr && write_count > 0) std::memset(ptr, 0, std::min(write_count, capacity));
    write_count  = 0;
    record_count = 0;
    published.store(0, std::memory_order_release);
}

void*
user_ring_record_header_buffer::shard::reserve_slot(size_t                        payload_size,
                                                    size_t                        payload_align,
                                                    rocprofiler_record_header_t** record)
{
    if(!ptr || !record) return nullptr;
    if(payload_size > std::numeric_limits<uint32_t>::max()) return nullptr;

    auto total_size = user_ring_record_header_buffer::slot_size(payload_size, payload_align);
    if(write_count + total_size > capacity) return nullptr;

    auto* base          = static_cast<std::byte*>(ptr) + write_count;
    auto  header_offset = user_ring_record_header_buffer::align_up(
        sizeof(slot_header), alignof(rocprofiler_record_header_t));
    auto payload_offset = user_ring_record_header_buffer::align_up(
        header_offset + sizeof(rocprofiler_record_header_t), payload_align);

    auto* slot           = reinterpret_cast<slot_header*>(base);
    slot->total_size     = static_cast<uint32_t>(total_size);
    slot->header_offset  = static_cast<uint32_t>(header_offset);
    slot->payload_offset = static_cast<uint32_t>(payload_offset);
    slot->payload_size   = static_cast<uint32_t>(payload_size);
    slot->sequence       = record_count;

    *record = reinterpret_cast<rocprofiler_record_header_t*>(base + header_offset);
    std::memset(*record, 0, sizeof(rocprofiler_record_header_t));

    auto* payload = base + payload_offset;
    write_count += total_size;
    ++record_count;

    slot->flags = 1;
    published.store(write_count, std::memory_order_release);
    return payload;
}

size_t
user_ring_record_header_buffer::align_up(size_t value, size_t alignment)
{
    if(alignment == 0) return value;
    auto remainder = value % alignment;
    return (remainder == 0) ? value : (value + alignment - remainder);
}

size_t
user_ring_record_header_buffer::page_aligned_size(size_t value)
{
    return align_up(value, units::get_page_size());
}

size_t
user_ring_record_header_buffer::slot_size(size_t payload_size, size_t payload_align)
{
    auto header_offset = align_up(sizeof(slot_header), alignof(rocprofiler_record_header_t));
    auto payload_offset =
        align_up(header_offset + sizeof(rocprofiler_record_header_t), payload_align);
    return align_up(payload_offset + payload_size, alignof(std::max_align_t));
}

bool
user_ring_record_header_buffer::use_shared_mmap()
{
    auto* envp = std::getenv("ROCPROFILER_EXPERIMENTAL_USER_RING_BUFFER_SHARED_MMAP");
    return envp && std::string_view{envp} != "0" && std::string_view{envp} != "false";
}

user_ring_record_header_buffer::shard*
user_ring_record_header_buffer::get_thread_shard()
{
    for(auto& itr : tls_shards)
    {
        if(itr.owner == this && itr.generation == m_generation)
            return static_cast<shard*>(itr.shard);
    }

    auto _lk = std::lock_guard<std::mutex>{m_mutex};
    auto* ret = create_shard_locked();
    if(ret) tls_shards.emplace_back(tls_shard_entry{this, m_generation, ret});
    return ret;
}

user_ring_record_header_buffer::shard*
user_ring_record_header_buffer::create_shard_locked()
{
    auto _shard = std::make_unique<shard>();
    if(!_shard->allocate(m_capacity, m_storage)) return nullptr;

    auto* ret = _shard.get();
    m_shards.emplace_back(std::move(_shard));
    return ret;
}

user_ring_record_header_buffer::shard*
user_ring_record_header_buffer::create_shard_for_load(size_t extra_bytes)
{
    auto _lk    = std::lock_guard<std::mutex>{m_mutex};
    auto _shard = std::make_unique<shard>();
    auto bytes  = std::max(m_capacity, page_aligned_size(extra_bytes + m_capacity));
    if(!_shard->allocate(bytes, m_storage)) return nullptr;

    auto* ret = _shard.get();
    m_shards.emplace_back(std::move(_shard));
    return ret;
}

void
user_ring_record_header_buffer::destroy_storage()
{
    auto _lk = std::lock_guard<std::mutex>{m_mutex};
    for(auto& itr : m_shards)
    {
        itr->destroy();
    }
    m_shards.clear();
    m_requested.store(0, std::memory_order_release);
    m_record_count.store(0, std::memory_order_release);
    m_capacity   = 0;
    m_init       = false;
    m_generation = next_generation().fetch_add(1, std::memory_order_acq_rel);
}

void
user_ring_record_header_buffer::move_from(user_ring_record_header_buffer&& rhs) noexcept
{
    auto lhs_lk = std::lock_guard<std::mutex>{m_mutex};
    auto rhs_lk = std::lock_guard<std::mutex>{rhs.m_mutex};

    m_requested.store(rhs.m_requested.load(std::memory_order_acquire), std::memory_order_release);
    m_record_count.store(rhs.m_record_count.load(std::memory_order_acquire),
                         std::memory_order_release);
    m_shards   = std::move(rhs.m_shards);
    m_capacity = rhs.m_capacity;
    m_storage  = rhs.m_storage;
    m_init     = rhs.m_init;

    rhs.m_requested.store(0, std::memory_order_release);
    rhs.m_record_count.store(0, std::memory_order_release);
    rhs.m_capacity   = 0;
    rhs.m_init       = false;
    rhs.m_generation = next_generation().fetch_add(1, std::memory_order_acq_rel);
}

size_t
user_ring_record_header_buffer::aggregate_count() const
{
    auto bytes = size_t{0};
    auto _lk   = std::lock_guard<std::mutex>{m_mutex};
    for(const auto& itr : m_shards)
    {
        bytes += itr->published.load(std::memory_order_acquire);
    }
    return bytes;
}
}  // namespace rocprofiler::common::container
