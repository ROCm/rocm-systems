// MIT License
//
/* Copyright (c) 2026 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#include "lib/common/environment.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace rocprofiler
{
namespace common
{
namespace lite_trace
{
inline bool
enabled()
{
    return ::rocprofiler::common::get_env("ROCPROF_LITE_TRACE", false);
}

inline bool
is_allowed_buffer_tracing_kind(rocprofiler_buffer_tracing_kind_t kind)
{
    return kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;
}

inline bool
is_allowed_callback_tracing_kind(rocprofiler_callback_tracing_kind_t kind)
{
    return kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
}

struct kernel_dispatch_record_t
{
    rocprofiler_thread_id_t            thread_id = 0;
    rocprofiler_timestamp_t            start     = 0;
    rocprofiler_timestamp_t            end       = 0;
    rocprofiler_kernel_dispatch_info_t dispatch_info{};
};

constexpr uint64_t record_store_magic    = 0x52504c4954453031ULL;
constexpr uint64_t record_store_capacity = 1ULL << 20;

struct record_store_header_t
{
    uint64_t magic    = 0;
    uint64_t pid      = 0;
    uint64_t capacity = 0;
    uint64_t count    = 0;
};

struct mapped_record_store_t
{
    record_store_header_t*    header  = nullptr;
    kernel_dispatch_record_t* records = nullptr;
    size_t                    bytes   = 0;
};

inline size_t
record_store_bytes()
{
    return sizeof(record_store_header_t) +
           (record_store_capacity * sizeof(kernel_dispatch_record_t));
}

inline std::array<char, 128>
record_store_path(const char* directory)
{
    auto path = std::array<char, 128>{};
    std::snprintf(path.data(),
                  path.size(),
                  "%s/rocprofiler-lite-trace-%ld.records",
                  directory,
                  long{getpid()});
    return path;
}

inline mapped_record_store_t
map_record_store_at(const std::array<char, 128>& path, bool create_if_missing)
{
    const auto bytes = record_store_bytes();
    auto       flags = O_RDWR | O_CLOEXEC;

    if(create_if_missing) flags |= O_CREAT;

    auto fd = ::open(path.data(), flags, S_IRUSR | S_IWUSR);
    if(fd < 0) return {};

    auto cleanup_fd = [&fd]() {
        if(fd >= 0)
        {
            ::close(fd);
            fd = -1;
        }
    };

    struct stat statbuf = {};
    if(::fstat(fd, &statbuf) != 0)
    {
        cleanup_fd();
        return {};
    }

    const bool resize = statbuf.st_size != static_cast<off_t>(bytes);
    if(resize && create_if_missing && ::ftruncate(fd, static_cast<off_t>(bytes)) != 0)
    {
        cleanup_fd();
        return {};
    }

    if(resize && !create_if_missing)
    {
        cleanup_fd();
        return {};
    }

    void* mapping = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    cleanup_fd();

    if(mapping == MAP_FAILED) return {};

    auto* header  = static_cast<record_store_header_t*>(mapping);
    auto* records = reinterpret_cast<kernel_dispatch_record_t*>(header + 1);

    const auto pid = static_cast<uint64_t>(getpid());
    if(create_if_missing || header->magic != record_store_magic || header->pid != pid ||
       header->capacity != record_store_capacity)
    {
        std::memset(mapping, 0, bytes);
        header->magic    = record_store_magic;
        header->pid      = pid;
        header->capacity = record_store_capacity;
        __atomic_store_n(&header->count, 0, __ATOMIC_RELAXED);
    }

    return mapped_record_store_t{header, records, bytes};
}

inline mapped_record_store_t
map_record_store(bool create_if_missing)
{
    for(const auto* directory : {"/dev/shm", "/tmp"})
    {
        auto store = map_record_store_at(record_store_path(directory), create_if_missing);
        if(store.header && store.records) return store;
    }

    return {};
}

inline mapped_record_store_t&
append_record_store()
{
    static auto store = map_record_store(true);
    return store;
}

inline mapped_record_store_t&
read_record_store()
{
    static auto store = map_record_store(false);
    return store;
}

inline bool
record_store_available()
{
    auto& store = append_record_store();
    return store.header && store.records;
}

inline bool
record_try_append(const kernel_dispatch_record_t& record)
{
    auto& store = append_record_store();
    if(!store.header || !store.records) return false;

    auto idx = __atomic_fetch_add(&store.header->count, 1, __ATOMIC_RELAXED);
    if(idx >= record_store_capacity) return false;

    store.records[idx] = record;
    return true;
}

inline const kernel_dispatch_record_t*
records(uint64_t* count)
{
    auto& store = read_record_store();
    if(!store.header || !store.records)
    {
        if(count) *count = 0;
        return nullptr;
    }

    auto available = std::min<uint64_t>(__atomic_load_n(&store.header->count, __ATOMIC_ACQUIRE),
                                        record_store_capacity);
    if(count) *count = available;
    return store.records;
}

inline void
unlink_record_store()
{
    for(const auto* directory : {"/dev/shm", "/tmp"})
    {
        const auto path = record_store_path(directory);
        ::unlink(path.data());
    }
}
}  // namespace lite_trace
}  // namespace common
}  // namespace rocprofiler
