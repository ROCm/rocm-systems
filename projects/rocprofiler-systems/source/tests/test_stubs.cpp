// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TEMPORARY WORKAROUND: Stub implementations for symbols that pmc-library and
// core-library reference through headers but are never called at runtime during
// unit tests. This allows the test binary to link without
// rocprofiler-systems-object-library, avoiding the static initializers in
// library.cpp that trigger thread_data instantiation and thread_deleter
// finalization under TSan.
//
// TODO: Decouple pmc-library and core-library from these object-library symbols
// so that this file is no longer needed.

#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <mutex>
#include <optional>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>

namespace rocprofsys
{
// --- runtime.cpp stubs ---

bool
push_enable_sampling_on_child_threads(bool _v)
{
    return _v;
}

bool
pop_enable_sampling_on_child_threads()
{
    return true;
}

pid_t
get_root_process_id()
{
    return getpid();
}

bool
is_root_process()
{
    return true;
}

bool
is_child_process()
{
    return false;
}

// --- thread_info.cpp stubs ---

const std::optional<thread_info>&
thread_info::get(int64_t, ThreadIdType)
{
    static const std::optional<thread_info> empty{};
    return empty;
}

uint64_t
thread_info::get_start() const
{
    return 0;
}

uint64_t
thread_info::get_stop() const
{
    return 0;
}

bool
thread_info::is_valid_time(uint64_t) const
{
    return false;
}

// --- tracing.cpp stubs ---

namespace tracing
{
std::unordered_map<hash_value_t, std::string>&
get_perfetto_track_uuids()
{
    static std::unordered_map<hash_value_t, std::string> empty{};
    return empty;
}

std::mutex&
get_perfetto_track_uuids_mutex()
{
    static std::mutex mtx{};
    return mtx;
}
}  // namespace tracing

}  // namespace rocprofsys
