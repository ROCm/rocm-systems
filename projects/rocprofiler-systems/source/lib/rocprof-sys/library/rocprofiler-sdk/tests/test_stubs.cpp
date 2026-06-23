// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Test-only stubs for symbols that live in rocprofiler-systems-object-library
// (compiled into librocprof-sys.so) but are absent from the unit-test link.
// Each stub delegates to g_runtime_stubs so tests can set EXPECT_CALLs when needed.

#include "test_stubs.hpp"
#include <cstdint>

#include <unistd.h>

namespace rocprofsys
{

// ---------------------------------------------------------------------------
// Backing objects for reference-returning ON_CALL defaults.
// Process-scoped statics; g_runtime_stubs.reset() leaves them intact.
// ---------------------------------------------------------------------------
namespace
{
std::unordered_map<tim::hash_value_t, std::string> s_track_uuids{};
std::mutex                                         s_track_uuids_mutex{};
std::optional<thread_info>                         s_empty_thread_info{};
}  // namespace

// ---------------------------------------------------------------------------
// Global mock definition
// ---------------------------------------------------------------------------
std::unique_ptr<::testing::NiceMock<gmock_runtime_stubs>> g_runtime_stubs;

// ---------------------------------------------------------------------------
// Helper — lazy accessor ensures stubs are alive before any stub function runs.
// ---------------------------------------------------------------------------
namespace
{
gmock_runtime_stubs&
stubs()
{
    if(!g_runtime_stubs)
    {
        reset_runtime_stubs();
    }
    return *g_runtime_stubs;
}
}  // namespace

void
reset_runtime_stubs()
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::ReturnArg;
    using ::testing::ReturnRef;

    g_runtime_stubs = std::make_unique<::testing::NiceMock<gmock_runtime_stubs>>();

    ON_CALL(*g_runtime_stubs, get_root_process_id()).WillByDefault(Return(::getpid()));
    ON_CALL(*g_runtime_stubs, is_root_process()).WillByDefault(Return(true));
    ON_CALL(*g_runtime_stubs, push_enable_sampling_on_child_threads(_))
        .WillByDefault(Return(false));
    ON_CALL(*g_runtime_stubs, pop_enable_sampling_on_child_threads())
        .WillByDefault(Return(false));
    ON_CALL(*g_runtime_stubs, get_perfetto_track_uuids())
        .WillByDefault(ReturnRef(s_track_uuids));
    ON_CALL(*g_runtime_stubs, get_perfetto_track_uuids_mutex())
        .WillByDefault(ReturnRef(s_track_uuids_mutex));
    ON_CALL(*g_runtime_stubs, thread_info_get(_, _))
        .WillByDefault(ReturnRef(s_empty_thread_info));
    ON_CALL(*g_runtime_stubs, thread_info_get_start(_)).WillByDefault(ReturnArg<0>());
    ON_CALL(*g_runtime_stubs, thread_info_get_stop(_)).WillByDefault(ReturnArg<0>());
    ON_CALL(*g_runtime_stubs, thread_index_data_as_string())
        .WillByDefault(Return(std::string{}));
    ON_CALL(*g_runtime_stubs, thread_info_as_string())
        .WillByDefault(Return(std::string{}));
}

// ---------------------------------------------------------------------------
// Free-function stubs
// ---------------------------------------------------------------------------

pid_t
get_root_process_id()
{
    return stubs().get_root_process_id();
}

bool
is_root_process()
{
    return stubs().is_root_process();
}

bool
push_enable_sampling_on_child_threads(bool enabled)
{
    return stubs().push_enable_sampling_on_child_threads(enabled);
}

bool
pop_enable_sampling_on_child_threads()
{
    return stubs().pop_enable_sampling_on_child_threads();
}

// ---------------------------------------------------------------------------
// thread_info member-function stubs
// ---------------------------------------------------------------------------

const std::optional<thread_info>&
thread_info::get(std::int64_t tid, ThreadIdType type)
{
    return stubs().thread_info_get(tid, type);
}

std::uint64_t
thread_info::get_start() const
{
    return stubs().thread_info_get_start(lifetime.first);
}

std::uint64_t
thread_info::get_stop() const
{
    return stubs().thread_info_get_stop(lifetime.second);
}

std::string
thread_index_data::as_string() const
{
    return stubs().thread_index_data_as_string();
}

std::string
thread_info::as_string() const
{
    return stubs().thread_info_as_string();
}

// ---------------------------------------------------------------------------
// tracing free-function stubs
// ---------------------------------------------------------------------------

namespace tracing
{

std::unordered_map<tim::hash_value_t, std::string>&
get_perfetto_track_uuids()
{
    return stubs().get_perfetto_track_uuids();
}

std::mutex&
get_perfetto_track_uuids_mutex()
{
    return stubs().get_perfetto_track_uuids_mutex();
}

}  // namespace tracing
}  // namespace rocprofsys
