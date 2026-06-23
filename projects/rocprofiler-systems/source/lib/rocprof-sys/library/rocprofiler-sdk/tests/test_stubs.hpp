// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/thread_info.hpp"

#include <gmock/gmock.h>
#include <timemory/hash/types.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <sys/types.h>
#include <unordered_map>

namespace rocprofsys
{

// ---------------------------------------------------------------------------
// GMock class — one MOCK_METHOD per symbol missing from the unit-test link.
// member-function stubs pass the live field value so the default can pass it
// through; tests that need different behaviour set an explicit ON_CALL.
// ---------------------------------------------------------------------------
struct gmock_runtime_stubs
{
    MOCK_METHOD(pid_t, get_root_process_id, ());
    MOCK_METHOD(bool, is_root_process, ());
    MOCK_METHOD(bool, push_enable_sampling_on_child_threads, (bool) );
    MOCK_METHOD(bool, pop_enable_sampling_on_child_threads, ());
    MOCK_METHOD((std::unordered_map<tim::hash_value_t, std::string>&),
                get_perfetto_track_uuids, ());
    MOCK_METHOD(std::mutex&, get_perfetto_track_uuids_mutex, ());
    MOCK_METHOD((const std::optional<thread_info>&), thread_info_get,
                (std::int64_t, ThreadIdType));
    // Passes the live lifetime field through so the default is a no-op passthrough.
    MOCK_METHOD(std::uint64_t, thread_info_get_start, (std::uint64_t));
    MOCK_METHOD(std::uint64_t, thread_info_get_stop, (std::uint64_t));
    MOCK_METHOD(std::string, thread_index_data_as_string, ());
    MOCK_METHOD(std::string, thread_info_as_string, ());
};

// Global instance — NiceMock suppresses "uninteresting call" warnings.
// Tests that need expectations can EXPECT_CALL(*g_runtime_stubs, ...) after
// calling reset_runtime_stubs() in their SetUp.
extern std::unique_ptr<::testing::NiceMock<gmock_runtime_stubs>> g_runtime_stubs;

// Creates a fresh NiceMock and installs safe default ON_CALLs.
// Call in SetUp; call reset (g_runtime_stubs.reset()) in TearDown.
void
reset_runtime_stubs();

}  // namespace rocprofsys
