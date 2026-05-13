// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace wall_clock_event_trace
{
void
session_reset();

void
push_region(std::int64_t thread_id, const std::string& name);
void
pop_region(std::int64_t thread_id, std::string_view name);

void
push_pthread_create(std::int64_t parent_thread_id, const std::string& name);
void
pop_pthread_create(std::int64_t parent_thread_id, std::string_view name);

void
push_start_thread(std::int64_t parent_thread_id, std::int64_t child_thread_id,
                  const std::string& name);
void
pop_start_thread(std::int64_t child_thread_id, std::string_view name);
}  // namespace wall_clock_event_trace
}  // namespace rocprofsys
