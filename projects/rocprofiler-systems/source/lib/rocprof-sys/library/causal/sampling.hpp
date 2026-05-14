// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/concepts.hpp"

#include <cstdint>
#include <memory>
#include <set>
#include <type_traits>

namespace rocprofsys
{
namespace causal
{
namespace sampling
{
std::set<int>
get_signal_types(std::int64_t _tid);

void
block_samples();

void
unblock_samples();

void
block_backtrace_samples();

void
unblock_backtrace_samples();

template <typename Tp = tim::scope::thread_scope>
void pause(Tp = {});

template <typename Tp = tim::scope::thread_scope>
void resume(Tp = {});

void block_signals(std::set<int> = {});

void unblock_signals(std::set<int> = {});

std::set<int>
setup();

std::set<int>
shutdown();

void
post_process();
}  // namespace sampling
}  // namespace causal
}  // namespace rocprofsys
