// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/timemory.hpp"
#include "library/components/backtrace.hpp"
#include "library/components/backtrace_metrics.hpp"
#include "library/components/backtrace_timestamp.hpp"
#include "library/components/callchain.hpp"
#include "library/thread_data.hpp"

#include <timemory/macros/language.hpp>
#include <timemory/variadic/types.hpp>

#include <cstdint>
#include <memory>
#include <set>
#include <type_traits>

namespace rocprofsys
{
namespace sampling
{
unique_ptr_t<std::set<int>>&
get_signal_types(std::int64_t _tid);

std::set<int>
setup();

std::set<int>
shutdown();

void
block_samples();

void
unblock_samples();

void block_signals(std::set<int> = {});

void unblock_signals(std::set<int> = {});

void
post_process();

void
postfork_parent_reinit();

void
postfork_child_cleanup();

void
pause();

void
resume();

}  // namespace sampling
}  // namespace rocprofsys
