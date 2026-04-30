// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "binary/analysis.hpp"
#include "common/defines.h"
#include "core/binary/fwd.hpp"
#include "core/containers/c_array.hpp"
#include "core/containers/static_vector.hpp"
#include "core/utility.hpp"
#include "library/causal/fwd.hpp"
#include "library/thread_data.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/utility/procfs/maps.hpp>
#include <timemory/utility/unwind.hpp>

#include <deque>
#include <dlfcn.h>
#include <map>

namespace rocprofsys
{
namespace causal
{
void
save_line_info(const settings::compose_filename_config&, int _verbose);

std::deque<binary::symbol>
get_line_info(uintptr_t _addr, bool include_discarded = true);

bool is_eligible_address(uintptr_t);

size_t set_current_selection(unwind_addr_t);
size_t set_current_selection(container::c_array<uint64_t>);

void
reset_sample_selection();

selected_entry
sample_selection(size_t _nitr = 1000, size_t _wait_ns = 100000);

void push_progress_point(std::string_view);

void pop_progress_point(std::string_view);

void
mark_progress_point(std::string_view, bool force = false);

uint16_t
sample_virtual_speedup();

void
start_experimenting();

void
finish_experimenting();
}  // namespace causal
}  // namespace rocprofsys
