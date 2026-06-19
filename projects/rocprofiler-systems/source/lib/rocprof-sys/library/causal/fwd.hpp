// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/binary/fwd.hpp"
#include "core/containers/static_vector.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/unwind/stack.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>

namespace rocprofsys
{
namespace unwind = ::tim::unwind;

namespace causal
{
static constexpr size_t unwind_depth  = ROCPROFSYS_MAX_UNWIND_DEPTH;
static constexpr size_t unwind_offset = 0;
using unwind_stack_t                  = unwind::stack<unwind_depth>;
using unwind_addr_t                   = container::static_vector<uintptr_t, unwind_depth>;
using hash_value_t                    = tim::hash_value_t;

struct selected_entry;
}  // namespace causal
}  // namespace rocprofsys
