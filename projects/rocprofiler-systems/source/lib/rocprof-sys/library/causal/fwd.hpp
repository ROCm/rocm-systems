// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "common/defines.h"
#include "core/binary/fwd.hpp"
#include "core/containers/static_vector.hpp"
#include "core/defines.hpp"

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
