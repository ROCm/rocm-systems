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
#include "core/common.hpp"
#include "core/defines.hpp"
#include "core/exception.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/unwind/bfd.hpp>
#include <timemory/unwind/types.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <variant>

namespace rocprofsys
{
namespace binary
{
namespace procfs = ::tim::procfs;  // NOLINT

using bfd_file     = ::tim::unwind::bfd_file;
using hash_value_t = ::tim::hash_value_t;

struct address_range;
struct address_multirange;
struct scope_filter;
struct symbol;
struct dwarf_entry;
struct binary_info;
}  // namespace binary
}  // namespace rocprofsys
