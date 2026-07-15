// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
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
