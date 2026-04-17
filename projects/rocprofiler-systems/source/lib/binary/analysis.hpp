// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/binary/fwd.hpp"
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

std::vector<binary_info>
get_binary_info(const std::vector<std::string>&, const std::vector<scope_filter>&,
                bool _process_dwarf = true, bool _process_bfd = true,
                bool _include_all = false);

template <bool ExcludeInternal>
std::optional<tim::unwind::processed_entry>
lookup_ipaddr_entry(uintptr_t, unw_context_t* = nullptr, tim::unwind::cache* = nullptr);
}  // namespace binary
}  // namespace rocprofsys
