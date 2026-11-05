// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <span>

namespace rocprofsys::collection
{
using source_id = std::uint32_t;
using schema_id = std::uint32_t;

struct sample_batch
{
    source_id               source;
    schema_id               schema;
    std::uint64_t           timestamp;
    std::span<const double> values;
};

struct gap_marker
{
    source_id     source;
    std::uint64_t begin_ns;
    std::uint64_t end_ns;
};
}  // namespace rocprofsys::collection
