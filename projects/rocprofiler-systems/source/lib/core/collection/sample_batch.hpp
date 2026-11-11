// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <limits>
#include <span>

namespace rocprofsys::collection
{
using schema_id = std::uint32_t;

/** Value of a column that has no value in a row. */
inline constexpr double k_no_value = std::numeric_limits<double>::quiet_NaN();

/** One row of a schema. `values` views a row buffer owned by the writer; it only has to
 * live until `sink_interface::write_batch` returns. */
struct sample_batch
{
    schema_id schema = 0;
    /** Timeline clock, nanoseconds. */
    std::uint64_t timestamp = 0;
    /** One per column, k_no_value = no value. */
    std::span<const double> values;
};
}  // namespace rocprofsys::collection
