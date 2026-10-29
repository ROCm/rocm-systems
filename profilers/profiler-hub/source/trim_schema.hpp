// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader_types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>

namespace profiler_hub
{

/** Copy phase for one table. Enumerator order is the order the trimmer runs
 *  the statements: a later phase reads rows an earlier phase already wrote. */
enum class row_kind_t
{
    copy,
    interval,
    sample,
    event,
    event_child,
};

/** One INSERT for a table the generic trimmer has already created. */
struct trim_statement_t
{
    row_kind_t  kind{ row_kind_t::copy };
    std::string sql;
};

/**
 * Row filter for one probed schema. The trimmer itself only copies the
 * database objects it finds; this chooses which rows of each table survive.
 */
class trim_schema_t
{
public:
    virtual ~trim_schema_t() = default;

    [[nodiscard]] virtual trim_statement_t statement_for(
        std::string_view                      table,
        const std::unordered_set<std::string>& tables,
        uint64_t                              window_start,
        uint64_t                              window_end) const = 0;
};

[[nodiscard]] std::unique_ptr<trim_schema_t>
make_trim_schema(reader_types::trace_schema_t schema, version_t version);

}  // namespace profiler_hub
