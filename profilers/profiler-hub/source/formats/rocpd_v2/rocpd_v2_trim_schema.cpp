// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocpd_v2_reader.hpp"
#include "trim_schema.hpp"

#include <string>

namespace profiler_hub
{
namespace
{

std::string
full_copy(std::string_view table)
{
    const std::string name{ table };
    return "INSERT INTO \"" + name + "\" SELECT * FROM oldDb.\"" + name + "\"";
}

// Unqualified table names in these statements read the trimmed database, so a
// link survives only when an earlier phase kept both of its ends.
class rocpd_v2_trim_schema_t final : public trim_schema_t
{
public:
    trim_statement_t statement_for(std::string_view                       table,
                                   const std::unordered_set<std::string>& tables,
                                   uint64_t                               window_start,
                                   uint64_t window_end) const override
    {
        if(table == "rocpd_api" || table == "rocpd_op" || table == "rocpd_monitor")
        {
            // start_ts is inclusive and end_ts is exclusive: a row that ends on
            // the start is kept, and a row that starts on the end is not.
            return trim_statement_t{ row_kind_t::interval,
                                     full_copy(table) + " WHERE \"start\" < " +
                                         std::to_string(window_end) + " AND \"end\" >= " +
                                         std::to_string(window_start) };
        }
        if(table == "rocpd_api_ops" && tables.count("rocpd_api") != 0 &&
           tables.count("rocpd_op") != 0)
        {
            return trim_statement_t{
                row_kind_t::event,
                full_copy(table) + " WHERE api_id IN (SELECT id FROM rocpd_api) AND "
                                   "op_id IN (SELECT id FROM rocpd_op)"
            };
        }
        if((table == "rocpd_kernelapi" || table == "rocpd_copyapi") &&
           tables.count("rocpd_api") != 0)
        {
            return trim_statement_t{
                row_kind_t::event_child,
                full_copy(table) + " WHERE api_ptr_id IN (SELECT id FROM rocpd_api)"
            };
        }
        return trim_statement_t{ row_kind_t::copy, full_copy(table) };
    }
};

}  // namespace

std::unique_ptr<trim_schema_t>
make_rocpd_v2_trim_schema()
{
    return std::make_unique<rocpd_v2_trim_schema_t>();
}

}  // namespace profiler_hub
