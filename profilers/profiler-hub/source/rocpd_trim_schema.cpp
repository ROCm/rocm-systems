// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "trim_schema.hpp"

#include <string>
#include <vector>

namespace profiler_hub
{
namespace
{

struct table_role_t
{
    std::string_view prefix;
    row_kind_t       kind;
};

// Longer prefixes come first so rocpd_pmc_event is not classified as rocpd_event.
constexpr table_role_t k_roles[] = {
    { "rocpd_pmc_event", row_kind_t::event_child },
    { "rocpd_kernel_dispatch", row_kind_t::interval },
    { "rocpd_memory_allocate", row_kind_t::interval },
    { "rocpd_memory_copy", row_kind_t::interval },
    { "rocpd_hip_event", row_kind_t::interval },
    { "rocpd_region", row_kind_t::interval },
    { "rocpd_sample", row_kind_t::sample },
    { "rocpd_event", row_kind_t::event },
    { "rocpd_arg", row_kind_t::event_child },
};

bool
has_role_name(std::string_view name, std::string_view prefix)
{
    if(name == prefix) return true;
    return name.size() > prefix.size() && name.substr(0, prefix.size()) == prefix &&
           name[prefix.size()] == '_';
}

const table_role_t*
role_of(std::string_view name)
{
    for(const table_role_t& role : k_roles)
    {
        if(has_role_name(name, role.prefix)) return &role;
    }
    return nullptr;
}

std::string
quote_ident(std::string_view name)
{
    std::string quoted = "\"";
    for(const char ch : name)
    {
        if(ch == '"') quoted += "\"\"";
        else quoted += ch;
    }
    quoted += '"';
    return quoted;
}

std::string
sibling(std::string_view name, std::string_view from, std::string_view to)
{
    if(!has_role_name(name, from)) return {};
    return std::string{ to } + std::string{ name.substr(from.size()) };
}

std::string
full_copy(std::string_view table)
{
    const std::string quoted = quote_ident(table);
    return "INSERT INTO " + quoted + " SELECT * FROM oldDb." + quoted;
}

class rocpd_v3_trim_schema_t final : public trim_schema_t
{
public:
    trim_statement_t statement_for(std::string_view                       table,
                                   const std::unordered_set<std::string>& tables,
                                   uint64_t                               window_start,
                                   uint64_t window_end) const override
    {
        const table_role_t* role = role_of(table);
        if(role == nullptr)
        {
            return trim_statement_t{ row_kind_t::copy, full_copy(table) };
        }

        const std::string quoted = quote_ident(table);
        const std::string start  = std::to_string(window_start);
        const std::string end    = std::to_string(window_end);
        switch(role->kind)
        {
            case row_kind_t::copy: return trim_statement_t{ role->kind, full_copy(table) };
            case row_kind_t::interval:
                return trim_statement_t{
                    role->kind,
                    "INSERT INTO " + quoted + " SELECT * FROM oldDb." + quoted +
                        " WHERE \"start\" < " + end + " AND \"end\" > " + start };
            case row_kind_t::sample: return sample_statement(*role, table, tables, start, end);
            case row_kind_t::event: return event_statement(*role, table, tables);
            case row_kind_t::event_child: return child_statement(*role, table, tables);
        }
        return trim_statement_t{ role->kind, full_copy(table) };
    }

private:
    static trim_statement_t sample_statement(const table_role_t&                    role,
                                             std::string_view                       table,
                                             const std::unordered_set<std::string>& tables,
                                             const std::string&                     start,
                                             const std::string&                     end)
    {
        const std::string quoted = quote_ident(table);
        std::string       sql    = "INSERT INTO " + quoted + " SELECT S.* FROM oldDb." +
                           quoted + " S WHERE (S.\"timestamp\" > " + start +
                           " AND S.\"timestamp\" < " + end + ")";
        const std::string region = sibling(table, "rocpd_sample", "rocpd_region");
        if(!region.empty() && tables.count(region) != 0)
        {
            sql += " OR S.\"timestamp\" IN (SELECT \"start\" FROM " + quote_ident(region) +
                   ")";
        }
        return trim_statement_t{ role.kind, std::move(sql) };
    }

    static trim_statement_t event_statement(const table_role_t&                    role,
                                            std::string_view                       table,
                                            const std::unordered_set<std::string>& tables)
    {
        const char* sources[] = { "rocpd_kernel_dispatch", "rocpd_memory_allocate",
                                  "rocpd_memory_copy",      "rocpd_hip_event",
                                  "rocpd_region",           "rocpd_sample" };
        std::vector<std::string> refs;
        for(const char* source : sources)
        {
            const std::string name = sibling(table, "rocpd_event", source);
            if(!name.empty() && tables.count(name) != 0) refs.push_back(name);
        }
        if(refs.empty()) return trim_statement_t{ role.kind, full_copy(table) };

        const std::string quoted = quote_ident(table);
        std::string       sql    = "INSERT INTO " + quoted + " SELECT DISTINCT E.* FROM oldDb." +
                           quoted + " E WHERE ";
        for(size_t index = 0; index < refs.size(); ++index)
        {
            if(index != 0) sql += " OR ";
            sql += "E.\"id\" IN (SELECT \"event_id\" FROM " + quote_ident(refs[index]) + ")";
        }
        return trim_statement_t{ role.kind, std::move(sql) };
    }

    static trim_statement_t child_statement(const table_role_t&                    role,
                                            std::string_view                       table,
                                            const std::unordered_set<std::string>& tables)
    {
        const table_role_t* matched = role_of(table);
        const std::string   event =
            sibling(table, matched == nullptr ? "" : matched->prefix, "rocpd_event");
        if(event.empty() || tables.count(event) == 0)
        {
            return trim_statement_t{ role.kind, full_copy(table) };
        }
        const std::string quoted = quote_ident(table);
        return trim_statement_t{
            role.kind,
            "INSERT INTO " + quoted + " SELECT X.* FROM oldDb." + quoted +
                " X WHERE X.\"event_id\" IN (SELECT \"id\" FROM " + quote_ident(event) + ")" };
    }
};

}  // namespace

std::unique_ptr<trim_schema_t>
make_trim_schema(reader_types::trace_schema_t schema, version_t version)
{
    if(schema == reader_types::trace_schema_t::rocpd && version.major == 3)
    {
        return std::make_unique<rocpd_v3_trim_schema_t>();
    }
    return nullptr;
}

}  // namespace profiler_hub
