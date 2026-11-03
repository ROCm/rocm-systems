// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "schema_cache.hpp"

#include "../../data_storage/backends/embedded_schema.hpp"

#include <cctype>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace profiler_hub::schema
{
namespace
{

// Remembers ' and " so commas and dashes inside a literal stay part of the literal.
struct sql_quotes
{
    bool in_single = false;
    bool in_double = false;

    // Records a quote so later scans ignore punctuation inside a literal.
    // Takes the current character. Returns nothing.
    void see(char ch)
    {
        if(ch == '\'' && !in_double)
            in_single = !in_single;
        else if(ch == '"' && !in_single)
            in_double = !in_double;
    }

    // True while the scan is between a matching pair of quotes.
    // No input. Returns true inside a literal.
    [[nodiscard]] bool inside() const { return in_single || in_double; }
};

// True when ch can sit in an unquoted SQL name.
// Takes one character. Returns true for a letter, a digit, or '_'.
[[nodiscard]] bool
is_name_char(char ch)
{
    return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

// Drops spaces and newlines from both ends.
// Takes a view. Returns a view of the same text, without the surrounding space.
[[nodiscard]] std::string_view
trim(std::string_view text)
{
    while(!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0)
    {
        text.remove_prefix(1);
    }
    while(!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0)
    {
        text.remove_suffix(1);
    }
    return text;
}

// Moves pos forward over spaces and newlines.
// Takes the text and the index to start at. Updates pos. Returns nothing.
void
skip_space(std::string_view text, std::size_t& pos)
{
    while(pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0)
    {
        ++pos;
    }
}

// Strips -- comments and lowercases keywords. Quoted text such as 'CPU' stays as written.
// Takes the raw schema SQL. Returns a new string.
[[nodiscard]] std::string
prepare_sql(std::string_view sql)
{
    std::string out;
    out.reserve(sql.size());
    sql_quotes quotes;

    for(std::size_t i = 0; i < sql.size(); ++i)
    {
        const char ch = sql[i];
        if(!quotes.inside() && ch == '-' && i + 1 < sql.size() && sql[i + 1] == '-')
        {
            while(i < sql.size() && sql[i] != '\n')
            {
                ++i;
            }
            if(i < sql.size()) out.push_back('\n');
            continue;
        }

        quotes.see(ch);
        if(quotes.inside())
        {
            out.push_back(ch);
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

// Reads one name, quoted or not. Accepts `, ", and ' quotes.
// Takes the text and the index to start at. pos moves past the name. Returns the name, or
// empty.
[[nodiscard]] std::string
read_name(std::string_view text, std::size_t& pos)
{
    skip_space(text, pos);
    if(pos >= text.size()) return {};

    const char quote = text[pos];
    if(quote == '`' || quote == '"' || quote == '\'')
    {
        ++pos;
        const std::size_t start = pos;
        while(pos < text.size() && text[pos] != quote)
        {
            ++pos;
        }
        std::string name{ text.substr(start, pos - start) };
        if(pos < text.size()) ++pos;
        return name;
    }

    const std::size_t start = pos;
    while(pos < text.size() && is_name_char(text[pos]))
    {
        ++pos;
    }
    return std::string{ text.substr(start, pos - start) };
}

// Removes the {{uuid}} marker from a table name.
// Takes rocpd_pmc_event{{uuid}}. Returns rocpd_pmc_event.
[[nodiscard]] std::string
drop_uuid_placeholder(std::string name)
{
    constexpr std::string_view marker = "{{uuid}}";
    for(auto at = name.find(marker); at != std::string::npos; at = name.find(marker))
    {
        name.erase(at, marker.size());
    }
    return name;
}

// Finds the ')' that closes the '(' at open_at. Parentheses inside quotes do not count.
// Takes the text and the index of '('. Returns that index, or npos if the paren never
// closes.
[[nodiscard]] std::size_t
closing_paren(std::string_view text, std::size_t open_at)
{
    int        depth = 0;
    sql_quotes quotes;
    for(std::size_t i = open_at; i < text.size(); ++i)
    {
        quotes.see(text[i]);
        if(quotes.inside()) continue;
        if(text[i] == '(')
            ++depth;
        else if(text[i] == ')')
        {
            --depth;
            if(depth == 0) return i;
        }
    }
    return std::string_view::npos;
}

// Reads the values from CHECK (... IN (...)).
// Takes one column definition. Returns CPU, GPU for that example, or nothing if there is
// no IN list.
[[nodiscard]] std::vector<std::string>
read_allowed_values(std::string_view column_sql)
{
    auto in_at = column_sql.find(" in (");
    if(in_at == std::string_view::npos) in_at = column_sql.find(" in(");
    if(in_at == std::string_view::npos) return {};

    const auto open = column_sql.find('(', in_at);
    if(open == std::string_view::npos) return {};
    const auto close = closing_paren(column_sql, open);
    if(close == std::string_view::npos) return {};

    const auto               list = column_sql.substr(open + 1, close - open - 1);
    std::vector<std::string> values;
    for(std::size_t i = 0; i < list.size(); ++i)
    {
        const char quote = list[i];
        if(quote != '\'' && quote != '"') continue;
        ++i;
        const std::size_t start = i;
        while(i < list.size() && list[i] != quote)
        {
            ++i;
        }
        values.emplace_back(list.substr(start, i - start));
    }
    return values;
}

// True for a table constraint rather than a column. Covers FOREIGN KEY, UNIQUE, and a
// bare CHECK. Takes one piece of the column list. Returns false when the piece is a
// column.
[[nodiscard]] bool
is_constraint(std::string_view clause)
{
    return clause.starts_with("foreign ") || clause.starts_with("constraint ") ||
           clause.starts_with("unique ") || clause.starts_with("check ") ||
           clause.starts_with("primary ");
}

// Stores one column: name, NOT NULL, DEFAULT, primary key, and any IN values.
// Takes the table to fill and the column clause. Returns nothing.
void
add_column(table_model& table, std::string_view clause)
{
    std::size_t pos  = 0;
    std::string name = read_name(clause, pos);
    if(name.empty()) return;

    column_model column;
    column.name           = name;
    column.not_null       = clause.find("not null") != std::string_view::npos;
    column.has_default    = clause.find("default") != std::string_view::npos;
    column.primary_key    = clause.find("primary key") != std::string_view::npos;
    column.allowed_values = read_allowed_values(clause);

    const std::string key = column.name;
    table.columns.insert_or_assign(key, std::move(column));
}

// Splits the column list on commas that are not inside parentheses or quotes.
// Takes the text between the outer parentheses. Returns the trimmed pieces.
[[nodiscard]] std::vector<std::string_view>
split_clauses(std::string_view body)
{
    std::vector<std::string_view> clauses;
    sql_quotes                    quotes;
    int                           depth = 0;
    std::size_t                   start = 0;

    for(std::size_t i = 0; i < body.size(); ++i)
    {
        quotes.see(body[i]);
        if(quotes.inside()) continue;

        if(body[i] == '(')
            ++depth;
        else if(body[i] == ')')
            --depth;
        else if(body[i] == ',' && depth == 0)
        {
            clauses.push_back(trim(body.substr(start, i - start)));
            start = i + 1;
        }
    }
    clauses.push_back(trim(body.substr(start)));
    return clauses;
}

// Adds one table and its columns. Constraint clauses are skipped.
// Takes the model, the table name, and the column-list body. Returns nothing.
void
add_table(schema_model& model, std::string name, std::string_view body)
{
    table_model table;
    table.name = std::move(name);

    for(const std::string_view clause : split_clauses(body))
    {
        if(clause.empty() || is_constraint(clause)) continue;
        add_column(table, clause);
    }

    const std::string key = table.name;
    model.tables.insert_or_assign(key, std::move(table));
}

// Reads every CREATE TABLE in the schema SQL.
// Takes the embedded tables script. Returns the tables and their columns.
[[nodiscard]] schema_model
parse_tables(std::string_view sql)
{
    const std::string text = prepare_sql(sql);
    schema_model      model;
    std::size_t       search = 0;

    while(search < text.size())
    {
        const auto at = text.find("create table", search);
        if(at == std::string::npos) break;
        // Skip a match that is the tail of a longer word.
        if(at > 0 && is_name_char(text[at - 1]))
        {
            search = at + 1;
            continue;
        }

        std::size_t pos = at + std::string_view{ "create table" }.size();
        skip_space(text, pos);
        if(text.substr(pos).starts_with("if not exists"))
        {
            pos += std::string_view{ "if not exists" }.size();
        }

        std::string table_name = drop_uuid_placeholder(read_name(text, pos));
        skip_space(text, pos);
        if(table_name.empty() || pos >= text.size() || text[pos] != '(')
        {
            search = pos + 1;
            continue;
        }

        const auto end = closing_paren(text, pos);
        if(end == std::string_view::npos) break;

        add_table(model, std::move(table_name), text.substr(pos + 1, end - pos - 1));
        search = end + 1;
    }
    return model;
}

}  // namespace

// Looks up a column on this table.
// Takes the column name. Returns the column, or null if the table does not have it.
const column_model*
table_model::find(std::string_view column) const
{
    const auto found = columns.find(std::string{ column });
    if(found == columns.end()) return nullptr;
    return &found->second;
}

// Looks up a table in this schema version.
// Takes the table name without {{uuid}}. Returns the table, or null if it is absent.
const table_model*
schema_model::find_table(std::string_view table) const
{
    const auto found = tables.find(std::string{ table });
    if(found == tables.end()) return nullptr;
    return &found->second;
}

// Parsed schema for one version. The embedded SQL is parsed on the first call.
// Takes a version such as 3.0.5. Returns that model. Throws if the library has no such
// version.
const schema_model&
schema_cache::get(profiler_hub::version_t version)
{
    const std::string version_text = version.to_string();
    std::lock_guard   lock{ m_mutex };

    if(const auto found = m_models.find(version_text); found != m_models.end())
    {
        return found->second;
    }

    const auto* embedded =
        rocpd::data_storage::schema::find_embedded_schema(version_text);
    if(embedded == nullptr || embedded->tables == nullptr)
    {
        throw std::invalid_argument("unsupported schema version " + version_text);
    }

    auto parsed = parse_tables(embedded->tables);
    ++m_parse_count;
    return m_models.emplace(version_text, std::move(parsed)).first->second;
}

// How many versions have been parsed.
// No input. Returns the count. A second get() of the same version does not add to it.
std::size_t
schema_cache::parse_count() const
{
    std::lock_guard lock{ m_mutex };
    return m_parse_count;
}

// Cache used by validation. One version is parsed once for the process.
// No input. Returns that cache.
schema_cache&
shared_cache()
{
    static schema_cache cache;
    return cache;
}

}  // namespace profiler_hub::schema
