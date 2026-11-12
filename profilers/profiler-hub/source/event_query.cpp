// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "event_query.hpp"

#include <algorithm>
#include <cctype>

namespace profiler_hub
{
namespace
{
bool
ascii_same(unsigned char left, unsigned char right)
{
    return std::tolower(left) == std::tolower(right);
}

bool
term_matches(std::string_view text, std::string_view term, bool substring)
{
    if(substring)
    {
        return std::search(text.begin(),
                           text.end(),
                           term.begin(),
                           term.end(),
                           ascii_same) != text.end();
    }
    return text.size() == term.size() &&
           std::equal(text.begin(), text.end(), term.begin(), ascii_same);
}
}  // namespace

bool
text_matches_terms(std::string_view                text,
                   const std::vector<std::string>& terms,
                   bool                            substring,
                   bool                            partial)
{
    if(terms.empty()) return false;

    if(partial)
    {
        for(const auto& term : terms)
        {
            if(term_matches(text, term, substring)) return true;
        }
        return false;
    }

    for(const auto& term : terms)
    {
        if(!term_matches(text, term, substring)) return false;
    }
    return true;
}

void
resolve_event_query_strings(const reader_catalog_t& catalog,
                            resolved_event_query&   query,
                            bool                    substring,
                            bool                    partial)
{
    query.string_ids.clear();
    query.kernel_symbol_ids.clear();
    if(query.search_strings.empty())
    {
        query.has_string_filter = false;
        return;
    }

    query.has_string_filter = true;
    for(const auto& [id, text] : catalog.strings_by_id)
    {
        if(text_matches_terms(text, query.search_strings, substring, partial))
        {
            query.string_ids.push_back(id);
        }
    }

    for(const auto& symbol : catalog.kernel_symbols)
    {
        if(!symbol) continue;
        const bool name_match =
            text_matches_terms(symbol->name, query.search_strings, substring, partial);
        const bool display_match = text_matches_terms(
            symbol->display_name, query.search_strings, substring, partial);
        if(name_match || display_match)
        {
            query.kernel_symbol_ids.push_back(symbol->id);
        }
    }
}

}  // namespace profiler_hub
