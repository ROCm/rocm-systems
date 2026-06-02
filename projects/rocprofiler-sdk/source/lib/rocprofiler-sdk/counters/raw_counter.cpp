// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/counters/raw_counter.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <fmt/core.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <string>
#include <string_view>

namespace rocprofiler
{
namespace counters
{
namespace
{
hsa_ven_amd_aqlprofile_id_query_t
get_block_id_from_name(rocprofiler_agent_id_t agent, const char* block_name)
{
    const auto* aql_agent = rocprofiler::agent::get_aql_agent(agent);
    if(!aql_agent)
        return hsa_ven_amd_aqlprofile_id_query_t{.name = nullptr, .id = 0, .instance_count = 0};
    auto profile = aqlprofile_pmc_profile_t{.agent = *aql_agent, .events = nullptr, .event_count = 0};
    auto query   = hsa_ven_amd_aqlprofile_id_query_t{block_name, 0, 0};
    if(aqlprofile_get_pmc_info(&profile, AQLPROFILE_INFO_BLOCK_ID, &query) != HSA_STATUS_SUCCESS)
        return hsa_ven_amd_aqlprofile_id_query_t{.name = nullptr, .id = 0, .instance_count = 0};
    return query;
}
// Trim leading and trailing ASCII whitespace from s in-place.
void
trim_inplace(std::string& s)
{
    auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

// Returns true when every character of s is an ASCII digit.
bool
all_digits(std::string_view s)
{
    if(s.empty()) return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

// Parse a non-negative integer from the string view. Returns false on failure.
bool
parse_uint(std::string_view sv, uint32_t& out)
{
    auto result = std::from_chars(sv.data(), sv.data() + sv.size(), out);
    return result.ec == std::errc{} && result.ptr == sv.data() + sv.size();
}
}  // namespace

std::string
normalize_block_name(std::string_view input)
{
    constexpr std::string_view hsa_prefix = "HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_";
    constexpr std::string_view aql_prefix = "AQLPROFILE_BLOCK_NAME_";

    if(input.size() > hsa_prefix.size() && input.substr(0, hsa_prefix.size()) == hsa_prefix)
        return std::string(input.substr(hsa_prefix.size()));
    if(input.size() > aql_prefix.size() && input.substr(0, aql_prefix.size()) == aql_prefix)
        return std::string(input.substr(aql_prefix.size()));
    return std::string(input);
}

std::optional<raw_counter_t>
parse_raw_counter_string(const std::string& input, rocprofiler_agent_id_t agent_id)
{
    // -----------------------------------------------------------------------
    // Try the all-numeric colon-separated format: "block_id:block_index:counter_id"
    // -----------------------------------------------------------------------
    {
        auto colon1 = input.find(':');
        if(colon1 != std::string::npos)
        {
            auto colon2 = input.find(':', colon1 + 1);
            if(colon2 != std::string::npos)
            {
                auto tok0 = std::string_view(input).substr(0, colon1);
                auto tok1 = std::string_view(input).substr(colon1 + 1, colon2 - colon1 - 1);
                auto tok2 = std::string_view(input).substr(colon2 + 1);

                uint32_t bid = 0, bidx = 0, cid = 0;
                if(parse_uint(tok0, bid) && parse_uint(tok1, bidx) && parse_uint(tok2, cid))
                {
                    return raw_counter_t{bid, bidx, cid};
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Try the comma-separated format: "BLOCK_NAME, block_index, counter_id"
    // The block token may be an enum name or a bare short name.
    // -----------------------------------------------------------------------
    {
        auto comma1 = input.find(',');
        if(comma1 == std::string::npos)
        {
            ROCP_WARNING << fmt::format("raw_counter: no comma or colon delimiter in '{}'", input);
            return std::nullopt;
        }
        auto comma2 = input.find(',', comma1 + 1);
        if(comma2 == std::string::npos)
        {
            ROCP_WARNING << fmt::format(
                "raw_counter: expected two commas in comma-separated spec '{}'", input);
            return std::nullopt;
        }

        auto block_tok  = std::string(input.substr(0, comma1));
        auto index_tok  = std::string(input.substr(comma1 + 1, comma2 - comma1 - 1));
        auto cid_tok    = std::string(input.substr(comma2 + 1));
        trim_inplace(block_tok);
        trim_inplace(index_tok);
        trim_inplace(cid_tok);

        uint32_t block_index = 0;
        uint32_t counter_id  = 0;
        if(!parse_uint(index_tok, block_index) || !parse_uint(cid_tok, counter_id))
        {
            ROCP_WARNING << fmt::format(
                "raw_counter: could not parse block_index/counter_id in '{}'", input);
            return std::nullopt;
        }

        // If the block token is purely numeric, use it directly.
        if(all_digits(block_tok))
        {
            uint32_t bid = 0;
            if(!parse_uint(block_tok, bid))
            {
                ROCP_WARNING << fmt::format("raw_counter: overflow parsing block_id in '{}'", input);
                return std::nullopt;
            }
            return raw_counter_t{bid, block_index, counter_id};
        }

        // Otherwise resolve the string through aqlprofile.
        auto short_name = normalize_block_name(block_tok);
        auto query      = get_block_id_from_name(agent_id, short_name.c_str());
        if(query.instance_count == 0)
        {
            ROCP_WARNING << fmt::format(
                "raw_counter: aqlprofile could not resolve block name '{}' (from '{}')",
                short_name,
                input);
            return std::nullopt;
        }
        return raw_counter_t{
            static_cast<uint32_t>(query.id), block_index, counter_id};
    }
}

}  // namespace counters
}  // namespace rocprofiler
