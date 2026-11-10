// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk-collector/join.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <string_view>
#include <utility>

namespace rocprofiler
{
namespace collector
{
namespace
{
using thread_key = std::pair<uint64_t, uint64_t>;  // (pid, tid)

constexpr std::string_view enter_suffix = "_enter";
constexpr std::string_view exit_suffix  = "_exit";

bool
ends_with(std::string_view text, std::string_view suffix)
{
    return text.size() > suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Fields the transport adds (_ms_ActivityId, _ms_Opcode): not arguments.
void
copy_arguments(const std::vector<ctf::field_value>& fields, std::vector<ctf::field_value>* out)
{
    out->clear();
    out->reserve(fields.size());
    for(const auto& itr : fields)
        if(itr.name.rfind("_ms_", 0) != 0) out->push_back(itr);
}

bool
context_value(const ctf::event_record& event, const char* name, uint64_t* value)
{
    const auto* field = ctf::find_field(event.context, name);
    if(field == nullptr) return false;
    *value = field->u64;
    return true;
}

uint64_t
trace_pid(const ctf::trace* source)
{
    if(source == nullptr) return 0;
    auto itr = source->description.env.find("vpid");
    if(itr == source->description.env.end()) return 0;
    return std::strtoull(itr->second.c_str(), nullptr, 10);
}

// The gaps of one process, sorted by begin, with the running maximum of their ends.
struct process_gaps
{
    std::vector<std::pair<uint64_t, uint64_t>> ranges  = {};  // (begin, end)
    std::vector<uint64_t>                      max_end = {};
};

std::map<uint64_t, process_gaps>
index_gaps(const std::vector<event_gap>& gaps)
{
    auto result = std::map<uint64_t, process_gaps>{};
    for(const auto& gap : gaps)
        result[gap.pid].ranges.emplace_back(gap.begin, gap.end);
    for(auto& [pid, entry] : result)
    {
        std::sort(entry.ranges.begin(), entry.ranges.end());
        auto running = uint64_t{0};
        for(const auto& range : entry.ranges)
        {
            running = std::max(running, range.second);
            entry.max_end.push_back(running);
        }
    }
    return result;
}

// Whether a gap overlaps [begin, end].
bool
overlaps(const process_gaps& gaps, uint64_t begin, uint64_t end)
{
    auto after = std::upper_bound(gaps.ranges.begin(),
                                  gaps.ranges.end(),
                                  std::make_pair(end, std::numeric_limits<uint64_t>::max()));
    if(after == gaps.ranges.begin()) return false;
    auto index = static_cast<size_t>(std::distance(gaps.ranges.begin(), after)) - 1;
    return gaps.max_end[index] >= begin;
}

class joiner
{
public:
    joiner(const api_call_callback&         on_call,
           std::map<uint64_t, process_gaps> gaps,
           join_statistics*                 stats)
    : m_on_call{on_call}
    , m_gaps{std::move(gaps)}
    , m_stats{stats}
    {}

    bool on_event(const ctf::event_record& event, std::string* error)
    {
        auto provider = std::string{};
        auto api      = std::string{};
        auto is_enter = false;
        if(!parse_api_event_name(event.description->name, &provider, &api, &is_enter))
        {
            ++m_stats->other_events;
            return true;
        }

        auto key = thread_key{};
        if(!context_value(event, "vpid", &key.first) || !context_value(event, "vtid", &key.second))
        {
            *error = "event " + event.description->name +
                     " has no vpid/vtid context (the collector's session adds them)";
            return false;
        }

        auto& stack = m_threads[key];
        if(is_enter)
        {
            auto call           = api_call{};
            call.provider       = std::move(provider);
            call.name           = std::move(api);
            call.pid            = key.first;
            call.tid            = key.second;
            call.correlation_id = ++m_next_id[key.first];
            call.parent_id      = stack.empty() ? 0 : stack.back().correlation_id;
            call.depth          = static_cast<uint32_t>(stack.size());
            call.start          = event.timestamp;
            copy_arguments(event.fields, &call.arguments);
            stack.push_back(std::move(call));
            return true;
        }

        // The innermost open call of the same API; the calls above it lost their exits.
        auto match = std::find_if(stack.rbegin(), stack.rend(), [&](const api_call& call) {
            return call.name == api && call.provider == provider;
        });
        if(match == stack.rend())
        {
            ++m_stats->unmatched_exits;
            return true;
        }
        auto index     = static_cast<size_t>(std::distance(match, stack.rend())) - 1;
        auto ambiguous = lost_between(key.first, match->start, event.timestamp);
        // The calls above the match lost their exits; when ambiguous, the exit may be of a
        // later call of the same API whose enter was lost, so the match did too.
        auto keep = ambiguous ? index : index + 1;
        while(stack.size() > keep)
        {
            ++m_stats->lost_exits;
            report(std::move(stack.back()));
            stack.pop_back();
        }
        if(ambiguous)
        {
            ++m_stats->ambiguous_exits;
            return true;
        }
        auto& call = stack.back();
        call.end   = event.timestamp;
        copy_arguments(event.fields, &call.results);
        report(std::move(call));
        stack.pop_back();
        return true;
    }

    // The calls still open when the trace ended, innermost first.
    void finish()
    {
        for(auto& [key, stack] : m_threads)
        {
            while(!stack.empty())
            {
                ++m_stats->open_at_stop;
                report(std::move(stack.back()));
                stack.pop_back();
            }
        }
    }

private:
    const api_call_callback&                    m_on_call;
    std::map<uint64_t, process_gaps>            m_gaps    = {};
    join_statistics*                            m_stats   = nullptr;
    std::map<thread_key, std::vector<api_call>> m_threads = {};
    std::map<uint64_t, uint64_t>                m_next_id = {};

    // Gaps of an unknown process (pid 0) apply to every process.
    bool lost_between(uint64_t pid, uint64_t begin, uint64_t end) const
    {
        for(auto key : {pid, uint64_t{0}})
        {
            auto itr = m_gaps.find(key);
            if(itr != m_gaps.end() && overlaps(itr->second, begin, end)) return true;
        }
        return false;
    }

    void report(api_call&& call)
    {
        ++m_stats->calls;
        if(m_on_call) m_on_call(call);
    }
};
}  // namespace

bool
parse_api_event_name(const std::string& event_name,
                     std::string*       provider,
                     std::string*       api,
                     bool*              is_enter)
{
    auto name  = std::string_view{event_name};
    auto colon = name.find(':');
    if(colon == std::string_view::npos || colon == 0) return false;
    auto event = name.substr(colon + 1);
    // TraceLogging encodes the keywords in the name: "<event>;k<bit>;...".
    if(auto semicolon = event.find(';'); semicolon != std::string_view::npos)
        event = event.substr(0, semicolon);

    if(ends_with(event, enter_suffix))
    {
        *is_enter = true;
        event.remove_suffix(enter_suffix.size());
    }
    else if(ends_with(event, exit_suffix))
    {
        *is_enter = false;
        event.remove_suffix(exit_suffix.size());
    }
    else
    {
        return false;
    }
    *provider = std::string{name.substr(0, colon)};
    *api      = std::string{event};
    return true;
}

bool
join_api_calls(const std::vector<ctf::trace_ptr>& traces,
               const api_call_callback&           on_call,
               std::vector<event_gap>*            gaps,
               join_statistics*                   statistics,
               std::string*                       error)
{
    auto all_gaps     = std::vector<event_gap>{};
    auto on_discarded = [&all_gaps](const ctf::discarded_record& record) {
        all_gaps.push_back({trace_pid(record.source), record.count, record.begin, record.end});
    };
    if(!ctf::scan_discarded(traces, on_discarded, error)) return false;

    auto stats    = join_statistics{};
    auto join     = joiner{on_call, index_gaps(all_gaps), &stats};
    auto failed   = std::string{};
    auto on_event = [&](const ctf::event_record& event) {
        if(failed.empty()) join.on_event(event, &failed);
    };
    if(!ctf::read_events(traces, on_event, nullptr, &stats.read, error)) return false;
    if(!failed.empty())
    {
        *error = failed;
        return false;
    }
    join.finish();
    if(gaps != nullptr) *gaps = std::move(all_gaps);
    if(statistics != nullptr) *statistics = stats;
    return true;
}
}  // namespace collector
}  // namespace rocprofiler
