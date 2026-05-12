// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/wall_clock_report_processor.hpp"
#include <cstdint>

#include "core/config.hpp"
#include "logger/debug.hpp"

#include <timemory/utility/filepath.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{
namespace
{
struct agg_key
{
    std::uint64_t thread_seq{};
    std::uint64_t path_hash{};

    bool operator==(const agg_key& o) const
    {
        return thread_seq == o.thread_seq && path_hash == o.path_hash;
    }
};

struct agg_key_hash
{
    std::size_t operator()(const agg_key& k) const noexcept
    {
        return std::hash<std::uint64_t>{}(k.thread_seq) ^
               (std::hash<std::uint64_t>{}(k.path_hash) << 1);
    }
};

struct agg_stats
{
    std::uint64_t thread_seq = 0;
    /// Minimum DFS preorder index among spans in this bucket (per-thread tree walk).
    std::uint64_t preorder_rank = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t count         = 0;
    double        sum_inc       = 0.0;
    double        sum_exc       = 0.0;
    double        min_inc       = std::numeric_limits<double>::infinity();
    double        max_inc       = 0.0;
    double        sum_sq_inc    = 0.0;
    double        sum_sq_exc    = 0.0;
    std::string   name;
    std::string   category;
    std::int64_t  depth = 0;
};

constexpr double k_sec_per_ns = 1.e-9;

double
safe_pct_self(double sum_exc, double sum_inc)
{
    if(sum_inc <= 0.0) return 0.0;
    return 100.0 * (sum_exc / sum_inc);
}

// Same banner caption timemory uses for wall_clock text (ASCII pipe table; no tim
// stream).
void
write_wall_clock_ascii_table(std::ostream& os, const std::vector<const agg_stats*>& rows)
{
    os << "# REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)\n";
    os << "| LABEL | COUNT | DEPTH | METRIC | UNITS | SUM | MEAN | MIN | MAX | VAR | "
          "STDDEV | % SELF |\n";
    os << std::fixed << std::setprecision(6);
    for(const agg_stats* as : rows)
    {
        const double n    = static_cast<double>(as->count);
        const double mean = (as->count > 0) ? (as->sum_inc / n) : 0.0;
        const double var  = (as->count > 0) ? ((as->sum_sq_inc / n) - mean * mean) : 0.0;
        const double stddev   = std::sqrt(std::max(0.0, var));
        const double pct      = safe_pct_self(as->sum_exc, as->sum_inc);
        const std::string lbl = "|" + std::to_string(as->thread_seq) + ">>> " + as->name;
        os << "| " << lbl << " | " << as->count << " | " << as->depth
           << " | wall_clock | sec | " << as->sum_inc << " | " << mean << " | "
           << as->min_inc << " | " << as->max_inc << " | " << var << " | " << stddev
           << " | " << pct << " |\n";
    }
}

std::int64_t
compute_span_depth(std::uint64_t                                           span_id,
                   const std::unordered_map<std::uint64_t, std::uint64_t>& parent_of)
{
    std::int64_t           d         = 0;
    std::uint64_t          cur       = span_id;
    constexpr std::int64_t guard_max = 1 << 22;
    for(std::int64_t guard = 0; guard < guard_max; ++guard)
    {
        auto pit = parent_of.find(cur);
        if(pit == parent_of.end()) break;
        const std::uint64_t p = pit->second;
        if(p == 0) break;
        ++d;
        cur = p;
    }
    return d;
}

}  // namespace

void
wall_clock_report_processor_t::compute_span_preorder_indices(
    const std::vector<completed_span>&                completed,
    std::unordered_map<std::uint64_t, std::uint64_t>& span_preorder_out)
{
    span_preorder_out.clear();

    std::unordered_map<std::uint64_t, std::vector<const completed_span*>> by_thread;
    by_thread.reserve(8);
    for(const auto& sp : completed)
        by_thread[sp.thread_seq_id].push_back(&sp);

    for(auto& tpair : by_thread)
    {
        auto& tvec = tpair.second;

        std::unordered_set<std::uint64_t> all_ids;
        all_ids.reserve(tvec.size() * 2);
        std::unordered_map<std::uint64_t, std::uint64_t> t_start_ns;
        t_start_ns.reserve(tvec.size() * 2);

        for(const completed_span* sr : tvec)
        {
            all_ids.insert(sr->span_id);
            t_start_ns[sr->span_id] = sr->t_start_ns;
        }

        std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> children;
        for(const completed_span* sr : tvec)
        {
            if(sr->parent_span_id != 0)
                children[sr->parent_span_id].push_back(sr->span_id);
        }

        for(auto& ch : children)
        {
            auto& vec = ch.second;
            std::sort(vec.begin(), vec.end(), [&](std::uint64_t a, std::uint64_t b) {
                return t_start_ns.at(a) < t_start_ns.at(b);
            });
        }

        std::vector<std::uint64_t> roots;
        for(const completed_span* sr : tvec)
        {
            if(sr->parent_span_id == 0 || all_ids.count(sr->parent_span_id) == 0)
                roots.push_back(sr->span_id);
        }

        std::sort(roots.begin(), roots.end(), [&](std::uint64_t a, std::uint64_t b) {
            return t_start_ns.at(a) < t_start_ns.at(b);
        });
        roots.erase(std::unique(roots.begin(), roots.end()), roots.end());

        std::vector<std::uint64_t> dfs_order;
        dfs_order.reserve(tvec.size());
        std::unordered_set<std::uint64_t> visited;
        visited.reserve(tvec.size() * 2);

        std::function<void(std::uint64_t)> dfs = [&](std::uint64_t sid) {
            if(visited.count(sid) != 0U) return;
            visited.insert(sid);
            dfs_order.push_back(sid);
            auto cit = children.find(sid);
            if(cit != children.end())
            {
                for(std::uint64_t cid : cit->second)
                    dfs(cid);
            }
        };

        for(std::uint64_t r : roots)
        {
            if(visited.count(r) == 0U) dfs(r);
        }

        for(const completed_span* sr : tvec)
        {
            const auto sid = sr->span_id;
            if(visited.count(sid) == 0U) dfs(sid);
        }

        for(size_t i = 0; i < dfs_order.size(); ++i)
            span_preorder_out.emplace(dfs_order[i], static_cast<std::uint64_t>(i));
    }
}

wall_clock_report_processor_t::wall_clock_report_processor_t(int pid, int ppid,
                                                             output_file_registry& out)
: m_pid(pid)
, m_ppid(ppid)
, m_output_registry(out)
{}

void
wall_clock_report_processor_t::prepare_for_processing()
{
    std::lock_guard<std::mutex> lk{ m_mutex };
    m_path_hash_for_span.clear();
    m_open_begin.clear();
    m_completed.clear();
}

std::uint64_t
wall_clock_report_processor_t::path_hash_combine(std::uint64_t      parent_path_hash,
                                                 const std::string& name,
                                                 const std::string& category)
{
    auto fnv = [](std::uint64_t h, const std::string& s) {
        constexpr std::uint64_t prime = 1099511628211ULL;
        for(unsigned char c : s)
        {
            h ^= c;
            h *= prime;
        }
        return h;
    };
    std::uint64_t h = fnv(parent_path_hash, name);
    h               = fnv(h, "|");
    return fnv(h, category);
}

void
wall_clock_report_processor_t::handle(const wall_clock_span_begin_sample& sample)
{
    std::lock_guard<std::mutex> lk{ m_mutex };

    std::uint64_t parent_ph = 0;
    if(sample.parent_span_id != 0)
    {
        auto pit = m_path_hash_for_span.find(sample.parent_span_id);
        if(pit != m_path_hash_for_span.end()) parent_ph = pit->second;
    }

    const std::uint64_t mph = path_hash_combine(parent_ph, sample.name, sample.category);

    m_path_hash_for_span[sample.span_id] = mph;

    span_begin_state st{};
    st.parent_span_id = sample.parent_span_id;
    st.thread_seq_id  = sample.thread_seq_id;
    st.path_hash      = mph;
    st.t_start_ns     = sample.timestamp_ns;
    st.name           = sample.name;
    st.category       = sample.category;

    m_open_begin.insert_or_assign(sample.span_id, std::move(st));
}

void
wall_clock_report_processor_t::handle(const wall_clock_span_end_sample& sample)
{
    std::lock_guard<std::mutex> lk{ m_mutex };

    auto it = m_open_begin.find(sample.span_id);
    if(it == m_open_begin.end())
    {
        LOG_WARNING("wall_clock_span_end without matching begin for span_id={}",
                    sample.span_id);
        return;
    }

    span_begin_state beg = it->second;
    m_open_begin.erase(it);

    if(sample.timestamp_ns < beg.t_start_ns)
    {
        LOG_WARNING("wall_clock_span_end timestamp before begin for span_id={}",
                    sample.span_id);
        return;
    }

    completed_span done;
    done.span_id        = sample.span_id;
    done.parent_span_id = beg.parent_span_id;
    done.thread_seq_id  = beg.thread_seq_id;
    done.path_hash      = beg.path_hash;
    done.t_start_ns     = beg.t_start_ns;
    done.t_end_ns       = sample.timestamp_ns;
    done.name           = std::move(beg.name);
    done.category       = std::move(beg.category);

    m_completed.emplace_back(std::move(done));
}

void
wall_clock_report_processor_t::write_report(const std::string& path) const
{
    if(m_completed.empty())
    {
        LOG_DEBUG("wall_clock_report_processor: no completed spans for pid={}", m_pid);
        return;
    }

    std::unordered_map<std::uint64_t, double>                     inclusive_sec;
    std::unordered_map<std::uint64_t, std::uint64_t>              parent_of;
    std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> children;

    for(const auto& sp : m_completed)
    {
        const double sec =
            static_cast<double>(sp.t_end_ns - sp.t_start_ns) * k_sec_per_ns;
        inclusive_sec[sp.span_id] = sec;
        parent_of.emplace(sp.span_id, sp.parent_span_id);
        if(sp.parent_span_id != 0) children[sp.parent_span_id].push_back(sp.span_id);
    }

    std::unordered_map<std::uint64_t, std::uint64_t> span_preorder;
    compute_span_preorder_indices(m_completed, span_preorder);

    std::unordered_map<agg_key, agg_stats, agg_key_hash> table;

    for(const auto& sp : m_completed)
    {
        double child_sum = 0.0;
        auto   cit       = children.find(sp.span_id);
        if(cit != children.end())
        {
            for(auto cid : cit->second)
            {
                auto inc_it = inclusive_sec.find(cid);
                if(inc_it != inclusive_sec.end()) child_sum += inc_it->second;
            }
        }

        const double inc = inclusive_sec.at(sp.span_id);
        const double exc = std::max(0.0, inc - child_sum);

        std::uint64_t pr = std::numeric_limits<std::uint64_t>::max();
        {
            const auto pit = span_preorder.find(sp.span_id);
            if(pit != span_preorder.end()) pr = pit->second;
        }

        agg_key k{ sp.thread_seq_id, sp.path_hash };
        auto    ait = table.find(k);
        if(ait == table.end())
        {
            agg_stats as{};
            as.thread_seq    = sp.thread_seq_id;
            as.preorder_rank = pr;
            as.name          = sp.name;
            as.category      = sp.category;
            as.depth         = compute_span_depth(sp.span_id, parent_of);
            as.count         = 1;
            as.sum_inc       = inc;
            as.sum_exc       = exc;
            as.min_inc       = inc;
            as.max_inc       = inc;
            as.sum_sq_inc += inc * inc;
            as.sum_sq_exc += exc * exc;
            table.emplace(k, std::move(as));
        }
        else
        {
            agg_stats& as = ait->second;
            ++as.count;
            as.preorder_rank = std::min(as.preorder_rank, pr);
            as.sum_inc += inc;
            as.sum_exc += exc;
            as.min_inc = std::min(as.min_inc, inc);
            as.max_inc = std::max(as.max_inc, inc);
            as.sum_sq_inc += inc * inc;
            as.sum_sq_exc += exc * exc;
        }
    }

    std::vector<const agg_stats*> rows;
    rows.reserve(table.size());
    for(const auto& kv : table)
        rows.push_back(&kv.second);

    std::sort(rows.begin(), rows.end(), [](const agg_stats* a, const agg_stats* b) {
        if(a->thread_seq != b->thread_seq) return a->thread_seq < b->thread_seq;
        if(a->preorder_rank != b->preorder_rank)
            return a->preorder_rank < b->preorder_rank;
        if(a->depth != b->depth) return a->depth < b->depth;
        return a->name < b->name;
    });

    const auto dir = tim::filepath::dirname(path);
    if(!dir.empty() && !tim::filepath::direxists(dir)) tim::filepath::makedir(dir);

    std::ofstream os(path);
    if(!os.good())
    {
        LOG_ERROR("Failed to open wall-clock report for writing: {}", path);
        return;
    }

    write_wall_clock_ascii_table(os, rows);

    os.close();
    LOG_INFO("Wrote native wall-clock report: {}", path);
}

void
wall_clock_report_processor_t::finalize_processing()
{
    std::lock_guard<std::mutex> lk{ m_mutex };

    const auto orphan_count = m_open_begin.size();
    if(orphan_count > 0)
    {
        LOG_DEBUG("wall_clock_report_processor: closing {} span begin(s) without end "
                  "(non-final thread or exit before pop); using zero-duration interval",
                  orphan_count);

        for(const auto& kv : m_open_begin)
        {
            const auto&             sid = kv.first;
            const span_begin_state& beg = kv.second;
            completed_span          done{};
            done.span_id        = sid;
            done.parent_span_id = beg.parent_span_id;
            done.thread_seq_id  = beg.thread_seq_id;
            done.path_hash      = beg.path_hash;
            done.t_start_ns     = beg.t_start_ns;
            done.t_end_ns       = beg.t_start_ns;
            done.name           = beg.name;
            done.category       = beg.category;
            m_completed.emplace_back(std::move(done));
        }
        m_open_begin.clear();
    }

    if(m_completed.empty()) return;

    const auto path = config::get_wall_clock_report_filename(m_pid);
    write_report(path);
    m_output_registry.register_file(path, output_format::text, "wall_clock");
}

}  // namespace trace_cache
}  // namespace rocprofsys
