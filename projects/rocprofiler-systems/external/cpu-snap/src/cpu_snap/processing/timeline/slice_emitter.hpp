#pragma once

/**
 * @file slice_emitter.hpp
 * @brief Backend-agnostic emitter that converts per-tid core::raw_sample
 *        vectors into a stream of nested `emitted_slice` records (one
 *        slice per frame-span, flame-graph view).
 *
 * The caller provides a resolver (IP -> function name) and a sink
 * callback. Symbolization, filtering,
 * coalescing live elsewhere; this module is purely the timeline-
 * shaping pass.
 *
 * Backends (Perfetto, pprof, flamegraph-collapsed, ...) consume
 * `emitted_slice` records and map them to their native record types.
 *
 * Single-threaded; safe to call after sampler::stop(). The templated
 * process() / emit_nested() entry points live here;
 * non-template helpers live in slice_emitter.cpp.
 */

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/types.h>

#include "cpu_snap/core/raw_sample.hpp"

namespace cpu_snap::processing::timeline
{

// NOLINTBEGIN(readability-identifier-naming)
inline constexpr std::uint64_t k_default_sample_period_ns = 1'000'000ULL;
// NOLINTEND(readability-identifier-naming)

/**
 * @brief One emitted slice handed to the backend sink.
 *
 * @note `name` views storage owned by the emitter; valid only inside
 *       the sink call.
 */
struct emitted_slice
{
    /** Linux thread id of the sampled thread the slice belongs to. */
    pid_t tid;
    /** Slice label (borrowed; valid only inside the sink call). */
    std::string_view name;
    /** Slice start timestamp in nanoseconds (stream clock). */
    std::uint64_t start_ts;
    /** Slice end timestamp in nanoseconds (stream clock). */
    std::uint64_t end_ts;
    /**
     * Nesting depth of this slice, 0 = outermost (root). A backend that
     * reconstructs begin/end ordering from completed spans MUST break
     * same-timestamp ties by this depth (open shallow-first, close
     * deep-first); timestamps alone do not encode nesting when a whole
     * sub-stack shares one sample's [start, end).
     */
    std::uint32_t depth;
};

/**
 * @brief Emitter behaviour knobs.
 */
struct slice_emitter_config
{
    /**
     * Nested mode only. When true (default), a frame that stays on the stack
     * across consecutive samples is one slice that spans all of them
     * (flame-graph view). When false, nothing is carried over between
     * samples: every sample emits its own full stack as nested slices that
     * span [its timestamp, the next sample's timestamp), so the timeline
     * shows exactly what each sample captured.
     */
    bool merge_samples = true;

    /** Period in ns used to bound the close of still-active frames and the
     *  last sample's region. */
    std::uint64_t sample_period_ns = k_default_sample_period_ns;  // 1 ms = 1 kHz fallback

    /** IP -> function-name resolver. Must not throw. Empty string =
     *  emitter falls back to "0xADDR". */
    std::function<std::string(std::uint64_t)> resolver;

    /**
     * Optional IP -> inline-frame expansion. When set, each
     * sampled frame IP is expanded into the returned innermost-first chain
     * of formatted frame-name strings and the emitter shapes one slice per
     * chain entry, with the root inline frame as the outermost slice. The
     * stack diff is then performed on frame names rather than IPs, and each
     * emitted_slice.name is the corresponding inline frame name. An empty
     * (or unset) resolver leaves the emitter in its IP-based behaviour.
     */
    std::function<std::vector<std::string>(std::uint64_t)> inline_resolver;

    /**
     * Optional IP -> symbol-identity key, used by the non-inline
     * emit_nested() stack-diff to decide whether two different IPs at
     * the same depth belong to the same frame (e.g. two call sites
     * inside one function). Two IPs are treated as the same frame when
     * this returns equal, non-nullopt keys for both; when either IP
     * fails to resolve (or this callback is unset), the diff falls back
     * to raw-IP equality.
     *
     * @note Honoured only by emit_nested()'s IP-based diff; unused when
     *       @c inline_resolver is set (emit_nested_inline() diffs on
     *       resolved names instead).
     */
    std::function<std::optional<std::uint64_t>(std::uint64_t)> symbol_identity;

    /**
     * Optional per-frame keep/drop predicate, applied to every raw
     * frame before diffing: `bool(ip, name)` returning false drops the
     * frame. `name` is always empty; the parameter exists so a single
     * predicate shape can be shared with cpu_snap::processing::symbols's frame
     * filters.
     *
     * Unset (default): no filtering. timeline does not construct or
     * own a module source itself; the caller builds this predicate
     * from whatever it has -- a live process's mappings or a decoded
     * capture's module map -- and passes it in ready-made.
     *
     * @see cpu_snap::processing::symbols::make_main_exe_only_filter, whose return
     *      value matches this field's signature directly.
     */
    std::function<bool(std::uint64_t, std::string_view)> library_frame_filter;
};

/**
 * @brief Convert per-tid `core::raw_sample` vectors into a stream of
 *        emitted_slice records and hand each to a sink callback.
 */
class slice_emitter
{
public:
    /**
     * @brief Construct an emitter from @p cfg.
     * @param cfg Emitter behaviour knobs. If @c cfg.resolver is empty,
     *            the emitter falls back to a hex-address resolver.
     */
    explicit slice_emitter(slice_emitter_config cfg);

    /**
     * @brief Walk every tid in @p samples, sort each per-tid vector by
     *        timestamp, and feed slices to @p sink.
     * @param samples Per-tid core::raw_sample storage produced by sampler.
     * @param sink    Callable invoked once per emitted_slice. Slice
     *                string views are valid only inside the call.
     * @return Total number of slices emitted.
     */
    // sink is used as a callback (called many times), so the parameter
    // takes it by lvalue reference rather than being forwarded.
    template <typename Sink>
    std::size_t process(const std::unordered_map<pid_t, std::vector<core::raw_sample>>& samples,
                        Sink&& sink)  // NOLINT(cppcoreguidelines-missing-std-forward)
    {
        std::size_t emitted = 0;
        for(const auto& [tid, vec_in] : samples)
        {
            if(vec_in.empty())
            {
                continue;
            }

            const auto by_timestamp = [](const core::raw_sample& aa, const core::raw_sample& bb) {
                return aa.timestamp_ns < bb.timestamp_ns;
            };

            if(!m_cfg.library_frame_filter &&
               std::is_sorted(vec_in.begin(), vec_in.end(), by_timestamp))
            {
                emitted += emit_nested(tid, vec_in, sink);
                continue;
            }

            std::vector<core::raw_sample> vec(vec_in);
            std::stable_sort(vec.begin(), vec.end(), by_timestamp);

            if(m_cfg.library_frame_filter)
            {
                for(core::raw_sample& smp : vec)
                {
                    filter_library_frames(smp);
                }
            }

            emitted += emit_nested(tid, vec, sink);
        }
        return emitted;
    }

private:
    /**
     * @brief Apply @c m_cfg.library_frame_filter to every frame of
     *        @p smp in place, compacting survivors and refreshing
     *        @c smp.ip to the new innermost frame.
     */
    void filter_library_frames(core::raw_sample& smp) const;

    /**
     * @brief Resolve @p ip to a function name via @c m_cfg.resolver,
     *        falling back to "0xADDR" if the resolver throws.
     */
    [[nodiscard]] std::string resolve_name(std::uint64_t ip) const;

    /**
     * @brief Whether @p lhs and @p rhs resolve to the same symbol via
     *        @c m_cfg.symbol_identity.
     *
     * @c false when the callback is unset or either IP fails to resolve
     * to a known symbol.
     */
    [[nodiscard]] bool resolves_to_same_symbol(std::uint64_t lhs, std::uint64_t rhs) const;

    template <typename Sink>
    std::size_t emit_nested(pid_t tid, const std::vector<core::raw_sample>& vec, Sink& sink)
    {
        if(!m_cfg.merge_samples)
        {
            return emit_nested_unmerged(tid, vec, sink);
        }
        if(m_cfg.inline_resolver)
        {
            return emit_nested_inline(tid, vec, sink);
        }

        struct open_frame
        {
            std::uint64_t ip;
            std::uint64_t start_ts;
        };
        std::vector<open_frame> active;
        std::uint64_t           last_ts = 0;
        std::size_t             count   = 0;

        auto emit_close = [&](const open_frame& fr, std::uint64_t end_ts,
                              std::uint32_t depth) {
            const std::string   name = resolve_name(fr.ip);
            const emitted_slice slc{ tid, name, fr.start_ts, end_ts, depth };
            sink(slc);
            ++count;
        };

        for(const auto& smp : vec)
        {
            const std::uint64_t ts = smp.timestamp_ns;

            std::vector<std::uint64_t> new_stack;
            new_stack.reserve(smp.frames.size());
            for(std::int32_t i = static_cast<std::int32_t>(smp.frames.size()) - 1; i >= 0; --i)
            {
                new_stack.push_back(smp.frames[i]);
            }

            std::size_t       cp     = 0;
            const std::size_t max_cp = std::min(active.size(), new_stack.size());
            while(cp < max_cp && active[cp].ip == new_stack[cp])
            {
                ++cp;
            }

            // A frame's own recorded PC can only differ across samples
            // once every callee below it has fully returned, so leniency
            // can apply at most once per transition, at the shallowest
            // divergence; anything deeper is a genuinely fresh call and
            // must match by raw IP.
            if(cp < max_cp && resolves_to_same_symbol(active[cp].ip, new_stack[cp]))
            {
                active[cp].ip = new_stack[cp];
                ++cp;
                while(cp < max_cp && active[cp].ip == new_stack[cp])
                {
                    ++cp;
                }
            }

            for(std::size_t n = active.size(); n > cp; --n)
            {
                emit_close(active.back(), ts, static_cast<std::uint32_t>(n - 1));
                active.pop_back();
            }
            for(std::size_t i = cp; i < new_stack.size(); ++i)
            {
                active.push_back({ new_stack[i], ts });
            }

            last_ts = ts;
        }

        const std::uint64_t final_ts = last_ts + m_cfg.sample_period_ns;
        while(!active.empty())
        {
            emit_close(active.back(), final_ts, static_cast<std::uint32_t>(active.size() - 1));
            active.pop_back();
        }
        return count;
    }

    /**
     * @brief Nested emission with inline-frame expansion.
     *
     * Each sampled frame IP is expanded via @c m_cfg.inline_resolver into
     * its innermost-first chain and pushed outermost-first so the root
     * inline frame becomes the outer slice. The stack diff runs on frame
     * names so slices that persist across consecutive samples (including
     * inlined sub-frames) stay open exactly as the IP-only path does.
     * An IP whose chain is empty falls back to a single hex-address name.
     *
     * @param tid        Linux tid the slices belong to.
     * @param vec        Timestamp-sorted samples for this tid.
     * @param sink       Sink invoked once per emitted_slice.
     * @return Number of slices emitted.
     */
    template <typename Sink>
    std::size_t emit_nested_inline(pid_t tid, const std::vector<core::raw_sample>& vec,
                                   Sink& sink)
    {
        struct open_frame
        {
            std::string   name;
            std::uint64_t start_ts;
        };
        std::vector<open_frame> active;
        std::uint64_t           last_ts = 0;
        std::size_t             count   = 0;

        auto emit_close = [&](const open_frame& fr, std::uint64_t end_ts,
                              std::uint32_t depth) {
            const emitted_slice slc{ tid, fr.name, fr.start_ts, end_ts, depth };
            sink(slc);
            ++count;
        };

        for(const auto& smp : vec)
        {
            const std::uint64_t ts = smp.timestamp_ns;

            std::vector<std::string> new_stack;
            new_stack.reserve(static_cast<std::size_t>(smp.frames.size()) * 2);
            for(std::int32_t i = static_cast<std::int32_t>(smp.frames.size()) - 1; i >= 0; --i)
            {
                std::vector<std::string> chain = m_cfg.inline_resolver(smp.frames[i]);
                if(chain.empty())
                {
                    new_stack.push_back(resolve_name(smp.frames[i]));
                    continue;
                }
                // chain is innermost-first; push outermost-first so the
                // root inline frame becomes the outer slice.
                for(auto it = chain.rbegin(); it != chain.rend(); ++it)
                {
                    new_stack.push_back(std::move(*it));
                }
            }

            std::size_t       cp     = 0;
            const std::size_t max_cp = std::min(active.size(), new_stack.size());
            while(cp < max_cp && active[cp].name == new_stack[cp])
            {
                ++cp;
            }

            for(std::size_t n = active.size(); n > cp; --n)
            {
                emit_close(active.back(), ts, static_cast<std::uint32_t>(n - 1));
                active.pop_back();
            }
            for(std::size_t i = cp; i < new_stack.size(); ++i)
            {
                active.push_back({ std::move(new_stack[i]), ts });
            }

            last_ts = ts;
        }

        const std::uint64_t final_ts = last_ts + m_cfg.sample_period_ns;
        while(!active.empty())
        {
            emit_close(active.back(), final_ts, static_cast<std::uint32_t>(active.size() - 1));
            active.pop_back();
        }
        return count;
    }

    /**
     * @brief Nested emission with no merging across samples.
     *
     * Every sample emits its full stack as nested slices over
     * [its timestamp, next sample's timestamp); the last sample lasts
     * @c m_cfg.sample_period_ns. A gap between samples is covered by the
     * earlier sample. Slices of one sample are emitted deepest first.
     * Honours @c m_cfg.inline_resolver (inline chain names); otherwise
     * names come from the IP.
     *
     * @param tid        Linux tid the slices belong to.
     * @param vec        Timestamp-sorted samples for this tid.
     * @param sink       Sink invoked once per emitted_slice.
     * @return Number of slices emitted.
     */
    template <typename Sink>
    std::size_t emit_nested_unmerged(pid_t tid, const std::vector<core::raw_sample>& vec,
                                     Sink& sink)
    {
        std::vector<std::string> stack;
        std::size_t              count = 0;

        for(std::size_t i = 0; i < vec.size(); ++i)
        {
            const auto&         smp   = vec[i];
            const std::uint64_t start = smp.timestamp_ns;
            const std::uint64_t end   = (i + 1 < vec.size())
                                            ? vec[i + 1].timestamp_ns
                                            : start + m_cfg.sample_period_ns;

            stack.clear();
            for(std::int32_t f = static_cast<std::int32_t>(smp.frames.size()) - 1; f >= 0; --f)
            {
                const std::uint64_t ip = smp.frames[f];
                if(!m_cfg.inline_resolver)
                {
                    stack.push_back(resolve_name(ip));
                    continue;
                }
                std::vector<std::string> chain = m_cfg.inline_resolver(ip);
                if(chain.empty())
                {
                    stack.push_back(resolve_name(ip));
                    continue;
                }
                // chain is innermost-first; push outermost-first.
                for(auto it = chain.rbegin(); it != chain.rend(); ++it)
                {
                    stack.push_back(std::move(*it));
                }
            }

            for(std::size_t depth = stack.size(); depth > 0; --depth)
            {
                const emitted_slice slc{ tid, stack[depth - 1], start, end,
                                         static_cast<std::uint32_t>(depth - 1) };
                sink(slc);
                ++count;
            }
        }
        return count;
    }

    slice_emitter_config m_cfg;
};

}  // namespace cpu_snap::processing::timeline
