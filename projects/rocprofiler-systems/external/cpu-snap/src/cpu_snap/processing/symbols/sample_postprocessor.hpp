#pragma once

/**
 * @file sample_postprocessor.hpp
 * @brief Offline post-processing for core::raw_sample records produced by
 *        cpu_snap::sampling::sampler.
 *
 * Splits expensive work (symbol resolution, predicate filtering,
 * user-callable sinks) OUT of the hot recording path. Run this AFTER
 * sampler::stop().
 *
 * Pipeline (per core::raw_sample):
 *   -# For each IP in the callchain: call config.resolver (cached
 *      per IP) to get a function name.
 *   -# config.frame_predicate(ip, name) may drop the frame.
 *   -# The sink callback is invoked for survivors.
 *
 * @note Symbol cache, frame scratch, and processed_sample storage are
 *       all owned by the postprocessor. The string_views inside
 *       processed_frame reference cache storage and are valid until
 *       the postprocessor is destroyed (or clear_cache() is called).
 * @note Header-only. C++20. Single-threaded.
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_snap/core/module_map.hpp"
#include "cpu_snap/core/raw_sample.hpp"
#include "cpu_snap/processing/symbols/frame_filters.hpp"
#include "cpu_snap/processing/symbols/proc_maps.hpp"

namespace cpu_snap::processing::symbols
{

/**
 * @brief One resolved frame, as emitted by the postprocessor.
 */
struct processed_frame
{
    /** Raw instruction pointer. */
    std::uint64_t    ip;
    /**
     * Function name as returned by the resolver. May be empty when no
     * resolver was configured or the resolver could not symbolize the
     * IP. View is stable for the lifetime of the postprocessor (or
     * until clear_cache()).
     */
    std::string_view resolved;
};

/**
 * @brief One sample after symbolization and filtering.
 */
struct processed_sample
{
    /** Linux tid of the sampled thread. */
    pid_t                            tid;
    /** Sample timestamp in nanoseconds. */
    std::uint64_t                    timestamp_ns;
    /** Instruction pointer at sample time (top of stack). */
    std::uint64_t                    ip;
    /** Innermost-first callchain. Span is valid only inside the sink call. */
    std::span<const processed_frame> frames;
};

/**
 * @brief Configuration for the postprocessor.
 */
struct sample_postprocessor_config
{
    /**
     * Resolves an IP to a function name. Called at most once per
     * unique IP; the result is cached. If null, all resolved
     * string_views are empty.
     */
    std::function<std::string(std::uint64_t ip)> resolver;

    /**
     * Return false to drop a frame from the processed sample. `name`
     * is the cached resolver output and may be empty.
     */
    std::function<bool(std::uint64_t ip, std::string_view name)> frame_predicate;

    /**
     * When true, the postprocessor lazily builds a library-frame
     * filter on first use and drops every frame whose IP belongs to a
     * shared-library mapping. Frames in the main executable mapping
     * (and frames whose IP falls outside any known mapping, e.g. JIT
     * pages) are kept.
     *
     * The filter's data source depends on @ref library_filter_modules:
     * when null (default), a live cpu_snap::processing::symbols::proc_maps is
     * constructed and reloaded from /proc/self/maps -- only correct in
     * the same, still-live process. When set, the filter resolves
     * through that module_map instead, at @ref
     * library_filter_timestamp_ns, against @ref main_exe_path -- the
     * offline / cross-process path.
     *
     * Composes with @ref frame_predicate via AND: a frame survives
     * iff the library filter accepts it AND the user predicate (when
     * set) accepts it.
     *
     * Default: false (no filtering; back-compat).
     *
     * @see cpu_snap::processing::symbols::make_main_exe_only_filter for the
     *      underlying predicate; setting this field is equivalent to
     *      chaining that filter manually via @ref frame_predicate.
     */
    bool drop_library_frames = false;

    /**
     * Module_map to resolve the library filter through instead of a
     * live /proc/self/maps read. Ignored unless @ref
     * drop_library_frames is true. Must outlive the postprocessor.
     */
    const cpu_snap::core::module_map* library_filter_modules = nullptr;

    /** Timestamp the library filter resolves every ip against; only
     *  used when @ref library_filter_modules is set. */
    std::uint64_t library_filter_timestamp_ns = 0;

    /** Path of the module considered "main executable"; only used
     *  when @ref library_filter_modules is set (module_map carries no
     *  such flag itself). */
    std::string main_exe_path;
};

/**
 * @brief Offline post-processor for core::raw_sample records.
 *
 * Symbolizes IPs, applies frame / sample predicates, and feeds
 * survivors to a sink. Symbol resolutions are cached per IP for the
 * lifetime of the instance.
 */
class sample_postprocessor
{
public:
    /**
     * @brief Construct a postprocessor from @p cfg.
     * @param cfg Resolver + predicates + library-filter toggle.
     */
    explicit sample_postprocessor(sample_postprocessor_config cfg)
    : m_cfg{ std::move(cfg) }
    {}

    ~sample_postprocessor()                                      = default;
    sample_postprocessor(const sample_postprocessor&)            = delete;
    sample_postprocessor& operator=(const sample_postprocessor&) = delete;
    sample_postprocessor(sample_postprocessor&&)                 = default;
    sample_postprocessor& operator=(sample_postprocessor&&)      = default;

    /**
     * @brief Process a span of core::raw_sample records.
     *
     * The sink is invoked once per surviving sample, in input order.
     *
     * @param raws Samples to iterate.
     * @param sink Invocable accepting `const processed_sample&`.
     *
     * @warning Pointer / span references inside processed_sample
     *          become invalid after the sink returns. Copy out if
     *          you need the data past that scope.
     */
    // sink is used as a callback (called many times per call), so the
    // parameter is consumed by lvalue reference rather than forwarded.
    template <typename Sink>
    void process(std::span<const core::raw_sample> raws,
                 Sink&& sink)  // NOLINT(cppcoreguidelines-missing-std-forward)
    {
        for(const core::raw_sample& rec : raws)
        {
            process_one(rec, sink);
        }
    }

    /**
     * @brief Iterate every tid in a per-tid sample map (as produced by
     *        sampler::samples()).
     * @param by_tid Map of tid -> vector<core::raw_sample>.
     * @param sink   Invocable accepting `const processed_sample&`.
     */
    template <typename Sink>
    void
    process_all(const std::unordered_map<pid_t, std::vector<core::raw_sample>>& by_tid,
                Sink&& sink)  // NOLINT(cppcoreguidelines-missing-std-forward)
    {
        for(const auto& [tid, vec] : by_tid)
        {
            process(std::span<const core::raw_sample>(vec), sink);
        }
    }

    /**
     * @brief Drop cached symbol resolutions.
     *
     * Useful if the binary mapping changes between runs.
     */
    void clear_cache() noexcept { m_symbol_cache.clear(); }

    /**
     * @brief Samples passed to the sink.
     * @return Cumulative delivered sample count.
     */
    [[nodiscard]] std::size_t delivered_samples() const noexcept { return m_delivered; }

    /**
     * @brief Frames discarded by frame_predicate across all samples.
     * @return Cumulative dropped frame count.
     */
    [[nodiscard]] std::size_t dropped_frames() const noexcept { return m_dropped_frames; }

    /**
     * @brief Number of unique IPs the resolver has been called on.
     * @return Symbol-cache size.
     */
    [[nodiscard]] std::size_t unique_symbols() const noexcept { return m_symbol_cache.size(); }

private:
    template <typename Sink>
    void process_one(const core::raw_sample& rec, Sink& sink)
    {
        ensure_library_filter();

        m_scratch.clear();
        m_scratch.reserve(rec.frames.size());
        for(std::uint32_t i = 0; i < rec.frames.size(); ++i)
        {
            const std::uint64_t    ip   = rec.frames[i];
            const std::string_view name = resolve_cached(ip);
            if(m_library_filter && !m_library_filter(ip, name))
            {
                ++m_dropped_frames;
                continue;
            }
            if(m_cfg.frame_predicate && !m_cfg.frame_predicate(ip, name))
            {
                ++m_dropped_frames;
                continue;
            }
            m_scratch.push_back({ ip, name });
        }

        processed_sample out{};
        out.tid          = rec.tid;
        out.timestamp_ns = rec.timestamp_ns;
        out.ip           = rec.ip;
        out.frames       = std::span<const processed_frame>(m_scratch);

        ++m_delivered;
        sink(out);
    }

    std::string_view resolve_cached(std::uint64_t ip)
    {
        if(!m_cfg.resolver)
        {
            return {};
        }
        if(auto it = m_symbol_cache.find(ip); it != m_symbol_cache.end())
        {
            return std::string_view(it->second);
        }
        std::string name;
        try
        {
            name = m_cfg.resolver(ip);
        } catch(...)  // NOLINT(bugprone-empty-catch)
        {
            // Resolver failure: keep name empty; downstream callers
            // treat unresolved IPs as anonymous frames.
        }
        auto [it, _] = m_symbol_cache.emplace(ip, std::move(name));
        return std::string_view(it->second);
    }

    void ensure_library_filter()
    {
        if(!m_cfg.drop_library_frames || m_library_filter)
        {
            return;
        }
        if(m_cfg.library_filter_modules != nullptr)
        {
            m_library_filter = cpu_snap::processing::symbols::make_main_exe_only_filter(
                *m_cfg.library_filter_modules, m_cfg.library_filter_timestamp_ns,
                m_cfg.main_exe_path);
            return;
        }
        m_library_maps   = std::make_unique<cpu_snap::processing::symbols::proc_maps>();
        m_library_filter = cpu_snap::processing::symbols::make_main_exe_only_filter(*m_library_maps);
    }

    sample_postprocessor_config                    m_cfg;
    std::unordered_map<std::uint64_t, std::string> m_symbol_cache;
    std::vector<processed_frame>                   m_scratch;
    std::size_t                                    m_delivered       = 0;
    std::size_t                                    m_dropped_frames  = 0;
    std::unique_ptr<cpu_snap::processing::symbols::proc_maps>  m_library_maps;
    std::function<bool(std::uint64_t, std::string_view)> m_library_filter;
};

/**
 * @brief Replace every frame IP in @p samples with a canonical IP
 *        per resolved function name.
 *
 * The kernel frame-pointer walker emits slightly different IPs
 * inside the same function across consecutive samples (different
 * basic blocks). Without this pass, downstream trace writers produce
 * one open/close cycle per sample even when the function did not
 * change, exploding the slice count.
 *
 * Apply AFTER recording, BEFORE writing nested slices to Perfetto /
 * flamegraph-collapsed / pprof. The resolver is called at most once
 * per unique IP (results cached internally).
 *
 * @tparam Resolver Invocable `std::string(std::uint64_t ip)`.
 * @param  samples  Per-tid samples to mutate in place.
 * @param  resolver IP -> function-name resolver.
 *
 * @note Workload-time cost is zero; this is an offline pass on
 *       already-recorded data. sampler emits raw IPs unchanged.
 */
// resolver is invoked many times across all samples; the parameter is
// consumed by lvalue reference rather than forwarded.
template <typename Resolver>
inline void
coalesce_frames_by_name(std::unordered_map<pid_t, std::vector<core::raw_sample>>& samples,
                        Resolver&& resolver)  // NOLINT(cppcoreguidelines-missing-std-forward)
{
    std::unordered_map<std::uint64_t, std::uint64_t> ip_to_canonical;
    std::unordered_map<std::string, std::uint64_t>   name_to_canonical;

    for(auto& [tid, vec] : samples)
    {
        for(auto& rec : vec)
        {
            for(std::uint32_t i = 0; i < rec.frames.size(); ++i)
            {
                const std::uint64_t ip = rec.frames[i];
                if(auto it = ip_to_canonical.find(ip); it != ip_to_canonical.end())
                {
                    rec.frames[i] = it->second;
                    continue;
                }
                std::string name;
                try
                {
                    name = resolver(ip);
                } catch(...)  // NOLINT(bugprone-empty-catch)
                {
                    // Resolver failure: treat IP as anonymous; falls
                    // through to the name.empty() branch below.
                }
                if(name.empty())
                {
                    ip_to_canonical.emplace(ip, ip);
                    continue;
                }
                auto [nit, inserted] = name_to_canonical.try_emplace(name, ip);
                ip_to_canonical.emplace(ip, nit->second);
                rec.frames[i] = nit->second;
            }
        }
    }
}

}  // namespace cpu_snap::processing::symbols
