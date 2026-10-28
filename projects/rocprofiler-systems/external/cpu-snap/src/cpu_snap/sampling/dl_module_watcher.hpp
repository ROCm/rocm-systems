#pragma once

/**
 * @file dl_module_watcher.hpp
 * @brief Streaming-time module observer for the v1 sampling pipeline.
 *
 * Walks dl_iterate_phdr to enumerate currently-loaded shared objects
 * (plus the main executable and the vDSO) and reads .note.gnu.build-id
 * out of each ELF image. snapshot() emits the full initial module set;
 * poll() diffs against the last-seen set and emits add/remove deltas.
 *
 * Offline symbolization can then pin every IP timestamp to the binary
 * that was mapped at that time, even when dlopen/dlclose races the
 * recording.
 *
 * proc_maps.hpp (utils/) is the offline equivalent and stays for the
 * symbolizer; dl_module_watcher is the streaming-time path.
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cpu_snap/core/records.hpp"
#include "cpu_snap/sampling/clock_kind.hpp"

namespace cpu_snap::sampling
{

/**
 * @brief Polls dl_iterate_phdr periodically and emits maps_init / maps_delta
 *        records describing the loaded-module set.
 *
 * Construction is cheap (no walk). Call snapshot() once at session
 * start to publish the initial module list; call poll() periodically
 * to publish add/remove deltas.
 *
 * @note Not thread-safe. The streaming pipeline owns the watcher and
 *       drives snapshot()/poll() from its single worker thread.
 */
class dl_module_watcher
{
public:
    /**
     * @brief One injected module entry, used only by set_iterator_for_tests().
     *
     * Production code obtains equivalent data from dl_iterate_phdr.
     */
    struct injected_module
    {
        std::uint64_t base_va     = 0;
        std::uint64_t size        = 0;
        std::uint64_t file_offset = 0;
        std::string   path;
    };

    /**
     * @brief Test-only walk function.
     *
     * The callback is invoked once per "loaded" module with an
     * injected_module describing it. Build-id is left empty for
     * injected modules.
     */
    using iterator_fn = std::function<void(std::function<void(const injected_module&)>)>;

    /** @param clock Clock all record timestamps are read from (sampler::config::clock). */
    explicit dl_module_watcher(clock_kind clock = clock_kind::monotonic) noexcept;
    ~dl_module_watcher();

    dl_module_watcher(const dl_module_watcher&)            = delete;
    dl_module_watcher& operator=(const dl_module_watcher&) = delete;
    dl_module_watcher(dl_module_watcher&&)                 = delete;
    dl_module_watcher& operator=(dl_module_watcher&&)      = delete;

    /**
     * @brief Walk loaded modules now, populate the known-set, and
     *        return the initial module list.
     * @return A populated core::maps_init_record. wall_ns is set to
     *         the observation time in the configured clock.
     */
    [[nodiscard]] core::maps_init_record snapshot();

    /**
     * @brief Re-walk loaded modules and diff against the previous known-set.
     *
     * Modules with the same path but a different base_va are reported
     * as a remove of the old entry followed by an add of the new one.
     * Same-key updates to size / file_offset / build_id emit no delta
     * (v1 simplification).
     *
     * When the dynamic linker's load/unload counters (dlpi_adds/dlpi_subs)
     * are unchanged since the last walk, poll() returns without walking.
     *
     * Timestamps are read from the configured clock, the clock of sample timestamps. An
     * add is stamped with the previous observation time so that samples
     * taken between the dlopen and this poll still resolve to the new
     * module; a remove is stamped with the observation time. If the same
     * base_va is both removed and re-added in one poll, the add is stamped
     * with the observation time to keep the two intervals disjoint.
     *
     * @note Limits: a dlopen followed by a dlclose between two polls is
     *       never reported, and a library unloaded then reloaded at the
     *       same address and path between polls yields no delta.
     *
     * @return Possibly-empty vector of deltas. Order: removes first,
     *         then adds, both in iteration order.
     */
    [[nodiscard]] std::vector<core::maps_delta_record> poll();

    /**
     * @brief Number of modules currently in the known-set.
     * @return Tracked module count.
     */
    [[nodiscard]] std::size_t known_module_count() const noexcept;

    /**
     * @brief Number of full module walks performed so far (snapshot() and
     *        every poll() that did not take the unchanged-generation shortcut).
     */
    [[nodiscard]] std::uint64_t full_walk_count() const noexcept;

    /**
     * @brief Replace the loaded-module enumerator with @p fn.
     *
     * Used by tests to drive deterministic add/remove scenarios
     * without touching the real dynamic linker.
     * @param fn Replacement enumerator, or an empty fn to restore the
     *           dl_iterate_phdr-backed walker.
     */
    void set_iterator_for_tests(iterator_fn fn);

private:
    [[nodiscard]] std::vector<core::module_info> collect_current_modules() const;
    [[nodiscard]] std::uint64_t            now_wall_ns() const noexcept;

    /** (dlpi_adds, dlpi_subs) at the last walk; changes iff a library was loaded or unloaded. */
    using generation_stamp = std::pair<unsigned long long, unsigned long long>;

    clock_kind                     m_clock;
    iterator_fn                    m_iterator;
    std::vector<core::module_info> m_known;
    std::optional<generation_stamp> m_generation;
    std::uint64_t                  m_last_observed_ns = 0;
    std::uint64_t                  m_full_walks       = 0;
};

}  // namespace cpu_snap::sampling
