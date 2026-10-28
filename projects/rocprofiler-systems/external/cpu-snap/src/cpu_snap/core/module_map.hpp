#pragma once

/**
 * @file module_map.hpp
 * @brief Time-aware loaded-module lookup built from MAPS_INIT / MAPS_DELTA
 *        records.
 *
 * A capture's module layout can change mid-stream (dlopen/dlclose), so a
 * sample must be resolved against the layout valid AT ITS TIMESTAMP, not
 * the final one. module_map keeps every module's [load_ts, unload_ts)
 * validity interval and answers resolve(ip, timestamp_ns) accordingly.
 *
 * Intervals are kept sorted by base_va; resolve() binary-searches the
 * address and scans only the modules whose base lies within the largest
 * module size below it.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "cpu_snap/core/records.hpp"

namespace cpu_snap::core
{

/**
 * @brief Time-aware view of the loaded-module set for one capture.
 *
 * Fed by every maps_init_record / maps_delta_record seen while decoding a
 * stream, in wire order. A profile that carries its module map alongside
 * its samples can be symbolized without reading /proc/<pid>/maps, even in
 * another process.
 */
class module_map
{
public:
    /** @brief One loaded module, as returned by resolve(). */
    struct module_ref
    {
        /** Runtime start address of the first executable mapping (page-aligned). */
        std::uint64_t base_va = 0;
        /** Span (in bytes) covered by the module's executable mappings. */
        std::uint64_t size = 0;
        /** ELF file offset at base_va. */
        std::uint64_t file_offset = 0;
        /** GNU build-id of the backing ELF, raw bytes. */
        std::vector<std::uint8_t> build_id;
        /** Absolute filesystem path of the backing ELF. */
        std::string path;
    };

    /** @brief Seed the map with the full module list observed at session start. */
    void apply(const maps_init_record& rec);

    /** @brief Apply one add/remove delta observed mid-capture. */
    void apply(const maps_delta_record& rec);

    /**
     * @brief Resolve an instruction pointer against the module loaded at @p timestamp_ns.
     *
     * A module is a candidate when `load_ts <= timestamp_ns` and, if it
     * was later removed, `timestamp_ns` is strictly before its
     * `unload_ts`. @p ip must additionally fall inside
     * `[base_va, base_va + size)`.
     *
     * @param ip Instruction pointer to resolve.
     * @param timestamp_ns Sample timestamp to resolve against.
     * @return The covering module, or std::nullopt if no module was loaded
     *         at that address at that time.
     */
    [[nodiscard]] std::optional<module_ref> resolve(std::uint64_t ip,
                                                     std::uint64_t timestamp_ns) const;

private:
    /** @brief One module's validity window, half-open: [load_ts, unload_ts). */
    struct interval
    {
        module_ref                   mod;
        std::uint64_t                load_ts = 0;
        std::optional<std::uint64_t> unload_ts;
    };

    void add_module(const module_info& mod, std::uint64_t load_ts);
    void remove_module(const module_info& mod, std::uint64_t unload_ts);

    /** Sorted by base_va; equal base_va keep insertion order. */
    std::vector<interval> m_intervals;
    std::uint64_t         m_max_size = 0;
};

}  // namespace cpu_snap::core
