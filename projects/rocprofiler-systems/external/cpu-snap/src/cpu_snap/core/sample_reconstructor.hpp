#pragma once

/**
 * @file sample_reconstructor.hpp
 * @brief Stateful reader -> (raw_sample, module_map) adapter.
 *
 * Wraps a core::reader: feed() decodes wire bytes and, for every SAMPLE
 * record, resolves its leaf trie node to an IP chain and appends a
 * raw_sample to the owning tid's vector (samples are decoded in capture
 * order, which is already chronological per stream, so no additional
 * sort is needed). MAPS_INIT / MAPS_DELTA records feed an internal
 * module_map. Replaces the reader -> raw_sample conversion that used to
 * be copy-pasted in every streaming demo.
 */

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)

#include "cpu_snap/core/module_map.hpp"
#include "cpu_snap/core/raw_sample.hpp"
#include "cpu_snap/core/reader.hpp"

namespace cpu_snap::core
{

/**
 * @brief Decodes a wire byte stream into per-tid raw_sample vectors plus
 *        the module_map observed along the way.
 */
class sample_reconstructor
{
public:
    sample_reconstructor();
    ~sample_reconstructor();

    sample_reconstructor(const sample_reconstructor&)            = delete;
    sample_reconstructor& operator=(const sample_reconstructor&) = delete;
    sample_reconstructor(sample_reconstructor&&)                 = delete;
    sample_reconstructor& operator=(sample_reconstructor&&)      = delete;

    /**
     * @brief Decode the next slice of wire bytes.
     * @param data Span of bytes from the wire stream. May be called
     *             repeatedly as bytes arrive.
     */
    void feed(std::span<const std::byte> data) noexcept;

    /** @brief Per-tid, time-sorted samples decoded so far. */
    [[nodiscard]] const std::unordered_map<pid_t, std::vector<raw_sample>>&
    samples() const noexcept;

    /** @brief Module layout observed so far (from MAPS_INIT / MAPS_DELTA). */
    [[nodiscard]] const module_map& modules() const noexcept;

    /** @brief Underlying reader's stream-health counters. */
    [[nodiscard]] const reader_counters& counters() const noexcept;

    /**
     * @brief Samples dropped because their leaf node could not be resolved
     *        to a stack (e.g. after a lost NEW_NODE chunk). Samples with
     *        leaf_node_id == 0 are intentionally empty and not counted.
     */
    [[nodiscard]] std::uint64_t unresolved_samples() const noexcept;

private:
    void handle_event(const record_variant& rec);

    reader                                             m_reader;
    std::unordered_map<pid_t, std::vector<raw_sample>> m_samples;
    module_map                                         m_modules;
    std::uint64_t                                      m_unresolved_samples = 0;
};

}  // namespace cpu_snap::core
