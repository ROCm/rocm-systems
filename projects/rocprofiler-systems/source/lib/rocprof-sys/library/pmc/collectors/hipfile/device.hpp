// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/hipfile/types.hpp"

#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace rocprofsys::pmc::collectors::hipfile
{

// Contract the hipFile collector requires of its backend (the data producer).
template <typename Backend>
concept hipfile_backend_contract = requires(Backend backend, std::uint64_t timestamp) {
    { backend.get_stats(timestamp) } -> std::convertible_to<const stats_snapshot&>;
    { backend.is_available() } -> std::convertible_to<bool>;
};

/**
 * @brief One GPU's view of the shared hipFile stats snapshot.
 *
 * hipFile indexes @c per_gpu_stats by HIP device ordinal (@c hipGetDevice()). The
 * profiler's GPU tracks and @c ROCPROFSYS_SAMPLING_GPUS use the agent's
 * @c device_type_index, which is not renamed when a visibility mask hides GPUs. Those
 * two numbers are stored separately: the hipFile slot is used only to read the
 * snapshot, and @c get_index() returns the profiler device index.
 *
 * Every device shares one backend, so an interval costs one hipFile query and all
 * devices in it observe the same snapshot.
 *
 * The device owns the only mutable state in the collector: the previous byte totals and
 * timestamp needed to turn cumulative counters into a wall-clock bandwidth. The
 * counters themselves need no history, since they are published cumulative.
 *
 * @tparam Backend hipFile backend session type (real or mock).
 */
template <hipfile_backend_contract Backend>
class device
{
public:
    using backend_type = Backend;

    device(std::shared_ptr<Backend> backend, std::size_t hipfile_slot,
           std::size_t device_index)
    : m_backend{ std::move(backend) }
    , m_hipfile_slot{ hipfile_slot }
    , m_index{ device_index }
    , m_name{ "GPU " + std::to_string(device_index) }
    {}

    /// Identity mapping: hipFile slot and profiler index are the same ordinal.
    device(std::shared_ptr<Backend> backend, std::size_t gpu_ordinal)
    : device(std::move(backend), gpu_ordinal, gpu_ordinal)
    {}

    /// @brief Profiler GPU index (@c device_type_index), for tracks, RocPD, and
    ///        @c ROCPROFSYS_SAMPLING_GPUS.
    [[nodiscard]] std::size_t get_index() const noexcept { return m_index; }

    /// @brief HIP ordinal / hipFile @c per_gpu_stats slot.
    [[nodiscard]] std::size_t get_hipfile_slot() const noexcept { return m_hipfile_slot; }

    [[nodiscard]] const std::string& get_name() const noexcept { return m_name; }

    /// @brief Whether this HIP ordinal addresses a real slot in the hipFile snapshot.
    [[nodiscard]] bool is_supported() const noexcept { return m_hipfile_slot < MAX_GPUS; }

    [[nodiscard]] enabled_metrics get_supported_metrics() const noexcept
    {
        enabled_metrics supported;
        supported.value = is_supported() ? ALL_HIPFILE_METRICS : 0U;
        return supported;
    }

    /**
     * @brief Read this GPU's counters and compute its bandwidth for the interval.
     *
     * @param timestamp Sample timestamp in nanoseconds. Doubles as the backend's
     *                  memoization key and as the bandwidth denominator.
     */
    [[nodiscard]] metrics get_metrics(const enabled_metrics& /*enabled*/,
                                      std::uint64_t timestamp)
    {
        metrics out{};

        if(!is_supported())
        {
            out.query_failed = true;
            return out;
        }

        const auto& snapshot = m_backend->get_stats(timestamp);
        if(!m_backend->is_available())
        {
            // hipFile stats are unreadable: the target performed no hipFile I/O, or
            // HIPFILE_STATS_LEVEL is off. Reporting zeros would be a measurement claim
            // we cannot make, so the sample is suppressed instead.
            out.query_failed = true;
            return out;
        }

        const auto& stats = snapshot.per_gpu[m_hipfile_slot];

        out.read_bytes       = stats.read_bytes;
        out.write_bytes      = stats.write_bytes;
        out.read_ops         = stats.read_ops;
        out.write_ops        = stats.write_ops;
        out.fastpath_reads   = stats.fastpath_reads;
        out.fastpath_writes  = stats.fastpath_writes;
        out.fallback_reads   = stats.fallback_reads;
        out.fallback_writes  = stats.fallback_writes;
        out.unaligned_reads  = stats.unaligned_reads;
        out.unaligned_writes = stats.unaligned_writes;
        out.read_errors      = stats.read_errors;
        out.write_errors     = stats.write_errors;

        // The first sample has no interval behind it, so it has no rate. Reporting the
        // lifetime total over a zero interval would open every run with a false spike.
        // The stamp is the process sampler's tick (tim::get_clock_real_now →
        // steady_clock), the same value every PMC collector gets that interval.
        // It does not step backward with NTP, so the unsigned subtraction cannot wrap
        // from a wall-clock rewind. A repeated stamp is already treated as zero elapsed.
        if(m_has_previous)
        {
            const auto elapsed_ns = timestamp - m_previous_timestamp;
            out.read_bandwidth =
                bandwidth(stats.read_bytes, m_previous_read_bytes, elapsed_ns);
            out.write_bandwidth =
                bandwidth(stats.write_bytes, m_previous_write_bytes, elapsed_ns);
        }

        m_previous_read_bytes  = stats.read_bytes;
        m_previous_write_bytes = stats.write_bytes;
        m_previous_timestamp   = timestamp;
        m_has_previous         = true;

        return out;
    }

private:
    /**
     * @brief Bandwidth in bytes/sec over one sampling interval.
     *
     * Normalised by the sampler interval rather than by hipFile's own I/O-call duration,
     * so the track is directly comparable to the AMD SMI instantaneous PCIe bandwidth
     * sitting beside it on the same GPU timeline.
     *
     * A counter that moved backwards means hipFile reset its stats; the interval is
     * unmeasurable rather than negative, so it reports zero.
     */
    [[nodiscard]] static double bandwidth(std::uint64_t current, std::uint64_t previous,
                                          std::uint64_t elapsed_ns) noexcept
    {
        if(elapsed_ns == 0 || current < previous)
        {
            return 0.0;
        }

        using elapsed_ns_t = std::chrono::duration<std::uint64_t, std::nano>;
        const auto elapsed_s =
            std::chrono::duration<double>(elapsed_ns_t{ elapsed_ns }).count();
        return static_cast<double>(current - previous) / elapsed_s;
    }

    std::shared_ptr<Backend> m_backend;
    const std::size_t        m_hipfile_slot;
    const std::size_t        m_index;
    const std::string        m_name;

    std::uint64_t m_previous_read_bytes  = 0;
    std::uint64_t m_previous_write_bytes = 0;
    std::uint64_t m_previous_timestamp   = 0;
    bool          m_has_previous         = false;
};

}  // namespace rocprofsys::pmc::collectors::hipfile
