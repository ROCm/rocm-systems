// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/hipfile/types.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rocprofsys::backends::hipfile
{

/**
 * @brief Contract a wrapper policy must satisfy.
 *
 * Checked where @c backend<T> or @c backend_factory<T> is instantiated, so a mismatch
 * reports at the template boundary rather than deep inside the template body.
 */
template <typename T>
concept wrapper_policy =
    requires { typename T::stats_l3_t; } && requires(T::stats_l3_t* out) {
        { T::get_stats_l3(out) } -> std::convertible_to<bool>;
        { T::runtime_version_supported() } -> std::convertible_to<bool>;
        { T::k_max_gpu_slots } -> std::convertible_to<std::size_t>;
    };

/**
 * @brief Session-level wrapper around hipFile's Level-3 stats query.
 *
 * Converts hipFile's raw struct into @c stats_snapshot and memoizes the result per
 * sampling timestamp. Every GPU device shares one backend, so a sampling interval costs
 * exactly one @c hipFileGetStatsL3 call regardless of GPU count, and every device in
 * that interval observes the same coherent snapshot.
 *
 * A failed query is not an error to propagate. hipFile also returns success with a
 * zero-filled struct and @c num_gpus == 0 when no GPU has performed hipFile I/O, and
 * when @c HIPFILE_STATS_LEVEL disables recording. Both are an expected steady state
 * rather than a fault: the snapshot stays zero-filled, @c is_available() reports false,
 * and the collector suppresses the sample. A stored zero means that counter was
 * measured and did not increase, which is only true once some GPU is active.
 *
 * @tparam Wrapper Raw hipFile C-API policy (e.g. @c wrapper).
 */
template <wrapper_policy Wrapper>
class backend
{
public:
    backend() noexcept = default;

    /**
     * @brief Level-3 snapshot for @p timestamp, querying hipFile at most once per
     *        distinct timestamp.
     *
     * @param timestamp Sampling timestamp in nanoseconds; doubles as the memoization key.
     */
    [[nodiscard]] const stats_snapshot& get_stats(std::uint64_t timestamp)
    {
        if(!m_has_snapshot || m_timestamp != timestamp)
        {
            m_snapshot     = query();
            m_timestamp    = timestamp;
            m_has_snapshot = true;
        }
        return m_snapshot;
    }

    /// @brief Whether the most recent query observed an active GPU.
    [[nodiscard]] bool is_available() const noexcept { return m_available; }

    void shutdown() noexcept {}

private:
    [[nodiscard]] stats_snapshot query()
    {
        stats_snapshot out{};

        // A libhipfile older than the headers were built against carries no stats API, so
        // there is nothing to call. Treated as unavailable rather than as an error, which
        // leaves a gap in the tracks instead of a run of misleading zeros.
        if(!Wrapper::runtime_version_supported())
        {
            m_available = false;
            return out;
        }

        // Single zero-init of the Level-3 struct; the wrapper does not memset again.
        typename Wrapper::stats_l3_t raw{};

        m_available = Wrapper::get_stats_l3(&raw);

        if(!m_available)
        {
            return out;
        }

<<<<<<< HEAD
        // Success with no active GPU is not a measurement. hipFile fills the struct
        // with zeros and returns success until a GPU is in use, including for the
        // whole run when HIPFILE_STATS_LEVEL disables the statistics server.
        if(raw.num_gpus == 0)
        {
            m_available = false;
            return out;
        }

        const auto slots = std::min<std::size_t>(Wrapper::MAX_GPU_SLOTS, MAX_GPUS);
=======
        const auto slots = std::min<std::size_t>(Wrapper::k_max_gpu_slots, k_max_gpus);
>>>>>>> c2bf847c1a (chore: bulk `clang-tidy` clean up)
        for(std::size_t i = 0; i < slots; ++i)
        {
            const auto& src = raw.per_gpu_stats[i];
            auto&       dst = out.per_gpu[i];

            dst.read_bytes       = src.read_bytes;
            dst.write_bytes      = src.write_bytes;
            dst.read_ops         = src.n_total_reads;
            dst.write_ops        = src.n_total_writes;
            dst.fastpath_reads   = src.n_nvfs_reads;
            dst.fastpath_writes  = src.n_nvfs_writes;
            dst.fallback_reads   = src.n_posix_reads;
            dst.fallback_writes  = src.n_posix_writes;
            dst.unaligned_reads  = src.n_unaligned_reads;
            dst.unaligned_writes = src.n_unaligned_writes;
            dst.read_errors      = src.n_reads_err;
            dst.write_errors     = src.n_writes_err;
        }
        return out;
    }

    stats_snapshot m_snapshot{};
    std::uint64_t  m_timestamp    = 0;
    bool           m_has_snapshot = false;
    bool           m_available    = false;
};

/**
 * @brief Factory for creating backend<Wrapper> session instances.
 */
template <wrapper_policy Wrapper>
struct backend_factory
{
    using backend_t = backend<Wrapper>;

    static std::shared_ptr<backend_t> create_backend()
    {
        return std::make_shared<backend_t>();
    }
};

}  // namespace rocprofsys::backends::hipfile
