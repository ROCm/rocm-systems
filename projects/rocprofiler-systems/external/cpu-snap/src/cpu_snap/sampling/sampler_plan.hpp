#pragma once

/**
 * @file sampler_plan.hpp
 * @brief Stateless resolution of a sampler::config into the effective
 *        configuration the sampler builds its objects from.
 */

#include <cstdint>

#include "cpu_snap/core/types.hpp"
#include "cpu_snap/sampling/backend_selection.hpp"
#include "cpu_snap/sampling/clock_kind.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/sampling/triggers/perf/kernel_drain.hpp"
#include "cpu_snap/sampling/triggers/signal/backend.hpp"
#include "cpu_snap/sampling/unwind/unwinder.hpp"
#include "cpu_snap/sampling/unwind_mode.hpp"

namespace cpu_snap::sampling::internal
{

/** @brief Effective configuration: backend, unwind mode and perf ring size resolved. */
struct resolved_plan
{
    sampler::config cfg;
    backend_choice  choice;
    bool            unwind_was_auto;
};

/**
 * @brief Validate @p cfg and resolve backend, unwind mode and perf ring sizing.
 *
 * @param probe Perf availability probe; not called for an explicit signal
 *        backend nor when @p trigger_injected.
 * @param trigger_injected The caller supplies its own trigger: automatic
 *        resolves to perf without probing.
 * @throws cpu_snap::config_error On an invalid configuration.
 * @throws cpu_snap::environment_error When perf is requested but unavailable.
 */
[[nodiscard]] resolved_plan
resolve_plan(sampler::config cfg, perf_probe_fn probe, bool trigger_injected);

[[nodiscard]] triggers::perf::kernel_drain::config
make_drain_config(const sampler::config& cfg) noexcept;

[[nodiscard]] triggers::signal::backend::config
make_signal_config(const sampler::config& cfg) noexcept;

[[nodiscard]] core::stream_clock
stream_clock_for(clock_kind clock) noexcept;

[[nodiscard]] std::uint16_t
streaming_flags_for(unwind_mode resolved, clock_kind clock) noexcept;

[[nodiscard]] unwind::capture_kind
capture_kind_for(unwind_mode m, backend_kind bk) noexcept;

}  // namespace cpu_snap::sampling::internal
