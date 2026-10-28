#pragma once

#include <cstdint>

namespace cpu_snap::sampling
{

/**
 * @brief Capture-mechanism trigger selector.
 *
 * Chooses which triggers::i_trigger the sampler builds at construction.
 * automatic (the default) prefers perf and falls back to signal when
 * perf_event_open is not permitted; perf and signal are explicit and override
 * that. perf uses perf_event_open; signal uses per-thread POSIX timers + a
 * realtime signal and needs no perf_event_open privilege.
 */
enum class backend_kind : std::uint8_t
{
    automatic,
    perf,
    signal,
};

}  // namespace cpu_snap::sampling
