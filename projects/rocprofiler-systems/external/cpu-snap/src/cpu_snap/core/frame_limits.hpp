#pragma once

/**
 * @file frame_limits.hpp
 * @brief Leaf header defining the callchain depth default and sanity bound.
 *
 * The depth in use is a runtime setting (`sampler::config::perf.callchain_depth`);
 * these constants only provide its default and the upper bound it is validated
 * against. Kept as a leaf so capture and storage can share them without pulling
 * in the decode machinery (<linux/perf_event.h>, epoll, mman).
 *
 * Header-only. C++20.
 */

#include <cstdint>

namespace cpu_snap::core
{

/** @brief Default number of frames captured per stack sample. */
inline constexpr std::uint32_t k_default_callchain_depth = 256;

/** @brief Sanity bound for a configured callchain depth. */
inline constexpr std::uint32_t k_max_callchain_depth = 4096;

}  // namespace cpu_snap::core
