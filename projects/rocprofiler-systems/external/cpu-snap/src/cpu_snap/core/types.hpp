#pragma once

/**
 * @file types.hpp
 * @brief Public function-alias types for the cpu_snap v1 wire pipeline.
 *
 * Used by sampling: host-supplied byte sink (output side) and
 * event callback (reader side). Both are type-erased via std::function
 * so the public ABI does not bind callers to a specific concrete callable.
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

#include "cpu_snap/core/records.hpp"

namespace cpu_snap::core
{

// NOLINTBEGIN(readability-identifier-naming): codebase k_*-prefix convention
/**
 * @brief Default target chunk size (1 MiB) used by chunk_framer.
 *
 * Lives in types.hpp (not chunk_framer.hpp) so that sampling's public
 * headers can use it as a config default without pulling chunk_framer's
 * heavy transitive include set.
 */
inline constexpr std::uint32_t k_default_chunk_bytes = 1U << 20;
// NOLINTEND(readability-identifier-naming)

/**
 * Host-supplied sink for the compressed byte stream produced by the
 * streaming pipeline.
 *
 * @note Invoked from the collector worker thread. Calls are serialized
 *       (never reentrant). The callback MUST NOT block; the span is
 *       valid only inside the call - copy out if the host needs it
 *       longer.
 */
using sink_fn = std::function<void(std::span<const std::byte>)>;

/**
 * Host-supplied event callback invoked by the reader as it parses
 * records from a fed byte stream.
 *
 * @note The record_variant reference is valid only inside the call;
 *       copy out if the host needs to retain it.
 */
using event_fn = std::function<void(const record_variant&)>;

}  // namespace cpu_snap::core
