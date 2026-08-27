// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file request.h
/// @brief What travels between the timed components.
///
/// @details One request is one memory instruction's worth of work, not one
/// cache line. The distinction is what makes an event-driven memory model
/// affordable inside a functional simulator: a wave64 access is up to
/// sixty-four lines, and a component graph that scheduled an event per line
/// would spend more time in the event queue than the emulator spends executing.
/// A request therefore carries the *set* of lines an instruction touched, each
/// level splits it into the parts it hits and the parts it misses, and only the
/// miss set travels onward.

#pragma once

#include "rocjitsu/vm/timing/vocabulary.h"

#include <cstdint>

namespace rocjitsu::timing {

/// @brief Which wavefront a request belongs to, so its completion can find it.
struct RequestOrigin {
  std::uint32_t compute_unit = 0;
  std::uint32_t wave_slot = 0;
  /// @brief Serial number of the access within the wavefront, so a completion
  ///        can be matched to the wait counter entry it retires.
  std::uint32_t sequence = 0;
};

/// @brief One memory instruction's traffic, as it moves down the hierarchy.
struct MemoryRequest {
  RequestOrigin origin;
  /// @brief Index into the plane's line pool: the addresses this request still
  ///        needs, contiguous, `line_count` of them.
  std::uint32_t line_base = 0;
  std::uint32_t line_count = 0;
  /// @brief Bytes per line at the level currently holding the request.
  std::uint32_t line_bytes = 64;
  bool is_load = true;
  bool non_temporal = false;
  bool instruction_fetch = false;
  /// @brief Which first-level cache serves this request. The plane routes on
  ///        it: a scalar access probes the scalar cache, whose geometry and
  ///        line size differ from the vector one, and routing it to the vector
  ///        cache would give it that cache's hit rate as well as its latency.
  MemorySpace space = MemorySpace::Global;
  /// @brief Tick the request entered the hierarchy, so a completion can report
  ///        the latency the wavefront actually saw rather than a nominal one.
  std::uint64_t issued_tick = 0;
  /// @brief Deepest level the request has reached, for reporting.
  std::uint8_t depth = 0;
  /// @brief Row activations this request forces on its memory channel.
  ///
  /// @details A DRAM channel holds one row open at a time and serves a request
  /// to that row far faster than one that has to close it and activate another.
  /// It is what separates a streaming access pattern from a scattered one, and
  /// in this model it is the only thing that does: both move the same bytes
  /// through the same channels, and only the number of rows they open differs.
  std::uint32_t row_misses = 0;
};

} // namespace rocjitsu::timing
