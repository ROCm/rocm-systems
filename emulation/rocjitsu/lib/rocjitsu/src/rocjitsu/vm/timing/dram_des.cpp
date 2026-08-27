// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/dram_des.h"

#include "rocjitsu/vm/timing/engine.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace rocjitsu::timing {
namespace {

/// @brief Cycles to move @p bytes at @p per_cycle, never fewer than one.
std::uint64_t cycles_for(std::uint64_t bytes, double per_cycle) {
  if (!(per_cycle > 0.0))
    return std::max<std::uint64_t>(1, bytes);
  const double cycles = std::ceil(static_cast<double>(bytes) / per_cycle);
  if (!(cycles > 1.0))
    return 1;
  return static_cast<std::uint64_t>(cycles);
}

} // namespace

ChannelDes::ChannelDes(std::string name, const simdojo::ClockDomain &domain, TimingEngine &engine,
                       double bytes_per_cycle, std::uint64_t latency_cycles,
                       std::uint64_t row_miss_cycles)
    : TimedComponent(std::move(name), domain, engine),
      bytes_per_cycle_(bytes_per_cycle > 0.0 ? bytes_per_cycle : 1.0),
      latency_cycles_(latency_cycles), row_miss_cycles_(row_miss_cycles) {}

std::uint64_t ChannelDes::advance(std::uint64_t now) {
  if (inbox().empty())
    return 0;

  // Only what was already queued is served at this tick, and each request is
  // copied out by value so a re-entrant arrival cannot reallocate the inbox
  // underneath the loop; see CacheDes::advance().
  const std::size_t queued = inbox().size();

  for (std::size_t entry = 0; entry < queued; ++entry) {
    const MemoryRequest request = inbox()[entry];
    const std::uint64_t bytes = static_cast<std::uint64_t>(request.line_count) * request.line_bytes;

    requests_.fetch_add(1, std::memory_order_relaxed);
    bytes_.fetch_add(bytes, std::memory_order_relaxed);

    // The server is occupied for as long as the bytes take to cross it. That
    // occupancy is the model's whole representation of queueing: a request
    // arriving behind others is served late because reserve() starts it when
    // this channel is next free, which is what a request behind others
    // actually experiences.
    // A row activation occupies the channel over and above the burst itself.
    // This is the only term that tells a streaming access pattern from a
    // scattered one: both move the same bytes through the same channels, and
    // only the number of rows they have to open differs. Without it a scattered
    // kernel reads at a streaming kernel's rate, which is what left the large
    // reductions and the strided copies at about half their measured duration.
    const std::uint64_t activations =
        row_miss_cycles_ != 0 ? static_cast<std::uint64_t>(request.row_misses) : 0;
    if (activations != 0)
      activations_.fetch_add(activations, std::memory_order_relaxed);
    const std::uint64_t done =
        reserve(now, cycles_for(bytes, bytes_per_cycle_) + activations * row_miss_cycles_);

    if (downstream_) {
      // A forwarding channel is the fabric crossing between the dies and
      // memory. It contributes bandwidth and nothing else: its latency is
      // already inside the downstream level's own latency, which the config
      // states as a delay measured at the requester rather than at this
      // channel's input. Constructing a forwarding channel with a non-zero
      // latency_cycles therefore has no effect, because a request carries no
      // arrival stamp to date the hand-off from.
      //
      // Charging this crossing at all is what stops a working set that happens
      // to fit in the memory-side cache from reading several times faster than
      // the part has ever been measured to go: on an eight megabyte copy it
      // produced 4.96 microseconds against a measured 10.88, and adding it
      // produced 11.72.
      MemoryRequest onward = request;
      onward.depth = request.depth == 0xFFu ? 0xFFu : static_cast<std::uint8_t>(request.depth + 1);
      downstream_(onward);
      continue;
    }

    if (completion_) {
      // As in CacheDes: the configured latency is the round trip the requester
      // sees, so it is a floor from the tick the request entered the hierarchy,
      // and what the channel adds on top of it is the time it spent waiting for
      // and occupying this server.
      const std::uint64_t nominal = request.issued_tick + cycles_to_ticks(latency_cycles_);
      completion_(request, std::max(nominal, done));
    }
  }

  inbox().erase(inbox().begin(), inbox().begin() + static_cast<std::ptrdiff_t>(queued));
  return 0;
}

} // namespace rocjitsu::timing
