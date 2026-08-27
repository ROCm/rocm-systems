// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/simulated_clock.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace rocjitsu::amdgpu {
namespace {

/// @brief Convert cycles to nanoseconds without letting a bad rate produce a
///        value the integer cast cannot represent.
std::uint64_t cycles_to_nanos(std::uint64_t cycles, double clock_ghz, double max_nanos,
                              double min_ghz, double max_ghz) {
  // The ordering matters: std::clamp propagates NaN unchanged, so a NaN has to
  // be rejected before it reaches the clamp rather than after. The rate comes
  // from a plugin, so it is untrusted.
  const double rate = std::isfinite(clock_ghz) && clock_ghz > 0.0 ? clock_ghz : min_ghz;
  const double ghz = std::clamp(rate, min_ghz, max_ghz);
  const double nanos = static_cast<double>(cycles) / ghz;
  return static_cast<std::uint64_t>(std::min(nanos, max_nanos));
}

} // namespace

SimulatedClock &SimulatedClock::instance() {
  static SimulatedClock clock;
  return clock;
}

std::uint64_t SimulatedClock::host_nanoseconds() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

std::uint64_t SimulatedClock::clamp_monotonic(std::atomic<std::uint64_t> &floor,
                                              std::uint64_t value) {
  std::uint64_t seen = floor.load(std::memory_order_relaxed);
  while (value > seen) {
    if (floor.compare_exchange_weak(seen, value, std::memory_order_relaxed))
      return value;
  }
  return std::max(seen, value);
}

const SimulatedClock::Binding &SimulatedClock::current() const {
  if (const Binding *bound = binding_.load(std::memory_order_acquire))
    return *bound;
  // The unbound state: host wall time, no offsets. Static rather than allocated
  // so a read that races the very first install still has something to use.
  static const Binding kHostTime{};
  return kHostTime;
}

void SimulatedClock::set_time_source(const timing::TimeSource *source) {
  // Rebase onto what has already been reported, so the new domain continues the
  // old one rather than restarting. Read through the public accessors so the
  // monotonic floors are the thing being continued.
  auto *binding = new Binding();
  binding->source = source;
  binding->ns_base = nanoseconds();
  binding->cycle_base = shader_cycles();

  if (source) {
    // Sanitised once, here, so every later reader of binding_->clock_ghz gets a
    // finite positive rate without repeating the check. std::clamp alone would
    // pass a NaN straight through to the integer casts downstream.
    const double advertised = source->clock_ghz();
    binding->clock_ghz = std::isfinite(advertised) && advertised > 0.0
                             ? std::clamp(advertised, kMinClockGhz, kMaxClockGhz)
                             : kMinClockGhz;
    binding->source_cycle_origin = source->current_cycles();
    binding->source_ns_origin = cycles_to_nanos(binding->source_cycle_origin, binding->clock_ghz,
                                                kMaxRepresentableNanos, kMinClockGhz, kMaxClockGhz);
  } else {
    // Returning to host wall time. Without an origin here the host's large
    // absolute nanosecond value would be added to the base and the clock would
    // leap forward by decades.
    binding->clock_ghz = 1.0;
    binding->source_ns_origin = host_nanoseconds();
    binding->source_cycle_origin = binding->source_ns_origin;
  }

  // The old binding is deliberately leaked; see Binding's documentation.
  binding_.store(binding, std::memory_order_release);
}

bool SimulatedClock::is_simulated() const { return current().source != nullptr; }

std::uint64_t SimulatedClock::nanoseconds() const {
  const Binding &binding = current();
  std::uint64_t source_ns = 0;
  if (binding.source) {
    source_ns = cycles_to_nanos(binding.source->current_cycles(), binding.clock_ghz,
                                kMaxRepresentableNanos, kMinClockGhz, kMaxClockGhz);
  } else {
    source_ns = host_nanoseconds();
  }
  const std::uint64_t elapsed =
      source_ns > binding.source_ns_origin ? source_ns - binding.source_ns_origin : 0;
  return clamp_monotonic(last_nanoseconds_, binding.ns_base + elapsed);
}

std::uint64_t SimulatedClock::counter_nanoseconds() const {
  const std::uint64_t now = nanoseconds();
  // Strictly increasing: bump the dedicated floor past whatever it held, so two
  // consecutive reads under a model that only advances at dispatch boundaries
  // still differ. Only this path pays for it.
  std::uint64_t previous = last_counter_nanoseconds_.load(std::memory_order_relaxed);
  std::uint64_t next = std::max(now, previous + 1);
  while (
      !last_counter_nanoseconds_.compare_exchange_weak(previous, next, std::memory_order_relaxed))
    next = std::max(now, previous + 1);

  // Carry the bump into the shared floor. The runtime correlates the value this
  // path returns against timestamps taken through nanoseconds() — a calibration
  // read here, a signal timestamp there — so letting this path run ahead on its
  // own would make a dispatch timestamp land below a calibration counter read
  // before it. That is time going backwards across two subsystems, which is the
  // failure this whole class exists to prevent, and it accumulates: under a
  // model whose clock only moves at dispatch boundaries, every read adds a
  // nanosecond of skew that nothing ever removes.
  clamp_monotonic(last_nanoseconds_, next);
  return next;
}

std::uint64_t SimulatedClock::shader_cycles() const {
  const Binding &binding = current();
  std::uint64_t source_cycles = 0;
  if (binding.source) {
    source_cycles = binding.source->current_cycles();
  } else {
    // No model: derive from host time at the nominal rate. The absolute value
    // is meaningless either way, but it has to keep advancing — a kernel that
    // spins until the counter changes would otherwise never make progress.
    source_cycles = host_nanoseconds();
  }
  const std::uint64_t elapsed =
      source_cycles > binding.source_cycle_origin ? source_cycles - binding.source_cycle_origin : 0;
  return clamp_monotonic(last_cycles_, binding.cycle_base + elapsed);
}

std::uint64_t SimulatedClock::wall_clock_ticks() const {
  // Derived from the nanosecond domain at the advertised rate, so the guest's
  // (ticks / kWallClockFrequencyHz) is the same duration a HIP event reports.
  static_assert(kWallClockFrequencyHz > 0 && kTimestampFrequencyHz >= kWallClockFrequencyHz,
                "the wall clock is derived from the nanosecond domain by integer division, so it "
                "has to be the slower of the two");
  return nanoseconds() / (kTimestampFrequencyHz / kWallClockFrequencyHz);
}

std::uint64_t SimulatedClock::shader_clock_hz() const {
  const Binding &binding = current();
  if (!binding.source)
    return 0;
  return static_cast<std::uint64_t>(binding.clock_ghz * 1.0e9);
}

} // namespace rocjitsu::amdgpu
