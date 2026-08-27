// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simulated_clock.h
/// @brief The device clock the guest observes.
///
/// @details Everything a program running under the emulator can use to measure
/// time resolves here: HIP event timing, the driver's clock counters, the DMA
/// engine's timestamp packets, and the in-kernel time instructions. Without a
/// timing model it reports host wall time, which is what rocjitsu has always
/// done. With one, it reports that model's simulated time, and the guest
/// measures the modelled machine rather than the machine the simulator happens
/// to be running on.
///
/// One clock rather than one per subsystem, because the guest subtracts values
/// taken from different subsystems: a HIP event pair brackets a dispatch whose
/// timestamps the completion tracker wrote, and the runtime converts them with
/// a frequency the driver advertised. Two domains that disagree produce a
/// plausible but wrong duration, which is much harder to notice than a broken
/// one.

#pragma once

#include "rocjitsu/vm/timing/time_source.h"

#include <atomic>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief The process-wide simulated device clock.
///
/// @details Process-wide because its readers are scattered across layers that
/// share no object — a free function in the HSA clock helper, the KFD ioctl
/// handlers, the command processor, an ISA execute body — and because rocjitsu
/// simulates one device timeline per process in every mode it supports.
/// Threading it through each of those call paths would be a large change for no
/// gain.
///
/// In daemon mode one process serves several guests through one of these, so
/// concurrent guests share a single simulated timeline and each other's
/// advancement. There is no per-client clock.
class SimulatedClock {
public:
  static SimulatedClock &instance();

  /// @brief Adopt @p source as the origin of simulated time, or return to host
  ///        wall time when it is null.
  ///
  /// @details Rebases rather than clamps: the new domain continues from the
  /// last value already reported, in both directions. Clamping alone would be a
  /// freeze, not a safety net — a cycle domain that has already reported a
  /// host-derived magnitude would sit above anything a model starting at zero
  /// could produce, and every in-kernel self-timing delta would be exactly
  /// zero for the rest of the process.
  ///
  /// Safe against concurrent readers. The uninstall genuinely races them: in
  /// daemon mode it runs on the engine thread while client threads are still
  /// answering driver queries.
  void set_time_source(const timing::TimeSource *source);

  bool is_simulated() const;

  /// @brief The current time in nanoseconds, in the guest's timestamp domain.
  ///
  /// @details Nanoseconds because that is the unit rocjitsu already advertises,
  /// so installing a model changes what the numbers say rather than how the
  /// runtime interprets them. Never moves backwards, whatever the model does.
  std::uint64_t nanoseconds() const;

  /// @brief A strictly increasing reading of the same domain, for the clock
  ///        counters ioctl.
  ///
  /// @details ROCR calibrates by reading this ioctl twice and dividing by the
  /// difference. A model whose clock only advances when kernels run returns the
  /// same value to both reads, and the division is by zero. Strict increase is
  /// a property of this one path rather than of the clock, because everywhere
  /// else two reads at the same instant should agree.
  std::uint64_t counter_nanoseconds() const;

  /// @brief The current time in shader-clock cycles, for the in-kernel time
  ///        instructions, so a kernel timing itself sees the same clock the
  ///        host sees around it.
  std::uint64_t shader_cycles() const;

  /// @brief The current value of the constant-rate device wall clock.
  ///
  /// @details A separate counter from shader_cycles() because the guest divides
  /// it by a separately advertised rate — hipDeviceAttributeWallClockRate,
  /// which the emulated topology reports as kWallClockFrequencyHz. Reporting
  /// shader cycles here would over-report elapsed time by the ratio of the two
  /// clocks: correctly typed, plausibly shaped, and silently wrong.
  std::uint64_t wall_clock_ticks() const;

  /// @brief The rate of the timestamp counter this class hands out, in Hz.
  ///
  /// @details Always 1 GHz, matching the nanosecond unit above. Reporting the
  /// modelled shader clock here would be wrong: the runtime uses this to
  /// convert the timestamps this class produces, and those are already
  /// nanoseconds.
  static constexpr std::uint64_t kTimestampFrequencyHz = 1'000'000'000ULL;

  /// @brief The rate of the constant-rate device wall clock, in Hz.
  ///
  /// @details Must match what the emulated topology advertises to libhsakmt,
  /// because the guest divides wall_clock_ticks() differences by it.
  static constexpr std::uint64_t kWallClockFrequencyHz = 100'000'000ULL;

  /// @brief The shader clock a model is driving, in Hz, or 0 when none is.
  ///
  /// @details The rate the driver should advertise as the engine clock, so a
  /// guest converting measured microseconds into cycles uses the clock of the
  /// machine that produced the microseconds.
  std::uint64_t shader_clock_hz() const;

private:
  SimulatedClock() = default;

  /// @brief One source, together with the offsets that rebase it.
  ///
  /// @details Immutable once published, and published as a unit through a
  /// single atomic pointer, so a reader can never pair a new source with a
  /// stale offset. Bindings are intentionally never freed: a reader may hold
  /// one across the uninstall, the objects are a few dozen bytes, and one is
  /// created per plugin load rather than per read.
  struct Binding {
    const timing::TimeSource *source = nullptr;
    /// @brief Reported values at the moment of binding — where the new domain
    ///        picks up.
    std::uint64_t ns_base = 0;
    std::uint64_t cycle_base = 0;
    /// @brief The new source's own values at that moment — what the offsets
    ///        above are measured from.
    std::uint64_t source_ns_origin = 0;
    std::uint64_t source_cycle_origin = 0;
    double clock_ghz = 1.0;
  };

  /// @brief The binding every read goes through. Never null after first use.
  const Binding &current() const;

  /// @brief Raise @p floor to at least @p value and return the result, so a
  ///        domain cannot retreat even if a model revises an estimate.
  static std::uint64_t clamp_monotonic(std::atomic<std::uint64_t> &floor, std::uint64_t value);

  static std::uint64_t host_nanoseconds();

  std::atomic<const Binding *> binding_{nullptr};
  mutable std::atomic<std::uint64_t> last_nanoseconds_{0};
  mutable std::atomic<std::uint64_t> last_cycles_{0};
  mutable std::atomic<std::uint64_t> last_counter_nanoseconds_{0};

  /// @brief Largest nanosecond value that survives the conversion from a
  ///        double, with room for the rebase offset. A double beyond the
  ///        integer range converts as undefined behaviour, so the value is
  ///        bounded before the cast rather than after.
  static constexpr double kMaxRepresentableNanos = 9.0e18;
  /// @brief Bounds the divisor from the other end: a denormal or absurd rate
  ///        would turn a modest cycle count into a value the conversion cannot
  ///        represent.
  static constexpr double kMinClockGhz = 1.0e-6;
  static constexpr double kMaxClockGhz = 1.0e6;
};

} // namespace rocjitsu::amdgpu
