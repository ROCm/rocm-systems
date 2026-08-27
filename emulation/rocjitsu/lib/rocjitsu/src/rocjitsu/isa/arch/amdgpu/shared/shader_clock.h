// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file shader_clock.h
/// @brief Hand-written policy for the ISA's time-reading instructions.
///
/// @details The instruction bodies under `generated/` are emitted by the amdisa
/// codegen pipeline and carry a DO NOT EDIT banner, so anything about *what
/// time is* has to live outside them or a regeneration reverts it silently.
/// That has happened here before: `s_sleep`'s architected delay was once
/// written straight into the generated header while the generator emitted a
/// bare yield, so the next regeneration would have turned it back into a
/// no-op with a still-green build. The generated bodies therefore call into
/// this header, and the modelling decision is version-controlled where a
/// reviewer can see it.
///
/// Both accessors read ::rocjitsu::amdgpu::SimulatedClock, which is the same
/// clock behind the HIP event timestamps, the KFD clock-counters ioctl and the
/// SDMA timestamp packet. A kernel that times itself and a host that brackets
/// the dispatch then measure the same machine.

#include "rocjitsu/vm/timing/simulated_clock.h"

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief The free-running shader-clock counter, for `s_memtime`,
///        `s_get_shader_cycles_u64` and HWREG SHADER_CYCLES.
///
/// @details These are LLVM's lowerings of `readcyclecounter`, whose unit is
/// shader cycles at the engine clock. Without a timing model installed the
/// value still advances, because a kernel that spins until the counter changes
/// would otherwise never make progress.
inline std::uint64_t shader_clock_ticks() { return SimulatedClock::instance().shader_cycles(); }

/// @brief The constant-rate device wall clock, for `s_memrealtime` and
///        `s_sendmsg_rtn MSG_RTN_GET_REALTIME`.
///
/// @details Deliberately a different counter from shader_clock_ticks(). These
/// are LLVM's lowerings of `readsteadycounter`, and a guest converts their
/// differences with the rate the driver advertised as the wall clock
/// (SimulatedClock::kWallClockFrequencyHz, reported through the amdgpu
/// `gpu_counter_freq` that libhsakmt turns into HsaNodeProperties::WallClockKHz)
/// rather than with the engine clock. Reporting shader cycles here would
/// over-report every measured interval by the ratio of the two clocks —
/// correctly typed, plausibly shaped, and silently wrong.
inline std::uint64_t device_wall_clock_ticks() {
  return SimulatedClock::instance().wall_clock_ticks();
}

} // namespace rocjitsu::amdgpu
