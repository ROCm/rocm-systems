// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file helper_error_sink.h
/// @brief Per-helper instruction error state, kept off the issuer-owned wave fields.

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Error reported by an instruction running on a matrix helper thread.
/// @details The issuer owns the wave's error fields and clears them for every
/// instruction it issues. A helper reports here instead; the issuer merges the
/// first failure onto the wave with its originating guest PC after joining.
struct HelperErrorSink {
  uint8_t error = 0; ///< InstructionExecutionError; zero is None.
  const char *detail = nullptr;
};

/// Non-null only on a helper thread while it executes an instruction.
inline thread_local HelperErrorSink *helper_error_sink = nullptr;

} // namespace rocjitsu::amdgpu
