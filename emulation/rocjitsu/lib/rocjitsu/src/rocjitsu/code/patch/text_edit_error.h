// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocjitsu {

enum class TextEditErrorCode {
  None,
  InvalidImage,
  UnsupportedTarget,
  UnsupportedReferences,
  InvalidRange,
  Overlap,
  StaleBytes,
  InvalidInstruction,
  Clause,
};

/// @brief Structural edit diagnostic with a source-relative failure location.
struct TextEditError {
  TextEditErrorCode code = TextEditErrorCode::None;
  uint64_t offset = 0; // .text-relative byte location when known; zero is also the default.
  std::string message;
};

} // namespace rocjitsu
