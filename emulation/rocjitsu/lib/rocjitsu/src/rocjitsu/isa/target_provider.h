// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file target_provider.h
/// @brief Helper for target-owned static ISA provider definitions.

#pragma once

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"

#include <memory>

namespace rocjitsu {

/// @brief Construct a decoder for @p Isa from a descriptor function pointer.
template <typename Isa>
std::unique_ptr<Decoder> make_isa_decoder(const IsaExecutionBackend *execution_backend = nullptr,
                                          uint64_t isa_features = 0) {
  return std::make_unique<IsaDecoder<Isa>>(execution_backend, isa_features);
}

} // namespace rocjitsu
