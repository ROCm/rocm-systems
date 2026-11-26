// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "concurrent_dispatch_fuzz_core.h"

namespace rocjitsu::concurrent_dispatch_fuzz {

util::FailureOr<RunStats> run_input(std::span<const uint8_t>, util::DiagnosticEmitter diagnostic) {
  return diagnostic.emit() << "injected concurrent dispatch oracle failure";
}

util::FailureOr<std::vector<uint8_t>> random_input(uint64_t, util::DiagnosticEmitter) {
  return std::vector<uint8_t>(kGeneratedInputSize, 0x5a);
}

util::FailureOr<std::vector<std::vector<uint8_t>>> seed_corpus(util::DiagnosticEmitter diagnostic) {
  return diagnostic.emit() << "unexpected seed corpus request";
}

} // namespace rocjitsu::concurrent_dispatch_fuzz
