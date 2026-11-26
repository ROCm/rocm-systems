// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/fault_injection/fault_mutation_context.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

struct Elf64_Ehdr;
struct FaultBarrierRecord;
struct TextEditError;

class FaultMutationContext::State {
public:
  std::once_flag analysis_once;
  FaultMutationError error; // Whole-image admission/ownership failure, shared by every discovery.
  bool analysis_ok = false;
  bool complete_ownership_needed = false;
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  std::vector<FaultSourceFacts> sources;
  std::once_flag drop_once;
  std::vector<std::shared_ptr<const FaultBarrierRecord>> kernels;
  // Borrow the context's source; forks share these pristine facts, not attempt state.
  std::unique_ptr<Instrumentor> instrumentor;
  std::once_flag ownership_once;
  FaultMutationError ownership_error;
  std::unique_ptr<KernelOwnership> ownership;
};

Result reject_fault_mutation(FaultMutationError &error, FaultMutationErrorCode code,
                             std::string message, uint64_t offset = 0);
Result validate_fault_mutation_source_header(std::span<const uint8_t> image, Elf64_Ehdr &header,
                                             FaultMutationError &error);
FaultMutationError translate_fault_mutation_edit_error(const TextEditError &edit);

} // namespace rocjitsu
