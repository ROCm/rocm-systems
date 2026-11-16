// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/result.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class BasicBlock;
class InstructionBoundaries;
class Instrumentor;

enum class FaultMutationKind { DropBarrier, LdsWrongAddress };
enum class FaultMutationErrorCode {
  None,
  BadIdentity,
  UnsupportedTarget,
  UnsupportedForm,
  AmbiguousSelection,
  IncompleteSelection,
  InvalidParameters,
  ValidationFailure,
  NoCandidates,
  OrdinalOutOfRange,
};
struct FaultMutationError {
  FaultMutationErrorCode code = FaultMutationErrorCode::None;
  uint64_t offset = 0; // .text-relative byte location when known; zero is also the default.
  std::string message;
};
/// @brief Source .text facts retained by an inventory after its context expires.
struct FaultMutationSection {
  uint64_t index = 0;       // ELF section-header table index, not a text-section ordinal.
  uint64_t file_offset = 0; // Byte offset from the start of the ELF image.
  uint64_t vaddr = 0;       // Declared section virtual address, before any runtime load bias.
  uint64_t size = 0;        // Section extent in bytes.
};
/// @brief One descriptor-backed hardware entry; offsets use the pristine context's image.
class FaultMutationOwner {
public:
  std::string entry;                   // Exact descriptor-derived kernel name used for discovery.
  uint64_t entry_offset = 0;           // .text-relative byte offset, not an ELF file offset.
  uint64_t descriptor_file_offset = 0; // Byte offset from the start of the ELF image.
  bool operator==(const FaultMutationOwner &) const = default;
};

/// @brief Facts for one independently owned, closed direct-CFG kernel.
/// @details Views and block pointers borrow the context; they must not outlive it.
/// The blocks are analysis inputs and must not be mutated by consumers.
struct FaultSourceFacts {
  FaultMutationSection section;
  FaultMutationOwner owner;
  KernelDescriptorInfo descriptor;
  std::vector<BasicBlock *> blocks;
  const InstructionBoundaries *boundaries = nullptr;
};

class FaultBarrierInventory;
class DropBarrierSelection;
class LdsInventory;
class LdsSelection;

/// @brief Own an immutable source snapshot and its selection lifetime.
/// @details Copies the input once. Independently constructed contexts have distinct
/// identities even for identical bytes. The owned code object supplies the target;
/// changing the caller's input buffer does not change this snapshot. The context
/// caches the fixed all-entry direct-ownership profile and retains an Instrumentor
/// seed whose full-text facts are shared by independent replacement batches.
/// It does not support alternate roots/ranges/options or helper/indirect-analysis profiles.
/// The caller/session should retain one context per immutable input across discovery,
/// selection and apply, until all related operations finish. Repeated applies produce
/// independent mutants of that original snapshot. To analyze a mutant, create a new
/// context; the pristine context and its facts are never retargeted to changed bytes.
class FaultMutationContext {
public:
  explicit FaultMutationContext(std::span<const uint8_t> image);
  ~FaultMutationContext();
  FaultMutationContext(const FaultMutationContext &) = delete;
  FaultMutationContext &operator=(const FaultMutationContext &) = delete;

  const AmdGpuCodeObject &source() const { return *source_; }
  std::span<const uint8_t> image() const;

  /// @brief Create a fresh request queue bound to the admitted pristine source.
  /// @details The context must outlive the returned Instrumentor. Forks share only
  /// immutable source facts; queues and patch attempts are independent.
  [[nodiscard]] FailureOr<std::unique_ptr<Instrumentor>>
  fork_instrumentor(FaultMutationError &error) const;

  /// @brief Find source facts independently of any fault recipe.
  /// @details The returned pointer and its views remain valid only while this context lives.
  [[nodiscard]] FailureOr<const FaultSourceFacts *> source_facts(std::string_view entry,
                                                                 FaultMutationError &error) const;

private:
  class State;
  Result analyze() const;
  void qualify_drop() const;
  // This context is the sole persistent strong source owner. The shared control block
  // lets records/selections observe expiry via weak_ptr, including after address reuse.
  const std::shared_ptr<const AmdGpuCodeObject> source_;
  std::unique_ptr<State> state_;
  friend FailureOr<struct DropBarrierResult> apply_drop_barrier(const FaultMutationContext &,
                                                                const DropBarrierSelection &,
                                                                FaultMutationError &);
  friend FailureOr<FaultBarrierInventory>
  discover_drop_barriers(const FaultMutationContext &, std::string_view, FaultMutationError &);
  friend FailureOr<LdsInventory> discover_lds_wrong_address(const FaultMutationContext &,
                                                            std::string_view, FaultMutationError &);
  friend FailureOr<struct LdsResult> apply_lds_wrong_address(const FaultMutationContext &,
                                                             const LdsSelection &,
                                                             std::optional<uint32_t>,
                                                             FaultMutationError &);
};

} // namespace rocjitsu
