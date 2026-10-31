// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/result.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {
class BasicBlock;
class InstructionBoundaries;

enum class StaticBarrierForm { BuiltinSplitPair, FullSingleton };

enum class StaticBarrierScope { Workgroup, Cluster };
enum class StaticBarrierErrorCode { None, UnsupportedForm, IncompleteBarrier };
struct StaticBarrierError {
  StaticBarrierErrorCode code = StaticBarrierErrorCode::None;
  uint64_t offset = 0; // .text-relative byte location when known; zero is also the default.
  std::string message;
};
/// @brief Original encoded instruction at a byte offset in the source .text.
class StaticBarrierMember {
public:
  uint64_t offset = 0;
  std::vector<uint8_t> bytes;
  bool operator==(const StaticBarrierMember &) const = default;
};
/// @brief One complete logical barrier, distinct from a multi-barrier drop group.
/// @details Form comes from qualification, never inferred from a caller's count.
/// Members are ordered physical instructions: two for a split pair, one for a
/// true full barrier. Equality includes form, owners' block and original bytes.
class StaticBarrier {
public:
  StaticBarrierForm form = StaticBarrierForm::BuiltinSplitPair;
  uint64_t block_offset = 0; // Containing block's start in .text-relative bytes.
  int32_t barrier_id = 0;    // Semantic builtin ID, not operand encoding; -1 for a full singleton.
  StaticBarrierScope scope = StaticBarrierScope::Workgroup;
  std::vector<StaticBarrierMember> members;
  bool operator==(const StaticBarrier &) const = default;
};

struct StaticBarrierBuiltin {
  int32_t id;
  StaticBarrierScope scope;
};

/// @brief Bounded mutation policy for a registry-bound ISA family.
/// @details Supplies form, admitted wave sizes and explicit user barrier IDs/scopes.
/// Split forms use -1 for workgroup scope; CDNA5 also admits -3 for cluster scope.
/// Trap-only IDs -2/-4 are not admitted. Full singleton barriers have logical ID -1
/// without a split selector. A cluster barrier is inert outside a clustered dispatch.
/// This is not CFG ownership, a selection heuristic or an execution qualification.
struct StaticBarrierPolicy {
  StaticBarrierForm form;
  std::span<const uint32_t> wave_sizes;
  std::span<const StaticBarrierBuiltin> builtins;
};

/// @brief Query family policy only for a concrete binding present in the registry.
/// @details Returned spans refer to static policy tables and do not borrow the registry.
[[nodiscard]] std::optional<StaticBarrierPolicy>
static_barrier_policy(rj_code_target_id_t target,
                      const IsaTargetRegistry &registry = default_isa_target_registry());

/// @brief Qualify complete builtin split pairs or full singleton barriers.
/// @details RDNA4 accepts immediate ID -1; CDNA5 also accepts -3 (cluster).
/// RDNA3/CDNA3/CDNA4 accept full s_barrier. Named lifecycle, dynamic operands, isfirst
/// (SCC side effect), trap IDs, interleaved and cross-block pairs fail closed.
/// Boundaries supply the same full-text instruction/clause facts used by checked edits.
/// Ownership and closed CFG checks belong to the caller. No convergence or race verdict.
[[nodiscard]] FailureOr<std::vector<StaticBarrier>>
qualify_static_barriers(std::span<BasicBlock *const> blocks, rj_code_target_id_t target,
                        const InstructionBoundaries &boundaries, StaticBarrierError &error);

} // namespace rocjitsu
