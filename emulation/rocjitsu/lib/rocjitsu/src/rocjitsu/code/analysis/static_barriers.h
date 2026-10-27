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

enum class StaticBarrierForm { BuiltinSplitPair, FullSingleton };

enum class StaticBarrierScope { Workgroup, Cluster };
enum class StaticBarrierErrorCode { None, UnsupportedForm, IncompleteBarrier };
struct StaticBarrierError {
  StaticBarrierErrorCode code = StaticBarrierErrorCode::None;
  uint64_t offset = 0;
  std::string message;
};
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
  uint64_t block_offset = 0;
  int32_t barrier_id = 0;
  StaticBarrierScope scope = StaticBarrierScope::Workgroup;
  std::vector<StaticBarrierMember> members;
  bool operator==(const StaticBarrier &) const = default;
};

/// @brief Bounded mutation policy for a registry-bound ISA family.
/// @details Supplies form, admitted wave sizes and the lowest semantic builtin ID.
/// Split forms use -1/-2 for workgroup scope; CDNA5 also admits -3/-4 for cluster
/// scope. Full singleton barriers have logical ID -1 without a split selector.
/// This is not CFG ownership, a selection heuristic or an execution qualification.
struct StaticBarrierPolicy {
  StaticBarrierForm form;
  std::span<const uint32_t> wave_sizes;
  int32_t minimum_builtin_id;
};

/// @brief Query family policy only for a concrete binding present in the registry.
[[nodiscard]] std::optional<StaticBarrierPolicy>
static_barrier_policy(rj_code_target_id_t target,
                      const IsaTargetRegistry &registry = default_isa_target_registry());

/// @brief Qualify complete builtin split pairs or full singleton barriers.
/// @details RDNA4 accepts immediate IDs -1/-2; CDNA5 also accepts -3/-4 (cluster).
/// RDNA3/CDNA3/CDNA4 accept full s_barrier. Named lifecycle, dynamic operands, isfirst
/// (SCC side effect), interleaved and cross-block pairs fail closed. Ownership
/// and closed CFG checks belong to the caller. No convergence or race verdict.
[[nodiscard]] FailureOr<std::vector<StaticBarrier>>
qualify_static_barriers(std::span<BasicBlock *const> blocks, rj_code_target_id_t target,
                        StaticBarrierError &error);

} // namespace rocjitsu
