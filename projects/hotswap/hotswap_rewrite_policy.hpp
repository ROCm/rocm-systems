//===- hotswap_rewrite_policy.hpp - HotSwap rewrite decision policy -------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Centralized policy for selecting whether HotSwap should call COMGR and which
// source/target ISA pair it should pass. COMGR owns the enabled rewrite passes.
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_REWRITE_POLICY_HPP
#define ROCR_HOTSWAP_REWRITE_POLICY_HPP

#include "hotswap_gfx_query.hpp"

#include <optional>
#include <string>

namespace rocr::hotswap {

struct RewriteOptions {
  bool entry_trampolines_requested = false;
};

struct RewriteDecision {
  std::string source_isa;
  std::string target_isa;
};

// HotSwap's default activation policy: B0-to-A0 rewriting is performed only
// for gfx1250 silicon at ASIC revision A0 (and only when the revision was
// successfully queried).
bool gate_allows_hotswap(const AgentGfxRevision &gfx);

// True for COMGR entry-trampoline targets: gfx12-5-generic or concrete gfx125
// processor names with a numeric suffix (for example, gfx1250 or gfx1251).
bool is_gfx12_5_entry_trampoline_target(const std::string &gfx_target);

// Agent-level precheck used by the loader to avoid source-ISA parsing when no
// COMGR rewrite can possibly apply.
bool has_candidate_hotswap_rewrite(const AgentGfxRevision &gfx,
                                   const RewriteOptions &options);

// Adds COMGR's hotswap-local gfx1250 stepping feature to an ISA name while
// preserving any existing feature suffixes. Non-gfx1250 ISA names and ISA
// names that already carry the stepping feature are returned unchanged.
std::string add_gfx1250_stepping_feature(const std::string &isa_name,
                                         bool is_b0);

// Returns the COMGR ISA pair for this load, or std::nullopt when the original
// code object should be loaded unchanged.
std::optional<RewriteDecision>
decide_hotswap_rewrite(const AgentGfxRevision &gfx,
                       const std::string &source_isa,
                       const std::string &target_isa,
                       const RewriteOptions &options);

} // namespace rocr::hotswap

#endif // ROCR_HOTSWAP_REWRITE_POLICY_HPP
