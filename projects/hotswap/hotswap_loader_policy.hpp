//===- hotswap_loader_policy.hpp - HotSwap loader decision policy ---------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Centralized loader policy for selecting whether HotSwap should call COMGR and
// which source/target ISA pair it should pass. COMGR owns validation and all
// code-object transformations after the call crosses this boundary.
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_LOADER_POLICY_HPP
#define ROCR_HOTSWAP_LOADER_POLICY_HPP

#include "hotswap_gfx_query.hpp"

#include <optional>
#include <string>

namespace rocr::hotswap {

struct RewriteOptions {
  bool gfx12_5_rewrite_requested = false;
};

struct RewriteDecision {
  std::string source_isa;
  std::string target_isa;
};

// HotSwap's baseline gfx1250 route is active only for gfx1250 silicon at ASIC
// revision A0 (and only when the revision was successfully queried).
bool gate_allows_hotswap(const AgentGfxRevision &gfx);

// Agent-level precheck used by the loader to avoid source-ISA parsing when no
// local routing condition can possibly apply.
bool has_candidate_hotswap_rewrite(const AgentGfxRevision &gfx,
                                   const RewriteOptions &options);

// Returns the COMGR ISA pair for this load, or std::nullopt when the original
// code object should be loaded unchanged. The decision is limited to loader
// routing and ISA-pair construction; COMGR decides which rewrite work, if any,
// is enabled for the request.
std::optional<RewriteDecision>
decide_hotswap_rewrite(const AgentGfxRevision &gfx,
                       const std::string &source_isa,
                       const std::string &target_isa,
                       const RewriteOptions &options);

} // namespace rocr::hotswap

#endif // ROCR_HOTSWAP_LOADER_POLICY_HPP
