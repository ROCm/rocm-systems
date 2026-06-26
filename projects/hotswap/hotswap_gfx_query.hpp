//===- hotswap_gfx_query.hpp - Agent gfx-target / ASIC-revision query -----===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Portable helpers for discovering an agent's gfx target and ASIC revision via
// the HSA runtime (HSA_AMD_AGENT_INFO_ASIC_REVISION).
//
//===----------------------------------------------------------------------===//

#ifndef ROCR_HOTSWAP_GFX_QUERY_HPP
#define ROCR_HOTSWAP_GFX_QUERY_HPP

#include <cstdint>
#include <string>

#include <hsa.h>
#include <hsa_ext_amd.h>

namespace rocr::hotswap {

// Architecture-agnostic description of an agent: its gfx target and ASIC
// revision (A0 == 0, A1 == 1, ...). revision_valid is false when the ASIC
// revision could not be queried from the runtime.
struct AgentGfxRevision {
  std::string gfx_target;
  uint32_t asic_revision = 0;
  bool revision_valid = false;
};

// Returns the full HSA ISA name reported for the agent (e.g.
// "amdgcn-amd-amdhsa--gfx1250:sramecc+:xnack-"), or an empty string on failure.
std::string get_agent_isa_name(hsa_agent_t agent);

// Extracts the gfx target (e.g. "gfx1250" or "gfx12-5-generic") from a full
// HSA ISA name. Returns an empty string when no gfx target is present. The
// returned token preserves hyphenated processor names and stops before feature
// suffixes (":sramecc+", etc.).
std::string extract_gfx_target(const std::string &isa_name);

// Queries the agent's gfx target and ASIC revision via the HSA runtime. The
// result is cached per agent handle, since code-object loads can be frequent.
// This function intentionally encodes no gating policy; callers apply
// gate_allows_hotswap() (below) to decide whether to act.
AgentGfxRevision query_agent_gfx_revision(hsa_agent_t agent);

// Clears the per-agent-handle cache used by query_agent_gfx_revision().
void reset_gfx_revision_cache();

// HotSwap's default activation policy: B0-to-A0 rewriting is performed only
// for gfx1250 silicon at ASIC revision A0 (and only when the revision was
// successfully queried).
bool gate_allows_hotswap(const AgentGfxRevision &gfx);

// Entry trampolines are a separate, opt-in rewrite that applies to the gfx12.5
// family (gfx125* and gfx12-5-generic) when
// AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES is enabled.
bool gate_allows_entry_trampolines(const AgentGfxRevision &gfx);

// True when a parsed gfx target is in the COMGR entry-trampoline family.
bool is_gfx12_5_entry_trampoline_target(const std::string &gfx_target);

// Combined activation policy for libhsa-hotswap.so.
bool gate_allows_hotswap_rewrite(const AgentGfxRevision &gfx,
                                 bool entry_trampolines_requested);

// Adds COMGR's hotswap-local gfx1250 stepping feature to an ISA name while
// preserving any existing feature suffixes. Non-gfx1250 ISA names and ISA
// names that already carry the stepping feature are returned unchanged.
std::string add_gfx1250_stepping_feature(const std::string &isa_name,
                                         bool is_b0);

} // namespace rocr::hotswap

#endif // ROCR_HOTSWAP_GFX_QUERY_HPP
