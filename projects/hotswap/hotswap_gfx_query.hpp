//===- hotswap_gfx_query.hpp - Agent gfx-target / ASIC-revision query -----===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Portable helpers for discovering an agent's gfx target and ASIC revision via
// the HSA runtime (HSA_AMD_AGENT_INFO_ASIC_REVISION). These rely solely on HSA
// and the standard library -- no <elf.h>, libdrm, or KFD sysfs access -- so the
// unit builds and is unit-testable on both Linux and Windows.
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

// Extracts the gfx target (e.g. "gfx1250") from a full HSA ISA name. Returns an
// empty string when no gfx target is present. The returned token stops at the
// first non-alphanumeric character so feature suffixes (":sramecc+", etc.) are
// dropped.
std::string extract_gfx_target(const std::string &isa_name);

// Queries the agent's gfx target and ASIC revision via the HSA runtime. The
// result is cached per agent handle, since code-object loads can be frequent.
// This function intentionally encodes no gating policy; callers decide which
// gfx target / revision to act on.
AgentGfxRevision query_agent_gfx_revision(hsa_agent_t agent);

} // namespace rocr::hotswap

#endif // ROCR_HOTSWAP_GFX_QUERY_HPP
