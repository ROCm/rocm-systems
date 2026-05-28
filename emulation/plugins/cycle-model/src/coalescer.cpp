// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cycle_model/Coalescer.h"

#include <algorithm>

namespace cycle_model {

std::vector<uint64_t> coalesce(const MemAccess& a, uint32_t line_bytes) {
  std::vector<uint64_t> lines;
  // Guard a misconfigured line size: line_bytes==0 would make `base += line_bytes`
  // loop forever and `~(line_bytes-1)` a garbage mask. No line size => no transactions.
  // (Configs are also validated power-of-two at load; this protects the public API.)
  if (line_bytes == 0) return lines;
  const uint64_t mask = ~static_cast<uint64_t>(line_bytes - 1);
  const uint64_t span = a.elem_bytes ? a.elem_bytes : 1;   // [addr, addr+span)
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!((a.lane_mask >> lane) & 1ull)) continue;
    const uint64_t lo = a.lane_addr[lane];
    const uint64_t first = lo & mask;
    const uint64_t last = (lo + span - 1) & mask;          // inclusive last touched line
    for (uint64_t base = first; base <= last; base += line_bytes)
      if (std::find(lines.begin(), lines.end(), base) == lines.end())
        lines.push_back(base);
  }
  return lines;
}

}  // namespace cycle_model
