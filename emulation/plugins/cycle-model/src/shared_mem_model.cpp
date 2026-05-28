// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cycle_model/SharedMemModel.h"

namespace cycle_model {

// L2 access for an L1-missing (or L1-bypassing) line, with the ordered
// per-transaction bandwidth scheduler (finding ②). `arrive_cyc` = when the request
// reaches L2. A pending L2 fill coalesces (no new BW booked). A hit pays the L2 BW
// queue + l2.hit_latency. A miss pays the L2 BW queue, then the line's HBM channel BW
// queue + hbm_access_latency, and installs the L2 tag. Each resource's next_free is
// advanced inside schedule() before the next line reads it, so same-resource
// transactions serialize while distinct ones overlap.
uint64_t SharedMemModel::l2_access(uint64_t l1_line_base, uint64_t arrive_cyc) {
  if (!l2_.configured()) return arrive_cyc + cfg_.l1v.miss_to_next_level;  // no L2 modeled (test cfg)
  uint64_t l2base = l1_line_base & ~static_cast<uint64_t>(l2_.line_bytes - 1);

  if (int slot = l2_.lookup(l2base); slot >= 0) {
    l2_.touch(static_cast<uint32_t>(slot));
    uint64_t rdy = l2_.ready(static_cast<uint32_t>(slot));
    if (rdy > arrive_cyc) return rdy;                                 // pending L2 fill: coalesce
    uint64_t l2_done = l2_bw_.schedule(arrive_cyc, l2_.line_bytes);   // L2 hit pays BW
    return l2_done + l2_.hit_latency;
  }

  // L2 miss -> L2 BW (discover the miss) + HBM channel, then install the L2 tag.
  uint64_t l2_done = l2_bw_.schedule(arrive_cyc, l2_.line_bytes);
  uint64_t line_done = hbm_access(l2base, l2_done + l2_.hit_latency);
  uint32_t v = l2_.victim(l2base);
  l2_.install(v, l2base);                                             // valid + tag + MRU
  l2_.ready(v) = line_done;
  return line_done;
}

// HBM service for an L2-line-sized fill arriving at `arrive`: the line's per-channel
// bandwidth queue + DRAM access latency. Channel = (l2line index) % channels. Falls
// back to a flat miss latency when no HBM is modeled (test cfg).
uint64_t SharedMemModel::hbm_access(uint64_t l2base, uint64_t arrive) {
  if (hbm_ch_.empty()) return arrive + l2_.miss_to_next;             // no HBM modeled (test cfg)
  uint32_t ch = static_cast<uint32_t>((l2base / l2_.line_bytes) % hbm_ch_.size());
  return hbm_ch_[ch].schedule(arrive, l2_.line_bytes) + cfg_.hbm_access_latency;
}

// Uncached (UC) access: bypasses BOTH L1 and L2 -> straight to HBM. Books no L1/L2 tag
// or bandwidth and pays no cache-probe latency; the access arrives at HBM at `arrive`.
uint64_t SharedMemModel::uc_access(uint64_t line_base, uint64_t arrive) {
  if (!l2_.configured()) return arrive + cfg_.vmem.base_latency;     // no modeled hierarchy (test cfg)
  uint64_t l2base = line_base & ~static_cast<uint64_t>(l2_.line_bytes - 1);
  return hbm_access(l2base, arrive);
}

}  // namespace cycle_model
