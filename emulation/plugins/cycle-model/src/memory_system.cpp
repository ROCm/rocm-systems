// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cycle_model/MemorySystem.h"

#include "cycle_model/Coalescer.h"
#include "cycle_model/CycleWaveState.h"

#include <algorithm>

namespace cycle_model {

namespace {
// rocjitsu Mtype: UC=0, NT=4. UC is fully uncached (bypasses L1 AND L2 -> straight to
// HBM); NT is L2-only (skips L1, still caches in L2). The per-access non_temporal hint
// is an NT-equivalent L1-skip.
bool bypass_l1(const MemAccess& m) {
  return m.mtype == 0 /*UC*/ || m.mtype == 4 /*NT*/ || m.non_temporal;
}
bool bypass_l2(const MemAccess& m) { return m.mtype == 0 /*UC*/; }

// A memory InstrEvent built by the adapter always carries a heap MemAccess; a few
// lib tests construct memory events with a null mem (don't care about addresses).
// Map null -> a shared empty access (lane_mask=0 => no active lanes), matching the
// old inline default-constructed MemAccess so those paths behave identically.
const MemAccess& access_of(const InstrEvent& inst) {
  static const MemAccess kEmpty{};
  return inst.mem ? *inst.mem : kEmpty;
}
}  // namespace

// L1-local walk only. Resident ready lines hit; pending fills coalesce (finding ①).
// UC/NT bypass L1 and emit SharedReqs; cold L1 misses emit a SharedReq, install a
// pending L1 victim (ready_cyc unresolved until completion), and HOLD an MSHR (free
// cycle unresolved). The shared L2/HBM hierarchy is serviced externally and reported
// back via on_mem_completion -> resolve_fills. LDS / unconfigured stay flat.
MemorySystem::AccessResult MemorySystem::access(const InstrEvent& inst, uint64_t now, CycleWaveState& ws) {
  AccessResult out;
  const MemAccess& ma = access_of(inst);   // heap mem for adapter-built events; empty for null
  if (inst.kind == InstrKind::LDS) { out.local_ready_cyc = now + lds_latency(ma); return out; }

  TimedTagCache& l1 = (inst.kind == InstrKind::SMEM) ? l1s_ : l1v_;
  if (!l1.configured()) {
    out.local_ready_cyc = now + ((inst.kind == InstrKind::SMEM) ? cfg_.smem.base_latency
                                                                : cfg_.vmem.base_latency);
    return out;
  }
  const bool skip_l1 = bypass_l1(ma);
  const bool skip_l2 = bypass_l2(ma);
  const bool uses_mshr = is_vector_mem(inst.kind) && !mshr_free_.empty();
  std::vector<uint64_t> lines = coalesce(ma, l1.line_bytes);
  if (lines.empty()) { out.local_ready_cyc = now + l1.hit_latency; return out; }  // no active lanes (defensive)

  out.local_ready_cyc = now;   // bumped by any L1-resolved line; shared lines dominate via max at completion
  for (uint64_t base : lines) {
    if (skip_l2) {                                   // UC: bypass L1+L2 -> HBM
      out.shared.push_back({base, now, true});
    } else if (skip_l1) {                            // NT: bypass L1, L2-only
      out.shared.push_back({base, now, false});
    } else if (int slot = l1.lookup(base); slot >= 0) {
      uint64_t rdy = l1.ready(static_cast<uint32_t>(slot));
      if (rdy == kUnresolved) {
        // Pending fill: coalesce onto the in-flight miss. Defer like a miss (resolved at
        // completion over the shared path), but the line is already resident -> claim no
        // new victim and no new MSHR. Recording base in l1_fill_lines heals this slot's
        // ready_cyc at completion, which also repairs a slot orphaned by a halted wave.
        out.shared.push_back({base, now + l1.hit_latency, false});
        out.l1_fill_lines.push_back(base);
      } else {                                       // resolved: real hit / resolved-future coalesce
        out.local_ready_cyc = std::max(out.local_ready_cyc, (rdy <= now) ? now + l1.hit_latency : rdy);
      }
      l1.touch(static_cast<uint32_t>(slot));
    } else {                                         // cold L1 miss -> shared, install pending L1 victim
      out.shared.push_back({base, now + l1.hit_latency, false});
      uint32_t v = l1.victim(base);
      l1.install(v, base);                           // valid + tag + MRU
      l1.ready(v) = kUnresolved;                     // resolved at completion
      out.l1_fill_lines.push_back(base);
      if (uses_mshr) {
        size_t i = std::min_element(mshr_free_.begin(), mshr_free_.end()) - mshr_free_.begin();
        mshr_free_[i] = kUnresolved;                 // HELD until completion resolves it
        mshr_owner_[i] = &ws;
        out.mshr_slots.push_back(static_cast<uint32_t>(i));
      }
    }
  }
  return out;
}

void MemorySystem::resolve_fills(const std::vector<uint32_t>& mshr_slots,
                                 const std::vector<uint64_t>& l1_fill_lines, bool is_smem,
                                 uint64_t complete) {
  for (uint32_t i : mshr_slots)
    if (i < mshr_free_.size()) mshr_free_[i] = complete;
  // One access = one cache (memory_system.cpp:32 picks l1 per inst.kind); route the
  // fill resolution to that same cache only. Scanning both would cross-resolve a
  // pending SMEM line sharing a numeric base with a pending VMEM line (review #6).
  TimedTagCache& l1 = is_smem ? l1s_ : l1v_;
  for (uint64_t base : l1_fill_lines) {
    if (int s = l1.lookup(base); s >= 0) {
      uint64_t& rc = l1.ready(static_cast<uint32_t>(s));
      if (rc == kUnresolved) rc = complete;
    }
  }
}

// LDS bank conflict: bank = (addr/4) % lds_banks. A bank serves one distinct word
// per cycle; lanes hitting the same word broadcast (one access). Conflict degree =
// max distinct words mapped to any single bank; extra serialized cycles = degree - 1.
// ds_*2 dual-access (a second per-lane address set) is not folded yet — deferred
// (would double the hot-path MemAccess payload for a rare op).
uint64_t MemorySystem::lds_latency(const MemAccess& m) const {
  const uint32_t banks = cfg_.lds_banks ? cfg_.lds_banks : 32;
  // Distinct dword addresses (broadcast dedup), then count per bank.
  std::vector<uint64_t> words;
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!((m.lane_mask >> lane) & 1ull)) continue;
    uint64_t word = m.lane_addr[lane] / 4;
    if (std::find(words.begin(), words.end(), word) == words.end()) words.push_back(word);
  }
  std::vector<uint32_t> per_bank(banks, 0);
  for (uint64_t w : words) ++per_bank[w % banks];
  uint32_t degree = 0;
  for (uint32_t c : per_bank) degree = std::max(degree, c);
  uint64_t conflict_cycles = degree > 0 ? degree - 1 : 0;
  return cfg_.lds_pipe.base_latency + conflict_cycles;
}

uint32_t MemorySystem::vector_misses_needing_mshr(const InstrEvent& inst) const {
  uint32_t needed = 0;
  for (uint64_t base : coalesce(access_of(inst), l1v_.line_bytes))
    if (!l1v_.present(base)) ++needed;                                 // tag-absent -> needs a new MSHR
  return needed;
}

// Free every MSHR a halting wave still holds: its fill will never arrive, so leaving
// the slot busy until its old free_cyc would spuriously block live waves (review #4).
void MemorySystem::release_wave_mshrs(const CycleWaveState* owner, uint64_t now) {
  for (size_t i = 0; i < mshr_free_.size(); ++i)
    if (mshr_owner_[i] == owner) {
      if (mshr_free_[i] > now) mshr_free_[i] = now;   // free now (drop the never-arriving fill)
      mshr_owner_[i] = nullptr;
    }
}

// Read-only MSHR admission gate (§2/§4c). A vector access whose cold-miss lines need
// more MSHRs than are free at `now` cannot admit: return the cycle enough free up.
// Mutates nothing. Non-vector / bypass / unconfigured -> always admit (return now).
uint64_t MemorySystem::admit_probe(const InstrEvent& inst, uint64_t now,
                                   const CycleWaveState& /*ws*/) const {
  if (!is_vector_mem(inst.kind) || bypass_l1(access_of(inst)) || !l1v_.configured() ||
      mshr_free_.empty())
    return now;

  uint32_t needed = vector_misses_needing_mshr(inst);
  if (needed == 0) return now;
  const uint32_t pool = static_cast<uint32_t>(mshr_free_.size());
  if (needed > pool) needed = pool;                                    // can't hold more than the pool at once

  uint32_t free = 0;
  for (uint64_t fc : mshr_free_) if (fc <= now) ++free;
  if (free >= needed) return now;

  // Need (needed - free) more to free up: the (needed-free)-th earliest busy MSHR.
  std::vector<uint64_t> busy;
  for (uint64_t fc : mshr_free_) if (fc > now) busy.push_back(fc);
  std::sort(busy.begin(), busy.end());
  return busy[needed - free - 1];
}

}  // namespace cycle_model
