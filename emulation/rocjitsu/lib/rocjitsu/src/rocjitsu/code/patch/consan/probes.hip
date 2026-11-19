// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/contract.h"
#include "rocjitsu/code/patch/consan/wave_state.h"

using u32 = unsigned int;
using u64 = unsigned long long;
using GlobalWords = volatile u32 __attribute__((address_space(1))) *;
namespace wire = rocjitsu::consan::wire;

namespace {

__device__ __attribute__((always_inline)) u32 uniform(u32 value) {
  return __builtin_amdgcn_readfirstlane(value);
}

__device__ __attribute__((always_inline)) u32 lane_id() {
  const u32 low = __builtin_amdgcn_mbcnt_lo(~0u, 0);
#if CONSAN_PROBE_WAVE_SIZE == 64
  return __builtin_amdgcn_mbcnt_hi(~0u, low);
#else
  return low;
#endif
}

__device__ __attribute__((always_inline)) GlobalWords context(u32 low, u32 high) {
  return reinterpret_cast<GlobalWords>(u64{uniform(low)} | (u64{uniform(high)} << 32));
}

__device__ __attribute__((always_inline)) GlobalWords row(GlobalWords dispatch, u32 id, u32 stride,
                                                          u32 workgroup) {
  const u32 waves =
      (dispatch[wire::kContextWorkgroup / 4] + CONSAN_PROBE_WAVE_SIZE - 1) / CONSAN_PROBE_WAVE_SIZE;
  const u64 report = u64{dispatch[wire::kContextReportPointer / 4]} |
                     (u64{dispatch[wire::kContextReportPointer / 4 + 1]} << 32);
  const u64 offset = rocjitsu::consan::kReportHeaderBytes +
                     (u64{uniform(workgroup)} * waves + uniform(id)) * uniform(stride);
  return reinterpret_cast<GlobalWords>(report + offset);
}

__device__ __attribute__((always_inline)) void publish(GlobalWords destination, u32 value) {
  __atomic_store_n(destination, value, __ATOMIC_RELEASE);
}

__device__ __attribute__((always_inline)) rocjitsu::consan::DeviceWaveState
wave_state(GlobalWords output) {
  return {output + wire::kRowFinalSequence / 4, output + wire::kRowLoss / 4,
          output + wire::kRowCounterExhausted / 4};
}

} // namespace

extern "C" __device__ __attribute__((noinline, used, visibility("default"))) void
rj_consan_entry(u32 low, u32 high, u32 first_workitem, u32 stride, u32 workgroup) {
  GlobalWords dispatch = context(low, high);
  const u32 wave = uniform(first_workitem) / CONSAN_PROBE_WAVE_SIZE;
  GlobalWords output = row(dispatch, wave, stride, workgroup);
  wave_state(output).initialize(lane_id());
  if (lane_id() == 0) {
    output[wire::kRowSerial / 4] = dispatch[wire::kContextSerial / 4];
    output[wire::kRowSerial / 4 + 1] = dispatch[wire::kContextSerial / 4 + 1];
    output[wire::kRowEntry / 4] = 0;
    output[wire::kRowWave / 4] = wave;
    output[wire::kRowWorkgroupX / 4] = uniform(workgroup);
    output[wire::kRowSyncLoss / 4] = 0;
    output[wire::kRowAddressLoss / 4] = 0;
    output[wire::kRowFinalPhase / 4] = 0;
    publish(output + wire::kRowStatus / 4, wire::kRowEntered);
  }
}

extern "C" __device__ __attribute__((noinline, used, visibility("default"))) void
rj_consan_access(u32 low, u32 high, u32 first_workitem, u32 stride, u32 workgroup, u32 site,
                 u32 address, u32 immediate, u32 mask_low, u32 mask_high, u32 width, u32 repeated) {
  GlobalWords dispatch = context(low, high);
  GlobalWords output =
      row(dispatch, uniform(first_workitem) / CONSAN_PROBE_WAVE_SIZE, stride, workgroup);
  u64 remaining = u64{uniform(mask_low)} | (u64{uniform(mask_high)} << 32);
  if (remaining == 0)
    return;
  const u32 sequence = wave_state(output).next_sequence(lane_id());
  for (u32 group = 0; sequence != 0 && group < rocjitsu::consan::kAddressGroupsPerSite && remaining;
       ++group) {
    const u32 first = __builtin_ctzll(remaining);
    const u32 selected = __builtin_amdgcn_readlane(address, first);
#if CONSAN_PROBE_WAVE_SIZE == 64
    const u64 matching = __builtin_amdgcn_ballot_w64(address == selected) & remaining;
#else
    const u64 matching = __builtin_amdgcn_ballot_w32(address == selected) & remaining;
#endif
    remaining &= ~matching;
    // Repeated sites retain lane-based representatives, not four iterations.
    // Each address group's lowest active lane selects a bank: lanes 0 and 4
    // both use bank 0. An occupied selected bank loses this representative even
    // if another bank is vacant; searching for a free bank changes the contract.
    const u32 bank = uniform(repeated) ? first % rocjitsu::consan::kAddressGroupsPerSite : group;
    const u32 slot = uniform(site) * rocjitsu::consan::kAddressGroupsPerSite + bank;
    GlobalWords record =
        output +
        (rocjitsu::consan::kRowHeaderBytes + slot * rocjitsu::consan::kAccessRecordBytes) / 4;
    // This wave owns its row. The committed bank itself is the persistent
    // occupancy bit, so a later invocation cannot overwrite its representative.
    if (uniform(repeated) && uniform(record[wire::kRecordCommit / 4]) != 0) {
      wave_state(output).record_loss(lane_id());
    } else if (lane_id() == 0) {
      const u64 offset = u64{selected} + uniform(immediate);
      record[wire::kRecordAnalysis / 4] = u32(rocjitsu::consan::AnalysisKind::LdsRaceDetection);
      record[wire::kRecordKind / 4] = u32(rocjitsu::consan::RecordKind::Access);
      record[wire::kRecordSite / 4] = uniform(site);
      record[wire::kRecordRange / 4] = 0;
      record[wire::kRecordSequence / 4] = sequence;
      record[wire::kRecordPhase / 4] = output[wire::kRowFinalPhase / 4];
      record[wire::kRecordLdsOffset / 4] = u32(offset);
      record[wire::kRecordWidth / 4] = uniform(width);
      record[wire::kRecordGuestMask / 4] = u32(matching);
      record[wire::kRecordGuestMask / 4 + 1] = u32(matching >> 32);
      record[wire::kRecordFlags / 4] = wire::kExactGuestMask | (offset >> 32 ? 2u : 0u);
      publish(record + wire::kRecordCommit / 4, wire::kAccessCommitted);
    }
  }
  // Remaining addresses count as one address-group exhaustion event; occupied
  // banks were counted individually above. If sequence allocation failed, no
  // groups were inspected, so count total loss without claiming address exhaustion.
  if (sequence != 0 && remaining && uniform(repeated))
    wave_state(output).advance(output + wire::kRowAddressLoss / 4, lane_id());
  if (remaining)
    wave_state(output).record_loss(lane_id());
}

extern "C" __device__ __attribute__((noinline, used, visibility("default"))) void
rj_consan_barrier(u32 low, u32 high, u32 first_workitem, u32 stride, u32 workgroup, u32 group,
                  u32 journal_offset, u32 group_count) {
  GlobalWords dispatch = context(low, high);
  GlobalWords output =
      row(dispatch, uniform(first_workitem) / CONSAN_PROBE_WAVE_SIZE, stride, workgroup);
  if (lane_id() == 0) {
    // Phase follows guest completions after the journal fills. Keep the first
    // kSyncCapacity witnesses immutable and count later completions as loss,
    // without wrapping to an earlier slot. Zero from advance() means counter
    // exhaustion, and must never publish a new witness.
    const u32 phase = wave_state(output).advance(output + wire::kRowFinalPhase / 4, 0);
    if (phase == 0 || phase > rocjitsu::consan::kSyncCapacity) {
      wave_state(output).advance(output + wire::kRowSyncLoss / 4, 0);
    } else {
      GlobalWords record =
          output + (uniform(journal_offset) + (phase - 1) * rocjitsu::consan::kSyncRecordBytes) / 4;
      record[wire::kSyncGroup / 4] = uniform(group);
      // All groups execute once per qualified loop cycle. Compute only retained
      // occurrences; the bounded ordinal cannot wrap or recreate a lost witness.
      // Runtime integer division can lower to FP operations, which would alter
      // guest exception state. Four integer steps suffice for a numerator < 16.
      static_assert(rocjitsu::consan::kSyncCapacity <= 16);
      const u32 groups = uniform(group_count);
      u32 remainder = phase - 1;
      u32 occurrence = 1;
#pragma unroll
      for (u32 shift = 4; shift != 0; --shift) {
        const u32 bit = 1u << (shift - 1);
        const u32 multiple = groups << (shift - 1);
        if (remainder >= multiple) {
          remainder -= multiple;
          occurrence += bit;
        }
      }
      record[wire::kSyncOccurrence / 4] = occurrence;
      record[wire::kSyncParticipants / 4] =
          (dispatch[wire::kContextWorkgroup / 4] + CONSAN_PROBE_WAVE_SIZE - 1) /
          CONSAN_PROBE_WAVE_SIZE;
      record[wire::kSyncSequence / 4] = output[wire::kRowFinalSequence / 4];
      record[wire::kSyncPhase / 4] = phase;
      publish(record + wire::kSyncCommit / 4, wire::kBarrierCompleted);
    }
  }
}

extern "C" __device__ __attribute__((noinline, used, visibility("default"))) void
rj_consan_exit(u32 low, u32 high, u32 first_workitem, u32 stride, u32 workgroup) {
  GlobalWords dispatch = context(low, high);
  GlobalWords output =
      row(dispatch, uniform(first_workitem) / CONSAN_PROBE_WAVE_SIZE, stride, workgroup);
  if (lane_id() == 0)
    publish(output + wire::kRowStatus / 4,
            wire::kRowEntered | wire::kRowExited |
                ((wave_state(output).has_loss() || output[wire::kRowSyncLoss / 4] ||
                  output[wire::kRowCounterExhausted / 4])
                     ? wire::kRowExhausted
                     : 0u));
}

// Retain standalone functions through HIP device internalization and linking.
__device__ auto rj_consan_entry_keepalive = &rj_consan_entry;
__device__ auto rj_consan_access_keepalive = &rj_consan_access;
__device__ auto rj_consan_exit_keepalive = &rj_consan_exit;

__device__ auto rj_consan_barrier_keepalive = &rj_consan_barrier;
