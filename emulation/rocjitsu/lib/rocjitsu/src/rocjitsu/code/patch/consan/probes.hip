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

__device__ __attribute__((always_inline)) GlobalWords row(GlobalWords dispatch, u32 id,
                                                          u32 stride) {
  const u64 report = u64{dispatch[wire::kContextReportPointer / 4]} |
                     (u64{dispatch[wire::kContextReportPointer / 4 + 1]} << 32);
  const u64 offset = rocjitsu::consan::kReportHeaderBytes + u64{uniform(id)} * uniform(stride);
  return reinterpret_cast<GlobalWords>(report + offset);
}

__device__ __attribute__((always_inline)) void publish(GlobalWords destination, u32 value) {
  __atomic_store_n(destination, value, __ATOMIC_RELEASE);
}

__device__ __attribute__((always_inline)) rocjitsu::consan::DeviceWaveState
wave_state(GlobalWords output) {
  return {output + wire::kRowFinalSequence / 4, output + wire::kRowLoss / 4};
}

} // namespace

extern "C" __device__ __attribute__((noinline, used, visibility("default")))
void rj_consan_entry(u32 low, u32 high, u32 first_workitem, u32 stride) {
  GlobalWords dispatch = context(low, high);
  const u32 wave = uniform(first_workitem) / CONSAN_PROBE_WAVE_SIZE;
  GlobalWords output = row(dispatch, wave, stride);
  wave_state(output).initialize(lane_id());
  if (lane_id() == 0) {
    output[wire::kRowSerial / 4] = dispatch[wire::kContextSerial / 4];
    output[wire::kRowSerial / 4 + 1] = dispatch[wire::kContextSerial / 4 + 1];
    output[wire::kRowEntry / 4] = 0;
    output[wire::kRowWave / 4] = wave;
    output[wire::kRowFinalPhase / 4] = 0;
    publish(output + wire::kRowStatus / 4, wire::kRowEntered);
  }
}

extern "C" __device__ __attribute__((noinline, used, visibility("default")))
void rj_consan_access(u32 low, u32 high, u32 first_workitem, u32 stride, u32 site,
                      u32 address, u32 immediate, u32 mask_low, u32 mask_high) {
  GlobalWords dispatch = context(low, high);
  GlobalWords output = row(dispatch, uniform(first_workitem) / CONSAN_PROBE_WAVE_SIZE, stride);
  u64 remaining = u64{uniform(mask_low)} | (u64{uniform(mask_high)} << 32);
  if (remaining == 0)
    return;
  const u32 sequence = wave_state(output).next_sequence(lane_id());
  for (u32 group = 0; group < rocjitsu::consan::kAddressGroupsPerSite && remaining; ++group) {
    const u32 first = __builtin_ctzll(remaining);
    const u32 selected = __builtin_amdgcn_readlane(address, first);
#if CONSAN_PROBE_WAVE_SIZE == 64
    const u64 matching = __builtin_amdgcn_ballot_w64(address == selected) & remaining;
#else
    const u64 matching = __builtin_amdgcn_ballot_w32(address == selected) & remaining;
#endif
    remaining &= ~matching;
    if (lane_id() == 0) {
      const u32 slot = uniform(site) * rocjitsu::consan::kAddressGroupsPerSite + group;
      GlobalWords record = output + (rocjitsu::consan::kRowHeaderBytes +
                                       slot * rocjitsu::consan::kAccessRecordBytes) / 4;
      const u64 offset = u64{selected} + uniform(immediate);
      record[wire::kRecordAnalysis / 4] = u32(rocjitsu::consan::AnalysisKind::LdsRaceDetection);
      record[wire::kRecordKind / 4] = u32(rocjitsu::consan::RecordKind::Access);
      record[wire::kRecordSite / 4] = uniform(site);
      record[wire::kRecordRange / 4] = 0;
      record[wire::kRecordSequence / 4] = sequence;
      record[wire::kRecordPhase / 4] = output[wire::kRowFinalPhase / 4];
      record[wire::kRecordLdsOffset / 4] = u32(offset);
      record[wire::kRecordWidth / 4] = 4;
      record[wire::kRecordGuestMask / 4] = u32(matching);
      record[wire::kRecordGuestMask / 4 + 1] = u32(matching >> 32);
      record[wire::kRecordFlags / 4] = wire::kExactGuestMask | (offset >> 32 ? 2u : 0u);
      publish(record + wire::kRecordCommit / 4, wire::kAccessCommitted);
    }
  }
  if (remaining)
    wave_state(output).record_loss(lane_id());
}

extern "C" __device__ __attribute__((noinline, used, visibility("default")))
void rj_consan_barrier(u32 low, u32 high, u32 first_workitem, u32 stride, u32 group,
                       u32 journal_offset) {
  GlobalWords dispatch = context(low, high);
  GlobalWords output = row(dispatch, uniform(first_workitem) / CONSAN_PROBE_WAVE_SIZE, stride);
  if (lane_id() == 0) {
    const u32 phase = output[wire::kRowFinalPhase / 4] + 1;
    output[wire::kRowFinalPhase / 4] = phase;
    GlobalWords record = output + (uniform(journal_offset) +
        uniform(group) * rocjitsu::consan::kSyncRecordBytes) / 4;
    record[wire::kSyncGroup / 4] = uniform(group);
    record[wire::kSyncOccurrence / 4] = 1;
    record[wire::kSyncParticipants / 4] =
        (dispatch[wire::kContextWorkgroup / 4] + CONSAN_PROBE_WAVE_SIZE - 1) / CONSAN_PROBE_WAVE_SIZE;
    record[wire::kSyncSequence / 4] = output[wire::kRowFinalSequence / 4];
    record[wire::kSyncPhase / 4] = phase;
    publish(record + wire::kSyncCommit / 4, wire::kBarrierCompleted);
  }
}

extern "C" __device__ __attribute__((noinline, used, visibility("default")))
void rj_consan_exit(u32 low, u32 high, u32 first_workitem, u32 stride) {
  GlobalWords dispatch = context(low, high);
  GlobalWords output = row(dispatch, uniform(first_workitem) / CONSAN_PROBE_WAVE_SIZE, stride);
  if (lane_id() == 0)
    publish(output + wire::kRowStatus / 4,
            wire::kRowEntered | wire::kRowExited |
                (wave_state(output).has_loss() ? wire::kRowExhausted : 0u));
}

// Retain standalone functions through HIP device internalization and linking.
__device__ auto rj_consan_entry_keepalive = &rj_consan_entry;
__device__ auto rj_consan_access_keepalive = &rj_consan_access;
__device__ auto rj_consan_exit_keepalive = &rj_consan_exit;

__device__ auto rj_consan_barrier_keepalive = &rj_consan_barrier;
