/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Staging Buffer Primitives — Device-Side CUDA Header
 *
 * This header-only file contains all __device__ __forceinline__ functions
 * used by the staging reshard kernel.  It is organized in layers:
 *
 *   Layer 1: PTX memory ordering helpers
 *            (st_release_sys, ld_acquire_sys, st_release_gpu, ld_acquire_gpu,
 *             st_release, ld_acquire)
 *
 *   Layer 2: Flow control primitives
 *            (rdma_wait_for_credits, lsa_wait_for_credits,
 *             lsa_rdma_wait_for_credits,
 *             rdma_signal, lsa_signal,
 *             staging_poll, rdma_poll,
 *             rdma_release, lsa_release,
 *             staging_fc_load_base)
 *
 *   Layer 3: Data movement primitives
 *            (staging_copy_contig, staging_copy_pack, staging_copy_unpack,
 *             staging_copy — unified dispatcher)
 *
 * Dependencies:
 *   staging_buffer.h  — StagingFlowCtrl, StagingRegion, StagingPeerInfo,
 *                        StagingTransferPlan, StagingKernelParams, constants
 *   nccl.h / nccl_device.h — ncclGin, ncclGinSignal_t, ncclWindow_t,
 *                              ncclTeam, ncclGetLocalPointer, ncclGetLsaPointer
 *
 * No dependency on reshard_3d_tensor.cu or MPI — purely device-side.
 *
 * See staging-primitives-impl-plan.md for the full design.
 ************************************************************************/

#ifndef NCCL_STAGING_PRIMITIVES_CUH
#define NCCL_STAGING_PRIMITIVES_CUH

#include "staging_types.h"

// ============================================================================
// Layer 1: PTX Memory Ordering Helpers
// ============================================================================
//
// These intrinsics provide acquire/release semantics for the SPSC ring buffer
// protocol.  Two scopes are needed:
//
//   System scope (st_release_sys / ld_acquire_sys):
//     Cross-GPU visibility — used for LSA remote peers and any memory that
//     must be visible across NVLink (different GPUs on the same node).
//
//   GPU scope (st_release_gpu / ld_acquire_gpu):
//     Same-GPU visibility — used for the local inter-warp pipeline
//     (Type 1→4 and Type 2→3), where producer and consumer are warps on
//     the same SM/GPU.  Lighter weight than system scope.
//
// The st_release / ld_acquire wrappers branch on isLocal so each primitive
// can select the correct scope without duplicating code.
// ============================================================================

// --- System scope (cross-GPU): for LSA / RDMA remote flow control ---

// Store with release semantics at system scope.
// Ensures all prior writes (data) are ordered before this store (signal).
__device__ __forceinline__ void st_release_sys(uint64_t* ptr, uint64_t val) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ptr), "l"(val) : "memory");
}

// Load with acquire semantics at system scope.
// Ensures all subsequent reads (data) are ordered after this load (signal).
__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* ptr) {
  uint64_t val;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(val) : "l"(ptr) : "memory");
  return val;
}

// --- GPU scope (same-GPU): for local inter-warp pipeline flow control ---

// Store with release semantics at GPU scope.
__device__ __forceinline__ void st_release_gpu(uint64_t* ptr, uint64_t val) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(ptr), "l"(val) : "memory");
}

// Load with acquire semantics at GPU scope.
__device__ __forceinline__ uint64_t ld_acquire_gpu(const uint64_t* ptr) {
  uint64_t val;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(val) : "l"(ptr) : "memory");
  return val;
}

// --- Scope-selecting wrappers (branch on isLocal) ---

// Store with release semantics — selects GPU or system scope.
__device__ __forceinline__ void st_release(uint64_t* ptr, uint64_t val, bool isLocal) {
  if (isLocal) {
    st_release_gpu(ptr, val);
  } else {
    st_release_sys(ptr, val);
  }
}

// Load with acquire semantics — selects GPU or system scope.
__device__ __forceinline__ uint64_t ld_acquire(const uint64_t* ptr, bool isLocal) {
  if (isLocal) {
    return ld_acquire_gpu(ptr);
  } else {
    return ld_acquire_sys(ptr);
  }
}

#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
#ifndef STAGING_PIPE_SYNC_TIMEOUT_CYCLES
#define STAGING_PIPE_SYNC_TIMEOUT_CYCLES 10000000000ULL
#endif

__device__ __forceinline__ void staging_pipe_sync_timeout(StagingKernelParams* params, uint64_t epoch, int channel_id,
                                                          const char* label, int peer, size_t progress,
                                                          uint64_t expected, uint64_t observed,
                                                          unsigned long long elapsed) {
  printf("[PIPE_SYNC_TIMEOUT] rank=%d local_rank=%d epoch=%llu ch=%d wait=%s peer=%d progress=%llu expected=%llu "
         "observed=%llu elapsed_cycles=%llu\n",
         params->myRank, params->myLocalRank, (unsigned long long)epoch, channel_id, label, peer,
         (unsigned long long)progress, (unsigned long long)expected, (unsigned long long)observed, elapsed);
  asm volatile("trap;");
}

__device__ __forceinline__ void staging_pipe_check_timeout(StagingKernelParams* params, uint64_t epoch,
                                                           int channel_id, const char* label, int peer,
                                                           size_t progress, uint64_t expected, uint64_t observed,
                                                           unsigned long long start) {
  const unsigned long long elapsed = clock64() - start;
  if (elapsed > STAGING_PIPE_SYNC_TIMEOUT_CYCLES) {
    staging_pipe_sync_timeout(params, epoch, channel_id, label, peer, progress, expected, observed, elapsed);
  }
}
#endif

// ============================================================================
// Layer 2: Flow Control Primitives
// ============================================================================

// ----------------------------------------------------------------------------
// 2a. wait_for_credits — Producer spins until consumer has freed ring slots
// ----------------------------------------------------------------------------

// RDMA variant: reads head from gin signal (remote RDMA) or staging buffer
// memory (local pipeline).
//
// Called by: Producer warps (Types 1, 4, 5)
//
// For local pipeline (isLocal == true):
//   Head pointer lives in the staging buffer's local pipeline control entry.
//   Read with GPU-scope acquire.
//
// For remote RDMA (isLocal == false):
//   Head counter is tracked by a gin signal.  gin.readSignal() returns the
//   current uint64_t value.  We subtract headSignalBase (read at kernel
//   start) to get a 0-based counter comparable with shadowTail.
__device__ __forceinline__ void rdma_wait_for_credits(ncclGin& gin, StagingRegion& region, StagingFlowCtrl& fc) {
  if (fc.isLocal) {
    // Local pipeline (same GPU): read head from staging buffer
    char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
    uint64_t* local_head_ptr = (uint64_t*)(staging_base + fc.localHeadOffset);

    while ((fc.shadowTail - ld_acquire_gpu(local_head_ptr)) >= (uint64_t)fc.peerNumSlots) {
      // Back-pressure: ring buffer full, wait for consumer to release
    }
  } else {
    // Remote RDMA: read head from gin signal
    while ((fc.shadowTail - (gin.readSignal(fc.localHeadSignal) - fc.headSignalBase)) >= (uint64_t)fc.peerNumSlots) {
      // Back-pressure: ring buffer full, wait for remote consumer
    }
  }
}

// LSA variant: reads head from staging buffer memory (both local pipeline
// and remote LSA use staging buffer memory for flow control).
//
// Called by: Producer warps (Types 2, 3)
//
// Structurally identical to rdma_wait_for_credits's local path, but uses
// the LSA region's window and selects scope via fc.isLocal.
__device__ __forceinline__ void lsa_wait_for_credits(StagingRegion& region, StagingFlowCtrl& fc) {
  char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
  uint64_t* local_head_ptr = (uint64_t*)(staging_base + fc.localHeadOffset);

  // GPU scope for local pipeline (isLocal=true), system scope for remote LSA.
  // Subtract lsaHeadBase so the credit math operates in delta space (with
  // base==0 the arithmetic is identical to today's absolute behavior).
  while ((fc.shadowTail - (ld_acquire(local_head_ptr, fc.isLocal) - fc.lsaHeadBase)) >= (uint64_t)fc.peerNumSlots) {
    // Back-pressure: ring buffer full, wait for consumer to release
  }
}

// Hybrid LSA-RDMA variant: producer reads gin counter (DMA local completion)
// instead of memory head pointer for credit tracking.
//
// Called by: Type 1 warp (producer) for the local inter-warp pipeline
// when the consumer (Type 4) does gin.put with CounterInc.
//
// The gin counter is incremented by the DMA engine when it finishes reading
// the local source buffer.  Reading this counter replaces both:
//   - the memory-based head store/load in the local pipeline
//   - the expensive gin.flush() that was previously needed after gin.put
//
// The gin counter accumulates within a devComm's lifetime but resets to 0
// when a new devComm is created (e.g., different ginSignalCount between
// tests).  The memory-based shadowTail, however, persists in the staging
// buffer's control region across ALL kernel launches regardless of devComm.
// To handle this epoch mismatch, the caller passes base_offset =
// (tail_base - counter_base), where tail_base is the memory tail read in
// the prologue and counter_base is the gin counter read in the prologue.
// The comparison normalizes both values to deltas since kernel start:
//   in_flight = (shadowTail - tail_base) - (readCounter - counter_base)
// which simplifies to: shadowTail - readCounter - base_offset.
//
// The consumer (Type 4) still uses staging_poll() for the tail (data
// availability) — that path is unchanged.
__device__ __forceinline__ void lsa_rdma_wait_for_credits(ncclGin& gin, StagingFlowCtrl& fc, uint64_t base_offset) {
  while ((fc.shadowTail - gin.readCounter(fc.localPutCounter) - base_offset) >= (uint64_t)fc.peerNumSlots) {
    // Back-pressure: ring buffer full, wait for DMA completion
  }
}

#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
/* Debug-only counterparts keep credit-wait instrumentation alongside the
 * shared primitives. The normal build continues to call the compact loops
 * above and does not reference the timeout machinery. */
__device__ __forceinline__ void rdma_wait_for_credits_debug(ncclGin& gin, StagingRegion& region, StagingFlowCtrl& fc,
                                                            StagingKernelParams* params, uint64_t epoch,
                                                            int channel_id, const char* label, int peer,
                                                            size_t progress) {
  const unsigned long long start = clock64();
  const uint64_t expected = fc.shadowTail - (uint64_t)fc.peerNumSlots + 1;
  if (fc.isLocal) {
    char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
    uint64_t* local_head_ptr = (uint64_t*)(staging_base + fc.localHeadOffset);
    uint64_t observed = ld_acquire_gpu(local_head_ptr);
    while ((fc.shadowTail - observed) >= (uint64_t)fc.peerNumSlots) {
      staging_pipe_check_timeout(params, epoch, channel_id, label, peer, progress, expected, observed, start);
      observed = ld_acquire_gpu(local_head_ptr);
    }
  } else {
    uint64_t observed = gin.readSignal(fc.localHeadSignal) - fc.headSignalBase;
    while ((fc.shadowTail - observed) >= (uint64_t)fc.peerNumSlots) {
      staging_pipe_check_timeout(params, epoch, channel_id, label, peer, progress, expected, observed, start);
      observed = gin.readSignal(fc.localHeadSignal) - fc.headSignalBase;
    }
  }
}

__device__ __forceinline__ void lsa_wait_for_credits_debug(StagingRegion& region, StagingFlowCtrl& fc,
                                                           StagingKernelParams* params, uint64_t epoch,
                                                           int channel_id, const char* label, int peer,
                                                           size_t progress) {
  char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
  uint64_t* local_head_ptr = (uint64_t*)(staging_base + fc.localHeadOffset);
  const unsigned long long start = clock64();
  const uint64_t expected = fc.shadowTail - (uint64_t)fc.peerNumSlots + 1;
  uint64_t observed = ld_acquire(local_head_ptr, fc.isLocal) - fc.lsaHeadBase;
  while ((fc.shadowTail - observed) >= (uint64_t)fc.peerNumSlots) {
    staging_pipe_check_timeout(params, epoch, channel_id, label, peer, progress, expected, observed, start);
    observed = ld_acquire(local_head_ptr, fc.isLocal) - fc.lsaHeadBase;
  }
}

__device__ __forceinline__ void lsa_rdma_wait_for_credits_debug(ncclGin& gin, StagingFlowCtrl& fc,
                                                                 uint64_t base_offset, StagingKernelParams* params,
                                                                 uint64_t epoch, int channel_id, const char* label,
                                                                 int peer, size_t progress) {
  const unsigned long long start = clock64();
  const uint64_t expected = fc.shadowTail - (uint64_t)fc.peerNumSlots + 1 - base_offset;
  uint64_t observed = gin.readCounter(fc.localPutCounter);
  while ((fc.shadowTail - observed - base_offset) >= (uint64_t)fc.peerNumSlots) {
    staging_pipe_check_timeout(params, epoch, channel_id, label, peer, progress, expected, observed, start);
    observed = gin.readCounter(fc.localPutCounter);
  }
}
#endif

__device__ __forceinline__ uint64_t staging_cursor_load(StagingRegion& region, size_t offset) {
  char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
  return ld_acquire_gpu((const uint64_t*)(staging_base + offset));
}

__device__ __forceinline__ void staging_cursor_store(StagingRegion& region, size_t offset, uint64_t value) {
  char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
  st_release_gpu((uint64_t*)(staging_base + offset), value);
}

// ----------------------------------------------------------------------------
// 2b. signal — Producer notifies consumer that a new chunk is ready (tail++)
// ----------------------------------------------------------------------------

// RDMA variant: writes tail via gin.signal (remote RDMA) or staging buffer
// memory (local pipeline).
//
// Called by: Producer warps (Types 1, 4, 5) — after writing data.
//
// For local pipeline (isLocal == true):
//   Tail pointer is in our own staging buffer.  The consumer warp reads it
//   via staging_poll with GPU-scope acquire.
//
// For remote RDMA (isLocal == false):
//   A standalone gin.signal(..., ncclGin_SignalInc{...}) increments the
//   peer's localTailSignal by 1.  RDMA ordering guarantees that the
//   signal is delivered after any prior gin.put to the same peer within
//   this CTA.
//
// Only lane 0 of the warp should call this (same convention as gin.signal).
__device__ __forceinline__ void rdma_signal(ncclGin& gin, ncclTeam world, StagingRegion& region, StagingFlowCtrl& fc) {
  fc.shadowTail++;

  if (fc.isLocal) {
    // Local pipeline (same GPU): direct store with GPU-scope release
    char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
    uint64_t* local_tail_scratch = (uint64_t*)(staging_base + fc.localTailOffset);
    st_release_gpu(local_tail_scratch, fc.shadowTail);
  } else {
    // Remote RDMA: standalone gin.signal to increment tail on the peer
    gin.signal(world, fc.remoteRank, ncclGin_SignalInc{fc.remoteTailSignal});
  }
}

// LSA variant: writes tail via LSA store (remote LSA) or staging buffer
// memory (local pipeline).
//
// Called by: Producer warps (Types 2, 3) — after writing data.
__device__ __forceinline__ void lsa_signal(StagingRegion& region, StagingFlowCtrl& fc) {
  fc.shadowTail++;

  // Wire write goes out as base + delta.  With lsaTailBase==0 (the
  // default for all paths except PIPE's LSA T6) this reduces to
  // today's absolute write of fc.shadowTail.
  uint64_t wire_val = fc.lsaTailBase + fc.shadowTail;

  if (fc.isLocal) {
    // Local pipeline (same GPU): direct store with GPU-scope release
    char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
    uint64_t* local_tail = (uint64_t*)(staging_base + fc.localTailOffset);
    st_release_gpu(local_tail, wire_val);
  } else {
    // Remote LSA (intra-node, cross-GPU): release store via LSA pointer
    uint64_t* remote_tail = (uint64_t*)ncclGetLsaPointer(region.window, fc.remoteTailOffset, fc.remoteRank);
    st_release_sys(remote_tail, wire_val);
  }
}

// ----------------------------------------------------------------------------
// 2c. poll — Consumer waits for new data from a producer
// ----------------------------------------------------------------------------

// staging_poll: reads tail from staging buffer memory.
// Works for local inter-warp pipelines (GPU-scope acquire) and remote LSA
// (system-scope acquire).
//
// Called by: Consumer warps (Types 3, 4, 7)
//
// Returns: number of new chunks available (0 if none).
// On success, *receive_offset is set to the absolute byte offset of the
// first new chunk in the staging buffer.
__device__ __forceinline__ int staging_poll(StagingRegion& region, StagingFlowCtrl& fc,
                                            size_t* receive_offset // OUT: offset of first new chunk in data region
) {
  char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
  uint64_t* tail_ptr = (uint64_t*)(staging_base + fc.localTailOffset);

  // Acquire load on tail pointer — ensures subsequent data reads see
  // the data that was written before the producer's release store of tail.
  // Uses GPU scope for local pipeline, system scope for remote LSA.
  // Subtract lsaTailBase so we operate in delta space (with base==0 the
  // arithmetic is identical to today's absolute behavior).
  uint64_t current_tail = ld_acquire(tail_ptr, fc.isLocal) - fc.lsaTailBase;

  if (current_tail <= fc.lastTailVal) {
    return 0; // No new chunks
  }

  int num_new = (int)(current_tail - fc.lastTailVal);

  // Compute offset of the first new chunk within this peer's sub-region.
  // Slot index = lastTailVal % peerNumSlots (circular buffer).
  int first_slot = (int)(fc.lastTailVal % fc.peerNumSlots);
  *receive_offset = fc.peerDataOffset + (size_t)first_slot * fc.peerChunkSize;

  fc.lastTailVal = current_tail;
  return num_new;
}

// rdma_poll: reads tail from a gin signal (remote RDMA producer).
//
// Called by: Consumer warps (Types 5, 6) — before processing data from
// a remote RDMA producer.
//
// Returns: number of new chunks available (0 if none).
// On success, *receive_offset is set to the absolute byte offset of the
// first new chunk in the staging buffer.
__device__ __forceinline__ int rdma_poll(ncclGin& gin, StagingRegion& region, StagingFlowCtrl& fc,
                                         size_t* receive_offset // OUT: offset of first new chunk in data region
) {
  // Read tail from gin signal (remote RDMA producer incremented it).
  // Subtract base to get 0-based counter.
  uint64_t current_tail = gin.readSignal(fc.localTailSignal) - fc.tailSignalBase;

  if (current_tail <= fc.lastTailVal) {
    return 0; // No new chunks
  }

  int num_new = (int)(current_tail - fc.lastTailVal);

  // Compute offset of the first new chunk within this peer's sub-region
  int first_slot = (int)(fc.lastTailVal % fc.peerNumSlots);
  *receive_offset = fc.peerDataOffset + (size_t)first_slot * fc.peerChunkSize;

  fc.lastTailVal = current_tail;
  return num_new;
}

// ----------------------------------------------------------------------------
// 2d. release — Consumer notifies producer that slots have been consumed
// ----------------------------------------------------------------------------

// RDMA variant: releases via gin.signal (remote RDMA) or staging buffer
// memory (local pipeline).
//
// Called by: Consumer warps (Types 4, 5, 6) — after consuming chunks from
// the RDMA region.
//
// Release policy: release when all slots since the last release have been
// consumed (consumed_since_release >= peerNumSlots).  For the final
// flush at transfer end, call rdma_release_flush() instead.
__device__ __forceinline__ void rdma_release(ncclGin& gin, ncclTeam world, StagingRegion& region, StagingFlowCtrl& fc) {
  uint64_t consumed_since_release = fc.lastTailVal - fc.localHeadVal;

  if (consumed_since_release < (uint64_t)fc.peerNumSlots) {
    return; // Not yet full, don't release
  }

  if (fc.isLocal) {
    // Local pipeline (same GPU): direct store with GPU-scope release
    char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
    uint64_t* local_head_scratch = (uint64_t*)(staging_base + fc.localHeadOffset);
    st_release_gpu(local_head_scratch, fc.lastTailVal);
  } else {
    // Remote RDMA: standalone gin.signal to increment head on the peer.
    // We increment once per consumed chunk to keep the remote
    // head_signal counter in sync.
    for (uint64_t i = 0; i < consumed_since_release; i++) {
      gin.signal(world, fc.remoteRank, ncclGin_SignalInc{fc.remoteHeadSignal});
    }
  }

  fc.localHeadVal = fc.lastTailVal;
}

// RDMA release flush: unconditional release of all consumed-but-unreleased
// slots.  Call at the END of a transfer to unblock the producer even if the
// buffer isn't completely full.
__device__ __forceinline__ void rdma_release_flush(ncclGin& gin, ncclTeam world, StagingRegion& region,
                                                   StagingFlowCtrl& fc) {
  uint64_t consumed_since_release = fc.lastTailVal - fc.localHeadVal;

  if (consumed_since_release == 0) {
    return; // Nothing to release
  }

  if (fc.isLocal) {
    char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
    uint64_t* local_head_scratch = (uint64_t*)(staging_base + fc.localHeadOffset);
    st_release_gpu(local_head_scratch, fc.lastTailVal);
  } else {
    for (uint64_t i = 0; i < consumed_since_release; i++) {
      gin.signal(world, fc.remoteRank, ncclGin_SignalInc{fc.remoteHeadSignal});
    }
  }

  fc.localHeadVal = fc.lastTailVal;
}

// LSA variant: releases via LSA store (remote LSA) or staging buffer
// memory (local pipeline).
//
// Called by: Consumer warps (Types 3, 7) — after consuming chunks from
// the LSA region.
__device__ __forceinline__ void lsa_release(StagingRegion& region, StagingFlowCtrl& fc) {
  uint64_t consumed_since_release = fc.lastTailVal - fc.localHeadVal;

  if (consumed_since_release < (uint64_t)fc.peerNumSlots) {
    return;
  }

  // Wire write goes out as base + delta.  With lsaHeadBase==0 (the
  // default for all paths except PIPE's LSA T7) this reduces to
  // today's absolute write of fc.lastTailVal.
  uint64_t wire_val = fc.lsaHeadBase + fc.lastTailVal;

  if (fc.isLocal) {
    // Local pipeline (same GPU): direct store with GPU-scope release
    char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
    uint64_t* local_head = (uint64_t*)(staging_base + fc.localHeadOffset);
    st_release_gpu(local_head, wire_val);
  } else {
    // Remote LSA (intra-node, cross-GPU): release store via LSA pointer
    uint64_t* remote_head = (uint64_t*)ncclGetLsaPointer(region.window, fc.remoteHeadOffset, fc.remoteRank);
    st_release_sys(remote_head, wire_val);
  }

  fc.localHeadVal = fc.lastTailVal;
}

// LSA release flush: unconditional release (same as rdma_release_flush but
// for LSA).  Call at the END of a transfer.
__device__ __forceinline__ void lsa_release_flush(StagingRegion& region, StagingFlowCtrl& fc) {
  uint64_t consumed_since_release = fc.lastTailVal - fc.localHeadVal;

  if (consumed_since_release == 0) {
    return;
  }

  // Wire write goes out as base + delta (see lsa_release).
  uint64_t wire_val = fc.lsaHeadBase + fc.lastTailVal;

  if (fc.isLocal) {
    char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
    uint64_t* local_head = (uint64_t*)(staging_base + fc.localHeadOffset);
    st_release_gpu(local_head, wire_val);
  } else {
    uint64_t* remote_head = (uint64_t*)ncclGetLsaPointer(region.window, fc.remoteHeadOffset, fc.remoteRank);
    st_release_sys(remote_head, wire_val);
  }

  fc.localHeadVal = fc.lastTailVal;
}

// ----------------------------------------------------------------------------
// 2e. staging_fc_load_base — Snapshot current counter values for multi-launch
// ----------------------------------------------------------------------------
//
// On the first kernel launch the staging buffer is freshly zeroed, so all
// shadow state (shadowTail, lastTailVal, localHeadVal) correctly starts
// at 0.  On *subsequent* launches the tail/head counters in staging memory
// still hold the final values from the previous launch.  Without adjusting
// shadow state the unsigned arithmetic in rdma_wait_for_credits / staging_poll
// wraps incorrectly and causes hangs.
//
// This primitive reads the current tail and head values from staging buffer
// memory and initialises ALL three shadow fields to match.  It is the
// staging-buffer-memory analogue of tailSignalBase / headSignalBase which
// serve the same purpose for GIN-signal-based flow control.
//
// Call ONCE per fc struct, right after copying it from params, before any
// flow-control call.  Works for any role (producer, consumer, or both):
//   - Producer uses shadowTail        → set to current tail
//   - Consumer uses lastTailVal      → set to current tail
//   - Consumer uses localHeadVal     → set to current head
//   - Unused fields are harmlessly set.
//
// For GIN-signal paths (useGinSignal==true) the base is already captured
// on the host side in tailSignalBase / headSignalBase, so this is a
// no-op.
//
// Parameters:
//   region — staging region whose window contains the counters
//   fc     — flow control struct to initialise (modified in-place)
__device__ __forceinline__ void staging_fc_load_base(StagingRegion& region, StagingFlowCtrl& fc) {
  if (fc.useGinSignal) {
    // GIN-signal path: base values are set by the host in
    // tailSignalBase / headSignalBase.  Shadow state starts
    // at 0 (relative to base).  Nothing to do.
    return;
  }

  // Memory-based path (local pipeline or LSA remote).
  char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);

  uint64_t tail = ld_acquire((const uint64_t*)(staging_base + fc.localTailOffset), fc.isLocal);
  uint64_t head = ld_acquire((const uint64_t*)(staging_base + fc.localHeadOffset), fc.isLocal);

  // Producer shadow state
  fc.shadowTail = tail;

  // Consumer shadow state
  fc.lastTailVal = tail;
  fc.localHeadVal = head;
}

// ============================================================================
// Layer 3: Data Movement Primitives
// ============================================================================
//
// Three copy modes corresponding to different data layout combinations:
//
//   PACK:   User buffer (strided) → Staging slot (contiguous)
//   CONTIG: Staging slot (contiguous) → Staging slot (contiguous)
//   UNPACK: Staging slot (contiguous) → User buffer (strided)
//
// Copies use alignment-aware vectorization: uint4 (128-bit) when both
// pointers are 16-byte aligned, uint32_t (32-bit) when both are 4-byte
// aligned, and byte copy otherwise.  The caller is responsible for
// resolving pointers (local staging, LSA-mapped remote staging, or user
// buffer) before calling these functions.
// ============================================================================

// Alignment-aware memcpy: picks the widest safe vector width at runtime.
__device__ __forceinline__ void staging_memcpy(char* dst, const char* src, size_t nbytes, int warp_group_threads,
                                               int thread_in_group) {
  uintptr_t align = (uintptr_t)src | (uintptr_t)dst;

  if ((align & 0xF) == 0) {
    size_t n16 = nbytes / 16;
    uint4* d = (uint4*)dst;
    const uint4* s = (const uint4*)src;
    for (size_t p = (size_t)thread_in_group; p < n16; p += (size_t)warp_group_threads) {
      d[p] = s[p];
    }
    for (size_t b = n16 * 16 + (size_t)thread_in_group; b < nbytes; b += (size_t)warp_group_threads) {
      dst[b] = src[b];
    }
  } else if ((align & 0x3) == 0) {
    size_t n4 = nbytes / 4;
    uint32_t* d = (uint32_t*)dst;
    const uint32_t* s = (const uint32_t*)src;
    for (size_t p = (size_t)thread_in_group; p < n4; p += (size_t)warp_group_threads) {
      d[p] = s[p];
    }
    for (size_t b = n4 * 4 + (size_t)thread_in_group; b < nbytes; b += (size_t)warp_group_threads) {
      dst[b] = src[b];
    }
  } else {
    for (size_t b = (size_t)thread_in_group; b < nbytes; b += (size_t)warp_group_threads) {
      dst[b] = src[b];
    }
  }
}

// Copy mode enum
enum StagingCopyMode {
  STAGING_COPY_PACK, // Case A: user buffer (strided) → staging (contiguous)
  STAGING_COPY_CONTIG, // Case B: staging → staging (contiguous both sides)
  STAGING_COPY_UNPACK // Case C: staging (contiguous) → user buffer (strided)
};

// ----------------------------------------------------------------------------
// 3a. staging_copy_contig — Contiguous copy (staging ↔ staging)
// ----------------------------------------------------------------------------
//
// Straight vectorized memcpy.  Used for staging-to-staging transfers
// (Types 3, 4, 5, 8 fan-out) and also for contiguous (same-dim sharding)
// pack/unpack where the data is one contiguous block.
__device__ __forceinline__ void staging_copy_contig(char* dst, const char* src, size_t num_bytes,
                                                    int warp_group_threads, int thread_in_group) {
  staging_memcpy(dst, src, num_bytes, warp_group_threads, thread_in_group);
}

// ----------------------------------------------------------------------------
// 3b. staging_copy_pack — Pack strided user buffer → contiguous staging
// ----------------------------------------------------------------------------
//
// Gathers strided data from the user's source buffer and packs it
// contiguously into a staging slot.  Uses the transfer plan's stride
// information to compute source offsets for each inner transfer iteration.
//
// Byte-range API: accepts (byte_start, num_bytes) into the flat packed
// representation.  Handles partial inner transfers at chunk boundaries,
// so chunkSize is not required to be >= innerSize.
//
// Parameters:
//   dst_staging      — contiguous staging slot pointer
//   src_user_buffer  — user's source buffer base pointer
//   plan             — transfer plan (strides, offsets, innerSize)
//   byte_start       — byte offset into the flat packed stream
//   num_bytes        — bytes to copy in this chunk
//   warp_group_threads, thread_in_group — thread indexing within the warp group
//
// IMPORTANT: Do NOT call this for contiguous transfers (plan.isContiguous
// == true).  Use staging_copy_contig with byte offsets instead.
__device__ __forceinline__ void staging_copy_pack(char* dst_staging, const char* src_user_buffer,
                                                  const StagingTransferPlan& plan, size_t byte_start, size_t num_bytes,
                                                  int warp_group_threads, int thread_in_group) {
  const size_t inner = plan.innerSize;
  size_t first_iter = byte_start / inner;
  size_t first_offset = byte_start % inner;

  size_t dst_write_offset = 0;
  size_t bytes_remaining = num_bytes;
  size_t iter = first_iter;

  while (bytes_remaining > 0) {
    // Compute source offset using the transfer plan's stride pattern.
    // Decompose the flat iteration index into per-dimension indices
    // and accumulate source strides.
    size_t src_offset = plan.srcBaseOffset;
    size_t tmp = iter;
    for (int d = plan.numOuterLoops - 1; d >= 0; d--) {
      size_t idx = tmp % plan.outerCounts[d];
      tmp /= plan.outerCounts[d];
      src_offset += idx * plan.outerSrcStrides[d];
    }

    // For the first iteration we may start mid-way through an inner
    // transfer; for subsequent iterations iter_start is always 0.
    size_t iter_start = (iter == first_iter) ? first_offset : 0;
    size_t avail = inner - iter_start;
    size_t iter_bytes = (avail < bytes_remaining) ? avail : bytes_remaining;

    const char* src_ptr = src_user_buffer + src_offset + iter_start;
    char* dst_ptr = dst_staging + dst_write_offset;

    staging_memcpy(dst_ptr, src_ptr, iter_bytes, warp_group_threads, thread_in_group);

    dst_write_offset += iter_bytes;
    bytes_remaining -= iter_bytes;
    iter++;
  }
}

// ----------------------------------------------------------------------------
// 3c. staging_copy_unpack — Unpack contiguous staging → strided user buffer
// ----------------------------------------------------------------------------
//
// Scatters contiguous data from a staging slot into the user's destination
// buffer, respecting the destination stride pattern from the transfer plan.
//
// Byte-range API: accepts (byte_start, num_bytes) into the flat packed
// representation.  Handles partial inner transfers at chunk boundaries.
// Parameters mirror staging_copy_pack but in reverse direction.
__device__ __forceinline__ void staging_copy_unpack(char* dst_user_buffer, const char* src_staging,
                                                    const StagingTransferPlan& plan, size_t byte_start,
                                                    size_t num_bytes, int warp_group_threads, int thread_in_group) {
  const size_t inner = plan.innerSize;
  size_t first_iter = byte_start / inner;
  size_t first_offset = byte_start % inner;

  size_t src_read_offset = 0;
  size_t bytes_remaining = num_bytes;
  size_t iter = first_iter;

  while (bytes_remaining > 0) {
    // Compute destination offset using the transfer plan's stride pattern
    size_t dst_offset = plan.dstBaseOffset;
    size_t tmp = iter;
    for (int d = plan.numOuterLoops - 1; d >= 0; d--) {
      size_t idx = tmp % plan.outerCounts[d];
      tmp /= plan.outerCounts[d];
      dst_offset += idx * plan.outerDstStrides[d];
    }

    // For the first iteration we may start mid-way through an inner
    // transfer; for subsequent iterations iter_start is always 0.
    size_t iter_start = (iter == first_iter) ? first_offset : 0;
    size_t avail = inner - iter_start;
    size_t iter_bytes = (avail < bytes_remaining) ? avail : bytes_remaining;

    const char* src_ptr = src_staging + src_read_offset;
    char* dst_ptr = dst_user_buffer + dst_offset + iter_start;

    staging_memcpy(dst_ptr, src_ptr, iter_bytes, warp_group_threads, thread_in_group);

    src_read_offset += iter_bytes;
    bytes_remaining -= iter_bytes;
    iter++;
  }
}

// ----------------------------------------------------------------------------
// 3b2. staging_copy_pack_parallel — Pack strided → contiguous, iterations
//      distributed across threads (optimized for small innerSize)
// ----------------------------------------------------------------------------
//
// When innerSize is small (e.g. 512 bytes), the standard staging_copy_pack
// wastes threads: 256 threads each copy 2 bytes of a single 512-byte inner
// transfer, then loop to the next.  This variant instead assigns different
// inner transfer iterations to different threads, so each thread independently
// copies innerSize bytes from its assigned iteration.
//
// Byte-range API: accepts (byte_start, num_bytes) into the flat packed
// representation, same as staging_copy_pack.
__device__ __forceinline__ void staging_copy_pack_parallel(char* dst_staging, const char* src_user_buffer,
                                                           const StagingTransferPlan& plan, size_t byte_start,
                                                           size_t num_bytes, int warp_group_threads,
                                                           int thread_in_group) {
  const size_t inner = plan.innerSize;
  if (inner == 0) {
    return;
  }

  // Map byte range to iteration range
  size_t first_iter = byte_start / inner;
  size_t last_byte = byte_start + num_bytes;
  size_t last_iter = (last_byte + inner - 1) / inner; // exclusive
  size_t num_iters = last_iter - first_iter;

  for (size_t i = (size_t)thread_in_group; i < num_iters; i += (size_t)warp_group_threads) {
    size_t iter = first_iter + i;

    // Compute source offset using the transfer plan's stride pattern
    size_t src_offset = plan.srcBaseOffset;
    size_t tmp = iter;
    for (int d = plan.numOuterLoops - 1; d >= 0; d--) {
      size_t idx = tmp % plan.outerCounts[d];
      tmp /= plan.outerCounts[d];
      src_offset += idx * plan.outerSrcStrides[d];
    }

    // Handle partial first/last iterations at chunk boundaries
    size_t iter_byte_start = iter * inner;
    size_t iter_byte_end = iter_byte_start + inner;
    size_t copy_start = (iter_byte_start >= byte_start) ? 0 : (byte_start - iter_byte_start);
    size_t copy_end = (iter_byte_end <= last_byte) ? inner : (last_byte - iter_byte_start);
    size_t copy_len = copy_end - copy_start;

    const char* src_ptr = src_user_buffer + src_offset + copy_start;
    size_t dst_write_pos = iter_byte_start + copy_start - byte_start;
    char* dst_ptr = dst_staging + dst_write_pos;

    // Single-thread sequential copy — for small innerSize (e.g. 512B)
    // this is more efficient than launching staging_memcpy with 1 thread.
    // Use widest available vector width based on alignment.
    uintptr_t align = (uintptr_t)src_ptr | (uintptr_t)dst_ptr;
    if ((align & 0xF) == 0 && copy_len >= 16) {
      size_t n16 = copy_len / 16;
      uint4* d = (uint4*)dst_ptr;
      const uint4* s = (const uint4*)src_ptr;
      for (size_t p = 0; p < n16; p++) {
        d[p] = s[p];
      }
      for (size_t b = n16 * 16; b < copy_len; b++) {
        dst_ptr[b] = src_ptr[b];
      }
    } else if ((align & 0x3) == 0 && copy_len >= 4) {
      size_t n4 = copy_len / 4;
      uint32_t* d = (uint32_t*)dst_ptr;
      const uint32_t* s = (const uint32_t*)src_ptr;
      for (size_t p = 0; p < n4; p++) {
        d[p] = s[p];
      }
      for (size_t b = n4 * 4; b < copy_len; b++) {
        dst_ptr[b] = src_ptr[b];
      }
    } else {
      for (size_t b = 0; b < copy_len; b++) {
        dst_ptr[b] = src_ptr[b];
      }
    }
  }
}

// ----------------------------------------------------------------------------
// 3c2. staging_copy_unpack_parallel — Unpack contiguous → strided, iterations
//      distributed across threads (optimized for small innerSize)
// ----------------------------------------------------------------------------
//
// Mirror of staging_copy_pack_parallel for the receive side.
__device__ __forceinline__ void staging_copy_unpack_parallel(char* dst_user_buffer, const char* src_staging,
                                                             const StagingTransferPlan& plan, size_t byte_start,
                                                             size_t num_bytes, int warp_group_threads,
                                                             int thread_in_group) {
  const size_t inner = plan.innerSize;
  if (inner == 0) {
    return;
  }

  size_t first_iter = byte_start / inner;
  size_t last_byte = byte_start + num_bytes;
  size_t last_iter = (last_byte + inner - 1) / inner;
  size_t num_iters = last_iter - first_iter;

  for (size_t i = (size_t)thread_in_group; i < num_iters; i += (size_t)warp_group_threads) {
    size_t iter = first_iter + i;

    // Compute destination offset using the transfer plan's stride pattern
    size_t dst_offset = plan.dstBaseOffset;
    size_t tmp = iter;
    for (int d = plan.numOuterLoops - 1; d >= 0; d--) {
      size_t idx = tmp % plan.outerCounts[d];
      tmp /= plan.outerCounts[d];
      dst_offset += idx * plan.outerDstStrides[d];
    }

    size_t iter_byte_start = iter * inner;
    size_t iter_byte_end = iter_byte_start + inner;
    size_t copy_start = (iter_byte_start >= byte_start) ? 0 : (byte_start - iter_byte_start);
    size_t copy_end = (iter_byte_end <= last_byte) ? inner : (last_byte - iter_byte_start);
    size_t copy_len = copy_end - copy_start;

    size_t src_read_pos = iter_byte_start + copy_start - byte_start;
    const char* src_ptr = src_staging + src_read_pos;
    char* dst_ptr = dst_user_buffer + dst_offset + copy_start;

    uintptr_t align = (uintptr_t)src_ptr | (uintptr_t)dst_ptr;
    if ((align & 0xF) == 0 && copy_len >= 16) {
      size_t n16 = copy_len / 16;
      uint4* d = (uint4*)dst_ptr;
      const uint4* s = (const uint4*)src_ptr;
      for (size_t p = 0; p < n16; p++) {
        d[p] = s[p];
      }
      for (size_t b = n16 * 16; b < copy_len; b++) {
        dst_ptr[b] = src_ptr[b];
      }
    } else if ((align & 0x3) == 0 && copy_len >= 4) {
      size_t n4 = copy_len / 4;
      uint32_t* d = (uint32_t*)dst_ptr;
      const uint32_t* s = (const uint32_t*)src_ptr;
      for (size_t p = 0; p < n4; p++) {
        d[p] = s[p];
      }
      for (size_t b = n4 * 4; b < copy_len; b++) {
        dst_ptr[b] = src_ptr[b];
      }
    } else {
      for (size_t b = 0; b < copy_len; b++) {
        dst_ptr[b] = src_ptr[b];
      }
    }
  }
}

// Threshold: when innerSize < this many bytes per thread, use parallel variant
#define STAGING_PARALLEL_INNER_THRESHOLD 128

// ----------------------------------------------------------------------------
// 3d. staging_copy — Unified dispatcher
// ----------------------------------------------------------------------------
//
// Routes to the appropriate copy function based on the mode.
// For STAGING_COPY_CONTIG, the plan/byte-range parameters are unused;
// contig_bytes is used instead.
// For PACK/UNPACK, byte_start and num_bytes specify the byte range within
// the flat packed representation (supports partial inner transfers).
__device__ __forceinline__ void staging_copy(StagingCopyMode mode, char* dst, const char* src,
                                             const StagingTransferPlan& plan, size_t byte_start, size_t num_bytes,
                                             size_t contig_bytes, // used only for STAGING_COPY_CONTIG
                                             int warp_group_threads, int thread_in_group) {
  switch (mode) {
  case STAGING_COPY_PACK:
    staging_copy_pack(dst, src, plan, byte_start, num_bytes, warp_group_threads, thread_in_group);
    break;
  case STAGING_COPY_CONTIG:
    staging_copy_contig(dst, src, contig_bytes, warp_group_threads, thread_in_group);
    break;
  case STAGING_COPY_UNPACK:
    staging_copy_unpack(dst, src, plan, byte_start, num_bytes, warp_group_threads, thread_in_group);
    break;
  }
}

// ============================================================================
// Layer 4: TMA (Tensor Memory Accelerator) Bulk Copy Primitives
// ============================================================================
//
// Async bulk copy via DMA engine — does not occupy warps during transfer.
// Requires sm_90+ (Hopper/Blackwell).
//
// Pattern:
//   1. staging_tma_mbarrier_init(mbar, 1)     — once per mbarrier
//   2. staging_tma_mbarrier_expect(mbar, N)    — register expected bytes
//   3. staging_tma_load(smem, gmem, mbar, N)  — async global→shared
//   4. staging_tma_mbarrier_wait(mbar, phase)  — block until load done
//   5. staging_tma_store(smem, gmem, N)        — async shared→global
//   6. staging_tma_store_wait<0>()             — block until store done
//
// These are standalone primitives — they do NOT modify existing staging_copy_*
// functions. Use them as drop-in replacements for staging_copy_contig when
// the copy is between global memory and shared memory.
// ============================================================================

#define STAGING_TMA_TILE_SIZE (32 * 1024) // 32KB default tile for TMA

// Cache hint constants for L2
__device__ static constexpr uint64_t kTmaEvictFirst = 0x12f0000000000000ULL;
__device__ static constexpr uint64_t kTmaEvictNormal = 0x1000000000000000ULL;

/* All TMA primitives below assemble PTX (mbarrier.*, cp.async.bulk.*,
 * fence.mbarrier_init) that ptxas only accepts when targeting sm_90 or
 * higher.  We still want sm_80 in NVCC_GENCODE so the default + direct
 * staging kernels work on Ampere — so the bodies become no-ops on older
 * archs (the Pipe launcher refuses to dispatch on those archs). */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define STAGING_TMA_AVAILABLE 1
#else
#define STAGING_TMA_AVAILABLE 0
#endif

// Initialize an mbarrier in shared memory.
// Must be called by exactly one thread per mbarrier.
__device__ __forceinline__ void staging_tma_mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
#if STAGING_TMA_AVAILABLE
  uint32_t mbar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count), "r"(mbar_smem));
#else
  (void)mbar_ptr;
  (void)arrive_count;
#endif
}

// Fence after mbarrier init (required before first use).
__device__ __forceinline__ void staging_tma_fence_init() {
#if STAGING_TMA_AVAILABLE
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

// Async bulk load: global memory → shared memory.
// Issues cp.async.bulk and links completion to the mbarrier.
__device__ __forceinline__ void staging_tma_load(void* smem_dst, const void* gmem_src, uint64_t* mbar_ptr,
                                                 int num_bytes) {
#if STAGING_TMA_AVAILABLE
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
               "[%0], [%1], %2, [%3], %4;\n" ::"r"(smem_addr),
               "l"(gmem_src), "r"(num_bytes), "r"(mbar_addr), "l"(kTmaEvictFirst)
               : "memory");
#else
  (void)smem_dst;
  (void)gmem_src;
  (void)mbar_ptr;
  (void)num_bytes;
#endif
}

// Register expected transaction bytes on the mbarrier.
// Call before staging_tma_load to register its expected transaction bytes.
__device__ __forceinline__ void staging_tma_mbarrier_expect(uint64_t* mbar_ptr, int num_bytes) {
#if STAGING_TMA_AVAILABLE
  uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;\n" ::"r"(num_bytes), "r"(mbar_addr));
#else
  (void)mbar_ptr;
  (void)num_bytes;
#endif
}

// Wait for mbarrier completion (parity-based).
// Blocks until all expected bytes have arrived. Toggles phase.
// Uses %= for unique asm labels to avoid collisions in loops.
__device__ __forceinline__ void staging_tma_mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase) {
#if STAGING_TMA_AVAILABLE
  uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "WAIT_LOOP_%=:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
               "@P1 bra WAIT_DONE_%=;\n\t"
               "bra WAIT_LOOP_%=;\n\t"
               "WAIT_DONE_%=:\n\t"
               "}\n" ::"r"(mbar_addr),
               "r"(phase), "r"(0x989680));
  phase ^= 1;
#else
  (void)mbar_ptr;
  (void)phase;
#endif
}

#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
__device__ __forceinline__ bool staging_tma_mbarrier_try_wait(uint64_t* mbar_ptr, uint32_t phase) {
#if STAGING_TMA_AVAILABLE
  uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
  uint32_t ready;
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2, %3;\n\t"
               "selp.u32 %0, 1, 0, P1;\n\t"
               "}\n"
               : "=r"(ready)
               : "r"(mbar_addr), "r"(phase), "r"(0x989680)
               : "memory");
  return ready != 0;
#else
  (void)mbar_ptr;
  (void)phase;
  return true;
#endif
}
#endif

// Async bulk store: shared memory → global memory.
// Issues cp.async.bulk and commits the transaction group.
__device__ __forceinline__ void staging_tma_store(const void* smem_src, void* gmem_dst, int num_bytes) {
#if STAGING_TMA_AVAILABLE
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_src));
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
               "[%0], [%1], %2, %3;\n" ::"l"(gmem_dst),
               "r"(smem_addr), "r"(num_bytes), "l"(kTmaEvictFirst)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
#else
  (void)smem_src;
  (void)gmem_dst;
  (void)num_bytes;
#endif
}

// Wait for outstanding TMA store transactions to complete.
// N = number of transactions that may still be pending (0 = wait for all).
template <int N>
__device__ __forceinline__ void staging_tma_store_wait() {
#if STAGING_TMA_AVAILABLE
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
#endif
}

// Async bulk store WITHOUT commit — for batching multiple stores into one group.
// Call staging_tma_store_commit() after issuing all stores, then
// staging_tma_store_wait<0>() to drain.
__device__ __forceinline__ void staging_tma_store_nocommit(const void* smem_src, void* gmem_dst, int num_bytes) {
#if STAGING_TMA_AVAILABLE
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_src));
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
               "[%0], [%1], %2, %3;\n" ::"l"(gmem_dst),
               "r"(smem_addr), "r"(num_bytes), "l"(kTmaEvictFirst)
               : "memory");
#else
  (void)smem_src;
  (void)gmem_dst;
  (void)num_bytes;
#endif
}

// Commit all pending (uncommitted) bulk stores into one tracking group.
__device__ __forceinline__ void staging_tma_store_commit() {
#if STAGING_TMA_AVAILABLE
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
#endif
}

// ============================================================================
// Layer 5: TMA-based strided unpack  (smem → scattered global)
// ============================================================================
//
// Replaces staging_copy_unpack for the TMA path: walks inner transfers in
// the transfer plan and issues one cp.async.bulk per inner transfer, then
// commits + waits once.  Called by a single thread (root warp lane 0).
//
// Use when innerSize >= STAGING_TMA_STORE_THRESHOLD; for smaller inner
// sizes the warp-level staging_copy_unpack_parallel is more efficient.

#define STAGING_TMA_STORE_THRESHOLD 2048

__device__ __forceinline__ void staging_tma_unpack(char* dst_user_buffer, const char* smem_staging,
                                                   const StagingTransferPlan& plan, size_t byte_start,
                                                   size_t num_bytes) {
  const size_t inner = plan.innerSize;
  size_t first_iter = byte_start / inner;
  size_t first_offset = byte_start % inner;

  size_t src_read_offset = 0;
  size_t bytes_remaining = num_bytes;
  size_t iter = first_iter;

  while (bytes_remaining > 0) {
    size_t dst_offset = plan.dstBaseOffset;
    size_t tmp = iter;
    for (int d = plan.numOuterLoops - 1; d >= 0; d--) {
      size_t idx = tmp % plan.outerCounts[d];
      tmp /= plan.outerCounts[d];
      dst_offset += idx * plan.outerDstStrides[d];
    }

    size_t iter_start = (iter == first_iter) ? first_offset : 0;
    size_t avail = inner - iter_start;
    size_t iter_bytes = (avail < bytes_remaining) ? avail : bytes_remaining;

    staging_tma_store_nocommit(smem_staging + src_read_offset, dst_user_buffer + dst_offset + iter_start,
                               (int)iter_bytes);

    src_read_offset += iter_bytes;
    bytes_remaining -= iter_bytes;
    iter++;
  }

  staging_tma_store_commit();
  staging_tma_store_wait<0>();
}

#endif // NCCL_STAGING_PRIMITIVES_CUH
