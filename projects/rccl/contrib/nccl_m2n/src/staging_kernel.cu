/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Staging Reshard kernels.
 *
 * This file contains the DIRECT staging kernel and the role-specialized
 * PIPE persistent-counter kernels.  PIPE source ranks launch a trainer
 * kernel, destination ranks launch a generator kernel, and mixed generator
 * ranks can split RDMA-receive/fanout and LSA-pull work across CTA phases
 * inside the same generator launch.
 ************************************************************************/

#include "nccl.h"
#include "nccl_device.h"
#include "cuda_runtime.h"

#include "m2n_checks.h"
#include "reshard_internal.h"
#include "staging_types.h"
#include "staging_primitives.cuh"

#include <cstdio>
#include <mutex>
#include <utility>

// ============================================================================
// Error-check macros (host-side only)
// ============================================================================

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define NCCLM2N_PIPE_HAS_LAUNCH_COMPLETION_EVENT 1
#else
#define NCCLM2N_PIPE_HAS_LAUNCH_COMPLETION_EVENT 0
#endif

#if NCCLM2N_PIPE_HAS_LAUNCH_COMPLETION_EVENT
namespace {

/* Persistent PIPE grids can occupy enough CTA admission capacity to keep a
 * peer grid only partially launched. Keep one reusable launch-completion event
 * per CUDA device, rather than per communicator or stream. A stream wait
 * captures the previous record before the next launch re-records the event.
 * This orders launch admission, not kernel completion, and kernels may still
 * execute concurrently. */
constexpr int kPipeLaunchOrderMaxDevices = 64;
constexpr int kPipeLaunchCompletionMinDriverVersion = 12030;

struct PipeLaunchOrderState {
  int cudaDev = -1;
  cudaEvent_t launchComplete = nullptr;
};

std::mutex gPipeLaunchOrderMutex;
PipeLaunchOrderState gPipeLaunchOrder[kPipeLaunchOrderMaxDevices];
int gPipeLaunchOrderCount = 0;

bool pipeLaunchCompletionEventSupported() {
  static std::once_flag once;
  static bool supported = false;
  std::call_once(once, [] {
    int driverVersion = 0;
    supported =
      cudaDriverGetVersion(&driverVersion) == cudaSuccess && driverVersion >= kPipeLaunchCompletionMinDriverVersion;
  });
  return supported;
}

bool pipeLaunchOrderingEnabled() {
  return reshardUsePipeLaunchOrder() && pipeLaunchCompletionEventSupported();
}

PipeLaunchOrderState* getPipeLaunchOrderState(int cudaDev) {
  for (int i = 0; i < gPipeLaunchOrderCount; i++) {
    if (gPipeLaunchOrder[i].cudaDev == cudaDev) {
      return &gPipeLaunchOrder[i];
    }
  }
  if (gPipeLaunchOrderCount >= kPipeLaunchOrderMaxDevices) {
    return nullptr;
  }
  PipeLaunchOrderState* state = &gPipeLaunchOrder[gPipeLaunchOrderCount++];
  state->cudaDev = cudaDev;
  state->launchComplete = nullptr;
  return state;
}

template <typename KernelFunc, typename... Args>
static ncclResult_t launchPipeKernelOrdered(dim3 grid, dim3 block, size_t dynamicSmemBytes, cudaStream_t stream,
                                            KernelFunc kernel, Args&&... args) {
  int cudaDev = -1;
  NCCL_M2N_CUDACHECK(cudaGetDevice(&cudaDev));

  std::lock_guard<std::mutex> lock(gPipeLaunchOrderMutex);
  PipeLaunchOrderState* state = getPipeLaunchOrderState(cudaDev);
  NCCL_M2N_CHECK_ARG(state != nullptr, -1, "PIPE launch-order state exhausted (%d CUDA devices)",
                     kPipeLaunchOrderMaxDevices);

  if (state->launchComplete == nullptr) {
    NCCL_M2N_CUDACHECK(cudaEventCreateWithFlags(&state->launchComplete, cudaEventDisableTiming));
  } else {
    NCCL_M2N_CUDACHECK(cudaStreamWaitEvent(stream, state->launchComplete, 0));
  }

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeLaunchCompletionEvent;
  attr.val.launchCompletionEvent.event = state->launchComplete;
  attr.val.launchCompletionEvent.flags = cudaEventRecordDefault;

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = dynamicSmemBytes;
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;

  cudaError_t launchResult = cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...);
  if (launchResult != cudaSuccess) {
    NCCL_M2N_FAIL(ncclInternalError, -1, "CUDA operation cudaLaunchKernelEx failed: %s",
                  cudaGetErrorString(launchResult));
  }
  return ncclSuccess;
}

} // namespace
#endif

void stagingPipeLaunchCompletionFinalize() {
#if NCCLM2N_PIPE_HAS_LAUNCH_COMPLETION_EVENT
  std::lock_guard<std::mutex> lock(gPipeLaunchOrderMutex);
  int savedCudaDev = -1;
  cudaError_t getDeviceResult = cudaGetDevice(&savedCudaDev);
  for (int i = 0; i < gPipeLaunchOrderCount; i++) {
    if (gPipeLaunchOrder[i].launchComplete != nullptr) {
      if (getDeviceResult == cudaSuccess && gPipeLaunchOrder[i].cudaDev >= 0) {
        NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(gPipeLaunchOrder[i].cudaDev));
      }
      NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(gPipeLaunchOrder[i].launchComplete));
      gPipeLaunchOrder[i].launchComplete = nullptr;
    }
    gPipeLaunchOrder[i].cudaDev = -1;
  }
  if (getDeviceResult == cudaSuccess && savedCudaDev >= 0) {
    NCCL_M2N_CUDACHECK_WARN(cudaSetDevice(savedCudaDev));
  }
  gPipeLaunchOrderCount = 0;
#endif
}

// Named barrier for a subset of threads within a CTA.
// __barrier_sync(id) syncs ALL CTA threads; to sync only `count` threads
// we need the PTX "bar.sync id, count" instruction directly.
__device__ __forceinline__ void barrier_sync_subset(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// ============================================================================
// Direct-Algorithm Kernel - StagingReshardKernel_Direct
// ============================================================================
//
// Stripped-down kernel for the direct algorithm (pure RDMA, no LSA, no ring).
// Eliminates: LSA fan-out (Type 6 fwd), ring forwarding, Type 7 (LSA recv),
// and all associated state, reducing register pressure and
// stack frame size compared to the PIPE kernel.
//
// Warp layout:
//   Source side: Type 1 (pack) + Type 4 (RDMA send)
//   Dest side:   Type 6 (RDMA recv + unpack), pure copy groups only
//
// ============================================================================

#define DIRECT_TYPE1_NUM_WARPS 8
#define DIRECT_TYPE1_THREADS (DIRECT_TYPE1_NUM_WARPS * 32)
#define DIRECT_TYPE1_BARRIER_ID 1

#define DIRECT_TYPE4_NUM_WARPS 4
#define DIRECT_TYPE4_WARP_START DIRECT_TYPE1_NUM_WARPS
#define DIRECT_SOURCE_TOTAL_WARPS (DIRECT_TYPE1_NUM_WARPS + DIRECT_TYPE4_NUM_WARPS)

#define DIRECT_TYPE6_NUM_GROUPS 4
#define DIRECT_TYPE6_WARPS_PER_GROUP 2
#define DIRECT_TYPE6_TOTAL_WARPS (DIRECT_TYPE6_NUM_GROUPS * DIRECT_TYPE6_WARPS_PER_GROUP)
#define DIRECT_TYPE6_THREADS_PER_GROUP (DIRECT_TYPE6_WARPS_PER_GROUP * 32)
#define DIRECT_TYPE6_BARRIER_ID_BASE 2

#define DIRECT_TOTAL_WARPS \
  (DIRECT_SOURCE_TOTAL_WARPS > DIRECT_TYPE6_TOTAL_WARPS ? DIRECT_SOURCE_TOTAL_WARPS : DIRECT_TYPE6_TOTAL_WARPS)

/* __launch_bounds__ caps register usage to keep the per-CTA register
 * budget below the 65536 SM limit. */
__global__ __launch_bounds__(DIRECT_TOTAL_WARPS * 32, 1) void StagingReshardKernel_Direct(
  StagingKernelParams* __restrict__ params, ncclDevComm devComm) {
  const int channel_id = (int)blockIdx.x;

  int ginContext = reshardMapCtaToGinContext((int)blockIdx.x, (int)gridDim.x, (int)devComm.ginContextCount);
  ncclGin gin{devComm, ginContext};
  ncclTeam world = ncclTeamWorld(devComm);

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  const bool isSource = params->isSource;
  const bool isDest = params->isDest;
  const int numRdmaTargets = params->numRdmaTargets;
  const int numRdmaSources = params->numRdmaSources;
  const size_t chunkSize = params->chunkSize;
  const int numChannels = (int)gridDim.x;

  __shared__ int type1_slot;
  __shared__ int type4_num_new[DIRECT_TYPE4_NUM_WARPS];
  __shared__ size_t type4_first_recv_offset[DIRECT_TYPE4_NUM_WARPS];
  __shared__ int type6_num_new[DIRECT_TYPE6_NUM_GROUPS];
  __shared__ size_t type6_first_recv_offset[DIRECT_TYPE6_NUM_GROUPS];

  __shared__ uint64_t rdma_head_bases[MAX_TARGETS];
  __shared__ uint64_t rdma_tail_bases[MAX_SOURCES];
  __shared__ uint64_t local_pipeline_tail_base[MAX_TARGETS];
  __shared__ uint64_t local_pipeline_head_base[MAX_TARGETS];
  __shared__ uint64_t local_put_counter_base[MAX_TARGETS];

  // ========================================================================
  // PROLOGUE: Read GIN signal base values + local pipeline base values
  // ========================================================================
  if (numRdmaTargets > 0 && (int)threadIdx.x < numRdmaTargets) {
    StagingFlowCtrl& fc = params->rdmaTargets[channel_id][threadIdx.x].fc;
    if (fc.useGinSignal) {
      rdma_head_bases[threadIdx.x] = gin.readSignal(fc.localHeadSignal);
    } else {
      rdma_head_bases[threadIdx.x] = 0;
    }
  }
  if (isSource && (int)threadIdx.x < numRdmaTargets) {
    StagingFlowCtrl& local_fc = params->localRdmaFc[channel_id][threadIdx.x];
    StagingRegion& region = params->rdmaRegions[channel_id];
    char* staging_base = (char*)ncclGetLocalPointer(region.window, 0);
    local_pipeline_tail_base[threadIdx.x] =
      ld_acquire((const uint64_t*)(staging_base + local_fc.localTailOffset), local_fc.isLocal);
    local_pipeline_head_base[threadIdx.x] =
      ld_acquire((const uint64_t*)(staging_base + local_fc.localHeadOffset), local_fc.isLocal);
    local_put_counter_base[threadIdx.x] = gin.readCounter(local_fc.localPutCounter);
  }
  if (isDest && (int)threadIdx.x < numRdmaSources) {
    StagingFlowCtrl& fc = params->rdmaSources[channel_id][threadIdx.x].fc;
    if (fc.useGinSignal) {
      rdma_tail_bases[threadIdx.x] = gin.readSignal(fc.localTailSignal);
    } else {
      rdma_tail_bases[threadIdx.x] = 0;
    }
  }
  __syncthreads();

  // ========================================================================
  // Initial barrier (all ranks, all CTAs)
  // ========================================================================
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  ncclGinBarrierSession<ncclCoopCta> bar{ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x};
#else
  ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
#endif
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  // ========================================================================
  // SOURCE SIDE — Type 1 (multi-warp pack) + Type 4 (RDMA send)
  // ========================================================================
  if (isSource && numRdmaTargets > 0) {
    StagingRegion rdma_region = params->rdmaRegions[channel_id];

    // ================================================================
    // Type 1: Multi-Warp Source Pack
    // ================================================================
    if (warp_id < DIRECT_TYPE1_NUM_WARPS) {
      const bool is_root_warp = (warp_id == 0);
      const int thread_in_group = warp_id * 32 + lane_id;

      for (int t = 0; t < numRdmaTargets; t++) {
        StagingFlowCtrl fc = params->localRdmaFc[channel_id][t];
        if (is_root_warp) {
          fc.shadowTail = local_pipeline_tail_base[t];
          fc.lastTailVal = local_pipeline_tail_base[t];
          fc.localHeadVal = local_pipeline_head_base[t];
        }
        const StagingTransferPlan plan = params->rdmaTargets[channel_id][t].plan;
        size_t total_chunks = (plan.totalBytes + chunkSize - 1) / chunkSize;
        size_t my_chunk_start = (total_chunks * (size_t)channel_id) / (size_t)numChannels;
        size_t my_chunk_end = (total_chunks * (size_t)(channel_id + 1)) / (size_t)numChannels;
        size_t my_num_chunks = my_chunk_end - my_chunk_start;

        for (size_t chunk = 0; chunk < my_num_chunks; chunk++) {
          size_t global_chunk = my_chunk_start + chunk;

          if (is_root_warp && lane_id == 0) {
            uint64_t base_offset = local_pipeline_tail_base[t] - local_put_counter_base[t];
            lsa_rdma_wait_for_credits(gin, fc, base_offset);
            type1_slot = (int)(fc.shadowTail % (uint64_t)fc.peerNumSlots);
          }

          barrier_sync_subset(DIRECT_TYPE1_BARRIER_ID, DIRECT_TYPE1_THREADS);

          int slot = type1_slot;
          char* staging_base = (char*)ncclGetLocalPointer(rdma_region.window, 0);
          char* staging_dst = staging_base + fc.peerDataOffset + (size_t)slot * fc.peerChunkSize;
          const char* user_src = (const char*)params->srcBuffer;

          {
            size_t byte_start = global_chunk * chunkSize;
            size_t remaining = plan.totalBytes - byte_start;
            size_t this_bytes = (chunkSize < remaining) ? chunkSize : remaining;

            if (plan.isContiguous) {
              staging_copy_contig(staging_dst, user_src + plan.srcBaseOffset + byte_start, this_bytes,
                                  DIRECT_TYPE1_THREADS, thread_in_group);
            } else {
              staging_copy_pack(staging_dst, user_src, plan, byte_start, this_bytes, DIRECT_TYPE1_THREADS,
                                thread_in_group);
            }
          }

          barrier_sync_subset(DIRECT_TYPE1_BARRIER_ID, DIRECT_TYPE1_THREADS);

          if (is_root_warp && lane_id == 0) {
            rdma_signal(gin, world, rdma_region, fc);
          }
        }
      }
    }

    // ================================================================
    // Type 4: Multi-Warp RDMA Send
    // ================================================================
    else if (warp_id >= DIRECT_TYPE4_WARP_START && warp_id < DIRECT_TYPE4_WARP_START + DIRECT_TYPE4_NUM_WARPS) {
      const int type4_idx = warp_id - DIRECT_TYPE4_WARP_START;

      const int my_tgt_start = (numRdmaTargets * type4_idx) / DIRECT_TYPE4_NUM_WARPS;
      const int my_tgt_end = (numRdmaTargets * (type4_idx + 1)) / DIRECT_TYPE4_NUM_WARPS;

      for (int t = my_tgt_start; t < my_tgt_end; t++) {
        StagingFlowCtrl local_fc = params->localRdmaFc[channel_id][t];
        local_fc.shadowTail = local_pipeline_tail_base[t];
        local_fc.lastTailVal = local_pipeline_tail_base[t];
        local_fc.localHeadVal = local_pipeline_head_base[t];
        StagingFlowCtrl rdma_fc = params->rdmaTargets[channel_id][t].fc;
        rdma_fc.headSignalBase = rdma_head_bases[t];

        const StagingTransferPlan& plan = params->rdmaTargets[channel_id][t].plan;
        const int target_rank = params->rdmaTargets[channel_id][t].peerWorldRank;

        size_t total_chunks = (plan.totalBytes + chunkSize - 1) / chunkSize;
        size_t my_chunk_start = (total_chunks * (size_t)channel_id) / (size_t)numChannels;
        size_t my_chunk_end = (total_chunks * (size_t)(channel_id + 1)) / (size_t)numChannels;
        size_t my_num_chunks = my_chunk_end - my_chunk_start;

        size_t chunks_done = 0;
        while (chunks_done < my_num_chunks) {
          size_t first_recv_offset = 0;
          int num_new = 0;
          if (lane_id == 0) {
            while ((num_new = staging_poll(rdma_region, local_fc, &first_recv_offset)) == 0) {
            }
          }
          if (lane_id == 0) {
            type4_num_new[type4_idx] = num_new;
            type4_first_recv_offset[type4_idx] = first_recv_offset;
          }
          __syncwarp();
          num_new = type4_num_new[type4_idx];
          first_recv_offset = type4_first_recv_offset[type4_idx];

          int batch = num_new;
          if ((size_t)batch > my_num_chunks - chunks_done) {
            batch = (int)(my_num_chunks - chunks_done);
          }

          int first_slot = (local_fc.peerChunkSize > 0) ?
                             (int)((first_recv_offset - local_fc.peerDataOffset) / local_fc.peerChunkSize) :
                             0;

          for (int bi = 0; bi < batch; bi++) {
            size_t chunk = chunks_done + (size_t)bi;
            size_t global_chunk = my_chunk_start + chunk;

            int this_slot = (first_slot + bi) % local_fc.peerNumSlots;
            size_t local_recv_offset = local_fc.peerDataOffset + (size_t)this_slot * local_fc.peerChunkSize;

            size_t byte_start = global_chunk * chunkSize;
            size_t remaining = plan.totalBytes - byte_start;
            size_t chunk_bytes = (chunkSize < remaining) ? chunkSize : remaining;

            if (lane_id == 0) {
              rdma_wait_for_credits(gin, rdma_region, rdma_fc);

              int remote_slot = (int)(rdma_fc.shadowTail % (uint64_t)rdma_fc.peerNumSlots);
              size_t remote_offset = rdma_fc.peerDataOffset + (size_t)remote_slot * rdma_fc.peerChunkSize;

              gin.put(world, target_rank, params->rdmaWindow, remote_offset, params->rdmaWindow, local_recv_offset,
                      chunk_bytes, ncclGin_None{}, ncclGin_CounterInc{local_fc.localPutCounter});
              rdma_signal(gin, world, rdma_region, rdma_fc);
            }
            __syncwarp();
          }

          chunks_done += (size_t)batch;
        }
      }
    }

  } // end source side

  // ========================================================================
  // DEST SIDE — Type 6 (Multi-Group RDMA Receive + Unpack)
  // ========================================================================
  if (isDest && numRdmaSources > 0) {
    StagingRegion rdma_region = params->rdmaRegions[channel_id];

    if (warp_id < DIRECT_TYPE6_TOTAL_WARPS) {
      const int group_id = warp_id / DIRECT_TYPE6_WARPS_PER_GROUP;
      const int warp_in_group = warp_id % DIRECT_TYPE6_WARPS_PER_GROUP;
      const bool is_root_warp = (warp_in_group == 0);
      const int group_barrier = DIRECT_TYPE6_BARRIER_ID_BASE + group_id;
      const int dst_copy_thread = warp_in_group * 32 + lane_id;
      const int dst_copy_threads = DIRECT_TYPE6_WARPS_PER_GROUP * 32;

      char* rdma_staging_base = (char*)ncclGetLocalPointer(rdma_region.window, 0);
      char* user_dst = (char*)params->dstBuffer;

      const int my_src_start = (numRdmaSources * group_id) / DIRECT_TYPE6_NUM_GROUPS;
      const int my_src_end = (numRdmaSources * (group_id + 1)) / DIRECT_TYPE6_NUM_GROUPS;

      for (int s = my_src_start; s < my_src_end; s++) {
        StagingFlowCtrl rdma_fc = params->rdmaSources[channel_id][s].fc;
        if (is_root_warp) {
          rdma_fc.tailSignalBase = rdma_tail_bases[s];
        }

        const StagingTransferPlan& plan = params->rdmaSources[channel_id][s].plan;

        size_t total_chunks = (plan.totalBytes + chunkSize - 1) / chunkSize;
        size_t my_chunk_start = (total_chunks * (size_t)channel_id) / (size_t)numChannels;
        size_t my_chunk_end = (total_chunks * (size_t)(channel_id + 1)) / (size_t)numChannels;
        size_t my_num_chunks = my_chunk_end - my_chunk_start;

        size_t chunks_done = 0;
        while (chunks_done < my_num_chunks) {
          size_t first_recv_offset = 0;
          int num_new = 0;
          if (is_root_warp && lane_id == 0) {
            while ((num_new = rdma_poll(gin, rdma_region, rdma_fc, &first_recv_offset)) == 0) {
            }
            type6_num_new[group_id] = num_new;
            type6_first_recv_offset[group_id] = first_recv_offset;
          }

          barrier_sync_subset(group_barrier, DIRECT_TYPE6_THREADS_PER_GROUP);

          num_new = type6_num_new[group_id];
          first_recv_offset = type6_first_recv_offset[group_id];

          int batch = num_new;
          if ((size_t)batch > my_num_chunks - chunks_done) {
            batch = (int)(my_num_chunks - chunks_done);
          }

          int first_slot = (rdma_fc.peerChunkSize > 0) ?
                             (int)((first_recv_offset - rdma_fc.peerDataOffset) / rdma_fc.peerChunkSize) :
                             0;

          for (int bi = 0; bi < batch; bi++) {
            int this_slot = (first_slot + bi) % rdma_fc.peerNumSlots;
            size_t recv_offset = rdma_fc.peerDataOffset + (size_t)this_slot * rdma_fc.peerChunkSize;

            size_t global_chunk = my_chunk_start + chunks_done + (size_t)bi;
            size_t byte_start = global_chunk * chunkSize;
            size_t remaining = plan.totalBytes - byte_start;
            size_t this_bytes = (chunkSize < remaining) ? chunkSize : remaining;

            const char* staging_src = rdma_staging_base + recv_offset;

            if (plan.isContiguous) {
              staging_copy_contig(user_dst + plan.dstBaseOffset + byte_start, staging_src, this_bytes, dst_copy_threads,
                                  dst_copy_thread);
            } else {
              staging_copy_unpack(user_dst, staging_src, plan, byte_start, this_bytes, dst_copy_threads,
                                  dst_copy_thread);
            }
          }

          barrier_sync_subset(group_barrier, DIRECT_TYPE6_THREADS_PER_GROUP);

          if (is_root_warp && lane_id == 0) {
            rdma_release_flush(gin, world, rdma_region, rdma_fc);
          }

          chunks_done += (size_t)batch;
        }
      }
    }

  } // end dest side

  // ========================================================================
  // Final barrier (all ranks, all CTAs)
  // ========================================================================
  __syncthreads();
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
}

// ============================================================================
// Host-Side Launch Wrapper — Direct Algorithm
// ============================================================================

ncclResult_t launchStagingReshardDirect(StagingKernelParams* devParams, struct ncclDevComm* devComm, int numCtas,
                                        cudaStream_t stream, bool verbose) {
  const int threads_per_cta = DIRECT_TOTAL_WARPS * 32;

  if (verbose) {
    printf("[STAGING_KERNEL_DIRECT] Launching: grid=%d, block=%d "
           "(Type1=%d, Type4=%d, Type6=%dx%d), stream=%p\n",
           numCtas, threads_per_cta, DIRECT_TYPE1_NUM_WARPS, DIRECT_TYPE4_NUM_WARPS, DIRECT_TYPE6_NUM_GROUPS,
           DIRECT_TYPE6_WARPS_PER_GROUP, (void*)stream);
    fflush(stdout);
  }

  StagingReshardKernel_Direct<<<numCtas, threads_per_cta, 0, stream>>>(devParams, *devComm);

  NCCL_M2N_CUDACHECK(cudaGetLastError());

  if (verbose) {
    printf("[STAGING_KERNEL_DIRECT] Kernel enqueued successfully\n");
    fflush(stdout);
  }

  return ncclSuccess;
}

__device__ __forceinline__ int pipe_gin_context(const struct ncclDevComm& devComm, int channel_id, int context_base,
                                                int context_count) {
  int available = (int)devComm.ginContextCount - context_base;
  if (available <= 0 || context_count <= 0) {
    return 0;
  }
  int span = min(context_count, available);
  return context_base + (channel_id % span);
}

__device__ __forceinline__ bool staging_peer_uses_split_b(const StagingPeerInfo& peer) {
  return peer.rdmaTransport == STAGING_RDMA_TRANSPORT_SPLIT_B;
}

__device__ __forceinline__ ncclWindow_t staging_peer_rdma_window(const StagingKernelParams* params,
                                                                 const StagingPeerInfo& peer) {
  return staging_peer_uses_split_b(peer) ? params->rdmaWindowB : params->rdmaWindow;
}

__device__ __forceinline__ bool staging_peer_active(const StagingPeerInfo& peer) {
  return peer.active && peer.channelCount > 0;
}

__device__ __forceinline__ bool pipe_edge_active(const StagingPipePeerEdge& edge) {
  return edge.active;
}

__device__ __forceinline__ bool pipe_edge_uses_split_b(const StagingPipePeerEdge& edge) {
  return edge.rdmaTransport == STAGING_RDMA_TRANSPORT_SPLIT_B;
}

__device__ __forceinline__ ncclWindow_t pipe_edge_rdma_window(const StagingKernelParams* params,
                                                              const StagingPipePeerEdge& edge) {
  return pipe_edge_uses_split_b(edge) ? params->rdmaWindowB : params->rdmaWindow;
}

__device__ __forceinline__ void staging_peer_chunk_range(const StagingPeerInfo& peer, size_t* chunkStart,
                                                         size_t* chunkEnd) {
  *chunkStart = peer.chunkStart;
  *chunkEnd = peer.chunkEnd;
}

__device__ __forceinline__ void pipe_edge_chunk_range(const StagingPipePeerEdge& edge, size_t* chunkStart,
                                                      size_t* chunkEnd) {
  *chunkStart = edge.chunkStart;
  *chunkEnd = edge.chunkEnd;
}

#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
#ifndef STAGING_PIPE_SYNC_TIMEOUT_CYCLES
#define STAGING_PIPE_SYNC_TIMEOUT_CYCLES 10000000000ULL
#endif

#define PIPE_SPIN_INIT(name) \
  unsigned long long name##_spin_start = clock64()
#define PIPE_SYNC_TIMEOUT(name, params, epoch, channel_id, peer, progress, expected, observed) \
  do { \
    const unsigned long long name##_spin_elapsed = clock64() - name##_spin_start; \
    if (name##_spin_elapsed > STAGING_PIPE_SYNC_TIMEOUT_CYCLES) { \
      staging_pipe_sync_timeout((params), (epoch), (channel_id), #name, (peer), (size_t)(progress), \
                                (uint64_t)(expected), (uint64_t)(observed), name##_spin_elapsed); \
    } \
  } while (0)

#else
#define PIPE_SPIN_INIT(name) ((void)0)
#define PIPE_SYNC_TIMEOUT(name, params, epoch, channel_id, peer, progress, expected, observed) ((void)0)
#endif

// ============================================================================
// PIPE kernels — role-specialized persistent-counter staging path
// ============================================================================
//
// The supported PIPE path uses a cached host/device transfer plan and
// persistent per-edge cursors.  It launches one trainer kernel on source ranks
// and one generator kernel on destination ranks; generator behavior is selected
// by compile-time role booleans at launch.
// ============================================================================

/* Per-CTA thread layout for source/trainer ranks.
 *   warp 0       : control path. It consumes locally packed chunks and issues
 *                  RDMA puts/signals to generator peers.
 *   warps 1..16  : pack path. They gather from the user source tensor into the
 *                  local staging FIFO and publish local tails to warp 0. */
#define PIPE_TRAINER_CONTROL_WARPS 1
#define PIPE_TRAINER_PACK_WARPS 16
#define PIPE_TRAINER_PACK_WARP_START PIPE_TRAINER_CONTROL_WARPS
#define PIPE_TRAINER_TOTAL_WARPS (PIPE_TRAINER_CONTROL_WARPS + PIPE_TRAINER_PACK_WARPS)
#define PIPE_TRAINER_THREADS (PIPE_TRAINER_TOTAL_WARPS * 32)
#define PIPE_TRAINER_PACK_THREADS (PIPE_TRAINER_PACK_WARPS * 32)
#define PIPE_TRAINER_PACK_BARRIER_ID 1

/* Per-CTA thread layout for destination/generator ranks.
 *   warp 0       : control path. It polls RDMA/LSA tails, optionally forwards
 *                  to ring/LSA peers, drives TMA pulls, and releases credits.
 *   warps 1..16  : unpack path. They scatter staged/TMA-pulled payload into
 *                  the user destination tensor. */
#define PIPE_GENERATOR_CONTROL_WARPS 1
#define PIPE_GENERATOR_UNPACK_WARPS 16
#define PIPE_GENERATOR_UNPACK_WARP_START PIPE_GENERATOR_CONTROL_WARPS
#define PIPE_GENERATOR_TOTAL_WARPS (PIPE_GENERATOR_CONTROL_WARPS + PIPE_GENERATOR_UNPACK_WARPS)
#define PIPE_GENERATOR_THREADS (PIPE_GENERATOR_TOTAL_WARPS * 32)
#define PIPE_GENERATOR_UNPACK_THREADS (PIPE_GENERATOR_UNPACK_WARPS * 32)
#define PIPE_GENERATOR_UNPACK_BARRIER_ID 1
#ifndef PIPE_GENERATOR_QUEUE_DEPTH
#define PIPE_GENERATOR_QUEUE_DEPTH 2
#endif
#define PIPE_GENERATOR_DEFAULT_TMA_TILE_SIZE (64 * 1024)

constexpr int kPipeGeneratorRdmaPhase = 0;
constexpr int kPipeGeneratorLsaPhase = 1;

struct PipeQueueDesc {
  /* Control warp -> unpack warps handoff.  Queue entries describe either an
   * RDMA staging slot or one TMA tile staged in shared memory. */
  size_t staging_offset;
  size_t byte_start;
  size_t bytes;
  int tile_buf;
  int last_tile;
};

struct PipeChunkRange {
  size_t start;
  size_t end;
  size_t count;
};

struct PipeGeneratorLaunchMap {
  int channel_id;
  int phase;
};

__device__ __forceinline__ PipeChunkRange pipe_chunk_range(const StagingPipePeerEdge& edge) {
  PipeChunkRange range{0, 0, 0};
  if (pipe_edge_active(edge)) {
    pipe_edge_chunk_range(edge, &range.start, &range.end);
    range.count = range.end - range.start;
  }
  return range;
}

__device__ __forceinline__ int pipe_slot_from_offset(const StagingFlowCtrl& fc, size_t offset) {
  return (fc.peerChunkSize > 0) ? (int)((offset - fc.peerDataOffset) / fc.peerChunkSize) : 0;
}

__device__ __forceinline__ int pipe_clip_batch(int polled, size_t remaining_chunks) {
  return ((size_t)polled > remaining_chunks) ? (int)remaining_chunks : polled;
}

__device__ __forceinline__ size_t pipe_min_size(size_t a, size_t b) {
  return (a < b) ? a : b;
}

template <bool DoRdmaSources, bool DoLsaSources, bool SplitMixedRoleCtas>
__device__ __forceinline__ PipeGeneratorLaunchMap pipe_generator_launch_map(const StagingPipeDevicePlan* pipe_plan,
                                                                            int launch_cta) {
  PipeGeneratorLaunchMap map{launch_cta, kPipeGeneratorRdmaPhase};
  if constexpr (SplitMixedRoleCtas && DoRdmaSources && DoLsaSources) {
    const int rdma_launch_channels = pipe_plan->numGeneratorRdmaLaunchChannels;
    const int lsa_launch_channels = pipe_plan->numGeneratorLsaLaunchChannels;
    if (launch_cta < rdma_launch_channels) {
      map.channel_id = pipe_plan->generatorRdmaLaunchChannels[launch_cta];
      map.phase = kPipeGeneratorRdmaPhase;
    } else if (launch_cta < rdma_launch_channels + lsa_launch_channels) {
      map.channel_id = pipe_plan->generatorLsaLaunchChannels[launch_cta - rdma_launch_channels];
      map.phase = kPipeGeneratorLsaPhase;
    } else {
      map.channel_id = -1;
    }
  } else if constexpr (DoRdmaSources && !DoLsaSources) {
    map.channel_id = (launch_cta < pipe_plan->numGeneratorRdmaLaunchChannels) ?
                       pipe_plan->generatorRdmaLaunchChannels[launch_cta] :
                       -1;
  } else if constexpr (!DoRdmaSources && DoLsaSources) {
    map.channel_id =
      (launch_cta < pipe_plan->numGeneratorLsaLaunchChannels) ? pipe_plan->generatorLsaLaunchChannels[launch_cta] : -1;
    map.phase = kPipeGeneratorLsaPhase;
  }
  return map;
}

__device__ __forceinline__ size_t pipe_ready_offset(const StagingKernelParams* params, int channel) {
  return (size_t)channel * params->pipeReadyChannelStride + params->pipeReadySlotOffset;
}

__device__ __forceinline__ void pipe_wait_ready(ncclGin& gin, ncclWindow_t window, size_t offset) {
  char* staging_base = (char*)ncclGetLocalPointer(window, 0);
  uint64_t* shadow = (uint64_t*)(staging_base + offset + CTRL_FIELD_PIPE_READY_SHADOW);
  const uint64_t generation = ld_acquire_sys(shadow) + 1;
  gin.waitSignal(ncclCoopThread(), window, offset + CTRL_FIELD_PIPE_READY, generation);
  st_release_sys(shadow, generation);
}

__device__ __forceinline__ void pipe_wait_target_ready(const StagingKernelParams* params,
                                                       const StagingPipeTrainerEdge* targets, int first, int end,
                                                       int channel, ncclGin& ginA, ncclGin& ginB) {
  for (int t = first; t < end; t++) {
    const StagingPipePeerEdge& target = targets[t].peer;
    if (!pipe_edge_active(target)) {
      continue;
    }
    ncclGin& gin = pipe_edge_uses_split_b(target) ? ginB : ginA;
    pipe_wait_ready(gin, pipe_edge_rdma_window(params, target), pipe_ready_offset(params, channel));
  }
}

__device__ __forceinline__ size_t pipe_layout_offset(const StagingPipeCopyLayout& layout, size_t iter) {
  size_t offset = layout.baseOffset;
  size_t tmp = iter;
  for (int d = layout.numOuterLoops - 1; d >= 0; d--) {
    size_t idx = tmp % layout.outerCounts[d];
    tmp /= layout.outerCounts[d];
    offset += idx * layout.outerStrides[d];
  }
  return offset;
}

__device__ __forceinline__ void pipe_copy_aligned(char* dst, const char* src, size_t bytes) {
  uintptr_t align = (uintptr_t)src | (uintptr_t)dst;
  if ((align & 0xF) == 0 && bytes >= 16) {
    size_t n16 = bytes / 16;
    uint4* d = (uint4*)dst;
    const uint4* s = (const uint4*)src;
    for (size_t p = 0; p < n16; p++) {
      d[p] = s[p];
    }
    for (size_t b = n16 * 16; b < bytes; b++) {
      dst[b] = src[b];
    }
  } else if ((align & 0x3) == 0 && bytes >= 4) {
    size_t n4 = bytes / 4;
    uint32_t* d = (uint32_t*)dst;
    const uint32_t* s = (const uint32_t*)src;
    for (size_t p = 0; p < n4; p++) {
      d[p] = s[p];
    }
    for (size_t b = n4 * 4; b < bytes; b++) {
      dst[b] = src[b];
    }
  } else {
    for (size_t b = 0; b < bytes; b++) {
      dst[b] = src[b];
    }
  }
}

__device__ __forceinline__ void pipe_copy_pack_layout(char* dst_staging, const char* src_user_buffer,
                                                      const StagingPipeCopyLayout& layout, size_t byte_start,
                                                      size_t num_bytes, int warp_group_threads, int thread_in_group) {
  const size_t inner = layout.innerSize;
  size_t first_iter = byte_start / inner;
  size_t first_offset = byte_start % inner;
  size_t dst_write_offset = 0;
  size_t bytes_remaining = num_bytes;
  size_t iter = first_iter;

  while (bytes_remaining > 0) {
    size_t src_offset = pipe_layout_offset(layout, iter);
    size_t iter_start = (iter == first_iter) ? first_offset : 0;
    size_t avail = inner - iter_start;
    size_t iter_bytes = pipe_min_size(avail, bytes_remaining);
    staging_memcpy(dst_staging + dst_write_offset, src_user_buffer + src_offset + iter_start, iter_bytes,
                   warp_group_threads, thread_in_group);
    dst_write_offset += iter_bytes;
    bytes_remaining -= iter_bytes;
    iter++;
  }
}

__device__ __forceinline__ void pipe_copy_unpack_layout(char* dst_user_buffer, const char* src_staging,
                                                        const StagingPipeCopyLayout& layout, size_t byte_start,
                                                        size_t num_bytes, int warp_group_threads, int thread_in_group) {
  const size_t inner = layout.innerSize;
  size_t first_iter = byte_start / inner;
  size_t first_offset = byte_start % inner;
  size_t src_read_offset = 0;
  size_t bytes_remaining = num_bytes;
  size_t iter = first_iter;

  while (bytes_remaining > 0) {
    size_t dst_offset = pipe_layout_offset(layout, iter);
    size_t iter_start = (iter == first_iter) ? first_offset : 0;
    size_t avail = inner - iter_start;
    size_t iter_bytes = pipe_min_size(avail, bytes_remaining);
    staging_memcpy(dst_user_buffer + dst_offset + iter_start, src_staging + src_read_offset, iter_bytes,
                   warp_group_threads, thread_in_group);
    src_read_offset += iter_bytes;
    bytes_remaining -= iter_bytes;
    iter++;
  }
}

__device__ __forceinline__ void pipe_copy_pack_layout_parallel(char* dst_staging, const char* src_user_buffer,
                                                               const StagingPipeCopyLayout& layout, size_t byte_start,
                                                               size_t num_bytes, int warp_group_threads,
                                                               int thread_in_group) {
  const size_t inner = layout.innerSize;
  if (inner == 0) {
    return;
  }

  size_t first_iter = byte_start / inner;
  size_t last_byte = byte_start + num_bytes;
  size_t last_iter = (last_byte + inner - 1) / inner;
  size_t num_iters = last_iter - first_iter;

  for (size_t i = (size_t)thread_in_group; i < num_iters; i += (size_t)warp_group_threads) {
    size_t iter = first_iter + i;
    size_t src_offset = pipe_layout_offset(layout, iter);
    size_t iter_byte_start = iter * inner;
    size_t iter_byte_end = iter_byte_start + inner;
    size_t copy_start = (iter_byte_start >= byte_start) ? 0 : (byte_start - iter_byte_start);
    size_t copy_end = (iter_byte_end <= last_byte) ? inner : (last_byte - iter_byte_start);
    size_t copy_len = copy_end - copy_start;
    const char* src_ptr = src_user_buffer + src_offset + copy_start;
    char* dst_ptr = dst_staging + (iter_byte_start + copy_start - byte_start);
    pipe_copy_aligned(dst_ptr, src_ptr, copy_len);
  }
}

__device__ __forceinline__ void pipe_copy_unpack_layout_parallel(char* dst_user_buffer, const char* src_staging,
                                                                 const StagingPipeCopyLayout& layout, size_t byte_start,
                                                                 size_t num_bytes, int warp_group_threads,
                                                                 int thread_in_group) {
  const size_t inner = layout.innerSize;
  if (inner == 0) {
    return;
  }

  size_t first_iter = byte_start / inner;
  size_t last_byte = byte_start + num_bytes;
  size_t last_iter = (last_byte + inner - 1) / inner;
  size_t num_iters = last_iter - first_iter;

  for (size_t i = (size_t)thread_in_group; i < num_iters; i += (size_t)warp_group_threads) {
    size_t iter = first_iter + i;
    size_t dst_offset = pipe_layout_offset(layout, iter);
    size_t iter_byte_start = iter * inner;
    size_t iter_byte_end = iter_byte_start + inner;
    size_t copy_start = (iter_byte_start >= byte_start) ? 0 : (byte_start - iter_byte_start);
    size_t copy_end = (iter_byte_end <= last_byte) ? inner : (last_byte - iter_byte_start);
    size_t copy_len = copy_end - copy_start;
    const char* src_ptr = src_staging + (iter_byte_start + copy_start - byte_start);
    char* dst_ptr = dst_user_buffer + dst_offset + copy_start;
    pipe_copy_aligned(dst_ptr, src_ptr, copy_len);
  }
}

__device__ __forceinline__ void pipe_pack_chunk(char* staging_dst, const char* user_src,
                                                const StagingPipeCopyLayout& layout, size_t byte_start, size_t bytes,
                                                int threads, int thread_in_group) {
  if (layout.isContiguous) {
    staging_copy_contig(staging_dst, user_src + layout.baseOffset + byte_start, bytes, threads, thread_in_group);
  } else if (layout.innerSize < (size_t)threads * STAGING_PARALLEL_INNER_THRESHOLD) {
    pipe_copy_pack_layout_parallel(staging_dst, user_src, layout, byte_start, bytes, threads, thread_in_group);
  } else {
    pipe_copy_pack_layout(staging_dst, user_src, layout, byte_start, bytes, threads, thread_in_group);
  }
}

__device__ __forceinline__ void pipe_unpack_chunk(char* user_dst, const char* staging_src,
                                                  const StagingPipeCopyLayout& layout, size_t byte_start, size_t bytes,
                                                  int threads, int thread_in_group) {
  if (layout.isContiguous) {
    staging_copy_contig(user_dst + layout.baseOffset + byte_start, staging_src, bytes, threads, thread_in_group);
  } else if (layout.innerSize < (size_t)threads * STAGING_PARALLEL_INNER_THRESHOLD) {
    pipe_copy_unpack_layout_parallel(user_dst, staging_src, layout, byte_start, bytes, threads, thread_in_group);
  } else {
    pipe_copy_unpack_layout(user_dst, staging_src, layout, byte_start, bytes, threads, thread_in_group);
  }
}

__global__ __launch_bounds__(PIPE_TRAINER_THREADS, 1) void StagingReshardKernel_PipeTrainer(
  StagingKernelParams* params, const StagingPipeDevicePlan* pipe_plan, StagingPipeCallParams call,
  struct ncclDevComm devCommA, struct ncclDevComm devCommB) {
  /* One active CTA owns one logical staging channel.  The launch map skips
   * idle channels so blockIdx.x is compact while channel_id remains stable. */
  int channel_id = -1;
  const int launch_cta = (int)blockIdx.x;
  if (launch_cta < pipe_plan->numTrainerRdmaLaunchChannels) {
    channel_id = pipe_plan->trainerRdmaLaunchChannels[launch_cta];
  }

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  __shared__ int pack_slot;

  if (!params->isSource || params->numRdmaTargets <= 0 || channel_id < 0 || channel_id >= params->numChannels) {
    return;
  }

  ncclGin ginA{devCommA, pipe_gin_context(devCommA, channel_id, 0, (int)devCommA.ginContextCount)};
  const int ctxPerSlotB = call.splitComm ? call.splitCommBContextCount : (int)devCommB.ginContextCount;
  ncclGin ginB{devCommB, pipe_gin_context(devCommB, channel_id, call.splitCommBContextBase, ctxPerSlotB)};
  ncclTeam worldA = ncclTeamWorld(devCommA);
  ncclTeam worldB = ncclTeamWorld(devCommB);
  StagingRegion rdma_region = params->rdmaRegions[channel_id];
#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
  const uint64_t trace_epoch = call.epoch;
#endif

  __syncthreads();

  /* Trainer stage A: control warp.
   *
   * Lane 0 waits for generator readiness, posts locally packed staging slots
   * to the generator FIFO, and increments the remote tail. */
  if (warp_id == 0) {
    if (lane_id == 0) {
      for (int t = 0; t < params->numRdmaTargets; t++) {
        const StagingPipeTrainerEdge& target_edge = pipe_plan->rdmaTargets[channel_id][t];
        const StagingPipePeerEdge& target = target_edge.peer;
        if (!pipe_edge_active(target)) {
          continue;
        }

        StagingFlowCtrl local_fc = target_edge.localFc;
        ncclGin& target_gin = pipe_edge_uses_split_b(target) ? ginB : ginA;
        ncclTeam target_world = pipe_edge_uses_split_b(target) ? worldB : worldA;
        ncclWindow_t target_window = pipe_edge_rdma_window(params, target);
        uint64_t local_put_counter_base = target_gin.readCounter(local_fc.localPutCounter);

        StagingFlowCtrl rdma_fc = target.fc;
        rdma_fc.shadowTail = staging_cursor_load(rdma_region, rdma_fc.cursorTailOffset);
        rdma_fc.headSignalBase = 0;
        /* Pack warps may publish this call's first local tail before the
         * readiness wait completes. Start from the last posted RDMA tail. */
        local_fc.shadowTail = rdma_fc.shadowTail;
        local_fc.lastTailVal = rdma_fc.shadowTail;

        pipe_wait_target_ready(params, pipe_plan->rdmaTargets[channel_id], t, t + 1, channel_id, ginA, ginB);

        const size_t chunkSize = target.logicalChunkSize;
        const PipeChunkRange chunks = pipe_chunk_range(target);

        size_t chunks_done = 0;
        while (chunks_done < chunks.count) {
          size_t first_recv_offset = 0;
          int num_new = 0;
          PIPE_SPIN_INIT(rdma_staging_poll_wait);
          while ((num_new = staging_poll(rdma_region, local_fc, &first_recv_offset)) == 0) {
            PIPE_SYNC_TIMEOUT(rdma_staging_poll_wait, params, trace_epoch, channel_id, target.peerWorldRank,
                              chunks_done, local_fc.lastTailVal + 1,
                              staging_cursor_load(rdma_region, local_fc.localTailOffset) - local_fc.lsaTailBase);
          }
          int batch = pipe_clip_batch(num_new, chunks.count - chunks_done);
          int first_slot = pipe_slot_from_offset(local_fc, first_recv_offset);

          for (int bi = 0; bi < batch; bi++) {
            int this_slot = (first_slot + bi) % local_fc.peerNumSlots;
            size_t local_recv_offset = local_fc.peerDataOffset + (size_t)this_slot * local_fc.peerChunkSize;
            size_t byte_start = (chunks.start + chunks_done + (size_t)bi) * chunkSize;
            size_t remaining = target.totalBytes - byte_start;
            size_t chunk_bytes = pipe_min_size(chunkSize, remaining);
#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
            rdma_wait_for_credits_debug(target_gin, rdma_region, rdma_fc, params, trace_epoch, channel_id,
                                        "rdma_credit_wait", target.peerWorldRank, chunks_done + (size_t)bi);
#else
            rdma_wait_for_credits(target_gin, rdma_region, rdma_fc);
#endif
            int remote_slot = (int)(rdma_fc.shadowTail % (uint64_t)rdma_fc.peerNumSlots);
            size_t remote_offset = rdma_fc.peerDataOffset + (size_t)remote_slot * rdma_fc.peerChunkSize;
            target_gin.put(target_world, target.peerWorldRank, target_window, remote_offset, target_window,
                           local_recv_offset, chunk_bytes, ncclGin_None{},
                           ncclGin_CounterInc{local_fc.localPutCounter});
            rdma_signal(target_gin, target_world, rdma_region, rdma_fc);
            staging_cursor_store(rdma_region, rdma_fc.cursorTailOffset, rdma_fc.shadowTail);
          }
          chunks_done += (size_t)batch;
        }
        if (chunks.count > 0) {
          uint64_t counter_done = local_put_counter_base + (uint64_t)chunks.count;
          PIPE_SPIN_INIT(rdma_put_counter_wait);
          while (target_gin.readCounter(local_fc.localPutCounter) < counter_done) {
            PIPE_SYNC_TIMEOUT(rdma_put_counter_wait, params, trace_epoch, channel_id, target.peerWorldRank,
                              chunks_done, counter_done, target_gin.readCounter(local_fc.localPutCounter));
          }
        }
      }
    }
  }

  /* Trainer stage B: pack warps.
   *
   * The pack group waits for local FIFO credit, gathers one logical chunk from
   * the source tensor into local staging memory, then publishes the local tail
   * consumed by the control warp above. */
  if (warp_id >= PIPE_TRAINER_PACK_WARP_START && warp_id < PIPE_TRAINER_PACK_WARP_START + PIPE_TRAINER_PACK_WARPS) {
    const int pack_warp = warp_id - PIPE_TRAINER_PACK_WARP_START;
    const int thread_in_group = pack_warp * 32 + lane_id;
    const bool is_root_warp = (pack_warp == 0);

    for (int t = 0; t < params->numRdmaTargets; t++) {
      const StagingPipeTrainerEdge& target_edge = pipe_plan->rdmaTargets[channel_id][t];
      const StagingPipePeerEdge& target = target_edge.peer;
      if (!pipe_edge_active(target)) {
        continue;
      }

      StagingFlowCtrl fc = target_edge.localFc;
      ncclGin& target_gin = pipe_edge_uses_split_b(target) ? ginB : ginA;
      ncclTeam target_world = pipe_edge_uses_split_b(target) ? worldB : worldA;
      uint64_t local_counter_base_offset = 0;
      if (is_root_warp) {
        fc.shadowTail = staging_cursor_load(rdma_region, fc.localTailOffset);
        fc.lastTailVal = fc.shadowTail;
        local_counter_base_offset = fc.shadowTail - target_gin.readCounter(fc.localPutCounter);
      }

      const size_t chunkSize = target.logicalChunkSize;
      const PipeChunkRange chunks = pipe_chunk_range(target);

      for (size_t chunk = 0; chunk < chunks.count; chunk++) {
        size_t global_chunk = chunks.start + chunk;
        if (is_root_warp && lane_id == 0) {
          uint64_t baseOffset = local_counter_base_offset;
#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
          lsa_rdma_wait_for_credits_debug(target_gin, fc, baseOffset, params, trace_epoch, channel_id,
                                          "lsa_rdma_credit_wait", target.peerWorldRank, chunk);
#else
          lsa_rdma_wait_for_credits(target_gin, fc, baseOffset);
#endif
          pack_slot = (int)(fc.shadowTail % (uint64_t)fc.peerNumSlots);
        }
        barrier_sync_subset(PIPE_TRAINER_PACK_BARRIER_ID, PIPE_TRAINER_PACK_THREADS);

        char* staging_base = (char*)ncclGetLocalPointer(rdma_region.window, 0);
        char* staging_dst = staging_base + fc.peerDataOffset + (size_t)pack_slot * fc.peerChunkSize;
        const char* user_src = (const char*)call.srcBuffer;
        size_t byte_start = global_chunk * chunkSize;
        size_t remaining = target.totalBytes - byte_start;
        size_t this_bytes = pipe_min_size(chunkSize, remaining);
        const StagingPipeCopyLayout& target_layout = pipe_plan->rdmaTargetLayouts[channel_id][target.copyLayoutIndex];
        pipe_pack_chunk(staging_dst, user_src, target_layout, byte_start, this_bytes, PIPE_TRAINER_PACK_THREADS,
                        thread_in_group);
        barrier_sync_subset(PIPE_TRAINER_PACK_BARRIER_ID, PIPE_TRAINER_PACK_THREADS);
        if (is_root_warp && lane_id == 0) {
          rdma_signal(target_gin, target_world, rdma_region, fc);
        }
      }
    }
  }

  __syncthreads();
  ginA.flush(ncclCoopCta());
  if (call.splitComm) {
    ginB.flush(ncclCoopCta());
  }
  __syncthreads();
}

/* Generator template knobs:
 *   TmaTileSize        : shared-memory tile size for LSA/TMA pulls.
 *   DoRdmaSources      : enable trainer/ring RDMA receive + unpack.
 *   DoRingForward      : forward RDMA-received chunks to another RDMA peer.
 *   DoLsaForward       : publish RDMA-received chunks to local LSA followers.
 *   DoLsaSources       : enable LSA/TMA pull + unpack from an in-domain leader.
 *   SplitMixedRoleCtas : for ranks with both RDMA and LSA inputs, launch
 *                        separate compact CTA sets for the two phases. */
template <int TmaTileSize, bool DoRdmaSources, bool DoRingForward, bool DoLsaForward, bool DoLsaSources,
          bool SplitMixedRoleCtas>
__global__ __launch_bounds__(PIPE_GENERATOR_THREADS, 1) void StagingReshardKernel_PipeGenerator(
  StagingKernelParams* params, const StagingPipeDevicePlan* pipe_plan, StagingPipeCallParams call,
  struct ncclDevComm devCommA, struct ncclDevComm devCommB) {
  /* Generator CTAs can be role-specialized at launch:
   *   RDMA phase: poll trainer/ring input and unpack/forward it.
   *   LSA phase : TMA-pull from an in-domain leader and unpack it.
   * Mixed roles use separate compact launch maps so one CTA does not have to
   * carry both RDMA and LSA work when the rank has both edge types. */
  const int num_channels = params->numChannels;
  const bool split_mixed_role_ctas = SplitMixedRoleCtas && DoRdmaSources && DoLsaSources;
  const PipeGeneratorLaunchMap launch_map =
    pipe_generator_launch_map<DoRdmaSources, DoLsaSources, SplitMixedRoleCtas>(pipe_plan, (int)blockIdx.x);
  const int channel_id = launch_map.channel_id;
  const int phase = launch_map.phase;
  const bool run_rdma_sources = DoRdmaSources && (!split_mixed_role_ctas || phase == kPipeGeneratorRdmaPhase);
  const bool run_lsa_sources = DoLsaSources && (!split_mixed_role_ctas || phase == kPipeGeneratorLsaPhase);
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const bool is_control = (warp_id == 0);
  const bool is_unpack = (warp_id >= PIPE_GENERATOR_UNPACK_WARP_START &&
                          warp_id < PIPE_GENERATOR_UNPACK_WARP_START + PIPE_GENERATOR_UNPACK_WARPS);
  const int unpack_warp = warp_id - PIPE_GENERATOR_UNPACK_WARP_START;
  const int unpack_thread = unpack_warp * 32 + lane_id;
  const bool is_unpack_leader = is_unpack && unpack_thread == 0;

  __shared__ volatile unsigned int q_tail;
  __shared__ volatile unsigned int q_done;
  __shared__ volatile unsigned int q_done_chunks;
  __shared__ PipeQueueDesc q_desc[PIPE_GENERATOR_QUEUE_DEPTH];
  __shared__ PipeQueueDesc cur_desc;
  __shared__ int rdma_ctrl_batch;
  __shared__ int rdma_ctrl_first_slot;
  __shared__ __align__(8) uint64_t tma_mbar[PIPE_GENERATOR_QUEUE_DEPTH];
  extern __shared__ char tma_tile_smem[];
  char* tma_tile_base = (char*)(((uintptr_t)tma_tile_smem + 127ULL) & ~127ULL);

  if (!params->isDest || channel_id < 0 || channel_id >= num_channels) {
    return;
  }

  if constexpr (DoLsaSources) {
    /* Only LSA-source CTAs use TMA.  Their control warp fills shared-memory
     * tiles, and unpack warps wait on these mbarriers before scattering. */
    if (threadIdx.x < PIPE_GENERATOR_QUEUE_DEPTH) {
      staging_tma_mbarrier_init(&tma_mbar[threadIdx.x], 1);
    }
    if (threadIdx.x == 0) {
      staging_tma_fence_init();
    }
  }
  __syncthreads();

  StagingRegion rdma_region = params->rdmaRegions[channel_id];
  StagingRegion lsa_region = params->lsaRegions[channel_id];
  char* rdma_staging_base = (char*)ncclGetLocalPointer(rdma_region.window, 0);
  char* user_dst = (char*)call.dstBuffer;
#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
  const uint64_t trace_epoch = call.epoch;
#endif

  if constexpr (DoRdmaSources) {
    if (run_rdma_sources) {
      ncclGin ginA{devCommA, pipe_gin_context(devCommA, channel_id, 0, (int)devCommA.ginContextCount)};
      const int ctxPerSlotB = call.splitComm ? call.splitCommBContextCount : (int)devCommB.ginContextCount;
      ncclGin ginB{devCommB, pipe_gin_context(devCommB, channel_id, call.splitCommBContextBase, ctxPerSlotB)};
      ncclTeam worldA = ncclTeamWorld(devCommA);
      ncclTeam worldB = ncclTeamWorld(devCommB);

      /* Reaching this kernel means this channel's prior consumption is done. */
      int ready_source = -1;
      for (int s = 0; s < params->numRdmaSources; s++) {
        if (pipe_edge_active(pipe_plan->rdmaSources[channel_id][s])) {
          ready_source = s;
          break;
        }
      }
      if (ready_source >= 0) {
        const StagingPipePeerEdge& source = pipe_plan->rdmaSources[channel_id][ready_source];
        ncclGin& source_gin = pipe_edge_uses_split_b(source) ? ginB : ginA;
        if (threadIdx.x == 0) {
          ncclTeam source_world = pipe_edge_uses_split_b(source) ? worldB : worldA;
          source_gin.signal(source_world, source.peerWorldRank,
                            ncclGin_WeakVASignalInc{pipe_edge_rdma_window(params, source),
                                                    pipe_ready_offset(params, channel_id) + CTRL_FIELD_PIPE_READY});
        }
        __syncthreads();
      }

      if (threadIdx.x == 0) {
        if constexpr (DoRingForward) {
          const int ring_start = params->numRdmaTargets - params->numRingTargets;
          pipe_wait_target_ready(params, pipe_plan->rdmaTargets[channel_id], ring_start, params->numRdmaTargets,
                                 channel_id, ginA, ginB);
        }
      }
      __syncthreads();

      for (int s = 0; s < params->numRdmaSources; s++) {
        const StagingPipePeerEdge& source = pipe_plan->rdmaSources[channel_id][s];
        const bool active = pipe_edge_active(source);
        ncclGin& source_gin = pipe_edge_uses_split_b(source) ? ginB : ginA;
        ncclTeam source_world = pipe_edge_uses_split_b(source) ? worldB : worldA;
        const size_t chunkSize = source.logicalChunkSize;
        const PipeChunkRange chunks = pipe_chunk_range(source);

        if (threadIdx.x == 0) {
          q_tail = 0;
          q_done = 0;
          rdma_ctrl_batch = 0;
          rdma_ctrl_first_slot = 0;
        }
        __syncthreads();

        if (active && chunks.count > 0 && is_control) {
          /* Generator RDMA stage A: control warp.
           *
           * Poll a trainer/ring producer tail, enqueue received staging slots
           * for unpack, optionally forward the same slot to the next ring peer
           * or LSA followers, then release producer credits after unpack and
           * forwarding have caught up. */
          StagingFlowCtrl poll_fc = source.fc;
          StagingFlowCtrl release_fc = source.fc;
          uint64_t rdma_release_head_base = staging_cursor_load(rdma_region, source.fc.cursorHeadOffset);
          poll_fc.tailSignalBase = 0;
          poll_fc.lastTailVal = rdma_release_head_base;
          poll_fc.localHeadVal = rdma_release_head_base;
          release_fc.tailSignalBase = 0;
          release_fc.lastTailVal = rdma_release_head_base;
          release_fc.localHeadVal = rdma_release_head_base;

          const int fwd_start = s * params->numLsaFollowers;

          const int ring_start = params->numRdmaTargets - params->numRingTargets;
          const int ring_idx = ring_start + s;
          StagingFlowCtrl ring_fc;
          uint64_t ring_counter_val = 0;
          int ring_target_rank = -1;
          const bool ring_active = DoRingForward && (params->numRingTargets > 0) && ring_idx >= 0 &&
                                   ring_idx < params->numRdmaTargets &&
                                   pipe_edge_active(pipe_plan->rdmaTargets[channel_id][ring_idx].peer);
          if (lane_id == 0 && ring_active) {
            const StagingPipePeerEdge& ring_target = pipe_plan->rdmaTargets[channel_id][ring_idx].peer;
            ring_fc = ring_target.fc;
            ncclGin& ring_gin = pipe_edge_uses_split_b(ring_target) ? ginB : ginA;
            ring_fc.shadowTail = staging_cursor_load(rdma_region, ring_fc.cursorTailOffset);
            ring_fc.headSignalBase = 0;
            ring_counter_val = ring_gin.readCounter(ring_fc.localPutCounter);
            ring_target_rank = ring_target.peerWorldRank;
          }

          size_t chunks_done = 0;
          while (chunks_done < chunks.count) {
            if (lane_id == 0) {
              size_t first_recv_offset = 0;
              int num_new = 0;
              PIPE_SPIN_INIT(rdma_poll_wait);
              while ((num_new = rdma_poll(source_gin, rdma_region, poll_fc, &first_recv_offset)) == 0) {
                PIPE_SYNC_TIMEOUT(rdma_poll_wait, params, trace_epoch, channel_id, s, chunks_done,
                                  poll_fc.lastTailVal + 1,
                                  source_gin.readSignal(poll_fc.localTailSignal) - poll_fc.tailSignalBase);
              }
              int batch = pipe_clip_batch(num_new, chunks.count - chunks_done);
              rdma_ctrl_batch = batch;
              rdma_ctrl_first_slot = pipe_slot_from_offset(poll_fc, first_recv_offset);
            }
            __syncwarp();

            int batch = rdma_ctrl_batch;
            int first_slot = rdma_ctrl_first_slot;
            for (int bi = 0; bi < batch; bi++) {
              size_t logical_chunk = chunks_done + (size_t)bi;
              if (lane_id == 0) {
                PIPE_SPIN_INIT(rdma_queue_wait);
                while ((q_tail - q_done) >= (unsigned int)PIPE_GENERATOR_QUEUE_DEPTH) {
                  PIPE_SYNC_TIMEOUT(rdma_queue_wait, params, trace_epoch, channel_id, source.peerWorldRank,
                                    logical_chunk, q_tail - PIPE_GENERATOR_QUEUE_DEPTH + 1, q_done);
                }
                unsigned int q_idx = q_tail % (unsigned int)PIPE_GENERATOR_QUEUE_DEPTH;
                int this_slot = (first_slot + bi) % poll_fc.peerNumSlots;
                size_t recv_offset = poll_fc.peerDataOffset + (size_t)this_slot * poll_fc.peerChunkSize;
                size_t global_chunk = chunks.start + logical_chunk;
                size_t byte_start = global_chunk * chunkSize;
                size_t remaining = source.totalBytes - byte_start;
                size_t this_bytes = pipe_min_size(chunkSize, remaining);
                q_desc[q_idx].staging_offset = recv_offset;
                q_desc[q_idx].byte_start = byte_start;
                q_desc[q_idx].bytes = this_bytes;
                q_desc[q_idx].tile_buf = -1;
                q_desc[q_idx].last_tile = 1;
                __threadfence_block();
                q_tail = q_tail + 1;

                if (ring_active) {
                  const StagingPipePeerEdge& ring_target = pipe_plan->rdmaTargets[channel_id][ring_idx].peer;
                  ncclGin& ring_gin = pipe_edge_uses_split_b(ring_target) ? ginB : ginA;
                  ncclTeam ring_world = pipe_edge_uses_split_b(ring_target) ? worldB : worldA;
                  ncclWindow_t ring_window = pipe_edge_rdma_window(params, ring_target);
#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
                  rdma_wait_for_credits_debug(ring_gin, rdma_region, ring_fc, params, trace_epoch, channel_id,
                                              "rdma_credit_wait", ring_target.peerWorldRank, logical_chunk);
#else
                  rdma_wait_for_credits(ring_gin, rdma_region, ring_fc);
#endif
                  int remote_slot = (int)(ring_fc.shadowTail % (uint64_t)ring_fc.peerNumSlots);
                  size_t remote_offset = ring_fc.peerDataOffset + (size_t)remote_slot * ring_fc.peerChunkSize;
                  ring_gin.put(ring_world, ring_target_rank, ring_window, remote_offset, ring_window, recv_offset,
                               this_bytes, ncclGin_None{}, ncclGin_CounterInc{ring_fc.localPutCounter});
                  rdma_signal(ring_gin, ring_world, rdma_region, ring_fc);
                  staging_cursor_store(rdma_region, ring_fc.cursorTailOffset, ring_fc.shadowTail);
                  ring_counter_val++;
                }
              }

              /* Publish each received RDMA chunk to LSA followers immediately,
               * allowing in-domain forwarding to overlap subsequent receives. */
              if constexpr (DoLsaForward) {
                for (int fwd_lane = lane_id; fwd_lane < params->numLsaFollowers; fwd_lane += 32) {
                  const int fwd_tgt_idx = fwd_start + fwd_lane;
                  if (fwd_tgt_idx >= params->numLsaTargets) {
                    continue;
                  }
                  const StagingPipePeerEdge& fwd_target = pipe_plan->lsaTargets[channel_id][fwd_tgt_idx];
                  if (!pipe_edge_active(fwd_target)) {
                    continue;
                  }
                  StagingFlowCtrl my_fwd_fc = fwd_target.fc;
                  my_fwd_fc.lsaHeadBase = 0;
                  my_fwd_fc.lsaTailBase = 0;
                  my_fwd_fc.shadowTail = staging_cursor_load(lsa_region, my_fwd_fc.cursorTailOffset);
                  my_fwd_fc.lastTailVal = my_fwd_fc.shadowTail;
                  my_fwd_fc.localHeadVal = staging_cursor_load(lsa_region, my_fwd_fc.localHeadOffset);
#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
                  lsa_wait_for_credits_debug(lsa_region, my_fwd_fc, params, trace_epoch, channel_id,
                                             "lsa_follower_credit_wait", fwd_target.peerWorldRank, logical_chunk);
#else
                  lsa_wait_for_credits(lsa_region, my_fwd_fc);
#endif
                  lsa_signal(lsa_region, my_fwd_fc);
                  staging_cursor_store(lsa_region, my_fwd_fc.cursorTailOffset, my_fwd_fc.shadowTail);
                }
              }
              __syncwarp();
            }
            chunks_done += (size_t)batch;

            /* Do not release producer RDMA slots until the unpack group and all
             * LSA followers have consumed this batch. */
            if (lane_id == 0) {
              PIPE_SPIN_INIT(rdma_q_done_wait);
              while (q_done < (unsigned int)chunks_done) {
                PIPE_SYNC_TIMEOUT(rdma_q_done_wait, params, trace_epoch, channel_id, source.peerWorldRank,
                                  chunks_done, chunks_done, q_done);
              }
            }
            if constexpr (DoLsaForward) {
              for (int fwd_lane = lane_id; fwd_lane < params->numLsaFollowers; fwd_lane += 32) {
                const int fwd_tgt_idx = fwd_start + fwd_lane;
                if (fwd_tgt_idx >= params->numLsaTargets) {
                  continue;
                }
                const StagingPipePeerEdge& fwd_target = pipe_plan->lsaTargets[channel_id][fwd_tgt_idx];
                if (!pipe_edge_active(fwd_target)) {
                  continue;
                }
                StagingFlowCtrl my_fwd_fc = fwd_target.fc;
                my_fwd_fc.lsaHeadBase = 0;
                const uint64_t desired_tail = staging_cursor_load(lsa_region, my_fwd_fc.cursorTailOffset);
                char* lsa_staging = (char*)ncclGetLocalPointer(lsa_region.window, 0);
                uint64_t* head_ptr = (uint64_t*)(lsa_staging + my_fwd_fc.localHeadOffset);
                PIPE_SPIN_INIT(rdma_fwd_head_wait);
                while ((ld_acquire(head_ptr, my_fwd_fc.isLocal) - my_fwd_fc.lsaHeadBase) < desired_tail) {
                  PIPE_SYNC_TIMEOUT(rdma_fwd_head_wait, params, trace_epoch, channel_id,
                                    fwd_target.peerWorldRank, chunks_done, desired_tail,
                                    ld_acquire(head_ptr, my_fwd_fc.isLocal) - my_fwd_fc.lsaHeadBase);
                }
              }
            }
            __syncwarp();
            if (lane_id == 0) {
              if (ring_active) {
                const StagingPipePeerEdge& ring_target = pipe_plan->rdmaTargets[channel_id][ring_idx].peer;
                ncclGin& ring_gin = pipe_edge_uses_split_b(ring_target) ? ginB : ginA;
                PIPE_SPIN_INIT(rdma_ring_counter_wait);
                while (ring_gin.readCounter(ring_fc.localPutCounter) < ring_counter_val) {
                  PIPE_SYNC_TIMEOUT(rdma_ring_counter_wait, params, trace_epoch, channel_id, ring_target.peerWorldRank,
                                    chunks_done, ring_counter_val, ring_gin.readCounter(ring_fc.localPutCounter));
                }
              }
              release_fc.lastTailVal = rdma_release_head_base + (uint64_t)chunks_done;
              rdma_release_flush(source_gin, source_world, rdma_region, release_fc);
              staging_cursor_store(rdma_region, release_fc.cursorHeadOffset, release_fc.localHeadVal);
            }
          }
        }

        if (active && chunks.count > 0 && is_unpack) {
          /* Generator RDMA stage B: unpack warps.
           *
           * A single unpack leader consumes queue descriptors from the control
           * warp; the full unpack group scatters each staged chunk into the
           * destination tensor and advances q_done for credit release. */
          for (size_t chunk = 0; chunk < chunks.count; chunk++) {
            if (is_unpack_leader) {
              PIPE_SPIN_INIT(rdma_unpack_queue_wait);
              while (q_tail <= (unsigned int)chunk) {
                PIPE_SYNC_TIMEOUT(rdma_unpack_queue_wait, params, trace_epoch, channel_id, source.peerWorldRank,
                                  chunk, chunk + 1, q_tail);
              }
              __threadfence_block();
              cur_desc = q_desc[chunk % (unsigned int)PIPE_GENERATOR_QUEUE_DEPTH];
            }
            /* Publish the control warp's queue descriptor to every unpack warp. */
            barrier_sync_subset(PIPE_GENERATOR_UNPACK_BARRIER_ID, PIPE_GENERATOR_UNPACK_THREADS);
            const char* staging_src = rdma_staging_base + cur_desc.staging_offset;
            const StagingPipeCopyLayout& source_layout =
              pipe_plan->rdmaSourceLayouts[channel_id][source.copyLayoutIndex];
            pipe_unpack_chunk(user_dst, staging_src, source_layout, cur_desc.byte_start, cur_desc.bytes,
                              PIPE_GENERATOR_UNPACK_THREADS, unpack_thread);
            /* Do not advance q_done until the full unpack group has consumed the slot. */
            barrier_sync_subset(PIPE_GENERATOR_UNPACK_BARRIER_ID, PIPE_GENERATOR_UNPACK_THREADS);
            if (is_unpack_leader) {
              __threadfence_block();
              q_done = (unsigned int)(chunk + 1);
            }
          }
        }
        __syncthreads();

        if (active) {
          source_gin.flush(ncclCoopCta());
          if (call.splitComm) {
            ginB.flush(ncclCoopCta());
          }
          __syncthreads();
        }
      }
    }
  }

  if constexpr (DoLsaSources) {
    if (run_lsa_sources) {
      uint32_t tma_consumer_phase[PIPE_GENERATOR_QUEUE_DEPTH];
#pragma unroll
      for (int i = 0; i < PIPE_GENERATOR_QUEUE_DEPTH; i++) {
        tma_consumer_phase[i] = 0;
      }
      for (int s = 0; s < params->numLsaSources; s++) {
        const StagingPipePeerEdge& source = pipe_plan->lsaSources[channel_id][s];
        const bool active = pipe_edge_active(source);
        const size_t chunkSize = source.logicalChunkSize;
        const PipeChunkRange chunks = pipe_chunk_range(source);
        const int leader_local_rank = source.peerLocalRank;

        if (threadIdx.x == 0) {
          q_tail = 0;
          q_done = 0;
          q_done_chunks = 0;
        }
        __syncthreads();

        if (active && chunks.count > 0 && is_control && lane_id == 0) {
          /* Generator LSA stage A: control warp lane 0.
           *
           * Poll the local NVLink-domain leader FIFO, issue TMA loads from the
           * leader's staging window into shared-memory tiles, enqueue each tile
           * for unpack, and release LSA credits chunk by chunk. */
          StagingFlowCtrl poll_fc = source.fc;
          StagingFlowCtrl release_fc = source.fc;
          const uint64_t head_base = staging_cursor_load(lsa_region, source.fc.cursorHeadOffset);
          poll_fc.lsaTailBase = 0;
          poll_fc.lsaHeadBase = 0;
          poll_fc.shadowTail = 0;
          poll_fc.lastTailVal = head_base;
          poll_fc.localHeadVal = head_base;
          release_fc.lsaTailBase = 0;
          release_fc.lsaHeadBase = 0;
          release_fc.shadowTail = 0;
          release_fc.lastTailVal = head_base;
          release_fc.localHeadVal = head_base;
          const uint64_t release_head_base = release_fc.localHeadVal;

          size_t chunks_done = 0;
          while (chunks_done < chunks.count) {
            size_t first_recv_offset = 0;
            int num_new = 0;
            PIPE_SPIN_INIT(lsa_poll_wait);
            while ((num_new = staging_poll(lsa_region, poll_fc, &first_recv_offset)) == 0) {
              PIPE_SYNC_TIMEOUT(lsa_poll_wait, params, trace_epoch, channel_id, s, chunks_done,
                                poll_fc.lastTailVal + 1,
                                staging_cursor_load(lsa_region, poll_fc.localTailOffset) - poll_fc.lsaTailBase);
            }
            int batch = pipe_clip_batch(num_new, chunks.count - chunks_done);
            int first_slot = pipe_slot_from_offset(poll_fc, first_recv_offset);

            for (int bi = 0; bi < batch; bi++) {
              size_t logical_chunk = chunks_done + (size_t)bi;
              int this_slot = (first_slot + bi) % poll_fc.peerNumSlots;
              size_t staging_slot_offset = poll_fc.peerDataOffset + (size_t)this_slot * poll_fc.peerChunkSize;
              size_t global_chunk = chunks.start + logical_chunk;
              size_t byte_start = global_chunk * chunkSize;
              size_t remaining = source.totalBytes - byte_start;
              size_t this_bytes = pipe_min_size(chunkSize, remaining);
              const char* remote_src =
                (const char*)ncclGetLsaPointer(lsa_region.window, staging_slot_offset, leader_local_rank);

              size_t tile_offset = 0;
              while (tile_offset < this_bytes) {
                PIPE_SPIN_INIT(lsa_queue_wait);
                while ((q_tail - q_done) >= (unsigned int)PIPE_GENERATOR_QUEUE_DEPTH) {
                  PIPE_SYNC_TIMEOUT(lsa_queue_wait, params, trace_epoch, channel_id, source.peerWorldRank,
                                    logical_chunk, q_tail - PIPE_GENERATOR_QUEUE_DEPTH + 1, q_done);
                }
                unsigned int q_idx = q_tail % (unsigned int)PIPE_GENERATOR_QUEUE_DEPTH;
                size_t tile_bytes = pipe_min_size((size_t)TmaTileSize, this_bytes - tile_offset);
                char* tma_tile_buf = tma_tile_base + (size_t)q_idx * (size_t)TmaTileSize;
                /* cp.async.bulk needs a multiple of 16 bytes; with a shorter tail the
                 * mbarrier never completes.  Load the padded size: the staging slot is
                 * chunk-sized, and the unpack reads only tile_bytes. */
                const int tma_bytes = (int)((tile_bytes + 15) & ~(size_t)15);
                staging_tma_mbarrier_expect(&tma_mbar[q_idx], tma_bytes);
                staging_tma_load(tma_tile_buf, remote_src + tile_offset, &tma_mbar[q_idx], tma_bytes);

                q_desc[q_idx].staging_offset = 0;
                q_desc[q_idx].byte_start = byte_start + tile_offset;
                q_desc[q_idx].bytes = tile_bytes;
                q_desc[q_idx].tile_buf = (int)q_idx;
                q_desc[q_idx].last_tile = (tile_offset + tile_bytes == this_bytes) ? 1 : 0;
                __threadfence_block();
                q_tail = q_tail + 1;
                tile_offset += tile_bytes;
              }
            }
            chunks_done += (size_t)batch;

            PIPE_SPIN_INIT(lsa_done_chunks_wait);
            while (q_done_chunks < (unsigned int)chunks_done) {
              PIPE_SYNC_TIMEOUT(lsa_done_chunks_wait, params, trace_epoch, channel_id, source.peerWorldRank,
                                chunks_done, chunks_done, q_done_chunks);
            }
            release_fc.lastTailVal = release_head_base + (uint64_t)chunks_done;
            lsa_release_flush(lsa_region, release_fc);
            staging_cursor_store(lsa_region, release_fc.cursorHeadOffset, release_fc.localHeadVal);
          }
        }

        if (active && chunks.count > 0 && is_unpack) {
          /* Generator LSA stage B: unpack warps.
           *
           * The unpack leader waits for the queued TMA tile mbarrier, then the
           * unpack group scatters the tile to destination memory.  The last tile
           * of each logical chunk advances q_done_chunks for LSA credit release. */
          size_t tiles_done = 0;
          size_t chunks_done = 0;
          while (chunks_done < chunks.count) {
            if (is_unpack_leader) {
              PIPE_SPIN_INIT(lsa_unpack_queue_wait);
              while (q_tail <= (unsigned int)tiles_done) {
                PIPE_SYNC_TIMEOUT(lsa_unpack_queue_wait, params, trace_epoch, channel_id, source.peerWorldRank,
                                  tiles_done, tiles_done + 1, q_tail);
              }
              __threadfence_block();
              cur_desc = q_desc[tiles_done % (unsigned int)PIPE_GENERATOR_QUEUE_DEPTH];
#ifdef STAGING_PIPE_SYNC_TIMEOUT_DEBUG
              PIPE_SPIN_INIT(lsa_tma_mbarrier_wait);
              while (!staging_tma_mbarrier_try_wait(&tma_mbar[cur_desc.tile_buf],
                                                    tma_consumer_phase[cur_desc.tile_buf])) {
                PIPE_SYNC_TIMEOUT(lsa_tma_mbarrier_wait, params, trace_epoch, channel_id, source.peerWorldRank,
                                  tiles_done, tma_consumer_phase[cur_desc.tile_buf],
                                  *reinterpret_cast<volatile uint64_t*>(&tma_mbar[cur_desc.tile_buf]));
              }
              tma_consumer_phase[cur_desc.tile_buf] ^= 1;
#else
              staging_tma_mbarrier_wait(&tma_mbar[cur_desc.tile_buf], tma_consumer_phase[cur_desc.tile_buf]);
#endif
            }
            /* Publish the completed TMA tile descriptor to every unpack warp. */
            barrier_sync_subset(PIPE_GENERATOR_UNPACK_BARRIER_ID, PIPE_GENERATOR_UNPACK_THREADS);
            const char* tile_src = tma_tile_base + (size_t)cur_desc.tile_buf * (size_t)TmaTileSize;
            const StagingPipeCopyLayout& source_layout =
              pipe_plan->lsaSourceLayouts[channel_id][source.copyLayoutIndex];
            pipe_unpack_chunk(user_dst, tile_src, source_layout, cur_desc.byte_start, cur_desc.bytes,
                              PIPE_GENERATOR_UNPACK_THREADS, unpack_thread);
            const int last_tile = cur_desc.last_tile;
            /* Keep the tile queued until the full unpack group has consumed it. */
            barrier_sync_subset(PIPE_GENERATOR_UNPACK_BARRIER_ID, PIPE_GENERATOR_UNPACK_THREADS);
            tiles_done++;
            if (last_tile) {
              chunks_done++;
            }
            if (is_unpack_leader) {
              if (last_tile) {
                q_done_chunks = (unsigned int)chunks_done;
              }
              __threadfence_block();
              q_done = (unsigned int)tiles_done;
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

// ============================================================================
// Host-Side Launch Wrapper — Pipe
// ============================================================================

static int pipeCountGeneratorRdmaLaunchChannels(const StagingKernelParams* params) {
  if (params == nullptr) return 0;
  int count = 0;
  for (int ch = 0; ch < params->numChannels && ch < STAGING_MAX_CHANNELS; ch++) {
    bool active = false;
    for (int s = 0; s < params->numRdmaSources && s < MAX_SOURCES; s++) {
      active = active || params->rdmaSources[ch][s].active;
    }
    count += active ? 1 : 0;
  }
  return count;
}

static int pipeCountGeneratorLsaLaunchChannels(const StagingKernelParams* params) {
  if (params == nullptr) return 0;
  int count = 0;
  for (int ch = 0; ch < params->numChannels && ch < STAGING_MAX_CHANNELS; ch++) {
    bool active = false;
    for (int s = 0; s < params->numLsaSources && s < MAX_SOURCES; s++) {
      active = active || params->lsaSources[ch][s].active;
    }
    count += active ? 1 : 0;
  }
  return count;
}

static int pipeCountTrainerRdmaLaunchChannels(const StagingKernelParams* params) {
  if (params == nullptr) return 0;
  int count = 0;
  for (int ch = 0; ch < params->numChannels && ch < STAGING_MAX_CHANNELS; ch++) {
    bool active = false;
    for (int t = 0; t < params->numRdmaTargets && t < MAX_TARGETS; t++) {
      active = active || params->rdmaTargets[ch][t].active;
    }
    count += active ? 1 : 0;
  }
  return count;
}

static int pipeCountGeneratorLaunchChannels(const StagingKernelParams* params) {
  if (params == nullptr) return 0;
  const int rdmaChannels = pipeCountGeneratorRdmaLaunchChannels(params);
  const int lsaChannels = pipeCountGeneratorLsaLaunchChannels(params);
  if (params->numRdmaSources > 0 && params->numLsaSources > 0) {
    return rdmaChannels + lsaChannels;
  }
  if (params->numRdmaSources > 0) {
    return rdmaChannels;
  }
  if (params->numLsaSources > 0) {
    return lsaChannels;
  }
  return 0;
}

template <int TmaTileSize, bool DoRdmaSources, bool DoRingForward, bool DoLsaForward, bool DoLsaSources,
          bool SplitMixedRoleCtas = false>
static ncclResult_t launchPipeGeneratorSpecialized(const StagingKernelParams* hostParams,
                                                   StagingKernelParams* devParams, StagingPipeDevicePlan* devPipePlan,
                                                   const StagingPipeCallParams& call, struct ncclDevComm* devCommA,
                                                   struct ncclDevComm* devCommB, int numCtas, cudaStream_t stream) {
  constexpr int dynamic_smem_bytes = DoLsaSources ? (PIPE_GENERATOR_QUEUE_DEPTH * TmaTileSize + 127) : 0;
  int grid = numCtas;
  if constexpr (SplitMixedRoleCtas) {
    grid = pipeCountGeneratorRdmaLaunchChannels(hostParams) + pipeCountGeneratorLsaLaunchChannels(hostParams);
  } else if constexpr (DoRdmaSources && !DoLsaSources) {
    grid = pipeCountGeneratorRdmaLaunchChannels(hostParams);
  } else if constexpr (!DoRdmaSources && DoLsaSources) {
    grid = pipeCountGeneratorLsaLaunchChannels(hostParams);
  }
  if (grid <= 0) {
    return ncclSuccess;
  }
  /* The template booleans below strip unused generator stages from the kernel
   * body.  That keeps RDMA-only, LSA-only, and mixed generator roles on the
   * same data-flow path without carrying every stage's live state. */
  if constexpr (DoLsaSources) {
    NCCL_M2N_CUDACHECK(
      cudaFuncSetAttribute(StagingReshardKernel_PipeGenerator<TmaTileSize, DoRdmaSources, DoRingForward, DoLsaForward,
                                                              DoLsaSources, SplitMixedRoleCtas>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_bytes));
  }
#if NCCLM2N_PIPE_HAS_LAUNCH_COMPLETION_EVENT
  if (pipeLaunchOrderingEnabled()) {
    NCCL_M2N_CHECK(
      (launchPipeKernelOrdered(dim3(grid), dim3(PIPE_GENERATOR_THREADS), dynamic_smem_bytes, stream,
                               StagingReshardKernel_PipeGenerator<TmaTileSize, DoRdmaSources, DoRingForward,
                                                                  DoLsaForward, DoLsaSources, SplitMixedRoleCtas>,
                               devParams, devPipePlan, call, *devCommA, *devCommB)));
  } else
#endif
  {
    StagingReshardKernel_PipeGenerator<TmaTileSize, DoRdmaSources, DoRingForward, DoLsaForward, DoLsaSources,
                                       SplitMixedRoleCtas>
      <<<grid, PIPE_GENERATOR_THREADS, dynamic_smem_bytes, stream>>>(devParams, devPipePlan, call, *devCommA,
                                                                     *devCommB);
  }
  NCCL_M2N_CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <int TmaTileSize>
static ncclResult_t launchPipeGeneratorForRole(const StagingKernelParams* hostParams, StagingKernelParams* devParams,
                                               StagingPipeDevicePlan* devPipePlan, const StagingPipeCallParams& call,
                                               struct ncclDevComm* devCommA, struct ncclDevComm* devCommB, int numCtas,
                                               cudaStream_t stream) {
  const bool has_rdma_source = hostParams->numRdmaSources > 0;
  const bool has_lsa_source = hostParams->numLsaSources > 0;
  const bool has_ring_forward = hostParams->numRingTargets > 0;
  const bool has_lsa_forward = hostParams->numLsaTargets > 0 && hostParams->numLsaFollowers > 0;

  /* Pick the smallest generator kernel variant that can express this rank's
   * graph.  The fallback keeps all stages compiled in only for unexpected
   * combinations. */
  if (has_rdma_source && !has_lsa_source && !has_ring_forward && !has_lsa_forward) {
    return launchPipeGeneratorSpecialized<TmaTileSize, true, false, false, false>(
      hostParams, devParams, devPipePlan, call, devCommA, devCommB, numCtas, stream);
  }
  if (has_rdma_source && !has_lsa_source && has_ring_forward && !has_lsa_forward) {
    return launchPipeGeneratorSpecialized<TmaTileSize, true, true, false, false>(
      hostParams, devParams, devPipePlan, call, devCommA, devCommB, numCtas, stream);
  }
  if (has_rdma_source && !has_lsa_source && !has_ring_forward && has_lsa_forward) {
    return launchPipeGeneratorSpecialized<TmaTileSize, true, false, true, false>(
      hostParams, devParams, devPipePlan, call, devCommA, devCommB, numCtas, stream);
  }
  if (has_rdma_source && has_lsa_source && !has_ring_forward && !has_lsa_forward) {
    return launchPipeGeneratorSpecialized<TmaTileSize, true, false, false, true, true>(
      hostParams, devParams, devPipePlan, call, devCommA, devCommB, numCtas, stream);
  }
  if (has_rdma_source && has_lsa_source && !has_ring_forward && has_lsa_forward) {
    return launchPipeGeneratorSpecialized<TmaTileSize, true, false, true, true, true>(
      hostParams, devParams, devPipePlan, call, devCommA, devCommB, numCtas, stream);
  }
  if (has_rdma_source && has_lsa_source && has_ring_forward && !has_lsa_forward) {
    return launchPipeGeneratorSpecialized<TmaTileSize, true, true, false, true, true>(
      hostParams, devParams, devPipePlan, call, devCommA, devCommB, numCtas, stream);
  }
  if (has_rdma_source && has_lsa_source && has_ring_forward && has_lsa_forward) {
    return launchPipeGeneratorSpecialized<TmaTileSize, true, true, true, true, true>(
      hostParams, devParams, devPipePlan, call, devCommA, devCommB, numCtas, stream);
  }
  if (!has_rdma_source && has_lsa_source) {
    return launchPipeGeneratorSpecialized<TmaTileSize, false, false, false, true>(
      hostParams, devParams, devPipePlan, call, devCommA, devCommB, numCtas, stream);
  }

  return launchPipeGeneratorSpecialized<TmaTileSize, true, true, true, true>(hostParams, devParams, devPipePlan, call,
                                                                             devCommA, devCommB, numCtas, stream);
}

ncclResult_t launchStagingReshardPipeSplit(const StagingKernelParams* hostParams, const StagingPipeCallParams* call,
                                           StagingKernelParams* devParams, StagingPipeDevicePlan* devPipePlan,
                                           struct ncclDevComm* devCommA, struct ncclDevComm* devCommB, int numCtas,
                                           int tmaTileSize, cudaStream_t stream) {
  if (hostParams == nullptr || call == nullptr || devParams == nullptr || devPipePlan == nullptr) {
    return ncclInvalidArgument;
  }
  if (devCommA == nullptr || devCommB == nullptr) {
    return ncclInvalidArgument;
  }

  const bool launch_trainer = hostParams->isSource && hostParams->numRdmaTargets > 0;
  const bool launch_generator = hostParams->isDest && (hostParams->numRdmaSources > 0 || hostParams->numLsaSources > 0);
  const int trainer_grid = launch_trainer ? pipeCountTrainerRdmaLaunchChannels(hostParams) : 0;
  const int generator_grid = launch_generator ? pipeCountGeneratorLaunchChannels(hostParams) : 0;

  if (launch_trainer && trainer_grid > 0) {
#if NCCLM2N_PIPE_HAS_LAUNCH_COMPLETION_EVENT
    if (pipeLaunchOrderingEnabled()) {
      NCCL_M2N_CHECK((launchPipeKernelOrdered(dim3(trainer_grid), dim3(PIPE_TRAINER_THREADS), 0, stream,
                                            StagingReshardKernel_PipeTrainer, devParams, devPipePlan, *call, *devCommA,
                                            *devCommB)));
    } else
#endif
    {
      StagingReshardKernel_PipeTrainer<<<trainer_grid, PIPE_TRAINER_THREADS, 0, stream>>>(devParams, devPipePlan, *call,
                                                                                          *devCommA, *devCommB);
    }
    NCCL_M2N_CUDACHECK(cudaGetLastError());
  }
  if (launch_generator && generator_grid > 0) {
    switch (tmaTileSize) {
    case 8 * 1024:
      NCCL_M2N_CHECK((launchPipeGeneratorForRole<8 * 1024>(hostParams, devParams, devPipePlan, *call, devCommA, devCommB,
                                                         numCtas, stream)));
      break;
    case 32 * 1024:
      NCCL_M2N_CHECK((launchPipeGeneratorForRole<32 * 1024>(hostParams, devParams, devPipePlan, *call, devCommA, devCommB,
                                                          numCtas, stream)));
      break;
    case 64 * 1024:
      NCCL_M2N_CHECK((launchPipeGeneratorForRole<64 * 1024>(hostParams, devParams, devPipePlan, *call, devCommA, devCommB,
                                                          numCtas, stream)));
      break;
    case 16 * 1024:
      NCCL_M2N_CHECK((launchPipeGeneratorForRole<16 * 1024>(hostParams, devParams, devPipePlan, *call, devCommA, devCommB,
                                                          numCtas, stream)));
      break;
    default:
      NCCL_M2N_CHECK((launchPipeGeneratorForRole<PIPE_GENERATOR_DEFAULT_TMA_TILE_SIZE>(
        hostParams, devParams, devPipePlan, *call, devCommA, devCommB, numCtas, stream)));
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t launchStagingReshardPipe(const StagingKernelParams* hostParams, const StagingPipeCallParams* call,
                                      StagingKernelParams* devParams, StagingPipeDevicePlan* devPipePlan,
                                      struct ncclDevComm* devComm, int numCtas, int tmaTileSize, cudaStream_t stream) {
  return launchStagingReshardPipeSplit(hostParams, call, devParams, devPipePlan, devComm, devComm, numCtas, tmaTileSize,
                                       stream);
}
