/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_KERNEL_TIMING_H_
#define NCCL_KERNEL_TIMING_H_

#include <stdint.h>

#include "debug.h"
#include "nccl.h"

struct ncclComm;
struct ncclKernelPlan;
struct ncclKernelTimingCtx;

/* One RCCL kernel dispatch, measured from the dispatch packet itself rather
 * than from stream markers. Timestamps are CLOCK_BOOTTIME nanoseconds -- the
 * same domain rocprof uses -- so records merge with an external trace with no
 * calibration, and are directly comparable across the GPUs of a node.
 *
 * Consumers may copy this layout rather than include this header; it is
 * append-only. */
typedef struct {
  uint64_t startNs;
  uint64_t endNs;
  uint64_t seq; /* per-communicator dispatch counter */
  uint64_t commHash;
  uint64_t count;     /* element count of the plan's first collective */
  uint32_t func;      /* ncclFunc_t */
  uint32_t datatype;  /* ncclDataType_t */
  uint32_t nChannels; /* grid.x */
  uint32_t nThreads;  /* block.x */
  int32_t rank;
  uint32_t nColls; /* collectives aggregated into this dispatch */
} ncclKernelTimingRecord;

/* Copies out and releases up to `max` completed records. `got` receives the
 * number written; `dropped` (optional) receives the running count of dispatches
 * that could not be reported, including queue/ring overflow and unreadable
 * timestamps. Returns ncclInvalidUsage if timing is not active on this
 * communicator.
 *
 * Not part of the versioned RCCL API: resolve it with dlsym if you want a
 * binary that also runs against builds without it. */
extern "C" ncclResult_t ncclKernelTimingDrain(ncclComm_t comm, ncclKernelTimingRecord* out, int max, int* got,
                                              uint64_t* dropped);

/* Internal */
bool ncclKernelTimingEnabled();
ncclResult_t ncclKernelTimingCommInit(struct ncclComm* comm);
ncclResult_t ncclKernelTimingCommFree(struct ncclComm* comm);

/* Reserves a slot and returns the event to attach as the dispatch's stop event,
 * or nullptr when this launch will not be timed. Only the ticket claim happens
 * here, so this stays on the critical path between getting the event and
 * ringing the doorbell for as little time as possible; everything the record
 * needs to describe the dispatch is filled in later by the matching Commit
 * call, once the launch itself is safely on its way. The reservation is only
 * armed by that Commit call, so a launch that fails after reserving is
 * discarded rather than reported with another dispatch's timestamps. */
cudaEvent_t ncclKernelTimingBeginLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, uint64_t* slot);

/* The same, for a dispatch that has no kernel plan to describe it: the DDA
 * backends and a few other collectives launch their kernels themselves, so the
 * caller says what the plan would have said -- but not until the matching
 * ncclKernelTimingCommitDispatch, for the same reason as above. */
cudaEvent_t ncclKernelTimingBeginDispatch(struct ncclComm* comm, cudaStream_t stream, uint64_t* slot);

/* Fills in the record from the plan and arms it. Called after the launch
 * succeeds, while plan/its coll task are still guaranteed live: reclaiming
 * them back into their memory pools only happens later, from a host-stream
 * callback queued behind this same dispatch. */
void ncclKernelTimingCommitLaunch(struct ncclComm* comm, uint64_t slot, struct ncclKernelPlan* plan,
                                  uint32_t nChannels, uint32_t nThreads);

/* The same, for the plan-less dispatch path: the caller already has the
 * scalars in hand (they are not behind a pointer that could be recycled), so
 * this just moves the writes off the pre-launch critical section. */
void ncclKernelTimingCommitDispatch(struct ncclComm* comm, uint64_t slot, uint32_t func, uint32_t datatype,
                                    uint64_t count, uint32_t nChannels, uint32_t nThreads);

void ncclKernelTimingCancelLaunch(struct ncclComm* comm, uint64_t slot);

#if defined(__HIP_PLATFORM_AMD__)
#include <tuple>
#include <utility>

/* Launches a kernel RCCL dispatches itself, with the timing stop event attached
 * to the dispatch packet when timing is on -- the same measurement the planned
 * kernels get, for the paths that do not go through a plan. `args` is the
 * hipLaunchKernel-style array of pointers to the kernel's arguments. */
inline ncclResult_t ncclKernelTimingLaunchArgs(struct ncclComm* comm, uint32_t func, uint32_t datatype, uint64_t count,
                                               const void* kernel, dim3 grid, dim3 block, void** args, size_t smem,
                                               cudaStream_t stream) {
  uint64_t slot = 0;
  cudaEvent_t stop = ncclKernelTimingBeginDispatch(comm, stream, &slot);
  hipError_t err = stop == nullptr ? cudaLaunchKernel(kernel, grid, block, args, smem, stream) :
                                     hipExtLaunchKernel(kernel, grid, block, args, smem, stream, nullptr, stop, 0);
  if (err != hipSuccess) {
    if (stop != nullptr) ncclKernelTimingCancelLaunch(comm, slot);
    WARN("Cuda failure %d '%s'", err, hipGetErrorString(err));
    return ncclUnhandledCudaError;
  }
  if (stop != nullptr) ncclKernelTimingCommitDispatch(comm, slot, func, datatype, count, grid.x, block.x);
  return ncclSuccess;
}

namespace ncclKernelTimingDetail {
template <typename Tuple, size_t... I>
ncclResult_t launch(struct ncclComm* comm, uint32_t func, uint32_t datatype, uint64_t count, const void* kernel,
                    dim3 grid, dim3 block, size_t smem, cudaStream_t stream, cudaEvent_t stop, uint64_t slot,
                    Tuple& args, std::index_sequence<I...>) {
  void* argv[] = {(void*)&std::get<I>(args)...};
  hipError_t err = stop == nullptr ? cudaLaunchKernel(kernel, grid, block, argv, smem, stream) :
                                     hipExtLaunchKernel(kernel, grid, block, argv, smem, stream, nullptr, stop, 0);
  if (err != hipSuccess) {
    if (stop != nullptr) ncclKernelTimingCancelLaunch(comm, slot);
    WARN("Cuda failure %d '%s'", err, hipGetErrorString(err));
    return ncclUnhandledCudaError;
  }
  if (stop != nullptr) ncclKernelTimingCommitDispatch(comm, slot, func, datatype, count, grid.x, block.x);
  return ncclSuccess;
}
} // namespace ncclKernelTimingDetail

/* The same, for a kernel named directly rather than through a void pointer.
 *
 * The arguments are converted to the kernel's own parameter types before their
 * addresses are taken, so a literal that would have been promoted at a
 * <<<>>> call site cannot silently pack the wrong width here. */
template <typename... Params, typename... Args>
ncclResult_t ncclKernelTimingLaunch(struct ncclComm* comm, uint32_t func, uint32_t datatype, uint64_t count,
                                    void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                                    Args&&... args) {
  static_assert(sizeof...(Params) == sizeof...(Args), "argument count does not match the kernel");
  uint64_t slot = 0;
  cudaEvent_t stop = ncclKernelTimingBeginDispatch(comm, stream, &slot);
  std::tuple<typename std::decay<Params>::type...> params(std::forward<Args>(args)...);
  return ncclKernelTimingDetail::launch(comm, func, datatype, count, (const void*)kernel, grid, block, smem, stream,
                                        stop, slot, params, std::index_sequence_for<Params...>{});
}
#endif

#endif
