/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Query interface of the KernelCh recorder profiler plugin, shared by the plugin
// and ProfilerKernelChMPITests. The test reaches these functions with dlopen on
// the same path RCCL loaded, so it reads the very records RCCL produced.

#ifndef RCCL_TEST_PROFILER_KERNELCH_RECORDER_H_
#define RCCL_TEST_PROFILER_KERNELCH_RECORDER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One recorded event, copied out of the plugin. Only the event types the tests
// reason about are recorded: Coll, P2p, KernelCh and KernelPhase.
typedef struct {
  uint64_t type;        // ncclProfileColl, ncclProfileP2p, ncclProfileKernelCh, ...
  uint64_t commId;      // commHash the owning communicator passed to init()
  int rank;
  int64_t parentIndex;  // snapshot index of the parent record, or -1
  uint64_t order;       // global start order across every recorded event
  long startTid;        // thread that delivered startEvent
  int stopEvents;       // stopEvent calls received for this handle
  int stopStates;       // KernelCh/KernelPhase stop states received
  // Coll / P2p
  char func[32];
  uint64_t seqNumber;
  size_t count;         // element count of the collective or send/recv
  int nChannels;        // the per-channel KernelCh count the core advertises
  int isSymColl;
  int peer;
  // KernelCh / KernelPhase
  int channelId;
  int phaseId;
  uint64_t startTimer;  // device timer at start
  uint64_t stopTimer;   // device timer at stop
} RcclKchRecord;

// Drop every record and counter. Only call while RCCL holds no event handle, that
// is with no communicator alive: the handles point into the storage this releases.
// Communicator contexts survive a reset.
void rcclKchRecorderReset(void);

// Copy up to cap records into out, in start order, and return the total number
// recorded (which may exceed cap).
size_t rcclKchRecorderSnapshot(RcclKchRecord* out, size_t cap);

// init() calls received since the last reset. Zero after a communicator was
// created means RCCL never loaded this plugin.
uint64_t rcclKchRecorderInitCount(void);

// ProxyOp startEvent calls since the last reset. Zero on a path that never
// involves the proxy thread.
uint64_t rcclKchRecorderProxyOpCount(void);

// Copy up to cap distinct threads that delivered a ProxyOp startEvent into out,
// and return how many there are.
size_t rcclKchRecorderProxyThreads(long* out, size_t cap);

// Calls the plugin could not reconcile: a stop or state for a handle it never
// issued or already stopped, or a KernelCh stop with no start.
uint64_t rcclKchRecorderAnomalies(void);

#ifdef __cplusplus
}
#endif

#endif  // RCCL_TEST_PROFILER_KERNELCH_RECORDER_H_
