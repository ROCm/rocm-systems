/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "transport.h"
#include "proxy.h"
#include "profiler.h"
#include "device.h"

static ncclResult_t profilerProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->shared = 0;
  return ncclSuccess;
}

// The following ncclProxySubArgs are overloaded by the profiler progress function:
// - base       : is set to the current value of workCounter[channelId]
// - posted     : is set to sub->nsteps to indicate that the profiler has started the event
// - transmitted: is set to sub->nsteps to indicate that the profiler has stopped the event
static ncclResult_t profilerProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      sub->base = sub->workCounter;
      sub->posted = sub->transmitted = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    // RCCL: teardown drain — skip kernel-channel events for channels whose GPU counters
    // were never written; without this, stopEvent dereferences an uninitialised collInfo.
    int stopping = proxyState->progressState.stop;
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct ncclDevProfiler* workStarted = (struct ncclDevProfiler *)sub->sendbuff;
      struct ncclDevProfiler* workCompleted = (struct ncclDevProfiler *)sub->recvbuff;
      if (sub->posted < sub->nsteps) {
        if (sub->base <= workStarted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
          ncclProfilerStartKernelChEvent(args, s, workStarted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
          sub->posted = sub->nsteps;
          continue; // allow events on every channel to start
        }
        if (!stopping) continue; // GPU counter not yet written; retry next call
        // Teardown in progress and GPU start counter never written for this channel.
        // Skip both start and stop events — calling stop without a prior start would
        // dereference an uninitialised kernelChInfo->collInfo.
        sub->posted = sub->transmitted = sub->nsteps;
        args->done++;
        continue;
      }
      if (sub->transmitted < sub->nsteps) {
        if (sub->base <= workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
          ncclProfilerStopKernelChEvent(args, s, workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
          sub->transmitted = sub->nsteps;
          args->done++;
        } else if (stopping) {
          // Start was fired but stop counter never written; skip stop event and drain.
          sub->transmitted = sub->nsteps;
          args->done++;
        }
        // else: not ready and not stopping — retry next call
      }
    }
    if (args->done == args->nsubs) args->state = ncclProxyOpNone;
  }
  return ncclSuccess;
}

struct ncclTransport profilerTransport = {
  "Prof",
  NULL,
  { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
  { NULL, NULL, NULL, NULL, NULL, profilerProxyConnect, NULL, profilerProxyProgress, NULL, NULL }
};
