/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel_fakes.h"

#include <algorithm>

#include "channel.h"
#include "fail_loud.h"
#include "signature-drift.h"

ncclResult_t g_initChannelResult = ncclSuccess;
int g_initChannelLastId = -1;
ncclResult_t initChannel(struct ncclComm*, int channelId) {
  g_initChannelLastId = channelId;
  return g_initChannelResult;
}

static ncclResult_t DefaultInitCollnetChannel(struct ncclComm*, int, struct ncclComm*, bool) {
  FailLoudUnfaked("channel_fakes", "initCollnetChannel");
}
std::function<ncclResult_t(struct ncclComm*, int, struct ncclComm*, bool)> g_initCollnetChannel =
    DefaultInitCollnetChannel;

ASSERT_HOOK_MATCHES_PROD(g_initCollnetChannel, initCollnetChannel);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share) {
  return g_initCollnetChannel(comm, channelId, parent, share);
}

ncclResult_t freeChannel(struct ncclChannel*, int, int, int, struct ncclComm*) { return ncclSuccess; }

// NCCL 2.32.6 moved the collective channel-layout arithmetic from enqueue.cc into src/channel.cc. It is pure and
// the enqueue planner's results depend on it, so the REAL bodies are compiled here, not copies. The rest of
// channel.cc (initChannel & co.) is faked above, so the file is #included inside an anonymous namespace: every
// definition gets internal linkage, clang emits only those the forwarders below reference, and the stateful
// functions never reach the linker. channel.cc's own #includes are pulled in first so their guards keep them out of
// the namespace.
#include "hip/hip_runtime.h"
#include "gdrwrap.h"
#include "param.h"
#include "transport.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
// Renamed too: a call such as initChannel(comm, ...) would otherwise also find the global fake by ADL.
#define initChannel unusedRealInitChannel
#define initNvlsChannel unusedRealInitNvlsChannel
#define initCollnetChannel unusedRealInitCollnetChannel
#define freeChannel unusedRealFreeChannel
namespace channel_cc {
namespace {
#include CHANNEL_CC_PATH
}  // namespace
}  // namespace channel_cc
#undef initChannel
#undef initNvlsChannel
#undef initCollnetChannel
#undef freeChannel
#pragma clang diagnostic pop

int ncclClampChannels(int nChannels, int minChannels, int maxChannels) {
  return channel_cc::ncclClampChannels(nChannels, minChannels, maxChannels);
}

int ncclFuncTrafficPerByte(ncclFunc_t func, int nRanks) { return channel_cc::ncclFuncTrafficPerByte(func, nRanks); }

size_t ncclCollTrafficPerChannel(size_t trafficBytes, int nChannels) {
  return channel_cc::ncclCollTrafficPerChannel(trafficBytes, nChannels);
}

struct ncclCollChannelLayout ncclComputeCollChannelLayout(size_t count, size_t elementSize, int trafficPerByte,
                                                          size_t trafficPerChannel, size_t currentTraffic,
                                                          int channelId, int maxChannels) {
  return channel_cc::ncclComputeCollChannelLayout(count, elementSize, trafficPerByte, trafficPerChannel,
                                                  currentTraffic, channelId, maxChannels);
}

int64_t rcclParamPxnOptQpUsage() { FailLoudUnfaked("channel_fakes", "rcclParamPxnOptQpUsage"); }

void ResetChannelFakes() {
  g_initChannelResult = ncclSuccess;
  g_initChannelLastId = -1;
  g_initCollnetChannel = DefaultInitCollnetChannel;
}
