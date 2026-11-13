/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel_fakes.h"

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

int64_t rcclParamPxnOptQpUsage() { FailLoudUnfaked("channel_fakes", "rcclParamPxnOptQpUsage"); }

void ResetChannelFakes() {
  g_initChannelResult = ncclSuccess;
  g_initChannelLastId = -1;
  g_initCollnetChannel = DefaultInitCollnetChannel;
}
