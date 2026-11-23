/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// NET_IB_GIN_CC_PATH for IB-MSEG: route the fork's spellings through gin-test.cc's seam window.
#define ncclNetIbMseg ncclNetIb
#define IbMsegInitDevices ncclIbInitDevices
#define IbMsegFinalizeDevices ncclIbFinalizeDevices
#define IbMsegGdrSupport ncclIbGdrSupport
#define IbMsegPeerMemSupport ncclIbPeerMemSupport
#define IbMsegDmaBufSupport ncclIbDmaBufSupport
#define IbMsegConnectImpl ncclIbConnectImpl
#define IbMsegAcceptImpl ncclIbAcceptImpl
#define IbMsegSetTrafficClass ncclIbSetTrafficClass
#define IbMsegCreateFlushQp ncclIbCreateFlushQp
#define IbMsegRegMrDmaBufInternal ncclIbRegMrDmaBufInternal

namespace {
// The develop scenes hand gin.cc an int as the listen comm, so the fork's GIN mark is a no-op unless a test hooks it.
std::function<void(void*, void*)> g_ginMarkListen = [](void*, void*) {};
void TestGinMarkListen(void* handle, void* listenComm) { g_ginMarkListen(handle, listenComm); }
}  // namespace
#define rcclIbMarkGinListen TestGinMarkListen

#include NET_IB_MSEG_GIN_CC_PATH

#undef rcclIbMarkGinListen
#undef IbMsegRegMrDmaBufInternal
#undef IbMsegCreateFlushQp
#undef IbMsegSetTrafficClass
#undef IbMsegAcceptImpl
#undef IbMsegConnectImpl
#undef IbMsegDmaBufSupport
#undef IbMsegPeerMemSupport
#undef IbMsegGdrSupport
#undef IbMsegFinalizeDevices
#undef IbMsegInitDevices
#undef ncclNetIbMseg
