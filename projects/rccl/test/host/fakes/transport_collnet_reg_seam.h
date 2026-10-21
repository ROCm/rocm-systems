/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Routes coll_reg.cc's CollNet registration calls through renamed wrappers.
// This keeps the registration hooks usable when another test TU in the same
// binary compiles the real coll_net.cc definitions.

#include "fakes/transport_reg_stubs.h"

ncclResult_t micro_ncclCollnetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* handle);
ncclResult_t micro_ncclCollnetLocalRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize,
                                                  int type, int* outRegBufUsed, void** outHandle);
ncclResult_t micro_ncclCollnetGraphRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize,
                                                  int type, int* outRegBufFlag, void** outHandle,
                                                  ncclCommCallbackQueue* cleanupQueue, int* nCleanupQueueElts);

#define ncclCollnetDeregBuffer micro_ncclCollnetDeregBuffer
#define ncclCollnetLocalRegisterBuffer micro_ncclCollnetLocalRegisterBuffer
#define ncclCollnetGraphRegisterBuffer micro_ncclCollnetGraphRegisterBuffer
