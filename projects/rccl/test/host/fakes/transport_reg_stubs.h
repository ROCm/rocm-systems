/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Buffer registration seams defined in transport_stubs.cc (src/transport/net.cc,
// coll_net.cc, nvls.cc). Kept out of transport_stubs.h because their signatures
// need comm.h, which init_fakes.h must not pull in ahead of init-test.cc's shims.
// Defaults fail loudly.

#ifndef RCCL_TEST_HOST_TRANSPORT_REG_STUBS_H_
#define RCCL_TEST_HOST_TRANSPORT_REG_STUBS_H_

#include <cstddef>
#include <functional>

#include "comm.h"
#include "nccl.h"

using ncclRegCleanupQueue = struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>;

extern std::function<ncclResult_t(ncclComm*, const void*, size_t, struct ncclConnector**, int, int*, void**)>
  g_ncclNetLocalRegisterBuffer;
extern std::function<ncclResult_t(ncclComm*, const void*, size_t, struct ncclConnector**, int, int*, void**,
                                  ncclRegCleanupQueue*, int*)>
  g_ncclNetGraphRegisterBuffer;

extern std::function<ncclResult_t(struct ncclComm*, const void*, size_t, int /*type*/, int*, void**)>
  g_ncclCollnetLocalRegisterBuffer;
extern std::function<ncclResult_t(struct ncclComm*, const void*, size_t, int /*type*/, int*, void**,
                                  ncclRegCleanupQueue*, int*)>
  g_ncclCollnetGraphRegisterBuffer;

extern std::function<ncclResult_t(struct ncclComm*, const void*, void*, size_t, size_t, int*, void**, void**)>
  g_ncclNvlsLocalRegisterBuffer;
extern std::function<ncclResult_t(struct ncclComm*, const void*, void*, size_t, size_t, int*, void**, void**,
                                  ncclRegCleanupQueue*, int*)>
  g_ncclNvlsGraphRegisterBuffer;
extern std::function<ncclResult_t(struct ncclComm*, ncclFunc_t, int* /*recChannels*/)> g_ncclNvlsRegResourcesQuery;

#endif  // RCCL_TEST_HOST_TRANSPORT_REG_STUBS_H_
