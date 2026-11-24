/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport_p2p_prime_fakes.h"

#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_ncclP2pPrimeBuffer, ncclP2pPrimeBuffer);

ncclResult_t EmulateP2pPrimeBuffer(const void*, size_t, hipStream_t) { return ncclSuccess; }

std::function<ncclResult_t(const void*, size_t, hipStream_t)> g_ncclP2pPrimeBuffer = EmulateP2pPrimeBuffer;

ncclResult_t ncclP2pPrimeBuffer(const void* buff, size_t size, hipStream_t stream) {
  return g_ncclP2pPrimeBuffer(buff, size, stream);
}

void ResetTransportP2pPrimeFakes() { g_ncclP2pPrimeBuffer = EmulateP2pPrimeBuffer; }
