/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/transport/p2p_prime.cu launcher; defaults to its Emulate* twin.

#ifndef RCCL_TEST_HOST_FAKES_TRANSPORT_P2P_PRIME_FAKES_H_
#define RCCL_TEST_HOST_FAKES_TRANSPORT_P2P_PRIME_FAKES_H_

#include <cstddef>
#include <functional>

#include "p2p.h"

// No GPU in the host tests, so there are no cache lines to prime: the emulator only reports success.
ncclResult_t EmulateP2pPrimeBuffer(const void* buff, size_t size, hipStream_t stream);

extern std::function<ncclResult_t(const void*, size_t, hipStream_t)> g_ncclP2pPrimeBuffer;

void ResetTransportP2pPrimeFakes();

#endif  // RCCL_TEST_HOST_FAKES_TRANSPORT_P2P_PRIME_FAKES_H_
