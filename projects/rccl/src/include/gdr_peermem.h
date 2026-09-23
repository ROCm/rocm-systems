/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GDR_PEERMEM_H_
#define NCCL_GDR_PEERMEM_H_

// Scan a NULL-terminated list of `memory_peers` base directories for a registered
// peer-memory client, returning 1 if one is found and 0 otherwise. Shared by the
// net_ib and net_ib_cast transports.
//
// Registering a client always creates a subdirectory named after that client holding a
// `version` attribute, so any subdirectory carrying a `version` is a client whatever its
// name; matching on `amdkfd` alone would miss every other client.
//
// The base-path list is a parameter rather than hardcoded so unit tests can aim the scan
// at a mock sysfs tree.
int ncclIbScanPeerMemClients(const char* const* basePaths);

// Scan the standard sysfs `memory_peers` locations for a registered peer-memory client,
// returning 1 if one is found and 0 otherwise. This is what the transports call; it owns
// the base-path list so net_ib and net_ib_cast do not each carry a copy of it.
int ncclIbScanDefaultPeerMemClients(void);

// Runtime fallback for when the sysfs scan finds nothing: attempts a real GPU memory
// registration against `context`. Returns 1/0 (definitive, cache it) or -1 if the test
// itself couldn't run (don't cache -- retry later).
int ncclIbProbeGdrSupport(struct ibv_context* context, int relaxedOrderingEnabled);

#endif  // NCCL_GDR_PEERMEM_H_
