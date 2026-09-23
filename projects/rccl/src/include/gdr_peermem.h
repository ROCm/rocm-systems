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

// Tries to register device memory on IB device `dev`, returning 1 on success and 0 otherwise.
typedef int (*ncclIbPeerMemRegProbe)(int dev, void* ctx);

// Returns 1 only if `probe` succeeds on every one of the `nDevs` devices, stopping at the first
// failure, and 0 when there are no devices. Registration working on one NIC says nothing about a
// different driver in a mixed-HCA host, so one passing device must not enable peermem for all.
//
// The probe is a parameter so unit tests can drive the decision without a GPU or a NIC.
int ncclIbProbePeerMemAllDevs(int nDevs, ncclIbPeerMemRegProbe probe, void* ctx);

#endif  // NCCL_GDR_PEERMEM_H_
