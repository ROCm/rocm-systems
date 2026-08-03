/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

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

#endif  // NCCL_GDR_PEERMEM_H_
