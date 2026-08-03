/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GDR_PEERMEM_H_
#define NCCL_GDR_PEERMEM_H_

// Scan a NULL-terminated list of `memory_peers` base directories for a registered
// peer-memory client. A real client registers itself as a named subdirectory (e.g.
// `amdkfd`), so the presence of any such subdirectory means a client is loaded.
//
// The base-path list is passed in (rather than hardcoded) so the detection can be
// exercised against a mock sysfs tree in unit tests. Returns 1 if a client
// subdirectory is found in any of the given paths, 0 otherwise.
int ncclIbScanPeerMemClients(const char* const* basePaths);

#endif  // NCCL_GDR_PEERMEM_H_
