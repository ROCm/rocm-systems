/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_OS_H_
#define NCCL_OS_H_

#include "nccl.h"

#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <mutex>

#ifdef NCCL_OS_WINDOWS
#include "os/windows.h"
#else
#include "os/linux.h"
#endif

uint64_t ncclOsGetPid();
uint64_t ncclOsGetTid();
size_t ncclOsGetPageSize();
ncclResult_t ncclOsInitialize();

/* Aligned memory allocation */
void* ncclOsAlignedAlloc(size_t alignment, size_t size);
void ncclOsAlignedFree(void* ptr);

std::tm* ncclOsLocaltime(const time_t* timer, std::tm* buf);

void ncclOsSetEnv(const char* name, const char* value);

/* Socket functions */
bool ncclOsSocketIsValid(struct ncclSocket* sock);
bool ncclOsSocketDescriptorIsValid(ncclSocketDescriptor sock);
ncclResult_t ncclOsFindInterfaces(const char* prefixList, char* names, union ncclSocketAddress *addrs, int sock_family,
  int maxIfNameSize, int maxIfs, int* found);
void ncclOsPollSocket(ncclSocketDescriptor sock, int op);
ncclResult_t ncclOsSocketPollConnect(struct ncclSocket* sock);
ncclResult_t ncclOsSocketStartConnect(struct ncclSocket* sock);
ncclResult_t ncclOsSocketSetFlags(struct ncclSocket* sock);
ncclResult_t ncclOsSocketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int block, int* closed);
ncclResult_t ncclOsSocketResetFd(struct ncclSocket* sock);
void ncclOsSocketResetAccept(struct ncclSocket* sock);
ncclResult_t ncclOsSocketTryAccept(struct ncclSocket* sock);

void ncclOsSetMutexCondShared(std::mutex &mutex, std::condition_variable &cond);

/* Affinity functions */
void ncclOsCpuZero(ncclAffinity& affinity);
int ncclOsCpuCount(const ncclAffinity affinity);
void ncclOsCpuSet(ncclAffinity& affinity, int cpu);
bool ncclOsCpuIsSet(const ncclAffinity affinity, int cpu);
ncclAffinity ncclOsCpuAnd(const ncclAffinity& a, const ncclAffinity& b);
ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity);
ncclResult_t ncclOsSetAffinity(const ncclAffinity affinity);
int ncclOsGetCpu();

#endif
