/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "socket_fakes.h"

#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_socketProgress, ncclSocketProgress);
#undef ASSERT_HOOK_MATCHES_PROD

static ncclResult_t DefaultSocketProgress(int, struct ncclSocket*, void*, int size, int* offset, int* closed) {
  *offset = size;
  if (closed) *closed = 0;
  return ncclSuccess;
}

std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, int*)> g_socketProgress =
    DefaultSocketProgress;

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed) {
  return g_socketProgress(op, sock, ptr, size, offset, closed);
}

const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, const int) {
  buf[0] = '\0';
  return buf;
}

void ResetSocketFakes() { g_socketProgress = DefaultSocketProgress; }
