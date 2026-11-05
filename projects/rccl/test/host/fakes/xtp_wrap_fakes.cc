/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "xtp_wrap_fakes.h"

#include "comm.h"
#include "xtp_wrap.h"

int g_rcclXtpClassAIntCalls = 0;
int g_rcclXtpRefreshCommCalls = 0;
int g_rcclXtpCommFreeCalls = 0;
struct ncclComm* g_rcclXtpRefreshLastComm = nullptr;
int g_rcclXtpRefreshLastNNodes = 0;
int g_rcclXtpRefreshLastLocalRanks = 0;
int g_rcclXtpRefreshLastNvlDomains = 0;
int g_rcclXtpRefreshLastMaxRanksPerNvlDomain = 0;

bool rcclXtpClassAInt(struct ncclComm*, const char*, int64_t*) {
  ++g_rcclXtpClassAIntCalls;
  return false;
}

void rcclXtpRefreshComm(struct ncclComm* comm) {
  ++g_rcclXtpRefreshCommCalls;
  g_rcclXtpRefreshLastComm = comm;
  if (comm == nullptr) {
    g_rcclXtpRefreshLastNNodes = 0;
    g_rcclXtpRefreshLastLocalRanks = 0;
    g_rcclXtpRefreshLastNvlDomains = 0;
    g_rcclXtpRefreshLastMaxRanksPerNvlDomain = 0;
    return;
  }
  g_rcclXtpRefreshLastNNodes = comm->nNodes;
  g_rcclXtpRefreshLastLocalRanks = comm->localRanks;
  g_rcclXtpRefreshLastNvlDomains = comm->nvlDomainInfo.nNvlDomains;
  g_rcclXtpRefreshLastMaxRanksPerNvlDomain = comm->nvlDomainInfo.maxRanksPerNvlDomain;
}

void rcclXtpCommFree(struct ncclComm*) { ++g_rcclXtpCommFreeCalls; }

void ResetXtpWrapFakes() {
  g_rcclXtpClassAIntCalls = 0;
  g_rcclXtpRefreshCommCalls = 0;
  g_rcclXtpCommFreeCalls = 0;
  g_rcclXtpRefreshLastComm = nullptr;
  g_rcclXtpRefreshLastNNodes = 0;
  g_rcclXtpRefreshLastLocalRanks = 0;
  g_rcclXtpRefreshLastNvlDomains = 0;
  g_rcclXtpRefreshLastMaxRanksPerNvlDomain = 0;
}
