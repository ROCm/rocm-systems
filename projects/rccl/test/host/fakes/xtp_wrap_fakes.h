/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for src/misc/xtp_wrap.cc. Named after the production TU. Defaults are
// fail-closed no-ops so wrap-test's rcclSetPxn still takes the built-in path.
// Refresh records the comm's topology at the call so initTransportsRank tests
// can assert it ran after initNvlDomainInfo.

#ifndef RCCL_TEST_HOST_XTP_WRAP_FAKES_H_
#define RCCL_TEST_HOST_XTP_WRAP_FAKES_H_

struct ncclComm;

extern int g_rcclXtpClassAIntCalls;
extern int g_rcclXtpRefreshCommCalls;
extern int g_rcclXtpCommFreeCalls;
extern struct ncclComm* g_rcclXtpRefreshLastComm;
extern int g_rcclXtpRefreshLastNNodes;
extern int g_rcclXtpRefreshLastLocalRanks;
extern int g_rcclXtpRefreshLastNvlDomains;
extern int g_rcclXtpRefreshLastMaxRanksPerNvlDomain;

void ResetXtpWrapFakes();

#endif  // RCCL_TEST_HOST_XTP_WRAP_FAKES_H_
