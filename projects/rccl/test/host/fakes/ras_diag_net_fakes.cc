/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "fakes/ras_diag_net_fakes.h"

#include "comm.h"
#include "fakes/fail_loud.h"
#include "fakes/signature-drift.h"
#include "graph/topo.h"
#include "transport/net_ib/common.h"

// The unit only reads the IB device table and compares the comm's plugin against &ncclNetIb.
ncclNet_t ncclNetIb = {.name = "IB"};
int ncclNIbDevs = 0;
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];

static ncclResult_t DefaultTopoGetNetPropertiesSnapshot(struct ncclComm*, bool, struct ncclTopoNetPropertiesSnapshot**,
                                                        int*) {
  FailLoudUnfaked("ras_diag_net_fakes", "ncclTopoGetNetPropertiesSnapshot");
}
std::function<ncclResult_t(struct ncclComm*, bool, struct ncclTopoNetPropertiesSnapshot**, int*)>
    g_topoGetNetPropertiesSnapshot = DefaultTopoGetNetPropertiesSnapshot;
ASSERT_HOOK_MATCHES_PROD(g_topoGetNetPropertiesSnapshot, ncclTopoGetNetPropertiesSnapshot);
ncclResult_t ncclTopoGetNetPropertiesSnapshot(struct ncclComm* comm, bool requireCommInitialized,
                                              struct ncclTopoNetPropertiesSnapshot** snapshots, int* nSnapshots) {
  return g_topoGetNetPropertiesSnapshot(comm, requireCommInitialized, snapshots, nSnapshots);
}

void ResetRasDiagNetFakes() {
  g_topoGetNetPropertiesSnapshot = DefaultTopoGetNetPropertiesSnapshot;
  ncclNIbDevs = 0;
  for (struct ncclIbDev& dev : ncclIbDevs) dev.currSpeed = 0;
}
