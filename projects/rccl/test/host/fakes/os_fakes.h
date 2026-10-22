/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams os_fakes.cc OWNS (src/os/linux.cc), declared once so a signature change
// is a compile error rather than a link mismatch.

#ifndef RCCL_TEST_HOST_OS_FAKES_H_
#define RCCL_TEST_HOST_OS_FAKES_H_

#include <functional>
#include <string>
#include <vector>

#include "nccl.h"
#include "os.h"  // ncclAffinity

// ncclOsCpuCount is load-bearing: initTransportsRank's exit: block calls it on EVERY path, so nothing in
// the function was testable until it was seamed, and its counter is the only way to see that :1488 skips exit:.
extern int g_ncclOsCpuCountValue;
extern int g_ncclOsCpuCountCalls;
// Every mask ncclOsCpuCount was handed, in call order. Which index is which call site is PATH-DEPENDENT:
// a path running :1607-1611 and reaching exit: gives [0]=:1608 and [1]=exit::2403; a path stopping before
// :1607 gives [0]=exit::2403; a path bypassing exit: (:1618) gives only :1608. Check .size() first.
extern std::vector<ncclAffinity> g_ncclOsCpuCountMasks;
extern ncclResult_t g_ncclOsSetAffinityResult;
// Every mask handed to ncclOsSetAffinity, in call order; same path-dependence as above. [0] is :1610
// only when :1607-1611 ran, otherwise it is exit::2404. A single "last" slot is not enough, because
// the exit: write masks whatever :1610 forwarded.
extern std::vector<ncclAffinity> g_ncclOsSetAffinityMasks;
extern std::function<ncclResult_t(ncclAffinity*)> g_ncclOsGetAffinity;

// ncclInit() NCCLCHECKs this before its own call_once, so it is the one per-call way to fail ncclInit().
extern ncclResult_t g_ncclOsTopoGetStrFromSysResult;
extern int g_ncclOsTopoGetStrFromSysCalls;

// fillInfo's MLOPart PCI-function fallback (init.cc:1092) probes sysfs for the class of its own BDF.
// The default is an accelerator class, i.e. what sysfs reports for a real GPU's own BDF; a test that
// wants the HIP-alias shape (BDF absent from sysfs, so the probe yields nothing) must ask for "".
// The call counter is the only way to see that the fn check short-circuited before the probe.
// See kDefaultPciDeviceClass in os_fakes.cc for why the default is not "".
extern std::string g_pciDeviceClass;
extern int g_pciDeviceClassCalls;
extern std::string g_lastPciDeviceClassBusId;

// fillInfo asks the physical device for its compute partition mode before falling back to the class
// probe above. The default is "SPX", i.e. an unpartitioned GPU. Set "CPX"/"DPX" for a partitioned
// device, or "" for a platform that reports no mode at all. The recorded busId is how a test sees
// that the probe targeted function 0 rather than the caller's own alias BDF.
extern std::string g_pciComputePartition;
extern int g_pciComputePartitionCalls;
extern std::string g_lastPciComputePartitionBusId;

void ResetOsFakes();

#endif  // RCCL_TEST_HOST_OS_FAKES_H_
