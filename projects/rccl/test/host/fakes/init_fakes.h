/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Init-only fake seams for the host-only `rccl-UnitTestsMicroInit` binary. See test/host/MICROTEST_README.md.

#ifndef RCCL_TEST_HOST_INIT_FAKES_H_
#define RCCL_TEST_HOST_INIT_FAKES_H_

#include <functional>
#include <string>
#include <vector>

#include "hip_fakes.h"
#include "nccl_fakes.h"
#include "os.h"  // ncclAffinity, for the initTransportsRank affinity seams below

struct ncclTopoSystem;

// Returns a pointer into the map's own std::string: re-scripting a name invalidates a pointer a caller may still hold.
const char* micro_getenv(const char* name);
void SetMicroEnv(const char* name, const char* value);
void SetMicroEnvAbsent(const char* name);
void ClearMicroEnv();

void SetGethostnameFail(bool fail);
void SetDladdrFail(bool fail);
size_t LastGethostnameLen();

// Default 1 is != VerSuccess(0), so showVersion()'s runtime-ROCm block is skipped.
extern int g_getROCmVersionResult;
extern unsigned int g_rocmVersionMajor;
extern unsigned int g_rocmVersionMinor;
extern unsigned int g_rocmVersionPatch;

extern bool g_ginHasError;

extern bool g_validHsaScratch;
extern const char* g_lastHsaScratchEnv;  // hsaScratchEnv as passed to validHsaScratchEnvSetting
extern int g_firmwareVersion;

extern int g_gdrSupportValue;
extern int g_gdrSupportCalls;

// fillInfo's MLOPart PCI-function fallback (init.cc:1092) probes sysfs for the class of its own BDF.
// The default is an accelerator class, i.e. what sysfs reports for a real GPU's own BDF; a test that
// wants the HIP-alias shape (BDF absent from sysfs, so the probe yields nothing) must ask for "".
// The call counter is the only way to see that the fn check short-circuited before the probe.
// See kDefaultPciDeviceClass in init_fakes.cc for why the default is not "".
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

extern bool g_bootstrapNetInitFail;

extern ncclResult_t g_ncclNetInitResult;
extern ncclResult_t g_ncclGinInitResult;
extern ncclResult_t g_ncclStrongStreamResult;
extern ncclResult_t g_ncclMemManagerInitResult;
extern ncclResult_t g_amdSmiInitResult;

// A std::function, not a result code: tests must write the allgathered (color, key) table into allData.
extern std::function<ncclResult_t(void* commState, void* allData, int size)>
    g_bootstrapAllGather;

extern ncclResult_t g_bootstrapGetUniqueIdResult;
extern ncclResult_t g_bcastGrowHandleResult;
extern uint64_t g_bootstrapHandleMagic;
extern int g_bcastGrowHandleCalls;
extern bool g_bcastGrowHandleIsRoot;

// The fake initChannel does NOT allocate ring->userRanks/rankToIndex like the real one; callers must supply storage.
extern ncclResult_t g_initChannelResult;
extern int g_initChannelLastId;

// -------------------------------------------------------------------------
// initTransportsRank() seams (init.cc:1386). All five were fail-loud stubs.
// ncclOsCpuCount is load-bearing: exit::2403 calls it on EVERY path, so nothing in the function was
// testable until it was seamed, and its counter is the only way to see that :1488 skips exit:.
// ncclTopoGetSystem stays defaulted to FAILURE on purpose -- it is the first call after the
// MNNVL/intra-proc block, so that default is what terminates the ladder and makes :1462-1565
// reachable. Its dumpXmlFile argument passes through so a test can tell :1573 from :1576.
// -------------------------------------------------------------------------
extern int g_ncclOsCpuCountValue;
extern int g_ncclOsCpuCountCalls;
// Every mask ncclOsCpuCount was handed, in call order. Which index is which call site is PATH-DEPENDENT:
// a path running :1607-1611 and reaching exit: gives [0]=:1608 and [1]=exit::2403; a path stopping before
// :1607 gives [0]=exit::2403; a path bypassing exit: (:1618) gives only :1608. Check .size() first.
extern std::vector<ncclAffinity> g_ncclOsCpuCountMasks;
extern ncclResult_t g_ncclOsSetAffinityResult;
// Every mask handed to ncclOsSetAffinity, in call order; same path-dependence as above -- [0] is :1610
// only when :1607-1611 ran, otherwise it is exit::2404. A single "last" slot is not enough, because
// the exit: write masks whatever :1610 forwarded.
extern std::vector<ncclAffinity> g_ncclOsSetAffinityMasks;
extern ncclResult_t g_ncclMnnvlCheckResult;
extern int g_ncclMnnvlCheckCalls;  // the oracle for the :1503-1509 enable/auto/disable logic
extern std::function<ncclResult_t(int*)> g_ncclGetUserP2pLevel;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTopoSystem**, const char*)> g_ncclTopoGetSystem;

// -------------------------------------------------------------------------
// Topology-detection / CPU-affinity seams (init.cc:1576-1648), rung 2 of the ladder.
// All default to success so a test can walk :1576-1648 and inject exactly one failure. ncclTopoCompute
// is the exception -- it defaults to FAILURE because it is now the terminator, the same role
// ncclTopoGetSystem played for rung 1. ncclTopoComputePaths gets a FailAt index rather than a result
// because :1591 and :1596 call it twice and a single knob cannot separate them.
// -------------------------------------------------------------------------
extern int g_tuningIndexValue;
extern std::string g_tuningIndexLastArch;  // :1577 forwards comm->archName; without this that is untested
extern int g_ncclTopoComputePathsCalls;
extern int g_ncclTopoComputePathsFailAt;   // -1 = never fail; 0 = the :1591 call, 1 = the :1596 one
extern ncclResult_t g_ncclTopoTrimSystemResult;
extern ncclResult_t g_ncclTopoSearchInitResult;
extern ncclResult_t g_ncclTopoComputeCommCPUResult;
extern ncclResult_t g_ncclTopoPrintResult;
extern std::function<ncclResult_t(struct ncclTopoSystem*, int, ncclAffinity*)> g_ncclTopoGetCpuAffinity;
extern int g_ncclTopoGetCpuAffinityLastRank;
extern std::function<ncclResult_t(ncclAffinity*)> g_ncclOsGetAffinity;
extern ncclResult_t g_ncclNvlsInitResult;
extern int g_ncclNvlsInitCalls;
// A std::function, not a result knob: rung 3 needs it to succeed AND write graph->nChannels, which
// :1671-1672 read back to size the tree graph. Defaults to failing, so it stays the rung-2 terminator.
extern std::function<ncclResult_t(struct ncclTopoSystem*, struct ncclTopoGraph*)> g_ncclTopoCompute;
extern int g_ncclTopoComputeCalls;
// Every ncclTopoGraph* handed to ncclTopoCompute, in call order; [0] is the :1648 ring compute.
extern std::vector<struct ncclTopoGraph*> g_ncclTopoComputeGraphs;

// -------------------------------------------------------------------------
// Graph-block seams (init.cc:1649-1774), rung 3 of the ladder.
// ncclTopoComputeP2pChannelsPerPeer terminates this rung and deliberately uses a DIFFERENT sentinel
// (ncclTimeout) from the ncclRemoteError rungs 1 and 2 share: a rung-3 test that forgot to arm
// g_ncclTopoCompute would stop at :1648 and return ncclRemoteError, which no rung-3 assertion accepts.
// -------------------------------------------------------------------------
extern ncclResult_t g_ncclTopoPrintGraphResult;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoPrintGraphGraphs;  // pairs 1:1 with the computes
extern ncclResult_t g_ncclTopoDumpGraphsResult;
extern int g_ncclTopoDumpGraphsCalls;
// The ngraphs :1764 passed. -1 until the dump runs; assert this rather than the vector length,
// which the fake clamps to the caller's array capacity.
extern int g_ncclTopoDumpGraphsNgraphs;
extern std::vector<struct ncclTopoGraph*> g_ncclTopoDumpGraphsArray;
extern ncclResult_t g_ncclTopoComputeP2pChannelsPerPeerResult;

void InstallCommAllocSuccess();

void InstallDevCommSetupSuccess();

void ResetInitFakes();

#endif  // RCCL_TEST_HOST_INIT_FAKES_H_
