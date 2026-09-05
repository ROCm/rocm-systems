/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the init-only fake seams. See init_fakes.h.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // RTLD_NEXT
#endif
#include <dlfcn.h>

#include "init_fakes.h"

#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>

// micro_getenv / SetMicroEnv / ClearMicroEnv / the getenv interposer / ncclGetEnv
// moved to env_fakes.cc so every microtest binary shares ONE env implementation:
// a second, map-only copy cannot intercept production's raw getenv() call sites.

// Arming gethostname failure + reaching fillInfo latches getHostName's hostHash call_once, poisoning later tests.
namespace {
bool g_gethostnameFail = false;
bool g_dladdrFail = false;
size_t g_lastGethostnameLen = 0;
}  // namespace

void SetGethostnameFail(bool fail) { g_gethostnameFail = fail; }
void SetDladdrFail(bool fail) { g_dladdrFail = fail; }
size_t LastGethostnameLen() { return g_lastGethostnameLen; }

extern "C" int gethostname(char* name, size_t len) {
  using Fn = int (*)(char*, size_t);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "gethostname"));
  g_lastGethostnameLen = len;
  if (g_gethostnameFail) {
    errno = ENAMETOOLONG;
    return -1;
  }
  return real ? real(name, len) : -1;
}

extern "C" int dladdr(const void* addr, Dl_info* info) {
  using Fn = int (*)(const void*, Dl_info*);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "dladdr"));
  if (g_dladdrFail) return 0;  // dladdr reports failure as 0, not -1
  return real ? real(addr, info) : 0;
}

// ncclParam* referenced by init.cc but not declared inside it, so the redirected NCCL_PARAM does not cover them.
// Each default mirrors production; the trailing comment names the definition it copies, so drift is checkable here.
int64_t ncclParamLaunchOrderImplicit() { return g_loadParam("LAUNCH_ORDER_IMPLICIT", 0); }  // enqueue.cc:1985
int64_t ncclParamNvlsEnable() { return g_loadParam("NVLS_ENABLE", 2); }                     // transport/nvls.cc:159
int64_t ncclParamNvtxDisable() { return g_loadParam("NVTX_DISABLE", 0); }                   // init_nvtx.cc:16
int64_t ncclParamPatEnable() { return g_loadParam("PAT_ENABLE", 0); }                       // graph/tuning.cc:1105
int64_t ncclParamSingleProcMemRegEnable() { return g_loadParam("SINGLE_PROC_MEM_REG_ENABLE", 0); }  // group.cc:605

// rccl::Recorder moved to recorder_fakes.cc: the ctor/dtor/instance() triple was
// copied verbatim into three fakes files, differing only in which record()
// overloads each target referenced. The init overload's argument recorder moved
// with it.

ncclResult_t ncclGroupStartInternal() { return ncclSuccess; }
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t*) { return ncclSuccess; }
ncclResult_t ncclGetUniqueId(ncclUniqueId* id) {
  if (id) std::memset(id, 0, sizeof(*id));
  return ncclSuccess;
}

static ncclResult_t DefaultNcclGroupJobAbort(struct ncclGroupJob*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclGroupJob*)> g_ncclGroupJobAbort = DefaultNcclGroupJobAbort;
ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* job) { return g_ncclGroupJobAbort(job); }
ncclResult_t ncclGroupJobComplete(struct ncclGroupJob*) { return ncclSuccess; }

bool g_ginHasError = false;
ncclResult_t ncclGinQueryLastError(struct ncclGinState*, bool* hasError) {
  if (hasError) *hasError = g_ginHasError;
  return ncclSuccess;
}

// TODO: the real impls live in rccl_wrap.cc, which pulls in ce_coll/dda/sym_kernels/dev_runtime/strongstream.
void rcclSetDefaultBuffSizes(struct ncclComm*, int* defaults) {
  defaults[0] = 1 << 18;  // LL
  defaults[1] = 1 << 18;  // LL128
  defaults[2] = 1 << 22;  // SIMPLE
}
void rcclSetP2pNetChunkSize(struct ncclComm*, int& sz) { sz = 1 << 17; }

bool g_validHsaScratch = true;
int g_firmwareVersion = 0;
// Records the argument so a test can observe that checkHsaEnvSetting actually read the environment.
const char* g_lastHsaScratchEnv = nullptr;
bool validHsaScratchEnvSetting(const char* hsaScratchEnv, int /*hipRuntimeVersion*/,
                               int /*firmwareVersion*/, const char* /*gcnArchName*/) {
  g_lastHsaScratchEnv = hsaScratchEnv;
  return g_validHsaScratch;
}
int getFirmwareVersion() { return g_firmwareVersion; }

ncclResult_t DefaultAmdSmiGetDeviceIndexByPciBusId(const char*, uint32_t* deviceIndex) {
  if (deviceIndex) *deviceIndex = static_cast<uint32_t>(-1);  // -1 -> skip fabric block
  return ncclSuccess;
}
std::function<ncclResult_t(const char*, uint32_t*)> g_amdSmiGetDeviceIndexByPciBusId =
    DefaultAmdSmiGetDeviceIndexByPciBusId;
ncclResult_t amd_smi_getDeviceIndexByPciBusId(const char* busId, uint32_t* deviceIndex) {
  return g_amdSmiGetDeviceIndexByPciBusId(busId, deviceIndex);
}

ncclResult_t DefaultAmdSmiGetFabricDeviceInfo(uint32_t, struct amdsmiFabricDeviceInfo*) {
  return ncclSuccess;
}
std::function<ncclResult_t(uint32_t, struct amdsmiFabricDeviceInfo*)> g_amdSmiGetFabricDeviceInfo =
    DefaultAmdSmiGetFabricDeviceInfo;
ncclResult_t amd_smi_getFabricDeviceInfo(uint32_t deviceIndex, struct amdsmiFabricDeviceInfo* info) {
  return g_amdSmiGetFabricDeviceInfo(deviceIndex, info);
}
ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported) {
  if (supported) *supported = false;
  return ncclSuccess;
}
int g_gdrSupportValue = 0;
int g_gdrSupportCalls = 0;
ncclResult_t ncclGpuGdrSupport(struct ncclComm*, int* gdrSupport) {
  ++g_gdrSupportCalls;
  if (gdrSupport) *gdrSupport = g_gdrSupportValue;
  return ncclSuccess;
}
// fillInfo MLOPart PCI-function fallback (init.cc ~1093). The default models what sysfs actually
// reports for a GPU's own BDF -- an accelerator class -- rather than an empty string. An empty
// default is what let the mloPart=0 stamp on non-partitioned GPUs ship green: with it, isGpu is 0
// for every test that does not set comm->busId, so the fn==0 arm of the fallback was unreachable and
// InitTransportsRank_NoPeerWithMloPart_LeavesHasMloPartUnset could not see the stamp on its own rank.
// A test that wants the HIP-alias shape (BDF absent from sysfs) must now ask for "" explicitly.
// Duplicates PCI_ACCELERATOR_CLASS (src/graph/xml.h), which this TU does not include.
static const char* const kDefaultPciDeviceClass = "0x120000";
std::string g_pciDeviceClass = kDefaultPciDeviceClass;
int g_pciDeviceClassCalls = 0;
std::string g_lastPciDeviceClassBusId;
ncclResult_t ncclOsGetPciDeviceClassByBusId(const char* busId, char* deviceClass, size_t maxLen) {
  ++g_pciDeviceClassCalls;
  g_lastPciDeviceClassBusId = busId ? busId : "";
  if (deviceClass && maxLen > 0) {
    std::strncpy(deviceClass, g_pciDeviceClass.c_str(), maxLen - 1);
    deviceClass[maxLen - 1] = '\0';
  }
  return ncclSuccess;
}
// fillInfo's compute-partition probe. "SPX" is the unpartitioned default, so the class-probe
// fallback stays on the path it had before partition detection existed.
static const char* const kDefaultComputePartition = "SPX";
std::string g_pciComputePartition = kDefaultComputePartition;
int g_pciComputePartitionCalls = 0;
std::string g_lastPciComputePartitionBusId;
ncclResult_t ncclOsGetPciDeviceComputePartitionByBusId(const char* busId, char* partition, size_t maxLen) {
  ++g_pciComputePartitionCalls;
  g_lastPciComputePartitionBusId = busId ? busId : "";
  if (partition && maxLen > 0) {
    std::strncpy(partition, g_pciComputePartition.c_str(), maxLen - 1);
    partition[maxLen - 1] = '\0';
  }
  return ncclSuccess;
}
ncclResult_t rocmLibraryInit(void) { return ncclSuccess; }
uint64_t ncclOsGetPid() { return 4321; }
// dmaBufSupported gate: NULL -> unsupported.
PFN_hsa_amd_portable_export_dmabuf pfn_hsa_amd_portable_export_dmabuf = nullptr;

// The NCCL_API dispatch symbol is emitted by the api-trace layer, which is not linked here.
extern ncclResult_t ncclCommGetAsyncError_impl(ncclComm_t comm, ncclResult_t* asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  return ncclCommGetAsyncError_impl(comm, asyncError);
}

bool g_bootstrapNetInitFail = false;
ncclResult_t bootstrapNetInit() { return g_bootstrapNetInitFail ? ncclSystemError : ncclSuccess; }
void initEnv() {}
ncclResult_t ncclOsInitialize() { return ncclSuccess; }
void initNvtxRegisteredEnums() {}
ncclResult_t g_ncclEnvPluginInitResult = ncclSuccess;
// Default reads the plain result seam so both styles work: set g_ncclEnvPluginInitResult, or ScopedHook the functor.
static ncclResult_t DefaultNcclEnvPluginInit() { return g_ncclEnvPluginInitResult; }
std::function<ncclResult_t()> g_ncclEnvPluginInit = DefaultNcclEnvPluginInit;
ncclResult_t ncclEnvPluginInit(void) { return g_ncclEnvPluginInit(); }
bool ncclIommuPassthroughOk(const char*) { return true; }
// ncclInit() strtok_r()s the /proc version read, so it must have >= 3 whitespace tokens.
ncclResult_t ncclTopoGetStrFromSys(const char* /*path*/, const char* fileName, char* strValue) {
  if (!strValue) return ncclSuccess;
  if (fileName && std::strcmp(fileName, "version") == 0)
    std::strcpy(strValue, "Linux version 6.8.0-microtest");
  else if (fileName && std::strcmp(fileName, "numa_balancing") == 0)
    std::strcpy(strValue, "0");
  else
    std::strcpy(strValue, "microtest");
  return ncclSuccess;
}

// ncclNetInit/ncclNetInitFromParent live in init-test.cc: they need the full ncclComm/ncclNet_t layout.
ncclResult_t g_ncclNetInitResult        = ncclSuccess;
ncclResult_t g_ncclGinInitResult        = ncclSuccess;
ncclResult_t g_ncclStrongStreamResult   = ncclSuccess;
ncclResult_t g_ncclMemManagerInitResult = ncclSuccess;
ncclResult_t g_amdSmiInitResult         = ncclSuccess;
// Defaults to failure so the bootstrapAllGather call sites no test reaches stay fail-fast.
std::function<ncclResult_t(void*, void*, int)> g_bootstrapAllGather =
    [](void*, void*, int) { return ncclInternalError; };

ncclResult_t g_bootstrapGetUniqueIdResult = ncclSuccess;
ncclResult_t g_bcastGrowHandleResult      = ncclSuccess;
uint64_t g_bootstrapHandleMagic           = 0xB007ULL;
int g_bcastGrowHandleCalls                = 0;
bool g_bcastGrowHandleIsRoot              = false;
int g_bootstrapGetUniqueIdCalls           = 0;

ncclResult_t DefaultBcastGrowHandle(struct ncclBootstrapHandle*, struct ncclComm*, bool) {
  return g_bcastGrowHandleResult;
}
std::function<ncclResult_t(struct ncclBootstrapHandle*, struct ncclComm*, bool)> g_bcastGrowHandle =
    DefaultBcastGrowHandle;

ncclResult_t g_initChannelResult        = ncclSuccess;
int g_initChannelLastId                 = -1;
// Default 1 (!= VerSuccess) means "version unknown".
int g_getROCmVersionResult = 1;
unsigned int g_rocmVersionMajor = 0;
unsigned int g_rocmVersionMinor = 0;
unsigned int g_rocmVersionPatch = 0;

// initTransportsRank() seams; the stubs live in nccl_stubs.cc / transport_stubs.cc / topo_stubs.cc.
// ncclOsCpuCount default 0 keeps exit::2404 from calling ncclOsSetAffinity unless a test asks for it.
int g_ncclOsCpuCountValue                = 0;
int g_ncclOsCpuCountCalls                = 0;
std::vector<ncclAffinity> g_ncclOsCpuCountMasks;
ncclResult_t g_ncclOsSetAffinityResult   = ncclSuccess;
std::vector<ncclAffinity> g_ncclOsSetAffinityMasks;
ncclResult_t g_ncclOsTopoGetStrFromSysResult = ncclSuccess;
int g_ncclOsTopoGetStrFromSysCalls       = 0;
ncclResult_t g_ncclMnnvlCheckResult      = ncclSuccess;
int g_ncclMnnvlCheckCalls                = 0;
// Non-zero default level: p2pLevel != 0 is what :1506 needs for the MNNVL auto scope to be reachable at all.
std::function<ncclResult_t(int*)> g_ncclGetUserP2pLevel =
    [](int* level) { *level = 3; return ncclSuccess; };
// Defaults to FAILURE -- it is the rung-1 terminator, and no test drives its call sites to success by default.
// ncclRemoteError is a SENTINEL: no init.cc path reachable from initTransportsRank produces it, so EXPECT_EQ
// on it proves execution reached a terminator rather than dying at AllGather1 or the :1554 intra-proc guard.
// The rung-2 terminator below shares it; see there for why the two cannot be confused.
std::function<ncclResult_t(struct ncclComm*, struct ncclTopoSystem**, const char*)> g_ncclTopoGetSystem =
    [](struct ncclComm*, struct ncclTopoSystem**, const char*) { return ncclRemoteError; };

// Topology-detection and CPU-affinity seams (:1576-1618). All succeed by default so a test can walk the
// block and inject exactly one failure; ncclTopoCompute is the exception -- it is the rung-2 terminator.
// g_tuningIndexValue / g_tuningIndexLastArch: tuning_fakes.cc, next to
// rcclGetTuningIndexForArch itself.
int g_ncclTopoComputePathsCalls             = 0;
int g_ncclTopoComputePathsFailAt            = -1;
ncclResult_t g_ncclTopoTrimSystemResult     = ncclSuccess;
ncclResult_t g_ncclTopoSearchInitResult     = ncclSuccess;
ncclResult_t g_ncclTopoComputeCommCPUResult = ncclSuccess;
ncclResult_t g_ncclTopoPrintResult          = ncclSuccess;
int g_ncclTopoGetCpuAffinityLastRank        = -1;
// Writes an EMPTY mask by default, so ncclOsCpuCount's 0 default stays consistent and :1609-1610 are skipped.
std::function<ncclResult_t(struct ncclTopoSystem*, int, ncclAffinity*)> g_ncclTopoGetCpuAffinity =
    [](struct ncclTopoSystem*, int, ncclAffinity* a) { CPU_ZERO(a); return ncclSuccess; };
std::function<ncclResult_t(ncclAffinity*)> g_ncclOsGetAffinity =
    [](ncclAffinity* a) { CPU_ZERO(a); return ncclSuccess; };
ncclResult_t g_ncclNvlsInitResult           = ncclSuccess;
int g_ncclNvlsInitCalls                     = 0;
// Same sentinel as the rung-1 terminator, and unambiguous for the same reason it is a sentinel at all:
// every rung-2 test calls installTopo(), which replaces g_ncclTopoGetSystem with a succeeding lambda,
// so ncclRemoteError here can only have come from :1648.
std::function<ncclResult_t(struct ncclTopoSystem*, struct ncclTopoGraph*)> g_ncclTopoCompute =
    [](struct ncclTopoSystem*, struct ncclTopoGraph*) { return ncclRemoteError; };  // rung-2 terminator
int g_ncclTopoComputeCalls                  = 0;
std::vector<struct ncclTopoGraph*> g_ncclTopoComputeGraphs;

// Graph-block seams (:1649-1774), rung 3. ncclTopoComputeP2pChannelsPerPeer is the terminator and uses
// ncclTimeout, NOT the ncclRemoteError the earlier rungs share, so a rung-3 assertion cannot be
// satisfied by a test that forgot to arm g_ncclTopoCompute and stopped at :1648 instead.
ncclResult_t g_ncclTopoPrintGraphResult     = ncclSuccess;
std::vector<struct ncclTopoGraph*> g_ncclTopoPrintGraphGraphs;
ncclResult_t g_ncclTopoDumpGraphsResult     = ncclSuccess;
int g_ncclTopoDumpGraphsCalls               = 0;
int g_ncclTopoDumpGraphsNgraphs             = -1;
std::vector<struct ncclTopoGraph*> g_ncclTopoDumpGraphsArray;
ncclResult_t g_ncclTopoComputeP2pChannelsPerPeerResult = ncclTimeout;  // rung-3 terminator

ncclResult_t ncclGinInit(struct ncclComm*) { return g_ncclGinInitResult; }
ncclResult_t ncclGinInitFromParent(struct ncclComm*, struct ncclComm*) { return g_ncclGinInitResult; }
ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream*) { return g_ncclStrongStreamResult; }
ncclResult_t amd_smi_init() { return g_amdSmiInitResult; }
size_t ncclOsGetPageSize() { return 4096; }
extern "C" ncclResult_t ncclMemManagerInit(struct ncclComm*) { return g_ncclMemManagerInitResult; }

ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream*) { return g_ncclStrongStreamResult; }

// commCleanup ordering oracle; the fakes that append to it live in nccl_stubs.cc. See init_fakes.h.
std::vector<std::string> g_cleanupCallOrder;
ncclResult_t g_ncclCeFinalizeResult = ncclSuccess;
struct ncclComm* g_ncclTunerPluginUnloadLastComm = nullptr;

void InstallCommAllocSuccess() {
  g_ncclNetInitResult = ncclSuccess;
  g_ncclGinInitResult = ncclSuccess;
  g_ncclStrongStreamResult = ncclSuccess;
  g_ncclMemManagerInitResult = ncclSuccess;
  g_amdSmiInitResult = ncclSuccess;
  g_hipDeviceGetAttributeResult = hipSuccess;
  g_hipDeviceGetPCIBusIdResult  = hipSuccess;
  g_hipEventCreateResult        = hipSuccess;
  g_hipMemPoolResult            = hipSuccess;
  g_hipStreamCreateResult       = hipSuccess;
}

void InstallDevCommSetupSuccess() {
  InstallCommAllocSuccess();
  g_hipAsyncOpsResult = hipSuccess;
}

void ResetInitFakes() {
  ResetHipFakes();
  ResetNcclFakes();
  ResetRecorderFakes();
  ResetRcclWrapFakes();
  ResetNcclStubs();
  ResetBootstrapStubs();
  ResetTransportStubs();
  ResetTuningFakes();
  ResetEnvFakes();
  g_ginHasError = false;
  g_ncclEnvPluginInit = DefaultNcclEnvPluginInit;
  g_ncclGroupJobAbort = DefaultNcclGroupJobAbort;
  g_bootstrapNetInitFail = false;
  g_validHsaScratch = true;
  g_lastHsaScratchEnv = nullptr;
  g_firmwareVersion = 0;
  g_gdrSupportValue = 0;
  g_gdrSupportCalls = 0;
  g_amdSmiGetDeviceIndexByPciBusId = DefaultAmdSmiGetDeviceIndexByPciBusId;
  g_amdSmiGetFabricDeviceInfo = DefaultAmdSmiGetFabricDeviceInfo;
  g_pciDeviceClass = kDefaultPciDeviceClass;
  g_pciDeviceClassCalls = 0;
  g_lastPciDeviceClassBusId.clear();
  g_pciComputePartition = kDefaultComputePartition;
  g_pciComputePartitionCalls = 0;
  g_lastPciComputePartitionBusId.clear();
  pfn_hsa_amd_portable_export_dmabuf = nullptr;
  g_ncclNetInitResult = ncclSuccess;
  g_ncclGinInitResult = ncclSuccess;
  g_ncclStrongStreamResult = ncclSuccess;
  g_ncclMemManagerInitResult = ncclSuccess;
  g_amdSmiInitResult = ncclSuccess;
  g_initChannelResult = ncclSuccess;
  g_initChannelLastId = -1;
  g_bootstrapGetUniqueIdResult = ncclSuccess;
  g_bcastGrowHandleResult = ncclSuccess;
  g_bootstrapHandleMagic = 0xB007ULL;
  g_bcastGrowHandleCalls = 0;
  g_bcastGrowHandleIsRoot = false;
  ResetBootstrapHandleTemplate();
  g_bootstrapGetUniqueIdCalls = 0;
  g_ncclEnvPluginInitResult = ncclSuccess;
  g_ncclOsTopoGetStrFromSysResult = ncclSuccess;
  g_ncclOsTopoGetStrFromSysCalls = 0;
  g_bcastGrowHandle = DefaultBcastGrowHandle;
  g_bootstrapAllGather = [](void*, void*, int) { return ncclInternalError; };
  g_gethostnameFail = false;
  g_dladdrFail = false;
  g_lastGethostnameLen = 0;
  g_getROCmVersionResult = 1;
  g_rocmVersionMajor = 0;
  g_rocmVersionMinor = 0;
  g_rocmVersionPatch = 0;
  g_ncclOsCpuCountValue = 0;
  g_ncclOsCpuCountCalls = 0;
  g_ncclOsCpuCountMasks.clear();
  g_ncclOsSetAffinityResult = ncclSuccess;
  g_ncclOsSetAffinityMasks.clear();
  g_ncclMnnvlCheckResult = ncclSuccess;
  g_ncclMnnvlCheckCalls = 0;
  g_ncclGetUserP2pLevel = [](int* level) { *level = 3; return ncclSuccess; };
  g_ncclTopoGetSystem = [](struct ncclComm*, struct ncclTopoSystem**, const char*) { return ncclRemoteError; };
  g_ncclTopoComputePathsCalls = 0;
  g_ncclTopoComputePathsFailAt = -1;
  g_ncclTopoTrimSystemResult = ncclSuccess;
  g_ncclTopoSearchInitResult = ncclSuccess;
  g_ncclTopoComputeCommCPUResult = ncclSuccess;
  g_ncclTopoPrintResult = ncclSuccess;
  g_ncclTopoGetCpuAffinityLastRank = -1;
  g_ncclTopoGetCpuAffinity = [](struct ncclTopoSystem*, int, ncclAffinity* a) { CPU_ZERO(a); return ncclSuccess; };
  g_ncclOsGetAffinity = [](ncclAffinity* a) { CPU_ZERO(a); return ncclSuccess; };
  g_ncclNvlsInitResult = ncclSuccess;
  g_ncclNvlsInitCalls = 0;
  g_ncclTopoCompute = [](struct ncclTopoSystem*, struct ncclTopoGraph*) { return ncclRemoteError; };
  g_ncclTopoComputeCalls = 0;
  g_ncclTopoComputeGraphs.clear();
  g_ncclTopoPrintGraphResult = ncclSuccess;
  g_ncclTopoPrintGraphGraphs.clear();
  g_ncclTopoDumpGraphsResult = ncclSuccess;
  g_ncclTopoDumpGraphsCalls = 0;
  g_ncclTopoDumpGraphsNgraphs = -1;
  g_ncclTopoDumpGraphsArray.clear();
  g_ncclTopoComputeP2pChannelsPerPeerResult = ncclTimeout;
  g_cleanupCallOrder.clear();
  g_ncclCeFinalizeResult = ncclSuccess;
  g_ncclTunerPluginUnloadLastComm = nullptr;
}
