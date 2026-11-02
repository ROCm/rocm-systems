/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/graph external fakes. See graph_fakes.h.

#include "graph_fakes.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fail_loud.h"

#include "bootstrap.h"
#include "comm.h"
#include "graph.h"
#include "net.h"
#include "os.h"
#include "transport.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = "";

std::vector<LogLine> g_logs;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  char buf[4096];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_logs.push_back({level, buf});
}

bool Logged(const char* needle) {
  for (const LogLine& l : g_logs) {
    if (l.text.find(needle) != std::string::npos) return true;
  }
  return false;
}

std::map<std::string, int64_t> g_params;
static int64_t DefaultLoadParam(const char* env, int64_t deftVal) {
  auto it = g_params.find(env);
  return it == g_params.end() ? deftVal : it->second;
}
std::function<int64_t(const char*, int64_t)> g_loadParam = DefaultLoadParam;

// Units compiled without param_redirect.h (utils.cc) still go through the cache protocol.
int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t, int64_t* cache, int8_t*) {
  const char* str = getenv(env);
  *cache = str ? strtoll(str, NULL, 0) : deftVal;
  return *cache;
}
const char* ncclGetEnv(const char* name) { return getenv(name); }

int64_t ncclParamP2pDisable() { return g_loadParam("P2P_DISABLE", 0); }
int64_t ncclParamMultiRankGpuEnable() { return g_loadParam("MULTI_RANK_GPU_ENABLE", 0); }
int64_t ncclParamWorkArgsBytes() { return g_loadParam("WORK_ARGS_BYTES", INT64_MAX); }
int64_t rcclParamIntraGraphGen() { return g_loadParam("RCCL_INTRA_GRAPH_GEN", 0); }
int64_t rcclParamInterGraphGen() { return g_loadParam("RCCL_INTER_GRAPH_GEN", 0); }

// --- OS layer, over the maps.
std::map<std::string, std::string> g_pciPaths;
std::map<std::string, std::string> g_sysfs;
std::map<std::string, std::string> g_realpaths;
std::map<std::string, std::vector<std::string>> g_bcmLinks;
std::map<unsigned, std::string> g_numaCpumap;
int g_numaCpuOffset = 0;
cpu_set_t g_processAffinity;

ncclResult_t ncclOsGetPciPath(const char* busId, char** path) {
  auto it = g_pciPaths.find(busId);
  if (it == g_pciPaths.end()) {
    *path = NULL;
    WARN("Could not find real path of %s", busId);
    return ncclSystemError;
  }
  *path = strdup(it->second.c_str());
  return ncclSuccess;
}

ncclResult_t ncclOsTopoGetStrFromSys(const char* path, const char* fileName, char* strValue, int maxLen) {
  auto it = g_sysfs.find(std::string(path) + "/" + fileName);
  snprintf(strValue, maxLen, "%s", it == g_sysfs.end() ? "" : it->second.c_str());
  return ncclSuccess;
}

ncclResult_t ncclOsGetPciDeviceClassByBusId(const char* busId, char* deviceClass, size_t maxLen) {
  auto it = g_pciPaths.find(busId);
  if (it == g_pciPaths.end()) {
    deviceClass[0] = '\0';
    return ncclSystemError;
  }
  return ncclOsTopoGetStrFromSys(it->second.c_str(), "class", deviceClass, (int)maxLen);
}

ncclResult_t ncclOsGetBcmLinks(const char* busId, int* nlinks, char** peers) {
  *nlinks = 0;
  *peers = NULL;
  auto it = g_bcmLinks.find(busId);
  if (it == g_bcmLinks.end()) return ncclSuccess;
  const size_t sz = sizeof("0000:00:00.0");
  *peers = (char*)calloc(it->second.size(), sz);
  for (const std::string& p : it->second) snprintf(*peers + sz * (*nlinks)++, sz, "%s", p.c_str());
  return ncclSuccess;
}

ncclResult_t ncclOsGetNumaNodeAffinity(unsigned int numaId, char* affinityStr, size_t maxLen, int* cpuOffset) {
  *cpuOffset = g_numaCpuOffset;
  auto it = g_numaCpumap.find(numaId);
  snprintf(affinityStr, maxLen, "%s", it == g_numaCpumap.end() ? "" : it->second.c_str());
  return ncclSuccess;
}

char* ncclOsRealpath(const char* path, char*) {
  auto it = g_realpaths.find(path);
  return it == g_realpaths.end() ? NULL : strdup(it->second.c_str());
}

void ncclOsCpuZero(ncclAffinity& affinity) { CPU_ZERO(&affinity); }
int ncclOsCpuCount(const ncclAffinity& affinity) { return CPU_COUNT(&affinity); }
void ncclOsCpuSet(ncclAffinity& affinity, int cpu) { CPU_SET(cpu, &affinity); }
bool ncclOsCpuIsSet(const ncclAffinity& affinity, int cpu) { return CPU_ISSET(cpu, &affinity); }
ncclAffinity ncclOsCpuAnd(const ncclAffinity& a, const ncclAffinity& b) {
  ncclAffinity r;
  CPU_AND(&r, &a, &b);
  return r;
}
ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity) {
  *affinity = g_processAffinity;
  return ncclSuccess;
}
uint64_t ncclOsGetPid() { return 4321; }
char* ncclOsStrSep(char** stringp, const char* delim) { return strsep(stringp, delim); }

// --- HIP and amd-smi come from hip_fakes.cc and amdsmi_fakes.cc; ResetGraphFakes() sets the graph defaults.
// The real hipDeviceProp_t default reports 304 CUs, so the topology code sees an MI300X-sized device.
static hipError_t DefaultGraphHipGetDeviceProperties(hipDeviceProp_t* prop, int) {
  *prop = hipDeviceProp_t{};
  prop->multiProcessorCount = 304;
  snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "gfx942:sramecc+:xnack-");
  return hipSuccess;
}

// --- NVML (only the PCI queries xml.cc makes).
static ncclResult_t DefaultNvmlGetHandle(const char*, nvmlDevice_t*) { return ncclSystemError; }
static ncclResult_t DefaultNvmlGetPciInfo(nvmlDevice_t, nvmlPciInfo_t*) { return ncclSystemError; }
static ncclResult_t DefaultNvmlGetLink(nvmlDevice_t, unsigned int*) { return ncclSystemError; }
std::function<ncclResult_t(const char*, nvmlDevice_t*)> g_nvmlGetHandleByPciBusId = DefaultNvmlGetHandle;
std::function<ncclResult_t(nvmlDevice_t, nvmlPciInfo_t*)> g_nvmlGetPciInfo = DefaultNvmlGetPciInfo;
std::function<ncclResult_t(nvmlDevice_t, unsigned int*)> g_nvmlGetPcieLinkGeneration = DefaultNvmlGetLink;
std::function<ncclResult_t(nvmlDevice_t, unsigned int*)> g_nvmlGetPcieLinkWidth = DefaultNvmlGetLink;

ncclResult_t ncclNvmlDeviceGetHandleByPciBusId(const char* busId, nvmlDevice_t* device) {
  return g_nvmlGetHandleByPciBusId(busId, device);
}
ncclResult_t ncclNvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pci) { return g_nvmlGetPciInfo(device, pci); }
ncclResult_t ncclNvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* gen) {
  return g_nvmlGetPcieLinkGeneration(device, gen);
}
ncclResult_t ncclNvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* width) {
  return g_nvmlGetPcieLinkWidth(device, width);
}

// --- Bootstrap.
static ncclResult_t DefaultAllGather(void*, void*, int) { FailLoudUnfaked("graph_fakes.cc", "bootstrapAllGather"); }
static ncclResult_t DefaultIntraNodeAllGather(void*, int*, int, int, void*, int) {
  FailLoudUnfaked("graph_fakes.cc", "bootstrapIntraNodeAllGather");
}
std::function<ncclResult_t(void*, void*, int)> g_bootstrapAllGather = DefaultAllGather;
std::function<ncclResult_t(void*, int*, int, int, void*, int)> g_bootstrapIntraNodeAllGather =
  DefaultIntraNodeAllGather;
ncclResult_t bootstrapAllGather(void* state, void* data, int size) { return g_bootstrapAllGather(state, data, size); }
ncclResult_t bootstrapIntraNodeAllGather(void* state, int* ranks, int rank, int nranks, void* data, int size) {
  return g_bootstrapIntraNodeAllGather(state, ranks, rank, nranks, data, size);
}

// --- Network plugin device counts.
static ncclResult_t DefaultGetDevCount(int, int* nPhys, int* nVirt) {
  *nPhys = 0;
  *nVirt = 0;
  return ncclSuccess;
}
std::function<ncclResult_t(int, int*, int*)> g_netGetDevCount = DefaultGetDevCount;
std::vector<int> g_setVirtDevCounts;
ncclResult_t ncclNetGetDevCount(int i, int* p, int* v) { return g_netGetDevCount(i, p, v); }
ncclResult_t ncclCollNetGetDevCount(int i, int* p, int* v) { return g_netGetDevCount(i, p, v); }
ncclResult_t ncclGinGetDevCount(int i, int* p, int* v) { return g_netGetDevCount(i, p, v); }
ncclResult_t ncclRmaGetDevCount(int i, int* p, int* v) { return g_netGetDevCount(i, p, v); }
ncclResult_t ncclNetSetVirtDevCount(int, int n) {
  g_setVirtDevCounts.push_back(n);
  return ncclSuccess;
}
ncclResult_t ncclCollNetSetVirtDevCount(int, int n) {
  g_setVirtDevCounts.push_back(n);
  return ncclSuccess;
}

// --- Transports.
static ncclResult_t DefaultCanConnect(int*, ncclComm*, ncclTopoGraph*, ncclPeerInfo*, ncclPeerInfo*) {
  FailLoudUnfaked("graph_fakes.cc", "canConnect");
}
std::function<ncclResult_t(int*, ncclComm*, ncclTopoGraph*, ncclPeerInfo*, ncclPeerInfo*)> g_p2pCanConnect =
  DefaultCanConnect;
std::function<ncclResult_t(int*, ncclComm*, ncclTopoGraph*, ncclPeerInfo*, ncclPeerInfo*)> g_shmCanConnect =
  DefaultCanConnect;
static ncclResult_t P2pCanConnect(int* r, ncclComm* c, ncclTopoGraph* g, ncclPeerInfo* a, ncclPeerInfo* b) {
  return g_p2pCanConnect(r, c, g, a, b);
}
static ncclResult_t ShmCanConnect(int* r, ncclComm* c, ncclTopoGraph* g, ncclPeerInfo* a, ncclPeerInfo* b) {
  return g_shmCanConnect(r, c, g, a, b);
}
static struct ncclTransport g_p2pTransport = {"P2P", P2pCanConnect, {}, {}};
static struct ncclTransport g_shmTransport = {"SHM", ShmCanConnect, {}, {}};
struct ncclTransport* ncclTransports[NTRANSPORTS] = {&g_p2pTransport, &g_shmTransport};

// --- Misc RCCL hooks.
std::vector<int> g_initChannelCalls;
ncclResult_t g_initChannelResult = ncclSuccess;
ncclResult_t initChannel(struct ncclComm*, int channelId) {
  g_initChannelCalls.push_back(channelId);
  return g_initChannelResult;
}
int g_rcclPxnValue = -1;
void rcclSetPxn(struct ncclComm*, int& v) { v = g_rcclPxnValue; }
bool g_useAinic = false;
bool rcclUseAinic() { return g_useAinic; }
int g_tuningOverrideCalls = 0;
void rcclApplyTuningOverrides(struct ncclTopoSystem*) { g_tuningOverrideCalls++; }

static ncclResult_t DefaultGenerateRings(int nNodes, uint32_t nChannels, int* order) {
  for (uint32_t c = 0; c < nChannels; c++) {
    for (int i = 0; i < nNodes; i++) order[c * nNodes + i] = i;
  }
  return ncclSuccess;
}
static void DefaultFindRingCutIndices(int nChannels, int nNodes, const int*, int* cut) {
  for (int c = 0; c < nChannels; c++) cut[c] = nNodes - 1;
}
std::function<ncclResult_t(int, uint32_t, int*)> g_generateRings = DefaultGenerateRings;
std::function<void(int, int, const int*, int*)> g_findRingCutIndices = DefaultFindRingCutIndices;
ncclResult_t generateRings(int nNodes, uint32_t nChannels, int* order) {
  return g_generateRings(nNodes, nChannels, order);
}
void findRingCutIndices(int nChannels, int nNodes, const int* order, int* cut) {
  g_findRingCutIndices(nChannels, nNodes, order, cut);
}
void permute_array_inplace(int* input, int length, int* permutation) {
  std::vector<int> copy(input, input + length);
  for (int i = 0; i < length; i++) input[i] = copy[permutation[i]];
}

void ResetGraphFakes() {
  ResetHipFakes();
  g_deviceCount = 0;
  g_hipDeviceGetPCIBusId = [](char*, int, int) { return hipErrorInvalidDevice; };
  g_hipGetDeviceProperties = DefaultGraphHipGetDeviceProperties;
  ResetAmdSmiFakes();
  g_amdSmiGetDeviceIndexByPciBusId = [](const char*, uint32_t* index) {
    *index = 0;
    return ncclSuccess;
  };
  g_amdSmiGetFabricDeviceInfo = [](uint32_t, struct amdsmiFabricDeviceInfo* info) {
    info->fabricSupported = false;
    return ncclSuccess;
  };
  g_logs.clear();
  g_params.clear();
  g_loadParam = DefaultLoadParam;
  g_pciPaths.clear();
  g_sysfs.clear();
  g_realpaths.clear();
  g_bcmLinks.clear();
  g_numaCpumap.clear();
  g_numaCpuOffset = 0;
  CPU_ZERO(&g_processAffinity);
  g_nvmlGetHandleByPciBusId = DefaultNvmlGetHandle;
  g_nvmlGetPciInfo = DefaultNvmlGetPciInfo;
  g_nvmlGetPcieLinkGeneration = DefaultNvmlGetLink;
  g_nvmlGetPcieLinkWidth = DefaultNvmlGetLink;
  g_bootstrapAllGather = DefaultAllGather;
  g_bootstrapIntraNodeAllGather = DefaultIntraNodeAllGather;
  g_netGetDevCount = DefaultGetDevCount;
  g_setVirtDevCounts.clear();
  g_p2pCanConnect = DefaultCanConnect;
  g_shmCanConnect = DefaultCanConnect;
  g_initChannelCalls.clear();
  g_initChannelResult = ncclSuccess;
  g_rcclPxnValue = -1;
  g_useAinic = false;
  g_tuningOverrideCalls = 0;
  g_generateRings = DefaultGenerateRings;
  g_findRingCutIndices = DefaultFindRingCutIndices;
}
