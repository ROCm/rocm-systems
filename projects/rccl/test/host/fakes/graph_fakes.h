/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for everything src/graph/*.cc calls outside src/graph, for rccl-UnitTestsMicroTopo. That binary
// compiles the real graph files, so it links none of the shared fakes, which define graph symbols.

#ifndef RCCL_TEST_HOST_GRAPH_FAKES_H_
#define RCCL_TEST_HOST_GRAPH_FAKES_H_

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "nccl.h"
#include "debug.h"
#include "nvmlwrap.h"
#include "amdsmi_wrap.h"

struct ncclComm;
struct ncclConnect;
struct ncclPeerInfo;
struct ncclTopoGraph;
struct ncclTopoSystem;

// Every NCCL_PARAM/RCCL_PARAM in the graph TUs (see param_redirect.h) reads g_loadParam on each call.
// The default returns g_params[env] when set, else the default value. NCCL_ names come without the
// "NCCL_" prefix, RCCL ones with "RCCL_".
extern std::function<int64_t(const char*, int64_t)> g_loadParam;
extern std::map<std::string, int64_t> g_params;

struct LogLine {
  ncclDebugLogLevel level;
  std::string text;
};
extern std::vector<LogLine> g_logs;
bool Logged(const char* needle);

// sysfs as the graph code sees it: PCI paths by bus id, then "<path>/<file>" contents.
extern std::map<std::string, std::string> g_pciPaths;
extern std::map<std::string, std::string> g_sysfs;
extern std::map<std::string, std::string> g_realpaths;
extern std::map<std::string, std::vector<std::string>> g_bcmLinks;
extern std::map<unsigned, std::string> g_numaCpumap;
extern int g_numaCpuOffset;
extern cpu_set_t g_processAffinity;

extern std::function<hipError_t(int*)> g_hipGetDeviceCount;
extern std::function<hipError_t(char*, int, int)> g_hipDeviceGetPCIBusId;
extern std::function<hipError_t(int*, const char*)> g_hipDeviceGetByPCIBusId;
extern std::function<hipError_t(hipDeviceProp_t*, int)> g_hipGetDeviceProperties;

extern ncclResult_t g_amdSmiInitResult;
extern int g_amdSmiInitCalls;
extern std::function<ncclResult_t(uint32_t*)> g_amdSmiGetNumDevice;
extern std::function<ncclResult_t(const char*, uint32_t*)> g_amdSmiGetDeviceIndexByPciBusId;
extern std::function<ncclResult_t(uint32_t, char*, size_t)> g_amdSmiGetDevicePciBusIdString;
extern std::function<ncclResult_t(uint32_t, struct amdsmiFabricDeviceInfo*)> g_amdSmiGetFabricDeviceInfo;
extern std::function<ncclResult_t(int, int, amdsmi_link_type_t*, int*, int*)> g_amdSmiGetLinkInfo;

extern std::function<ncclResult_t(const char*, nvmlDevice_t*)> g_nvmlGetHandleByPciBusId;
extern std::function<ncclResult_t(nvmlDevice_t, nvmlPciInfo_t*)> g_nvmlGetPciInfo;
extern std::function<ncclResult_t(nvmlDevice_t, unsigned int*)> g_nvmlGetPcieLinkGeneration;
extern std::function<ncclResult_t(nvmlDevice_t, unsigned int*)> g_nvmlGetPcieLinkWidth;

extern std::function<ncclResult_t(void*, void*, int)> g_bootstrapAllGather;
extern std::function<ncclResult_t(void*, int*, int, int, void*, int)> g_bootstrapIntraNodeAllGather;

// Plugin device counts, shared by the net, collnet, gin and rma getters.
extern std::function<ncclResult_t(int, int*, int*)> g_netGetDevCount;
extern std::vector<int> g_setVirtDevCounts;

// ncclTransports[TRANSPORT_P2P] / [TRANSPORT_SHM] canConnect.
extern std::function<ncclResult_t(int*, ncclComm*, ncclTopoGraph*, ncclPeerInfo*, ncclPeerInfo*)> g_p2pCanConnect;
extern std::function<ncclResult_t(int*, ncclComm*, ncclTopoGraph*, ncclPeerInfo*, ncclPeerInfo*)> g_shmCanConnect;

extern std::vector<int> g_initChannelCalls;
extern ncclResult_t g_initChannelResult;
extern int g_rcclPxnValue;
extern bool g_useAinic;
extern int g_tuningOverrideCalls;

// rccl_graph_gen.cc is tested elsewhere; generateRings defaults to the identity order per channel.
extern std::function<ncclResult_t(int, uint32_t, int*)> g_generateRings;
extern std::function<void(int, int, const int*, int*)> g_findRingCutIndices;

void ResetGraphFakes();

#endif  // RCCL_TEST_HOST_GRAPH_FAKES_H_
