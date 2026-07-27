// Link-time stubs for symbols referenced by paths_wrapper.cc and search_wrapper.cc.
// These functions are defined in other RCCL .cc files but are never actually called
// in the NullParentTests — they just need link-time definitions to satisfy the linker.

#include <cstdlib>
#include <cstring>
#include <cstdint>

// Forward declarations of RCCL types used in function signatures.
// These match the real RCCL types but we only need them for stub signatures.
struct ncclTopoSystem;
struct ncclTopoGraph;
struct ncclTopoNode;
struct ncclComm;
struct ncclXml;
enum ncclTopoGdrMode : int;
enum netDevsPolicy : int;
using ncclResult_t = int;
static constexpr int ncclSuccess = 0;

// --- Symbols from topo.cc ---
const char* topoNodeTypeStr[] = {"GPU", "PCI", "NVS", "CPU", "NIC", "NET", "GIN", "DEV"};
#ifdef __HIP_PLATFORM_AMD__
const char* topoPathTypeStr[] = {"LOC", "XGMI", "NVB", "C2C", "PIX", "PXB", "P2C", "PXN", "PHB", "SYS", "NET", "DIS"};
#else
const char* topoPathTypeStr[] = {"LOC", "NVL", "NVB", "C2C", "PIX", "PXB", "P2C", "PXN", "PHB", "SYS", "NET", "DIS"};
#endif

// --- Symbols from param.cc ---
const char* ncclGetEnv(const char* name) { return getenv(name); }
void ncclLoadParam(const char*, long, long, long* out, signed char*) { if(out) *out = 0; }

// --- Symbols from debug.cc ---
thread_local signed char ncclDebugNoWarn = 0;

// --- Symbols from alloc.h (template instantiation) ---
struct ncclAllocTracker { long totalAlloc; long totalAllocSize; };
ncclAllocTracker allocTracker[16] = {};

// --- Symbols from paths.cc dependencies ---
struct ncclTransport;
ncclTransport* ncclTransports[] = { nullptr };
ncclResult_t rcclSetPxn(ncclComm*, int&) { return ncclSuccess; }
ncclResult_t initChannel(ncclComm*, int) { return ncclSuccess; }
int IsArchMatch(const char*, const char*) { return 0; }

// --- Symbols from search.cc dependencies (rome_models.cc, topo.cc, xml.cc) ---
ncclResult_t parse1H16P(ncclTopoSystem*, ncclTopoGraph*) { return ncclSuccess; }
ncclResult_t parseA2a8P(ncclTopoSystem*, ncclTopoGraph*, const char*) { return ncclSuccess; }
ncclResult_t parse4H4P(ncclTopoSystem*, ncclTopoGraph*) { return ncclSuccess; }
ncclResult_t parseGraph(const char*, ncclTopoSystem*, ncclTopoGraph*, int*, int*, int) { return ncclSuccess; }
ncclResult_t parseGraphLight(const char*, ncclTopoSystem*, ncclTopoGraph*, int*) { return ncclSuccess; }
ncclResult_t parseChordalRing(ncclTopoSystem*, ncclTopoGraph*) { return ncclSuccess; }
// ncclTopoCheckGdr, ncclTopoSplitNvLink, ncclTopoGetGpuMaxPath, ncclTopoGetGpuMinPath,
// ncclTopoGetIntermediateRank — defined in real paths.cc, not stubbed here.
ncclResult_t ncclTopoGetLocal(ncclTopoSystem*, int, int, int, int*, int*, int*) { return ncclSuccess; }
ncclResult_t ncclTopoGetCompCap(ncclTopoSystem*, int*, int*) { return ncclSuccess; }
ncclResult_t ncclTopoGetLocalGpu(ncclTopoSystem*, long, int*) { return ncclSuccess; }
ncclResult_t ncclTopoGetLocalNet(ncclTopoSystem*, int, int, long*, int*) { return ncclSuccess; }
ncclResult_t ncclTopoDumpXmlToFile(const char*, ncclXml*) { return ncclSuccess; }
ncclResult_t ncclTopoGetNetDevsPolicy(netDevsPolicy*, int*) { return ncclSuccess; }
ncclResult_t ncclTopoGetXmlGraphFromFile(const char*, ncclXml*) { return ncclSuccess; }
void GcnArchNameFormat(char*, char*) {}
ncclResult_t ncclTopoReconcileGrowChannels(ncclComm*, int*) { return ncclSuccess; }
ncclResult_t ncclTopoCpuType(ncclTopoSystem*, int*, int*, int*) { return ncclSuccess; }
ncclResult_t parseRome4P2H(ncclTopoSystem*, ncclTopoGraph*, const char*) { return ncclSuccess; }
ncclResult_t parseGIOTopos(ncclTopoSystem*, ncclTopoGraph*) { return ncclSuccess; }
ncclResult_t ncclTopoGetStrFromSys(const char*, const char*, char*) { return ncclSuccess; }
ncclResult_t ncclTopoRemoveNode(ncclTopoSystem*, int, int) { return ncclSuccess; }
ncclResult_t getLocalNetCountByBw(ncclTopoSystem*, int, int*, float*) { return ncclSuccess; }
int ncclParamWorkArgsBytes() { return 0; }
