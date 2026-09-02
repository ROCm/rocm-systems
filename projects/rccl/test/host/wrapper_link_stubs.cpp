// Link-time stubs for symbols referenced by production .cc files compiled
// into the host UT binary. These are defined in other RCCL translation units
// that we do not compile; they satisfy the linker but are never called.
//
// This file uses forward declarations instead of real headers to avoid
// pulling in HIP runtime types. Functions that need real RCCL types
// (ncclGetErrorString, ncclCommSetAsyncError, ncclCommGetAsyncError)
// live in init_stubs.cpp which includes real headers.

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cerrno>

// Forward declarations of RCCL types used in function signatures.
struct ncclTopoSystem;
struct ncclTopoGraph;
struct ncclTopoNode;
struct ncclComm;
struct ncclXml;
enum ncclTopoGdrMode : int;
enum netDevsPolicy : int;
using ncclResult_t = int;
static constexpr int ncclSuccess = 0;
typedef enum { NCCL_LOG_NONE=0, NCCL_LOG_ERROR=1, NCCL_LOG_VERSION=2,
               NCCL_LOG_WARN=3, NCCL_LOG_INFO=4, NCCL_LOG_ABORT=5,
               NCCL_LOG_TRACE=6 } ncclDebugLogLevel;

// --- debug.cc ---
int ncclDebugLevel = 3;  // NCCL_LOG_WARN
uint64_t ncclDebugMask = 0x7fffffffffffffffULL;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags,
                  const char* filefunc, int line, const char* fmt, ...) {
    if (level > ncclDebugLevel) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

// --- param.cc ---
const char* ncclGetEnv(const char* name) { return getenv(name); }
int64_t ncclLoadParam(const char* env, int64_t deftVal, int64_t, int64_t* cache, signed char*) {
  const char* str = ncclGetEnv(env);
  int64_t value = deftVal;
  if (str && strlen(str) > 0) {
    errno = 0;
    value = strtoll(str, nullptr, 0);
    if (errno) value = deftVal;
  }
  if (cache) *cache = value;
  return value;
}

thread_local signed char ncclDebugNoWarn = 0;

// --- topo.cc ---
const char* topoNodeTypeStr[] = {"GPU", "PCI", "NVS", "CPU", "NIC", "NET", "GIN", "DEV"};
const char* topoPathTypeStr[] = {"LOC", "XGMI", "NVB", "C2C", "PIX", "PXB", "P2C", "PXN", "PHB", "SYS", "NET", "DIS"};

// --- alloc.h ---
struct ncclAllocTracker { long totalAlloc; long totalAllocSize; };
ncclAllocTracker allocTracker[16] = {};

// --- paths.cc deps ---
struct ncclTransport;
ncclTransport* ncclTransports[] = { nullptr };
ncclResult_t rcclSetPxn(ncclComm*, int&) { return ncclSuccess; }
ncclResult_t initChannel(ncclComm*, int) { return ncclSuccess; }
int IsArchMatch(const char*, const char*) { return 0; }

// --- search.cc deps (rome_models.cc, xml.cc) ---
ncclResult_t parse1H16P(ncclTopoSystem*, ncclTopoGraph*) { return ncclSuccess; }
ncclResult_t parseA2a8P(ncclTopoSystem*, ncclTopoGraph*, const char*) { return ncclSuccess; }
ncclResult_t parse4H4P(ncclTopoSystem*, ncclTopoGraph*) { return ncclSuccess; }
ncclResult_t parseGraph(const char*, ncclTopoSystem*, ncclTopoGraph*, int*, int*, int) { return ncclSuccess; }
ncclResult_t parseGraphLight(const char*, ncclTopoSystem*, ncclTopoGraph*, int*) { return ncclSuccess; }
ncclResult_t parseChordalRing(ncclTopoSystem*, ncclTopoGraph*) { return ncclSuccess; }
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

// --- HIP runtime stubs (return hipSuccess = 0) ---
extern "C" {
int hipHostFree(void*) { return 0; }
const char* hipGetErrorString(int) { return "stub"; }
int hipGetLastError() { return 0; }
int hipMemAddressFree(void*, size_t) { return 0; }
int hipMemUnmap(void*, size_t) { return 0; }
int hipMemRelease(uint64_t) { return 0; }
int hipMemCreate(uint64_t*, size_t, void*, unsigned long long) { return 0; }
int hipMemMap(void*, size_t, size_t, uint64_t, unsigned long long) { return 0; }
int hipMemSetAccess(void*, size_t, void*, size_t) { return 0; }
int hipMemExportToShareableHandle(void*, uint64_t, int, unsigned long long) { return 0; }
int hipMemAddressReserve(void**, size_t, size_t, void*, unsigned long long) { return 0; }
int hipMemGetAllocationGranularity(size_t* g, void*, int) { if (g) *g = 65536; return 0; }
int hipMemImportFromShareableHandle(uint64_t*, void*, int) { return 0; }
int hipMemRetainAllocationHandle(uint64_t*, void*) { return 0; }
int hipMemGetAddressRange(void**, size_t*, void*) { return 0; }
int hipDeviceSynchronize() { return 0; }
int hipMemcpy(void*, const void*, size_t, int) { return 0; }
int hipGetDevice(int* d) { if (d) *d = 0; return 0; }
int hipSetDevice(int) { return 0; }
int hipMalloc(void**, size_t) { return 0; }
int hipFree(void*) { return 0; }
int hipHostMalloc(void**, size_t, unsigned int) { return 0; }
int hipMemsetAsync(void*, int, size_t, void*) { return 0; }
int hipStreamCreateWithFlags(void**, unsigned int) { return 0; }
int hipStreamCreateWithPriority(void**, unsigned int, int) { return 0; }
int hipStreamDestroy(void*) { return 0; }
int hipStreamSynchronize(void*) { return 0; }
int hipThreadExchangeStreamCaptureMode(int*) { return 0; }
int hipDeviceGetAttribute(int* v, int, int) { if (v) *v = 0; return 0; }
int hipDeviceGetStreamPriorityRange(int* least, int* greatest) { if (least) *least = 0; if (greatest) *greatest = 0; return 0; }
int hipExtMallocWithFlags(void**, size_t, unsigned int) { return 0; }
int hipDeviceGet(int* d, int) { if (d) *d = 0; return 0; }
}

ncclResult_t CommCheck(ncclComm*, const char*, const char*) { return ncclSuccess; }
ncclResult_t ncclCommEnsureReady(ncclComm*) { return ncclSuccess; }

// --- group.cc deps ---
ncclResult_t ncclGroupStartInternal() { return ncclSuccess; }
struct ncclSimInfo_v22200;
ncclResult_t ncclGroupEndInternal(ncclSimInfo_v22200*) { return ncclSuccess; }

thread_local int ncclGroupDepth = 0;
thread_local ncclResult_t ncclGroupError = ncclSuccess;
thread_local ncclComm* ncclGroupCommHead[2] = {};
thread_local int ncclGroupBlocking = 0;

// bootstrap{Barrier,AllGather,Send,Recv} are provided by bootstrap_wrapper.cc
ncclResult_t ncclProxyClientGetFdBlocking(ncclComm*, int, void*, int*) { return ncclSuccess; }

int ncclCuMemEnable() { return 0; }
int ncclCuMemHandleType = 0;
ncclResult_t getBusId(int, int64_t* busId) { if (busId) *busId = 0; return ncclSuccess; }
