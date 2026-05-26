/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Minimal stubs for everything p2p.cc references but doesn't define, so
// rccl-UnitTestsMicro links without pulling in librccl.so.
//
// Philosophy:
//   - Globals get a definition with a sensible default.
//   - Logging is routed to stderr with a [fake] prefix so unexpected
//     WARN/INFO traffic during test runs is visible (not silently dropped).
//   - ncclLoadParam is a no-op, so every NCCL_PARAM(...) returns its
//     compile-time default.
//   - "Seam" functions (proxy, topo, shm, registration) return failure by
//     default. Tests that need to drive them should swap in their own
//     behaviour by overriding the function pointer hooks at the bottom of
//     this file (TODO -- not wired yet; add when the first test needs it).

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nccl.h"
#include "alloc.h"        // allocationTracker, ncclCuMemEnable
#include "debug.h"        // ncclDebugLog, ncclDebugLevel, ...
#include "param.h"        // ncclLoadParam
#include "rocmwrap.h"     // ncclCuMemHandleType
#include "archinfo.h"     // IsArchMatch
#include "utils.h"        // busIdToInt64
#include "graph.h"        // ncclTopoCheckP2p / Net / GetLinkType
#include "proxy.h"        // ncclProxy* family
#include "register.h"     // ncclRegLocalIsValid
#include "shm.h"          // ncclShm*
#include "comm.h"         // ncclCommGraphRegister / Deregister
#include "strongstream.h" // ncclStrongStream*

#include "p2p_fakes.h"     // controllable seam hooks

// ---------------------------------------------------------------------------
// Bucket A: trivial globals
// ---------------------------------------------------------------------------

// allocTracker is an array of per-device counters in alloc.h; size it to
// the same MAX_ALLOC_TRACK_NGPU the header uses. Zero-initialised.
struct allocationTracker allocTracker[32 /* MAX_ALLOC_TRACK_NGPU */] = {};

hipMemAllocationHandleType ncclCuMemHandleType =
    hipMemHandleTypePosixFileDescriptor;

// ---------------------------------------------------------------------------
// Bucket B: logging / param infrastructure
// ---------------------------------------------------------------------------

int                 ncclDebugLevel   = 0;   // NCCL_LOG_NONE
uint64_t            ncclDebugMask    = 0;
thread_local int    ncclDebugNoWarn  = 0;

void ncclDebugLog(ncclDebugLogLevel /*level*/,
                             unsigned long     /*flags*/,
                             const char*       filefunc,
                             int               line,
                             const char*       fmt,
                             ...)
{
    std::fprintf(stderr, "[fake] %s:%d ", filefunc ? filefunc : "?", line);
    std::va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
}

void ncclLoadParam(char const* /*env*/,
                   int64_t     /*deftVal*/,
                   int64_t     /*uninitialized*/,
                   int64_t*    /*cache*/)
{
    // No-op: leaves cache untouched so NCCL_PARAM callers see the default.
}

// ---------------------------------------------------------------------------
// Bucket C: seams worth controlling from tests (return failure by default)
// ---------------------------------------------------------------------------

int ncclCuMemEnable() { return 0; }

// --- Controllable seams: ncclProxyConnect / ncclProxyCallBlocking --------
// Default behaviour is the old stub: return ncclSystemError. Happy-path
// tests install a hook that succeeds and writes a canned rmtRegAddr into
// respBuff for ncclProxyMsgRegister.
static ncclResult_t DefaultProxyConnect(struct ncclComm*, int, int, int,
                                        struct ncclProxyConnector*)
{
    return ncclSystemError;
}

static ncclResult_t DefaultProxyCallBlocking(struct ncclComm*,
                                             struct ncclProxyConnector*,
                                             int, void*, int, void*, int)
{
    return ncclSystemError;
}

std::function<ncclResult_t(struct ncclComm*, int, int, int,
                           struct ncclProxyConnector*)>
    g_proxyConnect = DefaultProxyConnect;

std::function<ncclResult_t(struct ncclComm*, struct ncclProxyConnector*,
                           int, void*, int, void*, int)>
    g_proxyCallBlocking = DefaultProxyCallBlocking;

ncclResult_t ncclProxyConnect(struct ncclComm*           comm,
                              int                        transport,
                              int                        send,
                              int                        proxyRank,
                              struct ncclProxyConnector* proxyConn)
{
    return g_proxyConnect(comm, transport, send, proxyRank, proxyConn);
}

ncclResult_t ncclProxyCallBlocking(struct ncclComm*           comm,
                                   struct ncclProxyConnector* proxyConn,
                                   int                        type,
                                   void*                      reqBuff,
                                   int                        reqSize,
                                   void*                      respBuff,
                                   int                        respSize)
{
    return g_proxyCallBlocking(comm, proxyConn, type, reqBuff, reqSize,
                               respBuff, respSize);
}

// --- Controllable seams: hipMemGetAddressRange / hipIpcGetMemHandle ------
// p2p-test.cc macro-shims the p2p.cc call sites onto these hooks, so the
// test binary never reaches the real HIP runtime (no GPU). Defaults return
// hipErrorInvalidValue so unexpected call sites surface via CUCHECKGOTO.
static hipError_t DefaultHipMemGetAddressRange(hipDeviceptr_t*, std::size_t*,
                                               hipDeviceptr_t)
{
    return hipErrorInvalidValue;
}

static hipError_t DefaultHipIpcGetMemHandle(hipIpcMemHandle_t*, void*)
{
    return hipErrorInvalidValue;
}

std::function<hipError_t(hipDeviceptr_t*, std::size_t*, hipDeviceptr_t)>
    g_hipMemGetAddressRange = DefaultHipMemGetAddressRange;
std::function<hipError_t(hipIpcMemHandle_t*, void*)>
    g_hipIpcGetMemHandle = DefaultHipIpcGetMemHandle;

ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* /*comm*/,
                                          int              /*rank*/,
                                          void*            /*handle*/,
                                          int*             /*convertedFd*/)
{
    return ncclSystemError;
}

ncclResult_t ncclProxyClientQueryFdBlocking(struct ncclComm*           /*comm*/,
                                            struct ncclProxyConnector* /*proxyConn*/,
                                            int                        /*localFd*/,
                                            int*                       /*rmtFd*/)
{
    return ncclSystemError;
}

ncclResult_t ncclRegLocalIsValid(struct ncclReg* /*reg*/, bool* isValid)
{
    if (isValid) *isValid = false;
    return ncclSuccess;
}

ncclResult_t ncclShmImportShareableBuffer(struct ncclComm*  /*comm*/,
                                          int               /*proxyRank*/,
                                          ncclShmIpcDesc_t* /*desc*/,
                                          void**            /*hptr*/,
                                          void**            /*dptr*/,
                                          ncclShmIpcDesc_t* /*descOut*/)
{
    return ncclSystemError;
}

ncclResult_t ncclShmIpcClose(ncclShmIpcDesc_t* /*desc*/)
{
    return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Bucket D: topology / arch helpers
// ---------------------------------------------------------------------------

bool IsArchMatch(char const* /*arch*/, char const* /*target*/)
{
    return false;
}

ncclResult_t busIdToInt64(const char* /*busId*/, int64_t* id)
{
    if (id) *id = 0;
    return ncclSuccess;
}

ncclResult_t ncclTopoCheckP2p(struct ncclComm*       /*comm*/,
                              struct ncclTopoSystem* /*system*/,
                              int                    /*rank1*/,
                              int                    /*rank2*/,
                              int*                   p2p,
                              int*                   read,
                              int*                   intermediateRank,
                              int*                   cudaP2p)
{
    if (p2p)              *p2p              = 0;
    if (read)             *read             = 0;
    if (intermediateRank) *intermediateRank = -1;
    if (cudaP2p)          *cudaP2p          = 0;
    return ncclSuccess;
}

ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* /*system*/,
                              int                    /*rank1*/,
                              int                    /*rank2*/,
                              int*                   net)
{
    if (net) *net = 0;
    return ncclSuccess;
}

ncclResult_t getBusId(int /*cudaDev*/, int64_t* busId)
{
    if (busId) *busId = 0;
    return ncclSuccess;
}

ncclResult_t ncclCommGraphRegister(struct ncclComm* /*comm*/,
                                   void*            /*buff*/,
                                   size_t           /*size*/,
                                   void**           handle)
{
    if (handle) *handle = nullptr;
    return ncclSystemError;
}

ncclResult_t ncclCommGraphDeregister(struct ncclComm* /*comm*/,
                                     struct ncclReg*  /*reg*/)
{
    return ncclSuccess;
}

ncclResult_t ncclShmAllocateShareableBuffer(size_t            /*size*/,
                                            bool              /*legacy*/,
                                            ncclShmIpcDesc_t* /*desc*/,
                                            void**            /*hptr*/,
                                            void**            /*dptr*/)
{
    return ncclSystemError;
}

// --- Controllable seam: ncclStrongStreamAcquire ---------------------------
// Default behaviour preserves the old stub: succeed, hand back a null
// hipStream_t. Tests that want to drive failure into the
// devPeerRmtAddrs-allocation block override g_strongStreamAcquire in their
// fixture SetUp().
static ncclResult_t DefaultStrongStreamAcquire(struct ncclCudaGraph,
                                               struct ncclStrongStream*,
                                               bool,
                                               hipStream_t* stream)
{
    if (stream) *stream = nullptr;
    return ncclSuccess;
}

std::function<ncclResult_t(struct ncclCudaGraph,
                           struct ncclStrongStream*,
                           bool,
                           hipStream_t*)>
    g_strongStreamAcquire = DefaultStrongStreamAcquire;

ncclResult_t ncclStrongStreamAcquire(struct ncclCudaGraph graph,
                                     struct ncclStrongStream* ss,
                                     bool                 concurrent,
                                     hipStream_t*         stream)
{
    return g_strongStreamAcquire(graph, ss, concurrent, stream);
}

// --- Controllable seams: ncclCudaCallocAsync / ncclCudaMemcpyAsync --------
// Substitutes for the header-only function templates in alloc.h. The shim
// macros in p2p-test.cc route the ncclCudaCallocAsync / ncclCudaMemcpyAsync
// macros through these, type-erased to (void*, nbytes), so the test binary
// never reaches real HIP runtime.
//
// Defaults behave like an honest emulator: heap-allocate zeroed memory and
// memcpy bytes between host pointers. ResetP2pFakes() frees any allocations
// the default hook handed out so individual tests don't have to. Tests that
// install their own hook also take responsibility for any memory they hand
// out.
static std::vector<void*> g_fakeAllocations;

static ncclResult_t DefaultFakeCudaCallocAsync(void** ptr, std::size_t nbytes,
                                               hipStream_t /*stream*/)
{
    if (ptr == nullptr) return ncclInvalidArgument;
    void* p = std::calloc(1, nbytes);
    if (p == nullptr && nbytes > 0) return ncclSystemError;
    g_fakeAllocations.push_back(p);
    *ptr = p;
    return ncclSuccess;
}

static ncclResult_t DefaultFakeCudaMemcpyAsync(void* dst, void* src,
                                               std::size_t nbytes,
                                               hipStream_t /*stream*/)
{
    if (nbytes > 0 && (dst == nullptr || src == nullptr)) return ncclInvalidArgument;
    if (nbytes > 0) std::memcpy(dst, src, nbytes);
    return ncclSuccess;
}

std::function<ncclResult_t(void**, std::size_t, hipStream_t)>
    g_fakeCudaCallocAsync = DefaultFakeCudaCallocAsync;
std::function<ncclResult_t(void*, void*, std::size_t, hipStream_t)>
    g_fakeCudaMemcpyAsync = DefaultFakeCudaMemcpyAsync;

void ResetP2pFakes()
{
    g_strongStreamAcquire    = DefaultStrongStreamAcquire;
    g_fakeCudaCallocAsync    = DefaultFakeCudaCallocAsync;
    g_fakeCudaMemcpyAsync    = DefaultFakeCudaMemcpyAsync;
    g_proxyConnect           = DefaultProxyConnect;
    g_proxyCallBlocking      = DefaultProxyCallBlocking;
    g_hipMemGetAddressRange  = DefaultHipMemGetAddressRange;
    g_hipIpcGetMemHandle     = DefaultHipIpcGetMemHandle;
    for (void* p : g_fakeAllocations) std::free(p);
    g_fakeAllocations.clear();
}

ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph     /*graph*/,
                                     struct ncclStrongStream* /*ss*/,
                                     bool                     /*concurrent*/)
{
    return ncclSuccess;
}

ncclResult_t ncclStreamWaitStream(hipStream_t /*a*/,
                                  hipStream_t /*b*/,
                                  hipEvent_t  /*ev*/)
{
    return ncclSuccess;
}

ncclResult_t ncclTopoGetLinkType(struct ncclTopoSystem* /*system*/,
                                 int                    /*cudaDev1*/,
                                 int                    /*cudaDev2*/,
                                 bool*                  isXGMI,
                                 int                    /*maxInter*/,
                                 int                    /*nInter*/,
                                 int*                   /*inter*/)
{
    if (isXGMI) *isXGMI = false;
    return ncclSuccess;
}
