/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Reusable stubs for the NCCL (`nccl*`) symbols the micro-test binary
// references but doesn't define, so it links without librccl.so. Not
// p2p-specific; see nccl_fakes.h for the controllable-seam philosophy.

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "nccl.h"
#include "alloc.h"        // ncclCuMemEnable
#include "debug.h"        // ncclDebugLog, ncclDebugLevel, ...
#include "param.h"        // ncclLoadParam
#include "rocmwrap.h"     // ncclCuMemHandleType
#include "utils.h"        // busIdToInt64
#include "graph.h"        // ncclTopoCheckP2p / Net / GetLinkType
#include "proxy.h"        // ncclProxy* family
#include "register.h"     // ncclRegLocalIsValid
#include "shm.h"          // ncclShm*
#include "comm.h"         // ncclCommGraphRegister / Deregister
#include "strongstream.h" // ncclStrongStream*
#include "mem_manager.h"  // ncclMemTrack / ncclMemUntrack / ncclDynMemMarkExportToPeer

#include "nccl_fakes.h"   // controllable seam hooks

#include <type_traits>

// Signature-drift watchdog: assert each hook still matches the production
// symbol it shadows (templates + macro live in fakes/signature-drift.h).
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_proxyConnect,              ncclProxyConnect);
ASSERT_HOOK_MATCHES_PROD(g_proxyCallBlocking,         ncclProxyCallBlocking);
ASSERT_HOOK_MATCHES_PROD(g_proxyClientQueryFdBlocking, ncclProxyClientQueryFdBlocking);
ASSERT_HOOK_MATCHES_PROD(g_strongStreamAcquire,       ncclStrongStreamAcquire);
// ncclCuMemEnable: header declares `int ncclCuMemEnable()` (rocmwrap.h).
ASSERT_HOOK_MATCHES_PROD(g_cuMemEnable,               ncclCuMemEnable);

#undef ASSERT_HOOK_MATCHES_PROD

// ---------------------------------------------------------------------------
// Trivial globals
// ---------------------------------------------------------------------------

hipMemAllocationHandleType ncclCuMemHandleType =
    hipMemHandleTypePosixFileDescriptor;

// ---------------------------------------------------------------------------
// Logging / param infrastructure
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
    // No-op: leaves cache untouched so NCCL_PARAM callers in non-shimmed
    // TUs see the default. The NCCL_PARAM bodies that p2p-test.cc
    // redirects through g_loadParam bypass this entirely.
}

// Default returns deftVal verbatim -- preserves the pre-hook contract that
// every param sits at its compile-time default.
static int64_t DefaultLoadParam(const char* /*env*/, int64_t deftVal)
{
    return deftVal;
}

std::function<int64_t(const char*, int64_t)> g_loadParam = DefaultLoadParam;

// ---------------------------------------------------------------------------
// Seams worth controlling from tests (return failure by default)
// ---------------------------------------------------------------------------

// --- Controllable seam: ncclCuMemEnable ---------------------------------
// Default returns 0 (the historical stub behaviour) -- existing tests
// keep flowing into the legacy-IPC arm. Tests for the cuMem*-export arm
// install a hook returning 1.
static int DefaultCuMemEnable() { return 0; }
std::function<int()> g_cuMemEnable = DefaultCuMemEnable;
int ncclCuMemEnable() { return g_cuMemEnable(); }

// --- Controllable seam: ncclProxyClientQueryFdBlocking -------------------
// The cuMem* POSIX_FD arm of ipcRegisterBuffer calls this to ship the
// exported fd to the remote proxy and get an imported-fd handle back.
// Default returns ncclSystemError so unexpected calls fail loudly.
static ncclResult_t DefaultProxyClientQueryFdBlocking(
    struct ncclComm*, struct ncclProxyConnector*, int, int*)
{
    return ncclSystemError;
}
std::function<ncclResult_t(struct ncclComm*, struct ncclProxyConnector*,
                           int, int*)>
    g_proxyClientQueryFdBlocking = DefaultProxyClientQueryFdBlocking;

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

ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* /*comm*/,
                                          int              /*rank*/,
                                          void*            /*handle*/,
                                          int*             /*convertedFd*/)
{
    return ncclSystemError;
}

ncclResult_t ncclProxyClientQueryFdBlocking(struct ncclComm*           comm,
                                            struct ncclProxyConnector* proxyConn,
                                            int                        localFd,
                                            int*                       rmtFd)
{
    return g_proxyClientQueryFdBlocking(comm, proxyConn, localFd, rmtFd);
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
// Topology / graph helpers
// ---------------------------------------------------------------------------

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

ncclResult_t DefaultTopoGetLinkType(int, int, bool* isXGMI)
{
    if (isXGMI) *isXGMI = false;
    return ncclSuccess;
}
std::function<ncclResult_t(int, int, bool*)> g_ncclTopoGetLinkType = DefaultTopoGetLinkType;
int g_ncclTopoGetLinkTypeCalls = 0;

ncclResult_t ncclTopoGetLinkType(struct ncclTopoSystem* /*system*/,
                                 int                    cudaDev1,
                                 int                    cudaDev2,
                                 bool*                  isXGMI,
                                 int                    /*maxInter*/,
                                 int                    /*nInter*/,
                                 int*                   /*inter*/)
{
    g_ncclTopoGetLinkTypeCalls++;
    return g_ncclTopoGetLinkType(cudaDev1, cudaDev2, isXGMI);
}

// ---------------------------------------------------------------------------
// Memory-manager tracking stubs.
//
// The rebased p2p.cc reaches ncclCuMem{Alloc,Free} and the shareable-buffer
// import/export paths in alloc.h, which now route allocations through the
// ncclMemManager tracking layer. None of the microtests exercise real HIP
// allocation (the ncclCudaCallocAsync macro is shimmed away), so these are
// pure no-ops that satisfy the linker and report success.
// ---------------------------------------------------------------------------

ncclResult_t ncclMemTrack(struct ncclMemManager* /*manager*/,
                          void*                            /*ptr*/,
                          size_t                           /*size*/,
                          hipMemGenericAllocationHandle_t  /*handle*/,
                          hipMemAllocationHandleType       /*handleType*/,
                          ncclMemType_t                    /*memType*/)
{
    return ncclSuccess;
}

ncclResult_t ncclMemTrackImportFromPeer(struct ncclMemManager* /*manager*/,
                                        void*                           /*ptr*/,
                                        size_t                          /*size*/,
                                        hipMemGenericAllocationHandle_t /*handle*/,
                                        hipMemAllocationHandleType      /*handleType*/,
                                        ncclMemType_t                   /*memType*/,
                                        int                             /*ownerRank*/,
                                        int                             /*ownerDev*/,
                                        void*                           /*ownerPtr*/)
{
    return ncclSuccess;
}

ncclResult_t ncclMemUntrack(struct ncclMemManager* /*manager*/,
                            void*                  /*ptr*/,
                            size_t                 /*size*/)
{
    return ncclSuccess;
}

ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* /*manager*/,
                                        void*                  /*ptr*/,
                                        int                    /*peerRank*/)
{
    return ncclSuccess;
}

// Batch fd-query variant added upstream for multi-segment registration.
// Returns failure by default -- no microtest drives the multi-segment
// path (ncclParamMultiSegmentRegister is stubbed to 0 below).
ncclResult_t ncclProxyClientBatchQueryFdBlocking(struct ncclComm*           /*comm*/,
                                                 struct ncclProxyConnector* /*proxyConn*/,
                                                 int*                       /*localFds*/,
                                                 int*                       /*rmtFds*/,
                                                 int                        /*numSegments*/)
{
    return ncclSystemError;
}

// NCCL_PARAM(MultiSegmentRegister, ...) generated symbol. Return 0 so the
// `multiSegment && ... ncclParamMultiSegmentRegister()` guard in p2p.cc
// keeps the single-segment path the microtests exercise.
int64_t ncclParamMultiSegmentRegister() { return 0; }

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void ResetNcclFakes()
{
    g_strongStreamAcquire          = DefaultStrongStreamAcquire;
    g_proxyConnect                 = DefaultProxyConnect;
    g_proxyCallBlocking            = DefaultProxyCallBlocking;
    g_loadParam                    = DefaultLoadParam;
    g_cuMemEnable                  = DefaultCuMemEnable;
    g_proxyClientQueryFdBlocking   = DefaultProxyClientQueryFdBlocking;
    g_ncclTopoGetLinkType          = DefaultTopoGetLinkType;
    g_ncclTopoGetLinkTypeCalls     = 0;
}
