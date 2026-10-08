/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/rma/rma_proxy.cc.
//
// AICOMRCCL-1851: cover the NCCL 2.30.7 fix of NVIDIA/nccl PR #2187,
// "rma_proxy: fix MR registration type for host-NUMA cpuAccessSignals".
//
// The defect: ncclRmaProxyCtxAllocGraph always passed NCCL_PTR_CUDA when
// registering cpuAccessSignalsDev. But allocMemCPUAccessible only returns
// device memory when GDRCopy is enabled; otherwise it returns host-NUMA pinned
// memory. A net plugin asked to register host memory as device memory rejects
// the registration, so RMA-proxy context creation failed outright. The fix
// derives the type from the GDR handle allocMemCPUAccessible hands back:
// non-NULL means GDRCopy-mapped device memory, NULL means host.
//
// This matters more on ROCm than upstream: GDRCopy is off by default, so the
// host-NUMA arm is the *default* path here, not an edge case.
//
// ncclRmaProxyCtxAllocGraph is file-static, so this TU reaches it by
// #include-ing the production .cc directly (via RMA_PROXY_CC_PATH), the
// standard rccl-UnitTestsMicro pattern (see MICROTEST_README.md).
//
// Two seams make these real tests of the fix rather than argument assertions:
//
//   1. allocMemCPUAccessible is a static template in gdrwrap.h, so it cannot be
//      stubbed at link time -- which is an advantage. The test pivots the real
//      helper on the real discriminator, the extern global ncclGdrCopy, and the
//      GDR handle the production code branches on is the one production code
//      produced.
//   2. The network is already an ncclRma_t function-pointer vtable. FakeRma
//      records every registration and, in StrictPlugin*, rejects a host pointer
//      offered as NCCL_PTR_CUDA exactly as a real plugin does -- so the test
//      reproduces the reported failure, not just the wrong argument value.
//
// Why this has its own binary rather than joining rccl-UnitTestsMicro: see the
// rccl-UnitTestsMicroRmaProxy entry in MICROTEST_README.md.

#include <gtest/gtest.h>

#include <hsa/hsa.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/rma_fakes.h"

// rma_proxy.cc declares params of its own. Route the generated bodies through
// g_loadParam so they stay per-test controllable and the real ncclLoadParam is
// never linked. Must precede the unit under test.
#include "fakes/param_redirect.h"

#include "gin/gin_host.h"
#include "nccl.h"
#include "comm.h"
#include "rma/rma_proxy.h"

namespace {
// Settable behind the ncclParamDmaBufEnable() stub below. Default 0 keeps
// ncclRmaProxyRegMrSym on its plain regMrSym arm: the DMA-BUF arm reaches
// hsa_amd_portable_export_dmabuf through a function pointer no host-only build
// can satisfy. The DMA-BUF dispatch is not what this ticket is about.
int64_t g_paramDmaBufEnable = 0;
}  // namespace

// --- Link-satisfying stubs for rma_proxy.cc's non-static externals ---------
//
// Defined here rather than in a shared fakes/ .cc because none of them has an
// owning fakes file today and this is the only binary that needs them (same
// reasoning as p2p-test.cc's allocTracker and ras-test.cc's ncclSetThreadName).
// They must precede the #include of the unit under test.
int64_t ncclParamDmaBufEnable() { return g_paramDmaBufEnable; }
// NCCL_PARAM(IbDataDirect, "IB_DATA_DIRECT", 1), declared by src/transport/net_ib/init.cc.
int64_t ncclParamIbDataDirect() { return 1; }
// src/debug.cc. ncclRmaProxyConnectOnce names its progress thread; no behaviour to assert.
void ncclSetThreadName(std::thread&, const char*, ...) {}
// src/graph/topo.cc. Only ncclRmaProxyConnectOnce calls it, which these tests do not.
ncclResult_t ncclTopoGetLocalRmaDevs(struct ncclComm*, int*, int*) { return ncclInternalError; }
// src/rma/rma_proxy_progress.cc. Only the proxy thread calls it; never started here.
ncclResult_t ncclRmaProxyProgress(ncclRma_t*, void*) { return ncclInternalError; }
// libhsa-runtime64, reached from rma_proxy.cc's file-static getDmaBufFd via the
// HSACHECK macro (src/include/rocmwrap.h) -- a required HSA entry point, called
// directly rather than through a pfn gate. Host tests link no ROCr, and no RCCL
// TU defines it, so there is no owning fakes file. It must be defined even
// though no test reaches it: at -O0 (the host-test default, and what CI builds)
// the dead static survives and the reference is emitted. A Release build
// discards it via --gc-sections, which is why this only shows up in Debug.
extern "C" hsa_status_t hsa_status_string(hsa_status_t, const char** status_string) {
    if (status_string != nullptr) *status_string = "fake HSA status (host microtest)";
    return HSA_STATUS_SUCCESS;
}
// Defined by src/init.cc. Not a hook: allocMemCPUAccessible() is a header-static
// template, so this global is the only way to steer it. NULL is production's
// default (NCCL_GDRCOPY_ENABLE defaults to 0) and selects host-NUMA memory.
gdr_t ncclGdrCopy = NULL;

// Pull the unit under test in directly so its file-static functions are
// reachable. Must come after the stubs and headers above are in scope.
#include RMA_PROXY_CC_PATH

// alloc.h data symbol, referenced by the ncclCudaCalloc templates the
// GDRCopy-on arm walks through. Must follow the include, which brings
// allocationTracker into scope. Zero-initialised.
struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};

namespace {

// Sentinel GDRCopy handle. ncclGdrInit() on AMD returns this exact value for a
// supported arch (gdrwrap.h), so using it keeps the test honest about what a
// GDRCopy-enabled comm actually looks like here.
gdr_t const kGdrEnabled = (gdr_t)0x12345678L;

// ===========================================================================
// FakeRma -- stand-in for the RMA network behind the ncclRma_t vtable.
//
// Records every memory registration, and optionally enforces the type check a
// real plugin performs: a buffer that is host memory cannot be registered as
// NCCL_PTR_CUDA. `hostAddrs` is populated by the fixture's hipHostMalloc hook,
// so "is this host memory?" is answered by where the pointer actually came
// from, not by what the test guessed.
// ===========================================================================
struct Registration {
    void*    addr     = nullptr;
    size_t   size     = 0;
    int      type     = 0;
    uint64_t mrFlags  = 0;
};

struct FakeRma {
    std::vector<Registration> regs;
    std::vector<void*>        deregs;
    // When true, reject a registration whose declared type contradicts the
    // pointer's provenance -- what the net plugin in NVIDIA/nccl PR #2187 did.
    bool             strictMemType = false;
    // Refuse every registration past the first `rejectAfterNReg`. Used to prove
    // the success the other tests assert is a real oracle and not an
    // unconditional pass. -1 disables.
    int              rejectAfterNReg = -1;
    int              regAttempts     = 0;
    std::set<void*>* hostAddrs     = nullptr;
    int              destroyCtxCalls = 0;

    const Registration* find(const void* addr) const {
        for (const auto& r : regs) {
            if (r.addr == addr) return &r;
        }
        return nullptr;
    }

    ncclResult_t reg(void* data, size_t size, int type, uint64_t mrFlags, void** mhandle) {
        ++regAttempts;
        if ((rejectAfterNReg >= 0 && regAttempts > rejectAfterNReg) ||
            (strictMemType && type == NCCL_PTR_CUDA && hostAddrs != nullptr &&
             hostAddrs->count(data) != 0)) {
            // Mirror a real plugin: the registration is refused, nothing is
            // recorded, and no handle comes back.
            *mhandle = nullptr;
            return ncclInternalError;
        }
        regs.push_back({data, size, type, mrFlags});
        // Non-NULL and stable; production stores it but never dereferences it here.
        *mhandle = reinterpret_cast<void*>(regs.size());
        return ncclSuccess;
    }

    ncclRma_t vtable() {
        ncclRma_t v{};
        v.createContext  = &FakeRma::TrampCreateContext;
        v.regMrSym       = &FakeRma::TrampRegMrSym;
        v.regMrSymDmaBuf = &FakeRma::TrampRegMrSymDmaBuf;
        v.deregMrSym     = &FakeRma::TrampDeregMrSym;
        v.destroyContext = &FakeRma::TrampDestroyContext;
        return v;
    }

    // Production's rmaCtx and rmaCollComm are distinct objects: rmaCollComm from
    // connect(), rmaCtx from createContext(collComm, ...). Keep them distinct here
    // too, so production cannot pass one where the other belongs.
    struct Handle { FakeRma* rma; };
    Handle ctxH{this};
    Handle collH{this};

private:
    static FakeRma* Rma(void* h) { return static_cast<Handle*>(h)->rma; }
    static ncclResult_t TrampCreateContext(void* collComm, ncclRmaConfig_t*, void** rmaCtx) {
        *rmaCtx = &Rma(collComm)->ctxH;
        return ncclSuccess;
    }
    static ncclResult_t TrampRegMrSym(void* collComm, void* data, size_t size, int type,
                                      uint64_t mrFlags, void** mhandle) {
        return Rma(collComm)->reg(data, size, type, mrFlags, mhandle);
    }
    static ncclResult_t TrampRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type,
                                            uint64_t /*offset*/, int /*fd*/, uint64_t mrFlags,
                                            void** mhandle) {
        return Rma(collComm)->reg(data, size, type, mrFlags, mhandle);
    }
    static ncclResult_t TrampDeregMrSym(void* collComm, void* mhandle) {
        Rma(collComm)->deregs.push_back(mhandle);
        return ncclSuccess;
    }
    static ncclResult_t TrampDestroyContext(void* rmaCtx) {
        ++Rma(rmaCtx)->destroyCtxCalls;
        return ncclSuccess;
    }
};

// ===========================================================================
// Fixture: builds the minimum ncclComm a context needs, then drives the public
// lifecycle a production client uses -- ncclRmaProxyCreateContext, which runs
// ncclRmaProxyCtxAllocGraph, and ncclRmaProxyDestroyContext.
// ===========================================================================
class RmaProxyAllocGraphTest : public ::testing::Test {
protected:
    static constexpr int kNRanks    = 4;
    static constexpr int kNumRmaSig = 2;
    // The function derives this itself; recomputed here so the assertions do
    // not simply echo whatever production passed.
    static constexpr size_t kSignalsBufSize =
        static_cast<size_t>(kNRanks) * kNumRmaSig * sizeof(uint64_t);
    static constexpr size_t kFlushBufSize = static_cast<size_t>(kNRanks) * sizeof(uint64_t);

    std::unique_ptr<ncclComm>  comm_;
    // Owned by production: allocated in ncclRmaProxyCreateContext, released in
    // ncclRmaProxyDestroyContext.
    ncclRmaProxyCtx*           ctx_ = nullptr;
    ncclNetProperties_t        props_{};
    FakeRma                    rmaNet_;
    ncclRma_t                  rma_{};

    // Every pointer hipHostMalloc handed out during the test. FakeRma consults
    // this to decide whether a NCCL_PTR_CUDA registration is a lie.
    std::set<void*> hostAddrs_;
    std::unique_ptr<ScopedHook<hipError_t(void**, std::size_t, unsigned)>> hostMallocHook_;

    gdr_t savedGdrCopy_ = NULL;

    void SetUp() override {
        ResetHipFakes();
        ResetRmaFakes();
        ResetNcclFakes();

        savedGdrCopy_ = ncclGdrCopy;
        ncclGdrCopy   = NULL;  // production default: GDRCopy off
        g_paramDmaBufEnable = 0;

        // The GDRCopy-on arm runs ncclGdrCudaCalloc -> ncclCudaCallocDebug,
        // which walks getSideStream() (real getBusId, from src/misc/utils.cc)
        // and then the no-side-stream fallback: create stream, allocate,
        // memsetAsync, synchronize, destroy. Open each of those gates; the
        // allocation seams already malloc for real, so the buffer the test
        // later inspects is genuine memory.
        g_hipDeviceGetPCIBusIdResult = hipSuccess;
        g_hipStreamCreateResult      = hipSuccess;
        g_hipAsyncOpsResult          = hipSuccess;
        g_hipStreamSynchronize       = [](hipStream_t) { return hipSuccess; };
        // Keep ncclCudaCalloc off the cuMem arm, which needs a real driver.
        g_cuMemEnable = [] { return 0; };
        // ncclRmaProxyDestroyContext's GDR arm reaches ncclCudaFree, which
        // queries the allocation range for its accounting before cudaFree.
        g_hipMemGetAddressRange = [](hipDeviceptr_t* pbase, std::size_t* psize,
                                     hipDeviceptr_t dptr) {
            if (pbase != nullptr) *pbase = dptr;
            if (psize != nullptr) *psize = 0;
            return hipSuccess;
        };

        hostMallocHook_ = std::make_unique<ScopedHook<hipError_t(void**, std::size_t, unsigned)>>(
            g_hipHostMalloc, [this](void** ptr, std::size_t size, unsigned) {
                void* p = std::malloc(size);
                if (p == nullptr) return hipErrorOutOfMemory;
                *ptr = p;
                hostAddrs_.insert(p);
                return hipSuccess;
            });

        comm_ = std::make_unique<ncclComm>();
        comm_->rank              = 0;
        comm_->nRanks            = kNRanks;
        comm_->config.numRmaSig  = kNumRmaSig;
        comm_->memManager        = nullptr;
        ncclMemoryStackConstruct(&comm_->memPermanent);

        // A capable NIC, so no registration below can be steered by a missing
        // capability. The DMA-BUF bit is nonetheless never consulted here:
        // ncclRmaProxyRegMrSym tests ncclParamDmaBufEnable() first, and the
        // fixture pins that to 0 (see the note on g_paramDmaBufEnable above).
        props_.ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF;
        // Drives maxInflightRequests and the circular-buffer queue size; 0
        // would leave production sanitizing down to a zero-length queue.
        props_.maxRecvs   = 1;

        rmaNet_.hostAddrs = &hostAddrs_;
        rma_ = rmaNet_.vtable();
        comm_->rmaState.rmaProxyState.ncclRma = &rma_;
    }

    void TearDown() override {
        DestroyCtx();
        // persistentQueues / inProgressQueues live on this stack, not the heap.
        ncclMemoryStackDestruct(&comm_->memPermanent);
        hostMallocHook_.reset();
        ncclGdrCopy = savedGdrCopy_;
        g_paramDmaBufEnable = 0;
        ResetRmaFakes();
        ResetHipFakes();
        ResetNcclFakes();
    }

    // The entry point a production client calls. On failure production destroys
    // the partial context itself, so ctx_ is only adopted on success.
    ncclResult_t CreateCtx() {
        void* out = nullptr;
        const int destroysBefore = rmaNet_.destroyCtxCalls;
        ncclResult_t res =
            ncclRmaProxyCreateContext(comm_.get(), &rmaNet_.collH, props_, 0, &out);
        if (res == ncclSuccess) {
            ctx_ = static_cast<ncclRmaProxyCtx*>(out);
            // A successful create must not have torn the context back down.
            EXPECT_EQ(destroysBefore, rmaNet_.destroyCtxCalls);
        }
        return res;
    }

    void DestroyCtx() {
        if (ctx_ == nullptr) return;
        EXPECT_EQ(ncclSuccess, ncclRmaProxyDestroyContext(&rma_, ctx_));
        ctx_ = nullptr;
        // Production has released every CPU-accessible buffer it allocated, so
        // no recorded address is live. Stale entries would otherwise make the
        // next arm's provenance check fire on a recycled pointer.
        hostAddrs_.clear();
        // The registration log now refers to freed pointers.
        rmaNet_.regs.clear();
    }

    // Preamble shared by the two tests that sweep both GDR arms in a loop:
    // select the arm, tear the previous pass's context down (a no-op on the
    // first pass), and rebuild. Wrap call sites in ASSERT_NO_FATAL_FAILURE --
    // a failed ASSERT here only returns from this helper.
    void ReCreateCtxForArm(gdr_t gdr) {
        DestroyCtx();
        ncclGdrCopy = gdr;
        ASSERT_EQ(ncclSuccess, CreateCtx()) << "gdr=" << (void*)gdr;
    }

    // The cpuAccessSignals registration, located by the address production
    // actually allocated (not by position in the call sequence).
    const Registration* CpuAccessSignalsReg() const {
        return rmaNet_.find(ctx_->cpuAccessSignalsDev);
    }
    const Registration* FlushBufReg() const { return rmaNet_.find(ctx_->flushBufDev); }
};

// ---------------------------------------------------------------------------
// The regression anchor. GDRCopy off -- ROCm's default -- so
// allocMemCPUAccessible returns host-NUMA pinned memory and nulls the GDR
// handle. Pre-fix, this registration went out as NCCL_PTR_CUDA.
// ---------------------------------------------------------------------------
TEST_F(RmaProxyAllocGraphTest, GdrCopyOff_RegistersCpuAccessSignalsAsHost) {
    ncclGdrCopy = NULL;

    ASSERT_EQ(ncclSuccess, CreateCtx());

    // Premise check: the allocator really did take the host arm. Without this,
    // a change that made allocMemCPUAccessible always return device memory
    // would leave the type assertion below passing for the wrong reason.
    EXPECT_EQ(nullptr, ctx_->cpuAccessSignalsGdrHandle);
    ASSERT_NE(nullptr, ctx_->cpuAccessSignalsDev);

    const Registration* reg = CpuAccessSignalsReg();
    ASSERT_NE(nullptr, reg) << "cpuAccessSignalsDev was never registered";
    EXPECT_EQ(NCCL_PTR_HOST, reg->type)
        << "host-NUMA cpuAccessSignals registered as device memory (NVIDIA/nccl PR #2187)";
    EXPECT_EQ(kSignalsBufSize, reg->size);
}

// ---------------------------------------------------------------------------
// The other arm. GDRCopy on -> allocMemCPUAccessible returns GDRCopy-mapped
// device memory with a non-NULL handle, and the registration must stay CUDA.
// Pins the fix against an over-correction to an unconditional NCCL_PTR_HOST.
// ---------------------------------------------------------------------------
TEST_F(RmaProxyAllocGraphTest, GdrCopyOn_RegistersCpuAccessSignalsAsCuda) {
    ncclGdrCopy = kGdrEnabled;

    ASSERT_EQ(ncclSuccess, CreateCtx());

    ASSERT_NE(nullptr, ctx_->cpuAccessSignalsGdrHandle)
        << "GDRCopy path did not produce a handle; the premise of this test is gone";
    ASSERT_NE(nullptr, ctx_->cpuAccessSignalsDev);

    const Registration* reg = CpuAccessSignalsReg();
    ASSERT_NE(nullptr, reg) << "cpuAccessSignalsDev was never registered";
    EXPECT_EQ(NCCL_PTR_CUDA, reg->type);
    // ncclGdrCudaCalloc rounds its allocation up to GPU_PAGE_SIZE, so this is
    // the arm where a registration sized from the padded block instead of
    // signalsBufSize would otherwise go unnoticed.
    EXPECT_EQ(kSignalsBufSize, reg->size);
}

// ---------------------------------------------------------------------------
// The defect as reported, rather than as an argument value: a plugin that
// rejects a host pointer offered as device memory. Pre-fix this made
// ncclRmaProxyCtxAllocGraph -- and so RMA-proxy context creation -- fail.
// ---------------------------------------------------------------------------
TEST_F(RmaProxyAllocGraphTest, StrictPluginRejectsWrongMemoryType_AllocGraphStillSucceeds) {
    ncclGdrCopy          = NULL;
    rmaNet_.strictMemType = true;

    ASSERT_EQ(ncclSuccess, CreateCtx())
        << "net plugin rejected the cpuAccessSignals registration due to wrong memory type "
           "(NVIDIA/nccl PR #2187)";

    const Registration* reg = CpuAccessSignalsReg();
    ASSERT_NE(nullptr, reg);
    EXPECT_EQ(NCCL_PTR_HOST, reg->type);
}

// ---------------------------------------------------------------------------
// Negative control for the three tests above. They all assert context creation
// returns ncclSuccess, which is only evidence if a refused cpuAccessSignals
// registration can still make it fail -- i.e. if the NCCLCHECK around that
// registration really does propagate. Pin that oracle.
//
// Registration order under ncclRmaProxyCreateContext is signalsDev (from
// ncclRmaProxyCtxAlloc), then cpuAccessSignals, then flushBuf. Allowing exactly
// one registration therefore refuses the cpuAccessSignals one specifically; the
// size check below fails loudly if that order ever changes.
// ---------------------------------------------------------------------------
TEST_F(RmaProxyAllocGraphTest, PluginRefusesCpuAccessSignalsRegistration_CreateContextFails) {
    ncclGdrCopy             = NULL;
    rmaNet_.rejectAfterNReg = 1;

    EXPECT_NE(ncclSuccess, CreateCtx())
        << "a refused MR registration did not propagate out of "
           "ncclRmaProxyCtxAllocGraph; the ncclSuccess assertions in the other "
           "tests prove nothing";
    // Production stopped at the refusal: signalsDev got through, nothing after.
    EXPECT_EQ(1u, rmaNet_.regs.size());
}

// ---------------------------------------------------------------------------
// The flush buffer in the same function is allocated with
// hipExtMallocWithFlags(hipDeviceMallocFinegrained) regardless of GDRCopy, so
// its registration is unconditionally NCCL_PTR_CUDA. Guards against the fix
// leaking one line further down.
// ---------------------------------------------------------------------------
TEST_F(RmaProxyAllocGraphTest, FlushBufferStaysCudaOnBothGdrArms) {
    for (gdr_t gdr : {(gdr_t)NULL, kGdrEnabled}) {
        ASSERT_NO_FATAL_FAILURE(ReCreateCtxForArm(gdr));

        const Registration* reg = FlushBufReg();
        ASSERT_NE(nullptr, reg) << "flushBufDev was never registered, gdr=" << (void*)gdr;
        EXPECT_EQ(NCCL_PTR_CUDA, reg->type) << "gdr=" << (void*)gdr;
        EXPECT_EQ(kFlushBufSize, reg->size) << "gdr=" << (void*)gdr;
    }
}

// ---------------------------------------------------------------------------
// Whichever type is chosen, the proxy polls these signals from the host, so the
// registration must keep NCCL_NET_MR_FLAG_FORCE_SO.
// ---------------------------------------------------------------------------
TEST_F(RmaProxyAllocGraphTest, ForceStrongOrderingPreservedOnBothGdrArms) {
    for (gdr_t gdr : {(gdr_t)NULL, kGdrEnabled}) {
        ASSERT_NO_FATAL_FAILURE(ReCreateCtxForArm(gdr));

        const Registration* reg = CpuAccessSignalsReg();
        ASSERT_NE(nullptr, reg) << "cpuAccessSignalsDev was never registered, gdr="
                                << (void*)gdr;
        EXPECT_EQ(static_cast<uint64_t>(NCCL_NET_MR_FLAG_FORCE_SO), reg->mrFlags)
            << "gdr=" << (void*)gdr;
    }
}

}  // namespace
