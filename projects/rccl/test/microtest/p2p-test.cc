/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Microtests for src/transport/p2p.cc.
//
// This binary does NOT link librccl.so -- it compiles the hipified p2p.cc
// directly into the test TU via the #include below, with everything p2p.cc
// references (proxy, HIP driver shims, topology helpers, etc.) provided as
// stubs in fakes/. That gives the tests visibility into static helpers like
// ipcRegisterBuffer and the ability to drive their failure paths deterministically.
//
// See README.md in this directory for the full rationale, the fake-layer
// architecture, and the recipe for adding a new test.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "fakes/p2p_fakes.h"

// Pull in alloc.h NOW so its macros (ncclCudaCallocAsync etc.) are visible
// to be #undef'd. p2p.cc's transitive includes would otherwise be the first
// to see them, and the shim below would land too late.
#include "alloc.h"

// Same pattern for param.h: pull it in now so we can #undef NCCL_PARAM and
// replace it with a redirector that routes every generated ncclParamXxx()
// through g_loadParam on every call (no caching). Without this,
// ncclParamLegacyCudaRegister() and friends would cache their default on
// first call -- which means tests can't flip them between cases. The
// redirected body matches the real NCCL_PARAM signature
// (`int64_t ncclParam<name>()`) but skips the cache and uninitialized
// machinery entirely.
#include "param.h"
#undef NCCL_PARAM
#define NCCL_PARAM(name, env, deftVal) \
    int64_t ncclParam##name() { return g_loadParam((env), (deftVal)); }

// Macro shim: replace the header-only function templates ncclCudaCallocAsync
// and ncclCudaMemcpyAsync from alloc.h with thin trampolines that route
// through hookable fakes in fakes/p2p_fakes.cc. Without this, p2p.cc's call
// sites bind directly to the templates, which hit real HIP runtime (no GPU
// in this binary by design).
//
// The shims preserve type information at the call site via sizeof(**ptr) /
// sizeof(*dst); the fake hooks themselves are type-erased to (void*, nbytes).
// This mirrors the existing macro-intercept pattern used for ncclDebugLog
// and ncclLoadParam.
#undef ncclCudaCallocAsync
#undef ncclCudaMemcpyAsync
// ncclCudaCallocAsync is a variadic macro in alloc.h: (ptr, nelem, stream, ...)
// where the optional trailing arg is a memManager pointer added by RCCL. We
// absorb it with __VA_ARGS__ so both 3-arg and 4-arg call sites compile.
#define ncclCudaCallocAsync(ptr, nelem, stream, ...) \
    g_fakeCudaCallocAsync(reinterpret_cast<void**>(ptr), \
                          (nelem) * sizeof(**(ptr)), (stream))
#define ncclCudaMemcpyAsync(dst, src, nelem, stream) \
    g_fakeCudaMemcpyAsync(reinterpret_cast<void*>(dst), \
                          reinterpret_cast<void*>(src), \
                          (nelem) * sizeof(*(dst)), (stream))

// Macro shim: route the HIP driver entry points that ipcRegisterBuffer's
// fresh-registration arm calls (hipMemGetAddressRange, hipIpcGetMemHandle)
// through hookable fakes. The real symbols resolve at link time from
// hip::host but would need a real GPU at runtime. Same pattern as the
// ncclCudaCallocAsync shim above.
#define hipMemGetAddressRange(pbase, psize, dptr) \
    g_hipMemGetAddressRange((pbase), (psize), (dptr))
#define hipIpcGetMemHandle(handle, devPtr) \
    g_hipIpcGetMemHandle((handle), (devPtr))

// Same pattern for the three cuMem*-export entry points the ROCm 7+ arm
// of ipcRegisterBuffer calls. Without these macro shims, the call sites
// bind directly to the real hip::host symbols and need a GPU at runtime.
#define hipMemRetainAllocationHandle(handle, addr) \
    g_hipMemRetainAllocationHandle((handle), (addr))
#define hipMemExportToShareableHandle(shareableHandle, handle, handleType, flags) \
    g_hipMemExportToShareableHandle((shareableHandle), (handle), (handleType), (flags))
#define hipMemRelease(handle) \
    g_hipMemRelease((handle))

// Pull in the hipified copy of p2p.cc (cudaXxx -> hipXxx rewrites already
// applied by the hipify pass that runs as part of the main RCCL build).
// P2P_CC_PATH is defined by this target's CMakeLists.txt as a string
// literal pointing at ${PROJECT_BINARY_DIR}/hipify/src/transport/p2p.cc.
#include P2P_CC_PATH

// ===========================================================================
// Default fixture for every test in this file.
//
// Several tests install per-test hooks into the controllable seams declared
// in fakes/p2p_fakes.h (e.g. g_strongStreamAcquire). ResetP2pFakes() puts
// every hook back to its default in TearDown so tests don't leak state into
// each other.
// ===========================================================================
class P2pMicrotest : public ::testing::Test {
protected:
    void TearDown() override { ResetP2pFakes(); }
};

// ===========================================================================
// Helpers: lightweight builders for the recurring input/output shapes.
// ===========================================================================

namespace {

// ScopedHook -- RAII wrapper around a controllable seam (any of the
// std::function<...> hooks declared in fakes/p2p_fakes.h).
//
// Installs the test's behaviour on construction, counts calls via .calls,
// and restores the previous behaviour on destruction.
template <typename FnSig>
class ScopedHook;

template <typename R, typename... Args>
class ScopedHook<R(Args...)> {
public:
    template <typename Callable>
    ScopedHook(std::function<R(Args...)>& slot, Callable fn)
        : slot_(slot), saved_(std::move(slot))
    {
        slot_ = [this, fn = std::move(fn)](Args... args) -> R {
            ++calls;
            return fn(std::forward<Args>(args)...);
        };
    }
    ~ScopedHook() { slot_ = std::move(saved_); }

    ScopedHook(const ScopedHook&)            = delete;
    ScopedHook& operator=(const ScopedHook&) = delete;
    ScopedHook(ScopedHook&&)                 = delete;
    ScopedHook& operator=(ScopedHook&&)      = delete;

    int calls = 0;
private:
    std::function<R(Args...)>& slot_;
    std::function<R(Args...)>  saved_;
};

// CTAD: deduce R(Args...) from the std::function<R(Args...)>& argument.
template <typename R, typename... Args, typename Callable>
ScopedHook(std::function<R(Args...)>&, Callable) -> ScopedHook<R(Args...)>;

// CommBuilder -- fluent builder that owns the backing storage for the
// fields of ncclComm that ipcRegisterBuffer reads.
class CommBuilder {
public:
    CommBuilder& WithLocalRank(int peerRank, int peerLocalRank) {
        if (!rankToLocalRankInstalled_) {
            comm_.rankToLocalRank = rankToLocalRankStorage_.data();
            rankToLocalRankInstalled_ = true;
        }
        rankToLocalRankStorage_[peerRank] = peerLocalRank;
        return *this;
    }

    CommBuilder& WithMaxLocalRanks() {
        comm_.localRanks = NCCL_MAX_LOCAL_RANKS;
        return *this;
    }

    CommBuilder& WithSharedRes() {
        comm_.sharedRes = &sharedResStorage_;
        return *this;
    }

    // gproxyConn is a bare pointer in ncclComm; the real ncclCommInit
    // allocates it sized to comm->nRanks. Tests that drive the
    // fresh-registration arm hand-roll a backing array sized to nRanks
    // (must be > max peerRank the test will exercise).
    CommBuilder& WithProxyConnArray(int nRanks) {
        gproxyConnStorage_.assign(nRanks, ncclProxyConnector{});
        comm_.gproxyConn = gproxyConnStorage_.data();
        comm_.nRanks     = nRanks;
        return *this;
    }

    ncclComm& comm() { return comm_; }
    operator ncclComm&() { return comm_; }

    CommBuilder() = default;
    CommBuilder(const CommBuilder&)            = delete;
    CommBuilder& operator=(const CommBuilder&) = delete;

private:
    ncclComm comm_{};
    bool rankToLocalRankInstalled_ = false;
    std::array<int, NCCL_MAX_LOCAL_RANKS> rankToLocalRankStorage_{};
    ncclSharedResources sharedResStorage_{};
    std::vector<ncclProxyConnector> gproxyConnStorage_;
};

// ReusableIpcInfo -- owns the per-peer ncclIpcRegInfo + host-side
// remote-address slot that the reuse path keys off. Drop it into a
// regRecord with .InstallInto(regRecord).
struct ReusableIpcInfo {
    ncclIpcRegInfo info{};
    std::array<uintptr_t, NCCL_MAX_LOCAL_RANKS> hostPeerRmtAddrs{};
    int peerLocalRank;

    ReusableIpcInfo(int peerRank,
                    int peerLocalRank_,
                    uintptr_t rmtRegAddr,
                    bool legacyIpcCap)
        : peerLocalRank(peerLocalRank_)
    {
        info.peerRank             = peerRank;
        info.impInfo.rmtRegAddr   = reinterpret_cast<void*>(rmtRegAddr);
        info.impInfo.legacyIpcCap = legacyIpcCap;
        hostPeerRmtAddrs[peerLocalRank] = rmtRegAddr;
    }

    void InstallInto(ncclReg& regRecord)
    {
        // ipcInfos is now a dynamically-allocated pointer in ncclReg (not a
        // fixed array). Allocate it calloc-style so the function's
        // "ipcInfos == NULL" guard treats it as already-sized and skips
        // ncclRealloc. Size to NCCL_MAX_LOCAL_RANKS to match
        // comm.localRanks = NCCL_MAX_LOCAL_RANKS set in the test.
        if (regRecord.ipcInfos == nullptr) {
            regRecord.ipcInfos = static_cast<ncclIpcRegInfo**>(
                std::calloc(NCCL_MAX_LOCAL_RANKS, sizeof(ncclIpcRegInfo*)));
            regRecord.ipcInfosSize = NCCL_MAX_LOCAL_RANKS;
        }
        regRecord.ipcInfos[peerLocalRank]      = &info;
        regRecord.regIpcAddrs.hostPeerRmtAddrs = hostPeerRmtAddrs.data();
    }
};

// IpcRegOutputs -- the four OUT parameters of ipcRegisterBuffer, pre-seeded
// with sentinel values so accidental no-ops are visible as test failures.
struct IpcRegOutputs {
    static constexpr uintptr_t kSentinel = 0xDEADBEEFDEADBEEFull;
    int        regBufFlag    = static_cast<int>(kSentinel);
    uintptr_t  offsetOut     = kSentinel;
    uintptr_t* peerRmtAddrs  = reinterpret_cast<uintptr_t*>(kSentinel);
};

// CallIpcRegisterBuffer -- thin wrapper so test bodies aren't dominated by
// a 12-line argument list.
ncclResult_t CallIpcRegisterBuffer(ncclComm& comm,
                                   const void* userbuff,
                                   size_t buffSize,
                                   int* peerRanks,
                                   int nPeers,
                                   ncclIpcRegType type,
                                   ncclReg* regRecord,
                                   IpcRegOutputs& out,
                                   bool* isLegacyIpc)
{
    return ipcRegisterBuffer(&comm, userbuff, buffSize, peerRanks, nPeers,
                             type, regRecord,
                             &out.regBufFlag, &out.offsetOut,
                             &out.peerRmtAddrs, isLegacyIpc);
}

}  // namespace

// ===========================================================================
// Tests
// ===========================================================================

// Regression test for NVIDIA/nccl#1859 ("user-buffer p2p + coll rmtAddr
// nullptr error"), fixed by NCCL PR #1861 (commit ebd1e92), ported to
// RCCL in this branch.
//
// The bug lived in ipcRegisterBuffer's post-loop strong-stream block:
//
//   Pre-fix code (inside the COLLECTIVE branch only):
//     if (devPeerRmtAddrs == NULL || needUpdate)
//         ncclCudaCallocAsync(&devPeerRmtAddrs, ...);   // zero-filled
//     if (needUpdate)
//         ncclCudaMemcpyAsync(devPeerRmtAddrs, host...);// skipped when !needUpdate
//
// The fix (ebd1e92) replaced this with:
//   1. An unconditional `if (needUpdate)` block (alloc + memcpy) that fires
//      for ALL types -- so a SENDRECV fresh-registration always populates
//      devPeerRmtAddrs.
//   2. Deletion of the old COLLECTIVE-specific inner block -- on the reuse
//      path, devPeerRmtAddrs is already correct from step 1.
//
// This test drives the actual two-call sequence:
//   Call 1: ipcRegisterBuffer with NCCL_IPC_SENDRECV on a fresh regRecord
//           (ipcInfos[peer] == NULL). The fresh-registration arm fires,
//           sets needUpdate=true, and the `if (needUpdate)` block allocates
//           and populates devPeerRmtAddrs.
//   Call 2: ipcRegisterBuffer with NCCL_IPC_COLLECTIVE on the same regRecord
//           (ipcInfos[peer] now set -- reuse arm). needUpdate stays false.
//           devPeerRmtAddrs was already set in call 1; the function should
//           return it unchanged.
//
// With the pre-fix code call 1 never touches devPeerRmtAddrs (it was
// COLLECTIVE-only), so call 2 enters the buggy block with devPeerRmtAddrs==NULL
// and needUpdate==false: calloc fires but memcpy is skipped, leaving zeros.
TEST_F(P2pMicrotest, IpcRegisterBuffer_RegressionNcclIssue1859_P2pThenCollectivePopulatesDevTable)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr int       kNRanks        = kPeerRank + 1;
    constexpr uintptr_t kBegAddr       = 0x10000;
    constexpr uintptr_t kBaseAddr      = kBegAddr;  // hipMemGetAddressRange returns this
    constexpr size_t    kBaseSize      = 0x2000;    // covers kBegAddr + buffSize
    constexpr uintptr_t kBuffOffset    = 0x40;
    constexpr uintptr_t kRmtRegAddr    = 0xA000;    // canned remote addr from proxy

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    // ipcInfos starts NULL -- fresh-registration arm will fire on call 1.

    // Hooks for the fresh-registration path (legacy IPC arm).
    // g_loadParam: force ncclParamLegacyCudaRegister() == 1 so the
    //   HIP_VERSION < 71260540 branch sets legacyIpcCap = 1.
    // g_hipMemGetAddressRange: return a base/size that covers our buffer.
    // g_hipIpcGetMemHandle: succeed with a canned handle.
    // g_proxyConnect: succeed and mark the connector initialised.
    // g_proxyCallBlocking: succeed and write kRmtRegAddr into respBuff
    //   (the void* that ipcRegisterBuffer reads back as rmtRegAddr).
    ScopedHook loadParam(g_loadParam,
        [](const char* env, int64_t deftVal) -> int64_t {
            return std::strcmp(env, "LEGACY_CUDA_REGISTER") == 0 ? 1 : deftVal;
        });
    ScopedHook memGet(g_hipMemGetAddressRange,
        [](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t) -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t* h, void*) -> hipError_t {
            if (h) std::memset(h, 0x5A, sizeof(*h));
            return hipSuccess;
        });
    ScopedHook proxyConn(g_proxyConnect,
        [](ncclComm*, int, int, int, ncclProxyConnector* conn) -> ncclResult_t {
            if (conn) conn->initialized = true;
            return ncclSuccess;
        });
    ScopedHook proxyCall(g_proxyCallBlocking,
        [](ncclComm*, ncclProxyConnector*, int,
           void*, int, void* resp, int) -> ncclResult_t {
            // Write kRmtRegAddr as the rmtRegAddr the proxy hands back.
            if (resp) *reinterpret_cast<void**>(resp) =
                          reinterpret_cast<void*>(kRmtRegAddr);
            return ncclSuccess;
        });

    // -----------------------------------------------------------------------
    // Call 1: SENDRECV fresh-registration.
    // Fresh-reg arm: allocates ipcInfos[peerLocalRank], sets
    // hostPeerRmtAddrs[peerLocalRank]=kRmtRegAddr, needUpdate=true.
    // if (needUpdate) block: calloc + memcpy -> devPeerRmtAddrs[peerLocalRank]
    // should equal kRmtRegAddr after this call.
    // -----------------------------------------------------------------------
    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out1;
    bool isLegacyIpc = true;

    auto r1 = CallIpcRegisterBuffer(cb,
                                    reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                    512,
                                    peerRanks, 1,
                                    NCCL_IPC_SENDRECV,
                                    &regRecord, out1, &isLegacyIpc);

    ASSERT_EQ(r1, ncclSuccess);
    ASSERT_EQ(out1.regBufFlag, 1);
    // devPeerRmtAddrs must have been allocated and populated by the
    // if (needUpdate) block -- this is the invariant the fix establishes.
    ASSERT_NE(regRecord.regIpcAddrs.devPeerRmtAddrs, nullptr)
        << "devPeerRmtAddrs not allocated after SENDRECV fresh-registration. "
           "The if (needUpdate) block must calloc it for all IPC types.";
    EXPECT_EQ(regRecord.regIpcAddrs.devPeerRmtAddrs[kPeerLocalRank], kRmtRegAddr)
        << "devPeerRmtAddrs not populated after SENDRECV fresh-registration.";

    // -----------------------------------------------------------------------
    // Call 2: COLLECTIVE reuse.
    // ipcInfos[peerLocalRank] is now set -> reuse arm -> needUpdate stays false.
    // devPeerRmtAddrs is non-NULL and correct from call 1; the function must
    // return it as-is without zeroing it (the pre-fix bug would re-calloc
    // here and skip the memcpy because needUpdate==false).
    // -----------------------------------------------------------------------
    IpcRegOutputs out2;
    bool isLegacyIpc2 = false;

    auto r2 = CallIpcRegisterBuffer(cb,
                                    reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                    512,
                                    peerRanks, 1,
                                    NCCL_IPC_COLLECTIVE,
                                    &regRecord, out2, &isLegacyIpc2);

    ASSERT_EQ(r2, ncclSuccess);
    ASSERT_EQ(out2.regBufFlag, 1);
    EXPECT_EQ(regRecord.regIpcAddrs.devPeerRmtAddrs[kPeerLocalRank], kRmtRegAddr)
        << "devPeerRmtAddrs[" << kPeerLocalRank << "] was corrupted by the "
           "COLLECTIVE reuse call. Pre-fix code would re-calloc (zeroing) "
           "and skip the memcpy because needUpdate==false. "
           "See NCCL issue #1859 and PR #1861.";
    EXPECT_EQ(out2.peerRmtAddrs, regRecord.regIpcAddrs.devPeerRmtAddrs);
    EXPECT_EQ(out2.offsetOut, kBuffOffset);

    // Cleanup: ipcInfos array and each per-peer entry were allocated by
    // ncclCalloc/ncclRealloc (standard heap). devPeerRmtAddrs and
    // hostPeerRmtAddrs are tracked by g_fakeAllocations / ncclCalloc
    // respectively and freed by ResetP2pFakes() in TearDown.
    if (regRecord.ipcInfos) {
        for (int i = 0; i < regRecord.ipcInfosSize; ++i)
            std::free(regRecord.ipcInfos[i]);
        std::free(regRecord.ipcInfos);
        regRecord.ipcInfos = nullptr;
    }
    if (regRecord.regIpcAddrs.hostPeerRmtAddrs) {
        std::free(regRecord.regIpcAddrs.hostPeerRmtAddrs);
        regRecord.regIpcAddrs.hostPeerRmtAddrs = nullptr;
    }
}
