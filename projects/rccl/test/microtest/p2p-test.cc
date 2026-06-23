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

// DISABLED_ until the upstream NCCL PR #1861 fix is merged into RCCL.
// The test deliberately fails against the current buggy code (calloc'd dev
// table is never populated from the host table); re-enable by deleting the
// DISABLED_ prefix once the fix lands.
//
// Regression test for NVIDIA/nccl#1859 ("user-buffer p2p + coll rmtAddr
// nullptr error"). The same buggy code is present in RCCL's p2p.cc.
//
// Repro at the application level:
//   1. Call any P2P op (SENDRECV) with a registered user buffer. The
//      fresh-registration arm populates hostPeerRmtAddrs[peer] and sets
//      the local `needUpdate=true`, but devPeerRmtAddrs stays NULL
//      because the strong-stream block is COLLECTIVE-only.
//   2. Call any collective (e.g. AllReduce) on the SAME buffer.
//      Re-enters ipcRegisterBuffer via the reuse arm, so needUpdate
//      stays false. The post-loop strong-stream block fires because
//      devPeerRmtAddrs is still NULL. The buggy logic is then:
//          if (devPeerRmtAddrs == NULL)
//              ncclCudaCallocAsync(&devPeerRmtAddrs, ...);  // zero-filled
//          if (needUpdate)
//              ncclCudaMemcpyAsync(devPeerRmtAddrs, host..., ...);
//      With needUpdate==false the memcpy is skipped, so devPeerRmtAddrs
//      is allocated but full of zeros. The kernel then reads remote
//      addresses as NULL and crashes.
//
// The PR fix (nccl#1861) restructures the inner conditionals so that
// allocating a fresh devPeerRmtAddrs always implies copying the host
// table into it. That's the contract this test pins down: after a
// successful COLLECTIVE call from the post-SENDRECV state,
// devPeerRmtAddrs[peerLocalRank] must equal hostPeerRmtAddrs[peerLocalRank].
//
// The fake ncclCudaCallocAsync/MemcpyAsync hooks (defaults) are honest
// emulators -- calloc real heap memory, real memcpy -- so the test sees
// either zeros (buggy code: calloc but no memcpy) or kRmtRegAddr (fixed
// code: calloc followed by memcpy).
TEST_F(P2pMicrotest, DISABLED_IpcRegisterBuffer_RegressionNcclIssue1859_P2pThenCollectivePopulatesDevTable)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBegAddr       = 0x10000;
    constexpr uintptr_t kBuffOffset    = 0x40;
    // A plausible "peer's remote registered address" that step 1 would
    // have written into hostPeerRmtAddrs[peerLocalRank], and that step 2
    // is then supposed to copy into devPeerRmtAddrs[peerLocalRank].
    constexpr uintptr_t kRmtRegAddr    = 0xA000;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes();

    // Simulate the state left by a prior SENDRECV registration:
    //   - ipcInfos[peerLocalRank] populated (drives reuse arm in step 2)
    //   - hostPeerRmtAddrs[peerLocalRank] = kRmtRegAddr
    //   - devPeerRmtAddrs still NULL (SENDRECV doesn't allocate it)
    ReusableIpcInfo prior(kPeerRank, kPeerLocalRank, kRmtRegAddr,
                          /*legacyIpcCap=*/ false);
    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    prior.InstallInto(regRecord);
    ASSERT_EQ(regRecord.regIpcAddrs.devPeerRmtAddrs, nullptr);  // sanity

    // Leave g_strongStreamAcquire / g_fakeCudaCallocAsync /
    // g_fakeCudaMemcpyAsync at their defaults: the strong-stream block
    // runs to completion against real heap memory + real memcpy. The
    // test will then read the contents of the dev table back.

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = false;

    auto r = CallIpcRegisterBuffer(cb,
                                   /*userbuff=*/ reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                   /*buffSize=*/ 512,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    ASSERT_EQ(r, ncclSuccess);
    ASSERT_EQ(out.regBufFlag, 1);

    // The function must have allocated a fresh dev table -- this is the
    // entry condition for the bug.
    ASSERT_NE(regRecord.regIpcAddrs.devPeerRmtAddrs, nullptr);

    // The contract under test: the per-peer slot in the dev table must
    // hold the same remote address as the host table. With the bug it's
    // 0 (calloc'd but never written); with the fix it's kRmtRegAddr.
    EXPECT_EQ(regRecord.regIpcAddrs.devPeerRmtAddrs[kPeerLocalRank],
              kRmtRegAddr)
        << "devPeerRmtAddrs[" << kPeerLocalRank << "] was not populated "
           "from hostPeerRmtAddrs after a P2P-then-COLLECTIVE sequence. "
           "This is the failure mode tracked by NVIDIA/nccl#1859: a fresh "
           "devPeerRmtAddrs is allocated but the memcpy from the host "
           "table is gated on `needUpdate`, which is false on the reuse "
           "arm. See nccl PR #1861 for the fix.";

    // The function returned the dev table itself as peerRmtAddrs.
    EXPECT_EQ(out.peerRmtAddrs, regRecord.regIpcAddrs.devPeerRmtAddrs);
    EXPECT_EQ(out.offsetOut,    kBuffOffset);
}
