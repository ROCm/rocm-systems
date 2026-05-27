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

#include <unistd.h>  // dup, STDERR_FILENO -- POSIX_FD test hands close() a real fd

#include <array>
#include <cstdint>
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
#define ncclCudaCallocAsync(ptr, nelem, stream) \
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
// each other. Tests that don't currently install hooks still use this
// fixture -- it's the file-wide default so adding a hook to a test that
// previously didn't need one doesn't silently contaminate the next test.
// ===========================================================================
class P2pMicrotest : public ::testing::Test {
protected:
    void TearDown() override { ResetP2pFakes(); }
};

// ===========================================================================
// Helpers: lightweight builders for the recurring input/output shapes.
//
// Goal: each test body should read as "build the state that makes this test
// different, call the function, assert". Anything that's identical between
// tests lives here.
// ===========================================================================

namespace {

// ScopedHook -- RAII wrapper around a controllable seam (any of the
// std::function<...> hooks declared in fakes/p2p_fakes.h).
//
// Three jobs in one type:
//   1. Installs the test's behaviour on construction.
//   2. Counts calls automatically (.calls), so tests don't hand-roll a
//      separate `int xCalls = 0; ++xCalls` per hook.
//   3. Restores the previous behaviour on destruction. This means tests
//      don't have to inherit the fixture or rely on ResetP2pFakes() to
//      avoid contaminating each other -- the hook's lifetime ends with
//      the ScopedHook local, before the captured stack locals die.
//
// Usage (CTAD picks up the signature from the hook variable):
//   ScopedHook memGet(g_hipMemGetAddressRange,
//       [&](hipDeviceptr_t* pb, std::size_t* ps, hipDeviceptr_t) {
//           if (pb) *pb = ...;
//           return hipSuccess;
//       });
//   ...
//   EXPECT_EQ(memGet.calls, 1);
//
// Non-movable + non-copyable because the installed lambda captures `this`
// to bump the counter.
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

// CTAD: deduce R(Args...) from the std::function<R(Args...)>& argument so
// call sites don't have to spell out the signature.
template <typename R, typename... Args, typename Callable>
ScopedHook(std::function<R(Args...)>&, Callable) -> ScopedHook<R(Args...)>;

// RegRecordCleaner -- RAII guard that frees the allocations
// ipcRegisterBuffer makes *into* a ncclReg on the fresh-registration path:
//
//   - regRecord.ipcInfos[i]                       (per-peer ncclCalloc'd newInfo)
//   - regRecord.regIpcAddrs.hostPeerRmtAddrs      (lazily-ncclCalloc'd host table)
//
// regIpcAddrs.devPeerRmtAddrs is owned by g_fakeAllocations (the
// ncclCudaCallocAsync default registers it there), so this guard
// deliberately doesn't touch it.
//
// Use this on any test that drives the fresh-registration arm so that an
// ASSERT_* between the call and the explicit cleanup doesn't leak.
// Construct *after* the regRecord so destruction order is correct.
struct RegRecordCleaner {
    ncclReg& reg;
    explicit RegRecordCleaner(ncclReg& r) : reg(r) {}
    ~RegRecordCleaner() {
        for (auto*& info : reg.ipcInfos) {
            if (info) { std::free(info); info = nullptr; }
        }
        if (reg.regIpcAddrs.hostPeerRmtAddrs) {
            std::free(reg.regIpcAddrs.hostPeerRmtAddrs);
            reg.regIpcAddrs.hostPeerRmtAddrs = nullptr;
        }
    }
    RegRecordCleaner(const RegRecordCleaner&)            = delete;
    RegRecordCleaner& operator=(const RegRecordCleaner&) = delete;
};

// CommBuilder -- fluent builder that owns the backing storage for the
// fields of ncclComm that ipcRegisterBuffer reads. Tests grab a
// reference to the built comm via .comm and pass it to
// CallIpcRegisterBuffer.
//
// Each test sets only the slots it needs:
//
//     CommBuilder b;                                // bare ncclComm{}
//     CommBuilder b; b.WithLocalRank(peer, plr);    // + rankToLocalRank
//     CommBuilder b; b.WithLocalRank(...)           // + localRanks =
//                     .WithMaxLocalRanks();         //   NCCL_MAX_LOCAL_RANKS
//     CommBuilder b; ... .WithSharedRes();          // + sharedRes
//     CommBuilder b; ... .WithProxyConnArray(N);    // + gproxyConn[N]
//
// Storage outlives the comm because the builder owns it; the builder
// must outlive the test body that uses .comm.
class CommBuilder {
public:
    // rankToLocalRank table sized to NCCL_MAX_LOCAL_RANKS (the maximum
    // local rank index ipcRegisterBuffer can address); unassigned
    // entries stay 0.
    CommBuilder& WithLocalRank(int peerRank, int peerLocalRank) {
        if (!rankToLocalRankInstalled_) {
            comm_.rankToLocalRank = rankToLocalRankStorage_.data();
            rankToLocalRankInstalled_ = true;
        }
        rankToLocalRankStorage_[peerRank] = peerLocalRank;
        return *this;
    }

    // Most reuse-arm code paths read comm->localRanks but never index
    // anything by it; setting it to NCCL_MAX_LOCAL_RANKS is the
    // defensive default.
    CommBuilder& WithMaxLocalRanks() {
        comm_.localRanks = NCCL_MAX_LOCAL_RANKS;
        return *this;
    }

    // sharedRes must be non-null whenever the call path enters the
    // strong-stream block: the call site takes the address of
    // comm->sharedRes->hostStream.
    CommBuilder& WithSharedRes() {
        comm_.sharedRes = &sharedResStorage_;
        return *this;
    }

    // gproxyConn is a bare pointer in ncclComm; the real ncclCommInit
    // allocates it sized to comm->nRanks. Tests that drive the
    // fresh-registration arm hand-roll a backing array sized to
    // nRanks (must be > max peerRank the test will exercise).
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

// ReusableIpcInfo -- owns the per-peer ncclIpcRegInfo + the host-side
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
        regRecord.ipcInfos[peerLocalRank]      = &info;
        regRecord.regIpcAddrs.hostPeerRmtAddrs = hostPeerRmtAddrs.data();
    }
};

// IpcRegOutputs -- the four OUT parameters of ipcRegisterBuffer, pre-seeded
// with sentinel values. The sentinels matter: ipcRegisterBuffer is required
// to either populate them on success or zero them on failure, and an
// accidental no-op (or a fail path that forgets to clear) shows up as the
// sentinel surviving.
struct IpcRegOutputs {
    static constexpr uintptr_t kSentinel = 0xDEADBEEFDEADBEEFull;
    int        regBufFlag    = static_cast<int>(kSentinel);
    uintptr_t  offsetOut     = kSentinel;
    uintptr_t* peerRmtAddrs  = reinterpret_cast<uintptr_t*>(kSentinel);

    // Assert the function's documented failure-path contract: all three
    // outputs cleared. Called by every test that takes a fail: path.
    void ExpectZeroed() const
    {
        EXPECT_EQ(regBufFlag,   0);
        EXPECT_EQ(offsetOut,    0u);
        EXPECT_EQ(peerRmtAddrs, nullptr);
    }
};

// ForceLegacyCudaRegister -- the param-hook lambda that every fresh-reg
// test in the legacy-IPC arm needs. ncclParamLegacyCudaRegister() must
// return non-zero so:
//   (a) the `if (ncclParamLegacyCudaRegister()) legacyIpcCap = 1` write
//       fires under HIP_VERSION < 71260540, which is the precondition
//       for control reaching the `else if (legacyIpcCap)` arm, and
//   (b) the `comm->directMode || !ncclParamLegacyCudaRegister()` guard
//       inside that arm doesn't short-circuit to fail.
//
// Returned as a plain lambda (not a ScopedHook) so call sites compose:
//     ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());
auto ForceLegacyCudaRegister()
{
    // Note: NCCL_PARAM passes the env arg *without* the "NCCL_" prefix
    // (the real ncclLoadParam prepends it before getenv). Our redirector
    // doesn't go through ncclLoadParam, so the string we match here is
    // the raw arg from the NCCL_PARAM(...) call site:
    // `NCCL_PARAM(LegacyCudaRegister, "LEGACY_CUDA_REGISTER", 0)`.
    return [](const char* env, int64_t deftVal) -> int64_t {
        if (std::strcmp(env, "LEGACY_CUDA_REGISTER") == 0) return 1;
        return deftVal;
    };
}

// CallIpcRegisterBuffer -- thin wrapper so test bodies aren't dominated by
// a 12-line argument list. `isLegacyIpc` is in/out: callers initialise it
// to whatever value they want to see overwritten (or kept).
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

// ipcRegisterBuffer with regRecord == nullptr: cheapest real path. The
// function should fall through the whole per-peer loop without touching the
// proxy or driver, leaving all OUT params zeroed.
TEST_F(P2pMicrotest, IpcRegisterBuffer_NullRegRecordIsNoOp)
{
    CommBuilder cb;
    int peerRanks[] = {0};
    IpcRegOutputs out;
    bool isLegacyIpc = true;

    auto r = CallIpcRegisterBuffer(cb,
                                   /*userbuff=*/ reinterpret_cast<const void*>(0x1000),
                                   /*buffSize=*/ 4096,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   /*regRecord=*/ nullptr,
                                   out, &isLegacyIpc);

    EXPECT_EQ(r, ncclSuccess);
    out.ExpectZeroed();
    EXPECT_FALSE(isLegacyIpc);
}

// Reuse path for SENDRECV: per-peer loop hits the "already have IPC info"
// branch (no driver, no proxy, no device-stream work), and the post-loop
// arm returns the host-side remote address as *peerRmtAddrsOut.
TEST_F(P2pMicrotest, IpcRegisterBuffer_SendrecvReusesExistingIpcInfo)
{
    constexpr int       kPeerRank      = 3;
    constexpr int       kPeerLocalRank = 2;
    constexpr uintptr_t kBegAddr       = 0x10000;
    constexpr uintptr_t kBuffOffset    = 0x40;
    constexpr uintptr_t kRmtRegAddr    = 0xA000;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank);

    ReusableIpcInfo existing(kPeerRank, kPeerLocalRank, kRmtRegAddr,
                             /*legacyIpcCap=*/ true);
    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    existing.InstallInto(regRecord);

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = false;

    auto r = CallIpcRegisterBuffer(cb,
                                   /*userbuff=*/ reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_SENDRECV,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    // SENDRECV returns the raw remote address (cast to uintptr_t*), not a
    // pointer into a peer-address table.
    EXPECT_EQ(reinterpret_cast<uintptr_t>(out.peerRmtAddrs), kRmtRegAddr);
    EXPECT_TRUE(isLegacyIpc);  // propagated from existing.info.impInfo
}

// Reuse path for COLLECTIVE with a pre-populated device peer-address table:
// post-loop arm returns the *device-side* table itself rather than a single
// remote address, and the strong-stream allocation block stays skipped
// because both devPeerRmtAddrs is non-null and needUpdate is false.
TEST_F(P2pMicrotest, IpcRegisterBuffer_CollectiveReuseReturnsDevicePeerAddrTable)
{
    constexpr int       kPeerRank      = 1;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBegAddr       = 0x10000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kRmtRegAddr    = 0xA000;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks();  // unused on this path, set defensively

    ReusableIpcInfo existing(kPeerRank, kPeerLocalRank, kRmtRegAddr,
                             /*legacyIpcCap=*/ false);
    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x2000;
    existing.InstallInto(regRecord);

    // Pre-populated dev table -> needUpdate stays false, strong-stream
    // block stays skipped.
    std::array<uintptr_t, NCCL_MAX_LOCAL_RANKS> devPeerRmtAddrs{};
    regRecord.regIpcAddrs.devPeerRmtAddrs = devPeerRmtAddrs.data();

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;  // start true to see it cleared

    auto r = CallIpcRegisterBuffer(cb,
                                   /*userbuff=*/ reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                   /*buffSize=*/ 512,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    EXPECT_EQ(out.peerRmtAddrs, devPeerRmtAddrs.data());  // the table itself
    EXPECT_FALSE(isLegacyIpc);
}

// Reuse-COLLECTIVE with the device peer-address table missing: drives the
// `devPeerRmtAddrs == NULL || needUpdate` branch True so control enters
// the strong-stream allocation block.
//
// We *don't* let the block complete -- ncclCudaCallocAsync inside it is a
// header-only template that hits the real HIP runtime, which there is no
// GPU for here. Instead the test installs a per-test hook on
// ncclStrongStreamAcquire that:
//   (a) records that the block was entered, and
//   (b) returns an error on the first call so control flows cleanly
//       through NCCLCHECKGOTO into the fail: epilogue.
TEST_F(P2pMicrotest, IpcRegisterBuffer_CollectiveReuseStrongStreamAcquireFailurePropagates)
{
    constexpr int       kPeerRank      = 1;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBegAddr       = 0x10000;
    constexpr uintptr_t kBuffOffset    = 0x100;
    constexpr uintptr_t kRmtRegAddr    = 0xA000;

    // sharedRes must be non-null: the strong-stream call site takes the
    // address of comm->sharedRes->hostStream. The hook doesn't read it.
    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes();

    ReusableIpcInfo existing(kPeerRank, kPeerLocalRank, kRmtRegAddr,
                             /*legacyIpcCap=*/ false);
    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x4000;
    existing.InstallInto(regRecord);
    // devPeerRmtAddrs intentionally left null -- this is what makes the
    // strong-stream block fire.

    ScopedHook acquire(g_strongStreamAcquire,
        [](struct ncclCudaGraph, struct ncclStrongStream*, bool,
           hipStream_t* stream) -> ncclResult_t {
            if (stream) *stream = nullptr;
            return ncclSystemError;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;

    auto r = CallIpcRegisterBuffer(cb,
                                   /*userbuff=*/ reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                   /*buffSize=*/ 1024,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(acquire.calls, 1);    // strong-stream block was entered
    EXPECT_EQ(r, ncclSystemError);  // error from the hook propagated out
    out.ExpectZeroed();             // failure-path contract honoured
}

// Fresh-registration happy path (legacy IPC / cudaIpcGetMemHandle arm).
//
// This is the path that *produces* the IPC state every reuse test assumes
// already exists, and it lights up the entire body of the per-peer-loop
// `else` arm (the "register buffer with peerLocalRank" branch).
//
// The happy-path call has two distinct contracts worth pinning down
// separately:
//
//   (a) bookkeeping contract -- everything the function writes *into*
//       the caller-owned ncclReg (ipcInfos[], hostPeerRmtAddrs,
//       state bit).
//
//   (b) return-value contract -- the OUT parameters
//       (regBufFlag, offsetOut, peerRmtAddrs, isLegacyIpc) and the
//       seam-call counts that *produced* them.
//
// We split (a) and (b) into two TEST_Fs sharing a fixture so that a
// failure points at one of the two contracts unambiguously. The fixture
// runs the actual ipcRegisterBuffer call in SetUp() with all four
// seam hooks installed.
//
// Coverage-wise this lights up:
//   - hipMemGetAddressRange success
//   - ncclProxyConnect because gproxyConn[peerRank].initialized is false
//   - the `ncclCuMemEnable() == 0` path falling into the
//     `else if (legacyIpcCap)` legacy-export arm
//   - hipIpcGetMemHandle + the `ipcInfo.legacyIpcCap = true` write
//   - the legacy-arm's `if (isLegacyIpc) *isLegacyIpc = true` write
//     (the last unexercised isLegacyIpc write)
//   - ncclProxyCallBlocking returning a canned rmtRegAddr
//   - newInfo allocation + the `if (rmtRegAddr)` bookkeeping block:
//     ipcCalloc, ipcInfos[] install, state |= IPC_REG_COMPLETE
//   - hostPeerRmtAddrs lazy-allocation (the
//     `if (regIpcAddrs.hostPeerRmtAddrs == NULL)` arm)
//   - post-loop COLLECTIVE block with needUpdate=true driving
//     ncclCudaCallocAsync + ncclCudaMemcpyAsync
class FreshRegistrationLegacyIpcSucceedsFixture : public P2pMicrotest {
protected:
    static constexpr int       kPeerRank      = 2;
    static constexpr int       kPeerLocalRank = 1;
    static constexpr uintptr_t kBaseAddr      = 0x100000;
    static constexpr std::size_t kBaseSize    = 0x4000;
    static constexpr uintptr_t kBuffOffset    = 0x80;
    static constexpr uintptr_t kBegOffset     = 0x20;
    static constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    static constexpr uintptr_t kRmtRegAddr    = 0xCAFE0000ull;
    static constexpr int       kNRanks        = kPeerRank + 1;

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);

    // Storage that outlives the call. Heap-allocated via unique_ptr so
    // the fixture is movable-into-place in SetUp() without copying the
    // non-copyable members.
    std::unique_ptr<CommBuilder>       cb;
    std::unique_ptr<ncclReg>           regRecord;
    std::unique_ptr<RegRecordCleaner>  regCleanup;
    std::unique_ptr<ScopedHook<int64_t(const char*, int64_t)>> loadParam;
    std::unique_ptr<ScopedHook<hipError_t(hipDeviceptr_t*, std::size_t*, hipDeviceptr_t)>> memGet;
    std::unique_ptr<ScopedHook<hipError_t(hipIpcMemHandle_t*, void*)>> ipcGet;
    std::unique_ptr<ScopedHook<ncclResult_t(struct ncclComm*, int, int, int, struct ncclProxyConnector*)>> connect;
    std::unique_ptr<ScopedHook<ncclResult_t(struct ncclComm*, struct ncclProxyConnector*, int, void*, int, void*, int)>> proxy;

    IpcRegOutputs out;
    bool         isLegacyIpc = false;
    ncclResult_t result      = ncclSuccess;

    void SetUp() override
    {
        cb = std::make_unique<CommBuilder>();
        cb->WithLocalRank(kPeerRank, kPeerLocalRank)
           .WithMaxLocalRanks()
           .WithSharedRes()
           .WithProxyConnArray(kNRanks);

        regRecord = std::make_unique<ncclReg>();
        *regRecord = ncclReg{};
        regRecord->begAddr = kBegAddr;
        regRecord->endAddr = kBegAddr + 0x1000;
        // ipcInfos[kPeerLocalRank] is NULL -> fresh-registration arm.
        regCleanup = std::make_unique<RegRecordCleaner>(*regRecord);

        // Force the legacy-IPC arm to be entered (see ForceLegacyCudaRegister
        // for why this is required under HIP_VERSION < 71260540).
        loadParam = std::make_unique<ScopedHook<int64_t(const char*, int64_t)>>(
            g_loadParam, ForceLegacyCudaRegister());

        // hipMemGetAddressRange: returns baseAddr + baseSize. Production
        // contract: called with dptr == userbuff.
        memGet = std::make_unique<ScopedHook<hipError_t(hipDeviceptr_t*, std::size_t*, hipDeviceptr_t)>>(
            g_hipMemGetAddressRange,
            [this](hipDeviceptr_t* pbase, std::size_t* psize,
                   hipDeviceptr_t dptr) -> hipError_t {
                EXPECT_EQ(reinterpret_cast<const void*>(dptr), kUserbuff);
                if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
                if (psize) *psize = kBaseSize;
                return hipSuccess;
            });

        // hipIpcGetMemHandle: succeeds with a sentinel handle. Production
        // contract: called with *baseAddr*, NOT the (possibly mid-allocation)
        // userbuff -- legacy CUDA IPC handles always refer to whole
        // allocations.
        ipcGet = std::make_unique<ScopedHook<hipError_t(hipIpcMemHandle_t*, void*)>>(
            g_hipIpcGetMemHandle,
            [](hipIpcMemHandle_t* h, void* devPtr) -> hipError_t {
                EXPECT_EQ(devPtr, reinterpret_cast<void*>(kBaseAddr));
                if (h) std::memset(h, 0x5A, sizeof(*h));
                return hipSuccess;
            });

        // ncclProxyConnect: marks the gproxyConn slot initialized.
        connect = std::make_unique<ScopedHook<ncclResult_t(struct ncclComm*, int, int, int, struct ncclProxyConnector*)>>(
            g_proxyConnect,
            [](struct ncclComm* c, int transport, int /*send*/,
               int rank, struct ncclProxyConnector* pc) -> ncclResult_t {
                EXPECT_EQ(transport, TRANSPORT_P2P);
                EXPECT_EQ(rank, kPeerRank);
                EXPECT_EQ(pc, &c->gproxyConn[rank]);
                pc->initialized = true;
                return ncclSuccess;
            });

        // ncclProxyCallBlocking: canned rmtRegAddr in respBuff +
        // production-contract assertions on the request struct.
        proxy = std::make_unique<ScopedHook<ncclResult_t(struct ncclComm*, struct ncclProxyConnector*, int, void*, int, void*, int)>>(
            g_proxyCallBlocking,
            [](struct ncclComm*, struct ncclProxyConnector*,
               int type, void* req, int reqSize,
               void* resp, int respSize) -> ncclResult_t {
                EXPECT_EQ(type, ncclProxyMsgRegister);
                if (req == nullptr ||
                    static_cast<size_t>(reqSize) < sizeof(p2pIpcExpInfo)) {
                    ADD_FAILURE() << "malformed register-msg request: req="
                                  << req << " reqSize=" << reqSize;
                    return ncclInternalError;
                }
                auto* info = static_cast<p2pIpcExpInfo*>(req);
                EXPECT_TRUE(info->legacyIpcCap);
                EXPECT_EQ(info->size,   kBaseSize);
                EXPECT_EQ(info->offset, kBegOffset);
                EXPECT_GE(static_cast<size_t>(respSize), sizeof(void*));
                if (resp) std::memcpy(resp, &kRmtRegAddr, sizeof(void*));
                return ncclSuccess;
            });

        int peerRanks[] = {kPeerRank};
        result = CallIpcRegisterBuffer(*cb, kUserbuff,
                                       /*buffSize=*/ 256,
                                       peerRanks, 1,
                                       NCCL_IPC_COLLECTIVE,
                                       regRecord.get(), out, &isLegacyIpc);
    }

    void TearDown() override
    {
        // Tear hooks down before the fixture's other state so any hook
        // captures stay live for the duration of the call. (ScopedHook
        // dtor restores the prior slot.) Order matters: hooks first,
        // then the regRecord cleaner, then the regRecord itself.
        proxy.reset();
        connect.reset();
        ipcGet.reset();
        memGet.reset();
        loadParam.reset();
        regCleanup.reset();
        regRecord.reset();
        cb.reset();
        P2pMicrotest::TearDown();
    }
};

// (a) Return-value contract: the OUT parameters and the seam-call
// counts that produced them. A failure here points at the function's
// post-loop / output-writing code.
TEST_F(FreshRegistrationLegacyIpcSucceedsFixture, ReturnsCollectiveDevTable)
{
    // Every seam was called exactly once on the happy path.
    EXPECT_EQ(memGet->calls,  1);
    EXPECT_EQ(ipcGet->calls,  1);
    EXPECT_EQ(connect->calls, 1);
    EXPECT_EQ(proxy->calls,   1);

    EXPECT_EQ(result,         ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    // COLLECTIVE: returns the dev table (allocated by the strong-stream
    // block via g_fakeCudaCallocAsync's heap default), populated from the
    // host table by g_fakeCudaMemcpyAsync's real-memcpy default.
    ASSERT_NE(out.peerRmtAddrs, nullptr);
    EXPECT_EQ(out.peerRmtAddrs, regRecord->regIpcAddrs.devPeerRmtAddrs);
    EXPECT_EQ(out.peerRmtAddrs[kPeerLocalRank], kRmtRegAddr);
    EXPECT_TRUE(isLegacyIpc);  // the legacy-arm `*isLegacyIpc = true` write reached
}

// (b) Bookkeeping contract: everything written *into* the caller-owned
// ncclReg by the `if (rmtRegAddr)` block. A failure here points at
// the per-peer bookkeeping block, not the post-loop output code.
TEST_F(FreshRegistrationLegacyIpcSucceedsFixture, PopulatesIpcInfoRecord)
{
    ASSERT_EQ(result, ncclSuccess);  // setup precondition

    ASSERT_NE(regRecord->ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_EQ(regRecord->ipcInfos[kPeerLocalRank]->peerRank, kPeerRank);
    EXPECT_EQ(regRecord->ipcInfos[kPeerLocalRank]->baseAddr,
              reinterpret_cast<void*>(kBaseAddr));
    EXPECT_EQ(regRecord->ipcInfos[kPeerLocalRank]->impInfo.rmtRegAddr,
              reinterpret_cast<void*>(kRmtRegAddr));
    EXPECT_EQ(regRecord->ipcInfos[kPeerLocalRank]->impInfo.offset, kBegOffset);
    EXPECT_TRUE(regRecord->ipcInfos[kPeerLocalRank]->impInfo.legacyIpcCap);
    EXPECT_TRUE(regRecord->state & IPC_REG_COMPLETE);
    ASSERT_NE(regRecord->regIpcAddrs.hostPeerRmtAddrs, nullptr);
    EXPECT_EQ(regRecord->regIpcAddrs.hostPeerRmtAddrs[kPeerLocalRank],
              kRmtRegAddr);
}

// Inlined-body test removed -- see fixture above. Comment retained as
// a breadcrumb for grep so anyone searching for the historic name finds
// the split tests.
// IpcRegisterBuffer_FreshRegistrationLegacyIpcSucceeds -> split into
//   FreshRegistrationLegacyIpcSucceedsFixture.ReturnsCollectiveDevTable
//   FreshRegistrationLegacyIpcSucceedsFixture.PopulatesIpcInfoRecord


// Fresh-registration variant: ncclProxyCallBlocking returns success but
// writes rmtRegAddr=NULL into the response buffer. Drives the False arm
// of `if (rmtRegAddr)` -- the entire bookkeeping
// block (newInfo alloc, ipcInfos[] install, hostPeerRmtAddrs lazy alloc)
// must be skipped. Because *regBufFlag is then never set to 1, the
// post-loop strong-stream block also stays skipped, and the function
// returns ncclSuccess with all outputs left zeroed by the prologue.
TEST_F(P2pMicrotest, IpcRegisterBuffer_FreshRegistrationProxyReturnsNullRmtAddr)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr int       kNRanks        = kPeerRank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);
    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t)
            -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [&](hipIpcMemHandle_t* h, void*) -> hipError_t {
            if (h) std::memset(h, 0x5A, sizeof(*h));
            return hipSuccess;
        });
    ScopedHook connect(g_proxyConnect,
        [&](struct ncclComm*, int, int, int,
            struct ncclProxyConnector* pc) -> ncclResult_t {
            pc->initialized = true;
            return ncclSuccess;
        });
    // The key seam: success, but rmtRegAddr written back as NULL.
    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int,
            void*, int, void* resp, int respSize) -> ncclResult_t {
            void* nullAddr = nullptr;
            if (resp && static_cast<size_t>(respSize) >= sizeof(void*)) {
                std::memcpy(resp, &nullAddr, sizeof(void*));
            }
            return ncclSuccess;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;  // expect prologue to clear, no further writes

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(r, ncclSuccess);
    // Bookkeeping block was skipped:
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_EQ(regRecord.regIpcAddrs.hostPeerRmtAddrs, nullptr);
    EXPECT_EQ(regRecord.regIpcAddrs.devPeerRmtAddrs,  nullptr);
    EXPECT_FALSE(regRecord.state & IPC_REG_COMPLETE);
    // Outputs stay at the prologue-cleared values:
    EXPECT_EQ(out.regBufFlag,   0);
    EXPECT_EQ(out.offsetOut,    0u);
    EXPECT_EQ(out.peerRmtAddrs, nullptr);
    // legacyIpcCap path *was* taken (the legacy arm writes
    // `*isLegacyIpc = true` before ncclProxyCallBlocking is called); the
    // rmtRegAddr==NULL exit then short-circuits silently. Capturing this
    // pins down current behaviour -- if a future change clears
    // isLegacyIpc on the NULL-rmtRegAddr path, this assertion is the
    // place to update.
    EXPECT_TRUE(isLegacyIpc);
}

// Fresh-registration variant: pre-mark comm->gproxyConn[peerRank].initialized
// = true so the per-peer loop's `if (...initialized == false)` test takes
// the False arm and ncclProxyConnect is *not* called. The rest of the registration (handle export, proxy register,
// bookkeeping) proceeds normally.
TEST_F(P2pMicrotest, IpcRegisterBuffer_FreshRegistrationSkipsProxyConnectWhenAlreadyInitialized)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr uintptr_t kRmtRegAddr    = 0xCAFE0000ull;
    constexpr int       kNRanks        = kPeerRank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);
    // Key precondition: skip the connect.
    cb.comm().gproxyConn[kPeerRank].initialized = true;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);
    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t)
            -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [&](hipIpcMemHandle_t* h, void*) -> hipError_t {
            if (h) std::memset(h, 0x5A, sizeof(*h));
            return hipSuccess;
        });
    // Default g_proxyConnect returns ncclSystemError -- if it were called
    // we'd see the test fail with that error code. Wrap in a ScopedHook
    // anyway so we can assert .calls == 0 explicitly.
    ScopedHook connect(g_proxyConnect,
        [&](struct ncclComm*, int, int, int,
            struct ncclProxyConnector*) -> ncclResult_t {
            ADD_FAILURE() << "ncclProxyConnect must not be called when "
                             "gproxyConn[peerRank].initialized == true";
            return ncclSystemError;
        });
    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int,
            void*, int, void* resp, int respSize) -> ncclResult_t {
            EXPECT_GE(static_cast<size_t>(respSize), sizeof(void*));
            if (resp) std::memcpy(resp, &kRmtRegAddr, sizeof(void*));
            return ncclSuccess;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = false;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(connect.calls, 0);   // the contract under test
    EXPECT_EQ(memGet.calls,  1);
    EXPECT_EQ(ipcGet.calls,  1);
    EXPECT_EQ(proxy.calls,   1);
    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    ASSERT_NE(regRecord.ipcInfos[kPeerLocalRank], nullptr);
}

// Fresh-registration fail path with newInfo already allocated. Drives the
// True arm of `if (newInfo) free(newInfo)` in the `fail:` epilogue -- the
// leak-prevention contract on the fail: epilogue. Sequence:
//
//   1. Happy-path through hipMemGetAddressRange / hipIpcGetMemHandle /
//      proxyCallBlocking -> rmtRegAddr non-null -> newInfo ncclCalloc'd,
//      regRecord->ipcInfos[peerLocalRank] = newInfo.
//   2. Loop exits (one peer).
//   3. Post-loop COLLECTIVE strong-stream block fires (devPeerRmtAddrs is
//      NULL, needUpdate is true).
//   4. g_strongStreamAcquire hook returns failure -> NCCLCHECKGOTO into
//      `fail:` with newInfo non-null and the local var still holding the
//      most recent allocation.
//
// Without this test the True arm was never hit (existing fresh-reg
// failure test triggers fail: before newInfo is ever set).
//
// Note on the dangling pointer: the production fail: epilogue frees
// newInfo but leaves regRecord->ipcInfos[peerLocalRank] pointing at the
// freed memory. That's an existing pre-condition of the function (the
// fail: path doesn't roll back ipcInfos), not something this test is
// asserting. The RegRecordCleaner would normally double-free, so we
// null out the slot before the cleaner runs.
TEST_F(P2pMicrotest, IpcRegisterBuffer_FreshRegistrationFreesNewInfoOnPostLoopFailure)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr uintptr_t kRmtRegAddr    = 0xCAFE0000ull;
    constexpr int       kNRanks        = kPeerRank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);
    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t)
            -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [&](hipIpcMemHandle_t* h, void*) -> hipError_t {
            if (h) std::memset(h, 0x5A, sizeof(*h));
            return hipSuccess;
        });
    ScopedHook connect(g_proxyConnect,
        [&](struct ncclComm*, int, int, int,
            struct ncclProxyConnector* pc) -> ncclResult_t {
            pc->initialized = true;
            return ncclSuccess;
        });
    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int,
            void*, int, void* resp, int respSize) -> ncclResult_t {
            EXPECT_GE(static_cast<size_t>(respSize), sizeof(void*));
            if (resp) std::memcpy(resp, &kRmtRegAddr, sizeof(void*));
            return ncclSuccess;
        });
    // Fail in the post-loop strong-stream block. By this point newInfo is
    // allocated and installed into regRecord->ipcInfos[kPeerLocalRank].
    ScopedHook acquire(g_strongStreamAcquire,
        [](struct ncclCudaGraph, struct ncclStrongStream*, bool,
           hipStream_t* stream) -> ncclResult_t {
            if (stream) *stream = nullptr;
            return ncclSystemError;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = false;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    // Failure surfaced from the strong-stream block:
    EXPECT_EQ(acquire.calls, 1);
    EXPECT_EQ(r, ncclSystemError);
    out.ExpectZeroed();

    // The contract being pinned down: the function ran far enough that
    // newInfo *was* allocated (and stashed in ipcInfos) before the
    // failure, and the fail: epilogue then free()d it. We can't observe
    // the free directly without leak detection, but we *can* observe
    // that the precondition was reached -- otherwise the True arm of
    // `if (newInfo)` was never exercised.
    ASSERT_NE(regRecord.ipcInfos[kPeerLocalRank], nullptr)
        << "test scaffolding broken: newInfo was never allocated, so the "
           "True arm of `if (newInfo) free(newInfo)` is not reached";

    // Production leaves this slot pointing at freed memory on fail:.
    // Null it out so RegRecordCleaner doesn't double-free.
    regRecord.ipcInfos[kPeerLocalRank] = nullptr;
}

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

// Null-isLegacyIpc path: ipcRegisterBufferOnce (the public entry point at
// p2p.cc:1053) calls ipcRegisterBuffer with isLegacyIpc == NULL, so every
// `if (isLegacyIpc)` inside the function must take the False branch and
// skip the write -- otherwise we'd have a null-deref in production.
//
// All existing microtests pass a non-null pointer; this test pins down the
// nullptr arm. It reuses the cheap SENDRECV-reuse setup (no driver, no
// proxy) so the *only* thing it exercises is the gated-write contract.
TEST_F(P2pMicrotest, IpcRegisterBuffer_NullIsLegacyIpcPointerIsSkipped)
{
    constexpr int       kPeerRank      = 3;
    constexpr int       kPeerLocalRank = 2;
    constexpr uintptr_t kBegAddr       = 0x10000;
    constexpr uintptr_t kBuffOffset    = 0x40;
    constexpr uintptr_t kRmtRegAddr    = 0xA000;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank);

    ReusableIpcInfo existing(kPeerRank, kPeerLocalRank, kRmtRegAddr,
                             /*legacyIpcCap=*/ true);
    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    existing.InstallInto(regRecord);

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;

    // The contract: passing nullptr must not crash and must not affect any
    // OUT param other than isLegacyIpc itself.
    auto r = CallIpcRegisterBuffer(cb,
                                   /*userbuff=*/ reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_SENDRECV,
                                   &regRecord, out, /*isLegacyIpc=*/ nullptr);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(out.peerRmtAddrs), kRmtRegAddr);
}

// Fresh-registration variant: ncclProxyConnect returns failure. Drives the
// `else` (failure) arm of the NCCLCHECKGOTO at the ncclProxyConnect call
// site -- the per-peer loop must route to `fail:` *without* having
// allocated newInfo (the bookkeeping block under `if (rmtRegAddr)` is gated on
// rmtRegAddr, which we never get to). Complementary to the
// FreesNewInfoOnPostLoopFailure test: this one covers the True arm of
// `if (newInfo) free(newInfo)` False side -- newInfo is still NULL at
// fail:, so the epilogue must skip the free without crashing.
//
// Plan item A3.
TEST_F(P2pMicrotest, IpcRegisterBuffer_FreshRegistrationProxyConnectFailurePropagates)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr int       kNRanks        = kPeerRank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t)
            -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    // The seam under test: proxy connect refuses. Control should route
    // through NCCLCHECKGOTO into fail: without ever calling
    // hipIpcGetMemHandle or ncclProxyCallBlocking (both default to
    // failure -- if they're reached the assertion below would still
    // catch ret != ncclSystemError, but the .calls counters make the
    // contract explicit).
    ScopedHook connect(g_proxyConnect,
        [&](struct ncclComm*, int transport, int /*send*/,
            int rank, struct ncclProxyConnector*) -> ncclResult_t {
            EXPECT_EQ(transport, TRANSPORT_P2P);
            EXPECT_EQ(rank, kPeerRank);
            return ncclSystemError;
        });
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [&](hipIpcMemHandle_t*, void*) -> hipError_t {
            ADD_FAILURE() << "hipIpcGetMemHandle must not be reached when "
                             "ncclProxyConnect fails first";
            return hipErrorInvalidValue;
        });
    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int,
            void*, int, void*, int) -> ncclResult_t {
            ADD_FAILURE() << "ncclProxyCallBlocking must not be reached "
                             "when ncclProxyConnect fails first";
            return ncclSystemError;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(memGet.calls,  1);
    EXPECT_EQ(connect.calls, 1);
    EXPECT_EQ(ipcGet.calls,  0);
    EXPECT_EQ(proxy.calls,   0);

    EXPECT_EQ(r, ncclSystemError);
    out.ExpectZeroed();
    EXPECT_FALSE(isLegacyIpc);  // prologue cleared it; fail path didn't write

    // Bookkeeping must not have been touched -- newInfo was never
    // allocated, so neither slot was set.
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_EQ(regRecord.regIpcAddrs.hostPeerRmtAddrs, nullptr);
    EXPECT_FALSE(regRecord.state & IPC_REG_COMPLETE);
}

// Fresh-registration variant: hipMemGetAddressRange succeeds but
// hipIpcGetMemHandle fails in the legacy-IPC arm. Drives the CUDACHECKGOTO
// at the `hipIpcGetMemHandle(&ipcInfo.ipcDesc.devIpc, baseAddr)` call
// site (legacy-export arm -- the cuMem* arm is the alternative under
// ROCm 7+ when ncclCuMemEnable() returns non-zero, which our fake holds
// at 0). With the fake at ncclCuMemEnable() == 0 we always hit the
// `else if (legacyIpcCap)` arm, where this driver call sits.
//
// Pre-condition for the legacy-export arm to fire at all: legacyIpcCap
// must be true going in. Under our build (HIP_VERSION < 71260540) that
// comes from the NCCL_LEGACY_CUDA_REGISTER param. ncclLoadParam is a
// no-op so the param sits at its compile-time default; we don't depend
// on a specific value here -- the test installs g_hipIpcGetMemHandle
// returning failure, and if legacyIpcCap stays false the function takes
// the `nothing works, just return` goto fail in the trailing `else`
// instead, which is the same failure surface. Either way: fail: epilogue runs
// with newInfo still NULL.
//
// Plan item A4. Drives the failure arm of the legacy-export
// CUDACHECKGOTO(hipIpcGetMemHandle) in the `else if (legacyIpcCap)`
// branch. The hookable g_loadParam seam forces
// ncclParamLegacyCudaRegister() to return 1, which is the precondition
// for control reaching the legacy-export arm at all (the param
// defaults to 0, which routes instead through the trailing
// `// nothing works, just return` goto fail).
TEST_F(P2pMicrotest, IpcRegisterBuffer_FreshRegistrationLegacyIpcGetMemHandleFailurePropagates)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr int       kNRanks        = kPeerRank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);
    // Pre-mark gproxyConn initialized so the (covered-elsewhere)
    // proxyConnect call is skipped -- shaves a hook off this test.
    cb.comm().gproxyConn[kPeerRank].initialized = true;
    // directMode=false so the `comm->directMode || !ncclParam...` guard
    // inside the legacy-export arm doesn't short-circuit to fail before
    // the hipIpcGetMemHandle call.
    cb.comm().directMode = 0;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);

    // Force the legacy-IPC arm to be entered (see ForceLegacyCudaRegister).
    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t)
            -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    // The seam under test: legacy IPC export refuses.
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [&](hipIpcMemHandle_t*, void* devPtr) -> hipError_t {
            // Pin down the contract: called with the base address, not
            // the userbuff.
            EXPECT_EQ(devPtr, reinterpret_cast<void*>(kBaseAddr));
            return hipErrorInvalidValue;
        });
    // If proxyCall is reached we've over-shot the failure point.
    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int,
            void*, int, void*, int) -> ncclResult_t {
            ADD_FAILURE() << "ncclProxyCallBlocking must not be reached "
                             "when hipIpcGetMemHandle fails";
            return ncclSystemError;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(memGet.calls, 1);
    // The contract under test: we actually entered the legacy-export
    // arm. If g_loadParam wiring breaks, legacyIpcCap stays 0 and
    // control takes the `// nothing works` goto fail instead, leaving
    // ipcGet.calls at 0 -- this assertion is what catches that
    // regression.
    EXPECT_EQ(ipcGet.calls, 1);
    EXPECT_EQ(proxy.calls,  0);

    // CUDACHECKGOTO maps non-hipSuccess to ncclUnhandledCudaError.
    EXPECT_EQ(r, ncclUnhandledCudaError);
    out.ExpectZeroed();
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_FALSE(regRecord.state & IPC_REG_COMPLETE);
}

// Multi-peer loop: drive the per-peer loop with nPeers=2, both peers in
// the reuse arm (their ipcInfos[] slots are pre-populated). Pins down
// that the loop iterates correctly: both peers' regBufFlag bookkeeping
// is honoured, isLegacyIpc reflects the *last* peer (the loop overwrites
// each iteration), and the post-loop COLLECTIVE arm hands back the dev
// table (reuse + pre-populated dev table -> strong-stream block skipped).
//
// This is the first test that exercises nPeers > 1 -- previously the
// `for (int p = 0; p < nPeers; p++)` loop was only ever entered once,
// hiding any inter-iteration state leak.
//
// Plan item A6.
TEST_F(P2pMicrotest, IpcRegisterBuffer_MultiPeerReuseLoopIteratesCorrectly)
{
    constexpr int       kPeer0Rank      = 1;
    constexpr int       kPeer0LocalRank = 0;
    constexpr int       kPeer1Rank      = 3;
    constexpr int       kPeer1LocalRank = 2;
    constexpr uintptr_t kBegAddr        = 0x10000;
    constexpr uintptr_t kBuffOffset     = 0x80;
    constexpr uintptr_t kRmt0           = 0xA000;
    constexpr uintptr_t kRmt1           = 0xB000;

    CommBuilder cb;
    cb.WithLocalRank(kPeer0Rank, kPeer0LocalRank)
      .WithLocalRank(kPeer1Rank, kPeer1LocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes();

    // Hand-roll the per-peer ipcInfo + a *shared* host table so both
    // ReusableIpcInfo-style installs don't clobber each other.
    // ReusableIpcInfo.InstallInto overwrites regIpcAddrs.hostPeerRmtAddrs
    // unconditionally, so we can't use two of them naively.
    ncclIpcRegInfo info0{};
    info0.peerRank             = kPeer0Rank;
    info0.impInfo.rmtRegAddr   = reinterpret_cast<void*>(kRmt0);
    info0.impInfo.legacyIpcCap = true;
    ncclIpcRegInfo info1{};
    info1.peerRank             = kPeer1Rank;
    info1.impInfo.rmtRegAddr   = reinterpret_cast<void*>(kRmt1);
    info1.impInfo.legacyIpcCap = false;

    std::array<uintptr_t, NCCL_MAX_LOCAL_RANKS> hostPeerRmtAddrs{};
    hostPeerRmtAddrs[kPeer0LocalRank] = kRmt0;
    hostPeerRmtAddrs[kPeer1LocalRank] = kRmt1;

    // Pre-populated dev table -> needUpdate stays false, strong-stream
    // block skipped.
    std::array<uintptr_t, NCCL_MAX_LOCAL_RANKS> devPeerRmtAddrs{};
    devPeerRmtAddrs[kPeer0LocalRank] = kRmt0;
    devPeerRmtAddrs[kPeer1LocalRank] = kRmt1;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x2000;
    regRecord.ipcInfos[kPeer0LocalRank]   = &info0;
    regRecord.ipcInfos[kPeer1LocalRank]   = &info1;
    regRecord.regIpcAddrs.hostPeerRmtAddrs = hostPeerRmtAddrs.data();
    regRecord.regIpcAddrs.devPeerRmtAddrs  = devPeerRmtAddrs.data();

    // No hooks -- pure reuse arm, no driver, no proxy.
    int peerRanks[] = {kPeer0Rank, kPeer1Rank};
    IpcRegOutputs out;
    bool isLegacyIpc = false;

    auto r = CallIpcRegisterBuffer(cb,
                                   /*userbuff=*/ reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                   /*buffSize=*/ 512,
                                   peerRanks, 2,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    // COLLECTIVE post-loop arm returns the dev table itself.
    EXPECT_EQ(out.peerRmtAddrs, devPeerRmtAddrs.data());
    // Loop writes isLegacyIpc every iteration -- the surviving value is
    // from the *last* peer (peer1 -> legacyIpcCap=false). Pins down
    // current behaviour; if a future change changes the aggregation
    // semantics (OR across peers, etc.), this assertion is the place
    // to update.
    EXPECT_FALSE(isLegacyIpc);
}

// Fresh-registration entry path: ipcInfos[peerLocalRank] is NULL, so the
// per-peer loop takes the `else` branch instead of the reuse branch. The
// first thing it does is CUCHECKGOTO(hipMemGetAddressRange(...)), which is
// a real HIP runtime call -- not a PFN seam -- so passing a bogus userbuff
// fails it deterministically with no GPU required, and control routes
// through CUCHECKGOTO into the fail: epilogue.
TEST_F(P2pMicrotest, IpcRegisterBuffer_FreshRegistrationFailureClearsOutputs)
{
    constexpr int       kPeerRank          = 4;
    constexpr int       kPeerLocalRank     = 3;
    constexpr uintptr_t kBegAddr           = 0x10000;
    constexpr uintptr_t kBuffOffset        = 0x10;
    // Any non-registered pointer works: hipMemGetAddressRange returns an
    // error rather than crashing.
    constexpr uintptr_t kUnregisteredUserbuff = 0xBADADD0ull;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks();

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    // regRecord.ipcInfos[kPeerLocalRank] is NULL -> fresh-registration arm.

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;

    auto r = CallIpcRegisterBuffer(cb,
                                   /*userbuff=*/ reinterpret_cast<const void*>(kUnregisteredUserbuff + kBuffOffset),
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_SENDRECV,
                                   &regRecord, out, &isLegacyIpc);

    // CUCHECKGOTO maps any non-hipSuccess to ncclUnhandledCudaError.
    EXPECT_EQ(r, ncclUnhandledCudaError);
    out.ExpectZeroed();
    EXPECT_FALSE(isLegacyIpc);  // prologue cleared it before the failure
}

// Fresh-registration variant: directMode=true forces the legacy-export
// arm to short-circuit to fail *before* calling hipIpcGetMemHandle.
// Drives the True arm of `comm->directMode || !ncclParamLegacyCudaRegister()`
// in the `else if (legacyIpcCap)` block. All existing
// fresh-reg tests set directMode=0 implicitly (zero-initialised ncclComm)
// and force the param on, so this short-circuit's True arm was previously
// unhit.
//
// Plan item A4 follow-up (961-sub-branch). Complementary to
// LegacyIpcGetMemHandleFailurePropagates: that one fails at the
// hipIpcGetMemHandle call site itself; this one fails one line earlier,
// at the directMode guard.
TEST_F(P2pMicrotest, IpcRegisterBuffer_FreshRegistrationDirectModeShortCircuitsLegacyExport)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr int       kNRanks        = kPeerRank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);
    cb.comm().gproxyConn[kPeerRank].initialized = true;  // skip proxyConnect
    cb.comm().directMode = 1;                            // the seam under test

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);

    // Force legacyIpcCap=1 so control reaches the `else if (legacyIpcCap)`
    // arm. Without this it would take the `// nothing works` goto fail
    // one branch earlier, exercising a different failure surface.
    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t)
            -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    // The contract: hipIpcGetMemHandle MUST NOT be reached when directMode
    // is set -- the short-circuit fires first.
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [&](hipIpcMemHandle_t*, void*) -> hipError_t {
            ADD_FAILURE() << "hipIpcGetMemHandle must not be reached when "
                             "comm->directMode short-circuits the legacy "
                             "export arm to fail";
            return hipErrorInvalidValue;
        });
    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int,
            void*, int, void*, int) -> ncclResult_t {
            ADD_FAILURE() << "ncclProxyCallBlocking must not be reached "
                             "on the directMode short-circuit";
            return ncclSystemError;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(memGet.calls, 1);
    EXPECT_EQ(ipcGet.calls, 0);
    EXPECT_EQ(proxy.calls,  0);

    // The short-circuit is a bare `goto fail` (not NCCLCHECKGOTO), so ret
    // stays at its initial ncclSuccess. This pins down current behaviour
    // -- it's a quirk of the production code that a directMode-rejected
    // registration looks identical to a successful no-op from the
    // caller's perspective: ncclSuccess + regBufFlag=0.
    EXPECT_EQ(r, ncclSuccess);
    out.ExpectZeroed();
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_FALSE(regRecord.state & IPC_REG_COMPLETE);
}

// Multi-peer mixed loop: peer 0 in the reuse arm, peer 1 fresh-registers.
// Simulates the real lifecycle where a previous call already registered
// the buffer for peer 0 and a follow-up call brings peer 1 online.
//
// Three branch frontiers this is the first test to hit:
//   - line 990:15 `if (hostPeerRmtAddrs == NULL)` False arm: peer 0's
//     prior registration already allocated it, so peer 1's fresh-reg
//     bookkeeping skips the lazy alloc.
//   - line 1004:63 `needUpdate` True arm under a non-null devPeerRmtAddrs:
//     post-loop COLLECTIVE block enters even though devPeerRmtAddrs
//     != NULL, because peer 1 set needUpdate=true.
//   - line 1008:15 `if (devPeerRmtAddrs == NULL)` False arm: inside the
//     strong-stream block, skip the calloc but still memcpy.
//
// Plan item A6-mixed.
TEST_F(P2pMicrotest, IpcRegisterBuffer_MultiPeerMixedReuseAndFreshUpdatesDevTable)
{
    constexpr int       kPeer0Rank      = 1;
    constexpr int       kPeer0LocalRank = 0;
    constexpr int       kPeer1Rank      = 3;
    constexpr int       kPeer1LocalRank = 2;
    constexpr uintptr_t kBaseAddr       = 0x100000;
    constexpr std::size_t kBaseSize     = 0x4000;
    constexpr uintptr_t kBegOffset      = 0x20;
    constexpr uintptr_t kBegAddr        = kBaseAddr + kBegOffset;
    constexpr uintptr_t kBuffOffset     = 0x80;
    constexpr uintptr_t kRmt0           = 0xA000;
    constexpr uintptr_t kRmt1Fresh      = 0xCAFE0000ull;
    constexpr int       kNRanks         = kPeer1Rank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeer0Rank, kPeer0LocalRank)
      .WithLocalRank(kPeer1Rank, kPeer1LocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);
    cb.comm().gproxyConn[kPeer1Rank].initialized = true;

    // Peer 0's prior-registration state: reusable ipcInfo + host-table
    // slot. The host table was allocated by that prior fresh-reg, so it
    // is non-null going into this call (the contract we want to drive).
    ncclIpcRegInfo info0{};
    info0.peerRank             = kPeer0Rank;
    info0.impInfo.rmtRegAddr   = reinterpret_cast<void*>(kRmt0);
    info0.impInfo.legacyIpcCap = true;

    // hostPeerRmtAddrs has to be heap-allocated (production code path
    // uses ncclCalloc / free, and RegRecordCleaner free()s it).
    auto* hostTable = static_cast<uintptr_t*>(
        std::calloc(NCCL_MAX_LOCAL_RANKS, sizeof(uintptr_t)));
    ASSERT_NE(hostTable, nullptr);
    hostTable[kPeer0LocalRank] = kRmt0;

    // devPeerRmtAddrs pre-populated by the prior registration's
    // post-loop block. Mismatched-vs-host on purpose so that we can
    // assert the post-loop memcpy actually overwrote it.
    std::array<uintptr_t, NCCL_MAX_LOCAL_RANKS> devTable{};
    devTable[kPeer0LocalRank] = 0xDEADu;
    devTable[kPeer1LocalRank] = 0xDEADu;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    regRecord.ipcInfos[kPeer0LocalRank]    = &info0;
    regRecord.regIpcAddrs.hostPeerRmtAddrs = hostTable;
    regRecord.regIpcAddrs.devPeerRmtAddrs  = devTable.data();
    RegRecordCleaner regCleanup(regRecord);

    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t)
            -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [&](hipIpcMemHandle_t* h, void*) -> hipError_t {
            if (h) std::memset(h, 0x5A, sizeof(*h));
            return hipSuccess;
        });
    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int,
            void*, int, void* resp, int respSize) -> ncclResult_t {
            EXPECT_GE(static_cast<size_t>(respSize), sizeof(void*));
            if (resp) std::memcpy(resp, &kRmt1Fresh, sizeof(void*));
            return ncclSuccess;
        });

    int peerRanks[] = {kPeer0Rank, kPeer1Rank};
    IpcRegOutputs out;
    bool isLegacyIpc = false;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 2,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    // Peer 0: pure reuse, no driver/proxy. Peer 1: one trip through each
    // fresh-reg seam.
    EXPECT_EQ(memGet.calls, 1);
    EXPECT_EQ(ipcGet.calls, 1);
    EXPECT_EQ(proxy.calls,  1);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    // COLLECTIVE post-loop arm returns the dev table itself.
    EXPECT_EQ(out.peerRmtAddrs, devTable.data());

    // Peer 0's existing ipcInfo untouched.
    EXPECT_EQ(regRecord.ipcInfos[kPeer0LocalRank], &info0);
    // Peer 1's bookkeeping populated.
    ASSERT_NE(regRecord.ipcInfos[kPeer1LocalRank], nullptr);
    EXPECT_EQ(regRecord.ipcInfos[kPeer1LocalRank]->peerRank, kPeer1Rank);
    EXPECT_EQ(regRecord.ipcInfos[kPeer1LocalRank]->impInfo.rmtRegAddr,
              reinterpret_cast<void*>(kRmt1Fresh));

    // The 990:15 False-arm contract: peer 1's fresh-reg used the
    // already-allocated host table rather than re-allocating it.
    EXPECT_EQ(regRecord.regIpcAddrs.hostPeerRmtAddrs, hostTable);
    EXPECT_EQ(hostTable[kPeer0LocalRank], kRmt0);          // peer 0 untouched
    EXPECT_EQ(hostTable[kPeer1LocalRank], kRmt1Fresh);     // peer 1 inserted

    // The 1004:63 True / 1008:15 False / 1010:15 True contract: post-loop
    // strong-stream block fired (needUpdate=true), skipped the calloc
    // (devPeerRmtAddrs already non-null), but the memcpy from the host
    // table actually overwrote both slots.
    EXPECT_EQ(regRecord.regIpcAddrs.devPeerRmtAddrs, devTable.data());
    EXPECT_EQ(devTable[kPeer0LocalRank], kRmt0);
    EXPECT_EQ(devTable[kPeer1LocalRank], kRmt1Fresh);

    // legacyIpcCap reflects the *last* peer (peer 1, fresh-reg legacy arm).
    EXPECT_TRUE(isLegacyIpc);

    // RegRecordCleaner will free hostTable and the peer-1 ipcInfo.
    // Peer 0's ipcInfo is stack-allocated (&info0) -- null its slot so
    // the cleaner doesn't free() it. devTable is stack-owned; cleaner
    // doesn't touch devPeerRmtAddrs anyway, but the comment is here so
    // the next maintainer doesn't get surprised.
    regRecord.ipcInfos[kPeer0LocalRank] = nullptr;
}

// Reuse-COLLECTIVE with a missing dev table but no fresh registrations:
// drives the strong-stream block past the `if (needUpdate)` guard's False
// arm (branch 1010:15 False). All peers are reuse, so needUpdate stays
// false; devPeerRmtAddrs is null, so the outer `devPeerRmtAddrs == NULL ||
// needUpdate` enters the block; the inner `devPeerRmtAddrs == NULL` calloc
// fires (1008:15 True); the inner `if (needUpdate)` memcpy is skipped
// (1010:15 False); and *peerRmtAddrsOut returns the freshly-callocated dev
// table, still zeroed.
//
// This is the lifecycle path where a prior caller registered the host
// table (e.g. a SENDRECV) and a follow-up COLLECTIVE-reuse call is the
// first to materialise the device-side copy.
TEST_F(P2pMicrotest, IpcRegisterBuffer_CollectiveReuseAllocatesDevTableWithoutMemcpy)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBegAddr       = 0x10000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kRmtRegAddr    = 0xA000;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes();

    ReusableIpcInfo existing(kPeerRank, kPeerLocalRank, kRmtRegAddr,
                             /*legacyIpcCap=*/ false);
    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x2000;
    existing.InstallInto(regRecord);
    // devPeerRmtAddrs intentionally left null -- enters strong-stream block.

    // Count both seams so we can pin down the contract:
    //   calloc fires once (devPeerRmtAddrs == NULL True arm), memcpy is
    //   skipped (needUpdate False arm -- 1010:15 False).
    ScopedHook calloc(g_fakeCudaCallocAsync,
        [](void** ptr, std::size_t nbytes, hipStream_t) -> ncclResult_t {
            void* p = std::calloc(1, nbytes);
            if (!p) return ncclSystemError;
            if (ptr) *ptr = p;
            // Hand ownership to the test; we free it explicitly below.
            return ncclSuccess;
        });
    ScopedHook memcpy_(g_fakeCudaMemcpyAsync,
        [](void*, void*, std::size_t, hipStream_t) -> ncclResult_t {
            ADD_FAILURE() << "memcpy must not fire on pure-reuse path "
                          << "(needUpdate=false)";
            return ncclSystemError;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;  // start true to see it cleared

    auto r = CallIpcRegisterBuffer(cb,
                                   /*userbuff=*/ reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                   /*buffSize=*/ 512,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(calloc.calls,  1);
    EXPECT_EQ(memcpy_.calls, 0);  // the 1010:15-False contract
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    // Returns the freshly-callocated dev table; not the host one.
    EXPECT_EQ(out.peerRmtAddrs, regRecord.regIpcAddrs.devPeerRmtAddrs);
    ASSERT_NE(out.peerRmtAddrs, nullptr);
    // Skipped-memcpy contract: the slot is the calloc zero, NOT kRmtRegAddr.
    EXPECT_EQ(out.peerRmtAddrs[kPeerLocalRank], 0u);
    EXPECT_FALSE(isLegacyIpc);

    std::free(regRecord.regIpcAddrs.devPeerRmtAddrs);
    regRecord.regIpcAddrs.devPeerRmtAddrs = nullptr;
}

// Two-peer fresh-registration: drives the False arm of the function-local
// `if (baseAddr == NULL)` (branch 910:13 False) on the *second* loop
// iteration. Iteration p=0 calls hipMemGetAddressRange and writes the
// function-local baseAddr; iteration p=1 sees baseAddr already non-NULL
// from the prior peer and skips the re-call. This is the contract that
// keeps a multi-peer fresh registration from re-querying the driver N
// times for the same allocation.
TEST_F(P2pMicrotest, IpcRegisterBuffer_MultiPeerFreshRegistrationReusesBaseAddrAcrossLoop)
{
    constexpr int       kPeer0Rank      = 1;
    constexpr int       kPeer0LocalRank = 0;
    constexpr int       kPeer1Rank      = 3;
    constexpr int       kPeer1LocalRank = 2;
    constexpr uintptr_t kBaseAddr       = 0x100000;
    constexpr std::size_t kBaseSize     = 0x4000;
    constexpr uintptr_t kBegOffset      = 0x20;
    constexpr uintptr_t kBegAddr        = kBaseAddr + kBegOffset;
    constexpr uintptr_t kBuffOffset     = 0x80;
    constexpr uintptr_t kRmt0           = 0xCAFE0000ull;
    constexpr uintptr_t kRmt1           = 0xCAFE1000ull;
    constexpr int       kNRanks         = kPeer1Rank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeer0Rank, kPeer0LocalRank)
      .WithLocalRank(kPeer1Rank, kPeer1LocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);
    // Both gproxyConn slots pre-initialized so we don't have to fake
    // ncclProxyConnect for both peers; this test is focused on the
    // baseAddr-reuse contract, not the connect path.
    cb.comm().gproxyConn[kPeer0Rank].initialized = true;
    cb.comm().gproxyConn[kPeer1Rank].initialized = true;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    // Both ipcInfos[] slots NULL -> both peers take the fresh-reg arm.
    RegRecordCleaner regCleanup(regRecord);

    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize,
            hipDeviceptr_t dptr) -> hipError_t {
            EXPECT_EQ(reinterpret_cast<const void*>(dptr), kUserbuff);
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [&](hipIpcMemHandle_t* h, void* devPtr) -> hipError_t {
            // Both iterations must hand the driver the *same* baseAddr.
            // If a regression re-querying per peer ever lands, this is
            // the assertion that catches it (driving the True arm on
            // iter 1 again would still call ipcGet with kBaseAddr, but
            // the False-arm contract is that memGet only ran once --
            // checked on memGet.calls below).
            EXPECT_EQ(devPtr, reinterpret_cast<void*>(kBaseAddr));
            if (h) std::memset(h, 0x5A, sizeof(*h));
            return hipSuccess;
        });

    int proxyCallIdx = 0;
    const uintptr_t kRmts[] = {kRmt0, kRmt1};
    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int type,
            void* req, int reqSize, void* resp, int respSize) -> ncclResult_t {
            EXPECT_EQ(type, ncclProxyMsgRegister);
            if (req == nullptr ||
                static_cast<size_t>(reqSize) < sizeof(p2pIpcExpInfo)) {
                ADD_FAILURE() << "malformed register-msg request";
                return ncclInternalError;
            }
            auto* info = static_cast<p2pIpcExpInfo*>(req);
            // Both iterations must ship the same baseAddr-derived
            // ipcInfo (size = whole allocation, offset = begAddr-baseAddr).
            EXPECT_TRUE(info->legacyIpcCap);
            EXPECT_EQ(info->size,   kBaseSize);
            EXPECT_EQ(info->offset, kBegOffset);
            EXPECT_GE(static_cast<size_t>(respSize), sizeof(void*));
            if (resp && proxyCallIdx < 2) {
                std::memcpy(resp, &kRmts[proxyCallIdx], sizeof(void*));
            }
            ++proxyCallIdx;
            return ncclSuccess;
        });

    int peerRanks[] = {kPeer0Rank, kPeer1Rank};
    IpcRegOutputs out;
    bool isLegacyIpc = false;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 2,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    // ---- The 910:13-False contract: memGet fires once (iter 0 only).
    // ipcGet and the proxy fire once per peer; baseAddr is reused.
    EXPECT_EQ(memGet.calls, 1);
    EXPECT_EQ(ipcGet.calls, 2);
    EXPECT_EQ(proxy.calls,  2);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    EXPECT_EQ(out.peerRmtAddrs, regRecord.regIpcAddrs.devPeerRmtAddrs);
    ASSERT_NE(out.peerRmtAddrs, nullptr);
    EXPECT_EQ(out.peerRmtAddrs[kPeer0LocalRank], kRmt0);
    EXPECT_EQ(out.peerRmtAddrs[kPeer1LocalRank], kRmt1);
    EXPECT_TRUE(isLegacyIpc);

    // Both ipcInfos populated.
    ASSERT_NE(regRecord.ipcInfos[kPeer0LocalRank], nullptr);
    ASSERT_NE(regRecord.ipcInfos[kPeer1LocalRank], nullptr);
    EXPECT_EQ(regRecord.ipcInfos[kPeer0LocalRank]->impInfo.rmtRegAddr,
              reinterpret_cast<void*>(kRmt0));
    EXPECT_EQ(regRecord.ipcInfos[kPeer1LocalRank]->impInfo.rmtRegAddr,
              reinterpret_cast<void*>(kRmt1));
    // baseAddr written into both ipcInfos -- additional pin-down that
    // both peers saw the same function-local baseAddr.
    EXPECT_EQ(regRecord.ipcInfos[kPeer0LocalRank]->baseAddr,
              reinterpret_cast<void*>(kBaseAddr));
    EXPECT_EQ(regRecord.ipcInfos[kPeer1LocalRank]->baseAddr,
              reinterpret_cast<void*>(kBaseAddr));
}

// ===========================================================================
// cuMem*-export arm tests (plan item A7).
//
// All three tests share the same shape:
//   - g_cuMemEnable hook returns 1 -> cuMem* arm is entered (not legacy).
//   - g_hipMemRetainAllocationHandle hook succeeds with a sentinel handle.
//   - Per-test: drive sameProcess and ncclCuMemHandleType to pick which
//     of the three cuMem* sub-arms fires (sameProcess / POSIX_FD /
//     fabric).
//   - g_proxyCallBlocking returns a canned rmtRegAddr so the post-loop
//     bookkeeping fires (where applicable).
//
// These light up the entire `if (ncclCuMemEnable())` branch in
// ipcRegisterBuffer, which is the production path on ROCm 7+.
// ===========================================================================

#if ROCM_VERSION >= 70000

namespace {

// Sentinel value the Retain hook writes into the handle. The cuMem* arm
// shuttles this through hipMemExportToShareableHandle / hipMemRelease;
// the test asserts those hooks see the same value.
constexpr std::uintptr_t kSentinelHandleBits = 0xDEADC0DE12340000ull;

hipMemGenericAllocationHandle_t MakeSentinelHandle()
{
    hipMemGenericAllocationHandle_t h{};
    static_assert(sizeof(h) >= sizeof(std::uintptr_t),
                  "sentinel must fit in the handle");
    std::memcpy(&h, &kSentinelHandleBits, sizeof(std::uintptr_t));
    return h;
}

bool HandleHasSentinel(const hipMemGenericAllocationHandle_t& h)
{
    std::uintptr_t bits = 0;
    std::memcpy(&bits, &h, sizeof(std::uintptr_t));
    return bits == kSentinelHandleBits;
}

}  // namespace

// cuMem* arm, sameProcess=true: Retain -> memcpy handle into ipcInfo ->
// proxy register -> Release. Lights up the same-process sub-arm without
// touching hipMemExportToShareableHandle or the POSIX_FD/fabric branches.
TEST_F(P2pMicrotest, IpcRegisterBuffer_CuMemFreshRegistrationSameProcessSucceeds)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr uintptr_t kRmtRegAddr    = 0xCAFE0000ull;
    constexpr int       kNRanks        = kPeerRank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);
    // Skip the (covered-elsewhere) proxyConnect call; pre-mark the slot.
    cb.comm().gproxyConn[kPeerRank].initialized = true;
    // Drive the `if (proxyConn->sameProcess)` True arm.
    cb.comm().gproxyConn[kPeerRank].sameProcess = 1;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);

    // Enter the cuMem* arm.
    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t) -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });

    ScopedHook retain(g_hipMemRetainAllocationHandle,
        [](hipMemGenericAllocationHandle_t* h, void* addr) -> hipError_t {
            EXPECT_EQ(addr, reinterpret_cast<void*>(kBaseAddr));
            if (h) *h = MakeSentinelHandle();
            return hipSuccess;
        });
    // Same-process arm must NOT call export.
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
           unsigned long long) -> hipError_t {
            ADD_FAILURE() << "sameProcess arm must not call hipMemExportToShareableHandle";
            return hipErrorInvalidValue;
        });
    ScopedHook release(g_hipMemRelease,
        [](hipMemGenericAllocationHandle_t h) -> hipError_t {
            EXPECT_TRUE(HandleHasSentinel(h));
            return hipSuccess;
        });

    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int type,
            void* req, int reqSize, void* resp, int respSize) -> ncclResult_t {
            EXPECT_EQ(type, ncclProxyMsgRegister);
            if (req == nullptr ||
                static_cast<size_t>(reqSize) < sizeof(p2pIpcExpInfo)) {
                ADD_FAILURE() << "malformed register-msg request";
                return ncclInternalError;
            }
            auto* info = static_cast<p2pIpcExpInfo*>(req);
            // cuMem* arm clears legacyIpcCap (in contrast to the legacy arm).
            EXPECT_FALSE(info->legacyIpcCap);
            EXPECT_EQ(info->size,   kBaseSize);
            EXPECT_EQ(info->offset, kBegOffset);
            // Same-process arm memcpy'd the Retain handle into memHandle.
            EXPECT_TRUE(HandleHasSentinel(info->ipcDesc.memHandle));
            EXPECT_GE(static_cast<size_t>(respSize), sizeof(void*));
            if (resp) std::memcpy(resp, &kRmtRegAddr, sizeof(void*));
            return ncclSuccess;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;  // expect cuMem arm to clear

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(cuMemEnable.calls, 1);
    EXPECT_EQ(retain.calls,      1);
    EXPECT_EQ(xport.calls,       0);  // sameProcess arm skipped it
    EXPECT_EQ(release.calls,     1);
    EXPECT_EQ(proxy.calls,       1);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    EXPECT_FALSE(isLegacyIpc);  // cuMem arm cleared it

    ASSERT_NE(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_FALSE(regRecord.ipcInfos[kPeerLocalRank]->impInfo.legacyIpcCap);
}

// cuMem* arm, sameProcess=false, POSIX_FD handle type:
// Retain -> Export (hands back a real fd via dup(STDERR_FILENO) so the
// subsequent SYSCHECKGOTO(close(expFd)) succeeds) ->
// ncclProxyClientQueryFdBlocking -> close -> Release -> proxy register.
TEST_F(P2pMicrotest, IpcRegisterBuffer_CuMemFreshRegistrationPosixFdSucceeds)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr uintptr_t kRmtRegAddr    = 0xCAFE0000ull;
    constexpr int       kRmtImportedFd = 0x7e1e7;  // canned remote-fd value
    constexpr int       kNRanks        = kPeerRank + 1;

    // ncclCuMemHandleType is a fakes-owned global; existing tests don't
    // touch it (its default is hipMemHandleTypePosixFileDescriptor, which
    // is what we need here). Pin it down explicitly so a future fakes
    // change doesn't silently turn this into a fabric-arm test.
    auto saved_handle_type = ncclCuMemHandleType;
    ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);
    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().gproxyConn[kPeerRank].sameProcess = 0;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);

    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t) -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook retain(g_hipMemRetainAllocationHandle,
        [](hipMemGenericAllocationHandle_t* h, void*) -> hipError_t {
            if (h) *h = MakeSentinelHandle();
            return hipSuccess;
        });

    // Hand back a real fd so the subsequent SYSCHECKGOTO(close(expFd))
    // doesn't fail with EBADF. dup(STDERR_FILENO) is cheap and always
    // open during gtest runs.
    int duped_fd_seen = -1;
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [&duped_fd_seen](void* shareableHandle,
                         hipMemGenericAllocationHandle_t h,
                         hipMemAllocationHandleType handleType,
                         unsigned long long /*flags*/) -> hipError_t {
            EXPECT_TRUE(HandleHasSentinel(h));
            EXPECT_EQ(handleType, hipMemHandleTypePosixFileDescriptor);
            int fd = dup(STDERR_FILENO);
            if (fd < 0) return hipErrorInvalidValue;
            duped_fd_seen = fd;
            if (shareableHandle) {
                int* outFd = static_cast<int*>(shareableHandle);
                *outFd = fd;
            }
            return hipSuccess;
        });

    ScopedHook query(g_proxyClientQueryFdBlocking,
        [&duped_fd_seen](struct ncclComm*, struct ncclProxyConnector*,
                         int localFd, int* rmtFd) -> ncclResult_t {
            EXPECT_EQ(localFd, duped_fd_seen);
            if (rmtFd) *rmtFd = kRmtImportedFd;
            return ncclSuccess;
        });

    ScopedHook release(g_hipMemRelease,
        [](hipMemGenericAllocationHandle_t h) -> hipError_t {
            EXPECT_TRUE(HandleHasSentinel(h));
            return hipSuccess;
        });

    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int type,
            void* req, int reqSize, void* resp, int respSize) -> ncclResult_t {
            EXPECT_EQ(type, ncclProxyMsgRegister);
            if (req == nullptr ||
                static_cast<size_t>(reqSize) < sizeof(p2pIpcExpInfo)) {
                ADD_FAILURE() << "malformed register-msg request";
                return ncclInternalError;
            }
            auto* info = static_cast<p2pIpcExpInfo*>(req);
            EXPECT_FALSE(info->legacyIpcCap);
            // POSIX_FD arm writes the rmtFd handed back by
            // ncclProxyClientQueryFdBlocking into ipcInfo.impFd.
            EXPECT_EQ(info->impFd, kRmtImportedFd);
            EXPECT_GE(static_cast<size_t>(respSize), sizeof(void*));
            if (resp) std::memcpy(resp, &kRmtRegAddr, sizeof(void*));
            return ncclSuccess;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(retain.calls,  1);
    EXPECT_EQ(xport.calls,   1);
    EXPECT_EQ(query.calls,   1);
    EXPECT_EQ(release.calls, 1);
    EXPECT_EQ(proxy.calls,   1);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_FALSE(isLegacyIpc);

    // The duped fd was closed by ipcRegisterBuffer's SYSCHECKGOTO(close).
    // Verify by attempting a second close -- it should fail with EBADF.
    int second_close = ::close(duped_fd_seen);
    EXPECT_EQ(second_close, -1);
    EXPECT_EQ(errno, EBADF);

    ncclCuMemHandleType = saved_handle_type;
}

// cuMem* arm, sameProcess=false, fabric handle type, export failure:
// drives the `if (CUPFN(hipMemExportToShareableHandle(...)) != hipSuccess)`
// True arm in the fabric sub-branch. The contract: hipMemRelease must
// fire on the handle before `goto fail`, otherwise the handle leaks.
TEST_F(P2pMicrotest, IpcRegisterBuffer_CuMemFreshRegistrationFabricExportFailureReleasesHandle)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr int       kNRanks        = kPeerRank + 1;

    // Switch fakes-owned global to the non-POSIX_FD branch.
    auto saved_handle_type = ncclCuMemHandleType;
    ncclCuMemHandleType = hipMemHandleTypeWin32;  // anything != POSIX_FD

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);
    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().gproxyConn[kPeerRank].sameProcess = 0;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);

    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t) -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook retain(g_hipMemRetainAllocationHandle,
        [](hipMemGenericAllocationHandle_t* h, void*) -> hipError_t {
            if (h) *h = MakeSentinelHandle();
            return hipSuccess;
        });
    // Export fails on the fabric arm.
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [](void*, hipMemGenericAllocationHandle_t,
           hipMemAllocationHandleType handleType,
           unsigned long long) -> hipError_t {
            EXPECT_NE(handleType, hipMemHandleTypePosixFileDescriptor);
            return hipErrorInvalidValue;  // drives the error-arm goto fail
        });
    // The contract under test: Release fires on the sentinel handle
    // before goto fail (the handle-leak guard).
    ScopedHook release(g_hipMemRelease,
        [](hipMemGenericAllocationHandle_t h) -> hipError_t {
            EXPECT_TRUE(HandleHasSentinel(h));
            return hipSuccess;
        });
    // Proxy register must never fire on this failure path.
    ScopedHook proxy(g_proxyCallBlocking,
        [](struct ncclComm*, struct ncclProxyConnector*, int,
           void*, int, void*, int) -> ncclResult_t {
            ADD_FAILURE() << "proxy register must not fire after export failure";
            return ncclInternalError;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = true;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(retain.calls,  1);
    EXPECT_EQ(xport.calls,   1);
    EXPECT_EQ(release.calls, 1);  // the handle-leak guard contract
    EXPECT_EQ(proxy.calls,   0);

    // The fabric-arm export-failure path is a bare `goto fail` (not a
    // CUCHECKGOTO), so ret stays ncclSuccess; the fail: epilogue zeros
    // the outputs.
    EXPECT_EQ(r, ncclSuccess);
    out.ExpectZeroed();
    EXPECT_FALSE(isLegacyIpc);  // cleared by the cuMem arm before failing
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_FALSE(regRecord.state & IPC_REG_COMPLETE);

    ncclCuMemHandleType = saved_handle_type;
}

// cuMem* arm, Retain failure -> retry-as-legacy fallback. Plan item A8.
//
// Drives the `if (CUPFN(hipMemRetainAllocationHandle(...)) != hipSuccess)`
// True arm (branch 930:13 True) and the *inner* fallback path:
//   - the `comm->directMode || !ncclParamLegacyCudaRegister()` guard
//     short-circuits to its False arm (branch 932:17 False), because
//     directMode=false and the param is forced on. Control proceeds to
//     the legacy hipIpcGetMemHandle call.
//   - hipIpcGetMemHandle succeeds with a sentinel handle.
//   - `ipcInfo.legacyIpcCap = true` and the `*isLegacyIpc = true` write
//     (line 935) fire -- the *only* path that drives this isLegacyIpc
//     write inside the cuMem arm.
//   - proxy register sees legacyIpcCap=true (in contrast to the cuMem
//     happy-path tests, where it's false).
//
// Sister to the existing FreshRegistrationLegacyIpcSucceeds: that one
// stays out of the cuMem arm entirely; this one *enters* the cuMem arm,
// fails Retain, and falls back to legacy export within the cuMem block.
TEST_F(P2pMicrotest, IpcRegisterBuffer_CuMemFreshRegistrationRetainFailureFallsBackToLegacyExport)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr uintptr_t kRmtRegAddr    = 0xCAFE0000ull;
    constexpr int       kNRanks        = kPeerRank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);
    cb.comm().gproxyConn[kPeerRank].initialized = true;
    // directMode=false so the `directMode || !legacyParam` short-circuit's
    // first operand is False; combined with ForceLegacyCudaRegister
    // (second operand also False), the whole guard takes its False arm
    // and control proceeds to the legacy hipIpcGetMemHandle fallback.
    cb.comm().directMode = 0;

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);

    // Enter the cuMem arm.
    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    // Required for the inner fallback guard not to short-circuit.
    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t) -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });

    // Retain *fails* -- the key seam for this test.
    ScopedHook retain(g_hipMemRetainAllocationHandle,
        [](hipMemGenericAllocationHandle_t*, void*) -> hipError_t {
            return hipErrorInvalidValue;
        });
    // Export and Release must NOT fire on the fallback arm.
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [](void*, hipMemGenericAllocationHandle_t,
           hipMemAllocationHandleType, unsigned long long) -> hipError_t {
            ADD_FAILURE() << "retry-as-legacy must not call hipMemExportToShareableHandle";
            return hipErrorInvalidValue;
        });
    ScopedHook release(g_hipMemRelease,
        [](hipMemGenericAllocationHandle_t) -> hipError_t {
            ADD_FAILURE() << "retry-as-legacy must not call hipMemRelease "
                          << "(Retain failed -> no handle was acquired)";
            return hipErrorInvalidValue;
        });

    // The fallback's legacy export.
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t* h, void* devPtr) -> hipError_t {
            EXPECT_EQ(devPtr, reinterpret_cast<void*>(kBaseAddr));
            if (h) std::memset(h, 0x5A, sizeof(*h));
            return hipSuccess;
        });

    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int type,
            void* req, int reqSize, void* resp, int respSize) -> ncclResult_t {
            EXPECT_EQ(type, ncclProxyMsgRegister);
            if (req == nullptr ||
                static_cast<size_t>(reqSize) < sizeof(p2pIpcExpInfo)) {
                ADD_FAILURE() << "malformed register-msg request";
                return ncclInternalError;
            }
            auto* info = static_cast<p2pIpcExpInfo*>(req);
            // The retry-as-legacy contract: legacyIpcCap=true even though
            // we entered via the cuMem arm.
            EXPECT_TRUE(info->legacyIpcCap);
            EXPECT_EQ(info->size,   kBaseSize);
            EXPECT_EQ(info->offset, kBegOffset);
            EXPECT_GE(static_cast<size_t>(respSize), sizeof(void*));
            if (resp) std::memcpy(resp, &kRmtRegAddr, sizeof(void*));
            return ncclSuccess;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = false;  // start false to see the retry arm set it

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(cuMemEnable.calls, 1);
    EXPECT_EQ(retain.calls,      1);
    EXPECT_EQ(xport.calls,       0);
    EXPECT_EQ(release.calls,     0);
    EXPECT_EQ(ipcGet.calls,      1);  // the fallback's legacy export fired
    EXPECT_EQ(proxy.calls,       1);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    // The line-935 `*isLegacyIpc = true` write fired -- this is the
    // only path inside the cuMem arm that sets it.
    EXPECT_TRUE(isLegacyIpc);

    ASSERT_NE(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_TRUE(regRecord.ipcInfos[kPeerLocalRank]->impInfo.legacyIpcCap);
    EXPECT_TRUE(regRecord.state & IPC_REG_COMPLETE);
}

// Companion to the retry-as-legacy test: Retain fails AND the inner
// `directMode || !ncclParamLegacyCudaRegister()` guard short-circuits
// to its True arm via directMode=true. Drives branch 932:17 True
// (the bare-`goto fail` exit from the cuMem arm when no fallback is
// permitted). Mirror of the existing
// FreshRegistrationDirectModeShortCircuitsLegacyExport test, which
// exercises the analogous guard in the *legacy* arm.
TEST_F(P2pMicrotest, IpcRegisterBuffer_CuMemFreshRegistrationRetainFailureDirectModeShortCircuits)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr int       kNRanks        = kPeerRank + 1;

    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);
    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().directMode = 1;  // drives the guard's True arm

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    RegRecordCleaner regCleanup(regRecord);

    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    // legacyParam value doesn't matter for short-circuit semantics --
    // directMode=true makes the first operand True.
    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t) -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook retain(g_hipMemRetainAllocationHandle,
        [](hipMemGenericAllocationHandle_t*, void*) -> hipError_t {
            return hipErrorInvalidValue;
        });
    // Nothing else may fire: the guard short-circuits to goto fail
    // before any of these are touched.
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t*, void*) -> hipError_t {
            ADD_FAILURE() << "short-circuit must not reach hipIpcGetMemHandle";
            return hipErrorInvalidValue;
        });
    ScopedHook proxy(g_proxyCallBlocking,
        [](struct ncclComm*, struct ncclProxyConnector*, int,
           void*, int, void*, int) -> ncclResult_t {
            ADD_FAILURE() << "short-circuit must not reach proxy register";
            return ncclInternalError;
        });

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = false;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(retain.calls, 1);
    EXPECT_EQ(ipcGet.calls, 0);
    EXPECT_EQ(proxy.calls,  0);

    // Bare `goto fail` exit -- same quirk as the legacy-arm directMode
    // short-circuit: ret stays ncclSuccess, fail: epilogue zeros outputs,
    // caller sees ncclSuccess + regBufFlag=0.
    EXPECT_EQ(r, ncclSuccess);
    out.ExpectZeroed();
    EXPECT_FALSE(isLegacyIpc);  // the line-935 write never fired
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_FALSE(regRecord.state & IPC_REG_COMPLETE);
}

#endif  // ROCM_VERSION >= 70000
