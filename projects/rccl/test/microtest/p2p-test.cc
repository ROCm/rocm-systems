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
#include <initializer_list>
#include <utility>
#include <vector>

#include "fakes/p2p_fakes.h"

// Pull in alloc.h NOW so its macros (ncclCudaCallocAsync etc.) are visible
// to be #undef'd. p2p.cc's transitive includes would otherwise be the first
// to see them, and the shim below would land too late.
#include "alloc.h"

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
// This is the path that *produces* the IPC state every reuse test assumes
// already exists. Coverage-wise it lights up the entire body of the
// `else` arm at line ~905 of p2p.cc, including:
//   - hipMemGetAddressRange success (911)
//   - ncclProxyConnect (921-922) because gproxyConn is uninitialised
//   - ncclCuMemEnable() == 0 path -> legacy export arm (979)
//   - cudaIpcGetMemHandle + ipcInfo.legacyIpcCap=true (961-964)
//   - line 964 `if (isLegacyIpc) *isLegacyIpc = true` (the last
//     unexercised isLegacyIpc write)
//   - ncclProxyCallBlocking returning a canned rmtRegAddr (985)
//   - newInfo allocation + regRecord->ipcInfos install (986-1000)
//   - hostPeerRmtAddrs lazy-allocation (994)
//   - post-loop COLLECTIVE block with needUpdate=true driving
//     ncclCudaCallocAsync + ncclCudaMemcpyAsync
//
// All four runtime-driver entry points (hipMemGetAddressRange,
// hipIpcGetMemHandle, ncclProxyConnect, ncclProxyCallBlocking) are driven
// via per-test hooks installed on the fakes.
TEST_F(P2pMicrotest, IpcRegisterBuffer_FreshRegistrationLegacyIpcSucceeds)
{
    constexpr int       kPeerRank      = 2;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBaseAddr      = 0x100000;
    constexpr std::size_t kBaseSize    = 0x4000;
    constexpr uintptr_t kBuffOffset    = 0x80;
    constexpr uintptr_t kBegOffset     = 0x20;  // regRecord->begAddr - baseAddr
    constexpr uintptr_t kBegAddr       = kBaseAddr + kBegOffset;
    constexpr uintptr_t kRmtRegAddr    = 0xCAFE0000ull;

    // nRanks just has to be big enough to index gproxyConn[kPeerRank].
    constexpr int kNRanks = kPeerRank + 1;
    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(kNRanks);

    ncclReg regRecord{};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    // regRecord.ipcInfos[kPeerLocalRank] is NULL -> fresh-registration arm.
    // Construct *after* regRecord so its dtor runs first (LIFO), freeing
    // anything ipcRegisterBuffer leaves behind before regRecord itself dies.
    RegRecordCleaner regCleanup(regRecord);

    // ---- hook 1: hipMemGetAddressRange returns baseAddr + baseSize.
    // Production contract: called with dptr == userbuff (so the driver
    // can find the enclosing allocation). Assert that here -- if a
    // future refactor mis-wires this, we want the test to catch it.
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

    // ---- hook 2: hipIpcGetMemHandle succeeds with a sentinel handle.
    // p2p.cc copies the handle into ipcInfo.ipcDesc.devIpc and ships it
    // off via ncclProxyCallBlocking; the fake on the other side never
    // inspects the bytes, so any pattern works.
    //
    // Production contract: called with the *baseAddr* the driver returned
    // from hipMemGetAddressRange above, NOT the (possibly mid-allocation)
    // userbuff. This is load-bearing: legacy CUDA IPC handles always
    // refer to whole allocations, so passing userbuff here would either
    // fail at runtime or hand the peer a handle to the wrong region.
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [&](hipIpcMemHandle_t* h, void* devPtr) -> hipError_t {
            EXPECT_EQ(devPtr, reinterpret_cast<void*>(kBaseAddr));
            if (h) std::memset(h, 0x5A, sizeof(*h));
            return hipSuccess;
        });

    // ---- hook 3: ncclProxyConnect marks the gproxyConn slot initialized.
    // The real implementation does more, but for the unit under test the
    // only thing that matters post-call is that proxyConn is non-null,
    // which it already is (it's &comm->gproxyConn[peerRank]).
    ScopedHook connect(g_proxyConnect,
        [&](struct ncclComm* c, int transport, int /*send*/,
            int rank, struct ncclProxyConnector* pc) -> ncclResult_t {
            EXPECT_EQ(transport, TRANSPORT_P2P);
            EXPECT_EQ(rank, kPeerRank);
            EXPECT_EQ(pc, &c->gproxyConn[rank]);
            pc->initialized = true;
            return ncclSuccess;
        });

    // ---- hook 4: ncclProxyCallBlocking returns a canned rmtRegAddr in
    // respBuff. Without this the post-call `if (rmtRegAddr)` block
    // (which is where the real registration bookkeeping happens) stays
    // skipped.
    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*,
            int type, void* req, int reqSize,
            void* resp, int respSize) -> ncclResult_t {
            // Production contract for ncclProxyMsgRegister: the request
            // is a fully-populated p2pIpcExpInfo describing the
            // allocation we want the peer to import. Pin down the
            // fields the unit under test is responsible for setting --
            // a regression that ships a half-built request or wires the
            // wrong base address would otherwise sail through.
            EXPECT_EQ(type, ncclProxyMsgRegister);
            // Can't use ASSERT_* inside a non-void-returning lambda; do a
            // manual early-return on the precondition that the cast below
            // depends on.
            if (req == nullptr ||
                static_cast<size_t>(reqSize) < sizeof(p2pIpcExpInfo)) {
                ADD_FAILURE() << "malformed register-msg request: req="
                              << req << " reqSize=" << reqSize;
                return ncclInternalError;
            }
            auto* info = static_cast<p2pIpcExpInfo*>(req);
            EXPECT_TRUE(info->legacyIpcCap);              // legacy arm set this
            EXPECT_EQ(info->size,   kBaseSize);           // whole-allocation
            EXPECT_EQ(info->offset, kBegOffset);          // begAddr - baseAddr

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

    // ---- contract: every seam was called exactly once on the happy path.
    EXPECT_EQ(memGet.calls,  1);
    EXPECT_EQ(ipcGet.calls,  1);
    EXPECT_EQ(connect.calls, 1);
    EXPECT_EQ(proxy.calls,   1);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    // COLLECTIVE: returns the dev table (allocated by the strong-stream
    // block via g_fakeCudaCallocAsync's heap default), populated from the
    // host table by g_fakeCudaMemcpyAsync's real-memcpy default.
    ASSERT_NE(out.peerRmtAddrs, nullptr);
    EXPECT_EQ(out.peerRmtAddrs, regRecord.regIpcAddrs.devPeerRmtAddrs);
    EXPECT_EQ(out.peerRmtAddrs[kPeerLocalRank], kRmtRegAddr);
    EXPECT_TRUE(isLegacyIpc);  // line 964 write reached

    // ---- contract: regRecord bookkeeping populated as documented.
    ASSERT_NE(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank]->peerRank, kPeerRank);
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank]->baseAddr,
              reinterpret_cast<void*>(kBaseAddr));
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank]->impInfo.rmtRegAddr,
              reinterpret_cast<void*>(kRmtRegAddr));
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank]->impInfo.offset, kBegOffset);
    EXPECT_TRUE(regRecord.ipcInfos[kPeerLocalRank]->impInfo.legacyIpcCap);
    EXPECT_TRUE(regRecord.state & IPC_REG_COMPLETE);
    ASSERT_NE(regRecord.regIpcAddrs.hostPeerRmtAddrs, nullptr);
    EXPECT_EQ(regRecord.regIpcAddrs.hostPeerRmtAddrs[kPeerLocalRank],
              kRmtRegAddr);

    // Cleanup is handled by regCleanup (RAII) + ResetP2pFakes() (for the
    // dev-table calloc owned by g_fakeAllocations). Both run even if an
    // ASSERT_* above bails early.
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
