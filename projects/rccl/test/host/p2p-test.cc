/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "ScopedHook.h"
#include "fakes/nccl_fakes.h"
#include "fakes/hip_fakes.h"
#include "../common/ProcessIsolatedTestRunner.hpp"

// Pull in alloc.h NOW so its macros (ncclCudaCallocAsync etc.) are visible
// to be #undef'd. p2p.cc's transitive includes would otherwise be the first
// to see them, and the shim below would land too late.
#include "alloc.h"

// Macro shim: replace the header-only ncclCalloc template so a test can force
// its allocation-failure arm. ncclCalloc's alloc.h template calls libc malloc
// directly (there is no external symbol or hookable seam to control it), so
// the failure branch (`if (p == NULL) return ncclSystemError`) is otherwise
// unreachable in a no-GPU test. Mirrors the ncclCalloc shim in init-test.cc.
//
// Armed by exact request size (byte count of `nelem * sizeof(T)`), not a call
// index, so a test targets one specific allocation without depending on how
// many callocs the surrounding code happens to make. SetP2pCallocFailSize(0)
// disarms; the fixture's TearDown resets it. Both statics are TU-local, so
// this never affects any other test binary.
static std::size_t g_callocFailSize = 0;  // 0 = never fail
template <typename T>
static ncclResult_t MicroCalloc(const char* file, int line, const char* fn,
                                T** ptr, std::size_t nelem) {
    if (g_callocFailSize != 0 && nelem * sizeof(T) == g_callocFailSize) {
        g_callocFailSize = 0;  // fire once, then disarm
        return ncclSystemError;
    }
    return ncclCallocDebug(ptr, nelem, file, line, fn, true);
}
static void SetP2pCallocFailSize(std::size_t size) { g_callocFailSize = size; }
#undef ncclCalloc
#define ncclCalloc(...) MicroCalloc(__FILE__, __LINE__, __func__, __VA_ARGS__)

// NCCL_PARAM redirector (shared with init-test.cc): routes every generated
// ncclParamXxx() through g_loadParam on each call so tests can flip a param's
// value between cases (without this, ncclParamLegacyCudaRegister() and friends
// would cache their default on first call). See param_redirect.h.
#include "fakes/param_redirect.h"

// Controllable seams for the two header-only alloc.h templates. Declared
// here (defined below, after P2P_CC_PATH) because these hooks and the
// honest-emulator machinery behind them are used only by this test file;
// the generic, cross-module fakes live under fakes/.
extern std::function<ncclResult_t(void** ptr, std::size_t nbytes, hipStream_t)>
    g_fakeCudaCallocAsync;
extern std::function<ncclResult_t(void* dst, void* src, std::size_t nbytes, hipStream_t)>
    g_fakeCudaMemcpyAsync;

// Controllable seam for the header-only ncclCuMemAlloc template, which the
// cuMem arm of ncclP2pAllocateShareableBuffer calls. The real template drives
// a long chain of HIP driver primitives (cuMemCreate / reserve / map /
// setAccess / zeroing side stream) that has no GPU here; the macro shim below
// routes the call site to this seam instead. Type-erased to (void** ptr,
// handle*, size). Default hands back a heap buffer and a sentinel handle.
extern std::function<ncclResult_t(void** ptr,
                                  hipMemGenericAllocationHandle_t* handlep,
                                  std::size_t size)>
    g_fakeCuMemAlloc;

// Restore every hook this file drives (plus, transitively, the nccl* and HIP
// hooks) to its default. Called from the file-wide fixture's TearDown().
// Defined below, after P2P_CC_PATH.
static void ResetP2pFakes();

// Macro shim: replace the header-only function templates ncclCudaCallocAsync
// and ncclCudaMemcpyAsync from alloc.h with thin trampolines that route
// through hookable fakes defined in this file. Without this, p2p.cc's call
// sites bind directly to the templates, which hit real HIP runtime (no GPU
// in this binary by design).
//
// The shims preserve type information at the call site via sizeof(**ptr) /
// sizeof(*dst); the fake hooks themselves are type-erased to (void*, nbytes).
// This mirrors the existing macro-intercept pattern used for ncclDebugLog
// and ncclLoadParam.
#undef ncclCudaCallocAsync
#undef ncclCudaMemcpyAsync
// Variadic: production's ncclCudaCallocAsync now takes trailing
// manager/memType args (e.g. comm->memManager). We ignore them here --
// the fake is type-erased to (void**, nbytes, stream) -- but the macro
// must still swallow the extra arguments so the call site expands.
#define ncclCudaCallocAsync(ptr, nelem, stream, ...) \
    g_fakeCudaCallocAsync(reinterpret_cast<void**>(ptr), \
                          (nelem) * sizeof(**(ptr)), (stream))
// ncclCudaCalloc (non-async): the legacy-IPC arm of
// ncclP2pAllocateShareableBuffer allocates the buffer through this before
// hipIpcGetMemHandle. Route it through the same honest emulator (a NULL
// stream stands in for the synchronous call). The variadic tail swallows the
// trailing manager/memType/flags arguments.
#undef ncclCudaCalloc
#define ncclCudaCalloc(ptr, nelem, ...) \
    g_fakeCudaCallocAsync(reinterpret_cast<void**>(ptr), \
                          (nelem) * sizeof(**(ptr)), nullptr)
// ncclCuMemAlloc: the cuMem arm of ncclP2pAllocateShareableBuffer allocates
// through this. Route it to the file-local g_fakeCuMemAlloc seam so the call
// site never reaches the real VMM primitives. The variadic tail swallows the
// trailing manager/memType arguments.
#define ncclCuMemAlloc(ptr, handlep, type, size, ...) \
    g_fakeCuMemAlloc(reinterpret_cast<void**>(ptr), (handlep), (size))
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

// hipPointerGetAttribute: HIP_VERSION >= 71260540 fresh-reg arm queries
// legacy-IPC capability through this instead of ncclParamLegacyCudaRegister().
#define hipPointerGetAttribute(data, attribute, ptr) \
    g_hipPointerGetAttribute((data), (attribute), (ptr))

// ---------------------------------------------------------------------------
// Recording shim for the alloc.h free primitives (ncclCuMemFreeAddr,
// ncclCudaFree).
//
// These are header-only functions in alloc.h, so there is no external symbol
// or hookable seam to observe them. But the p2p.cc free paths (send/recv
// connector free and proxyFree) call them with the retained device pointer as
// the first argument -- exactly the observable a test needs to prove the right
// handle was released on the right arm. Without this, a test can only assert
// ncclSuccess, which the shutdown-flag short-circuit (ShutdownFlagGuard)
// guarantees regardless of what the free path actually did, so deleting the
// free call under test leaves the test green.
//
// The wrappers record (kind, ptr) and then delegate to the real inline. The
// ShutdownFlagGuard those tests install makes the real inline return before it
// reaches any HIP symbol, so recording does not change behaviour -- it only
// makes the release observable. g_freeCalls is cleared in ResetP2pFakes().
//
// Defined here, while the real inline names are still callable (the #define
// shims below would otherwise make the wrapper bodies recurse).
namespace {
enum class FreeKind { CuMemFreeAddr, CudaFree };
struct FreeCall {
    FreeKind    kind;
    const void* ptr;
    bool operator==(const FreeCall& o) const { return kind == o.kind && ptr == o.ptr; }
};
std::vector<FreeCall> g_freeCalls;

inline ncclResult_t RecordCuMemFreeAddr(void* ptr, struct ncclMemManager* mgr,
                                        int numSegments = 1) {
    g_freeCalls.push_back({FreeKind::CuMemFreeAddr, ptr});
    return ncclCuMemFreeAddr(ptr, mgr, numSegments);
}
template <typename T>
inline ncclResult_t RecordCudaFree(T* ptr, struct ncclMemManager* mgr,
                                   int numSegments = 1) {
    g_freeCalls.push_back({FreeKind::CudaFree, static_cast<const void*>(ptr)});
    return ncclCudaFree(ptr, mgr, numSegments);
}
}  // namespace

// Route every call site in the included p2p.cc through the recording wrappers.
// Variadic so both the 2-arg (manager only) and 3-arg (manager, numSegments)
// call sites expand.
// ncclCuMemFree has no bare call site in p2p.cc (every cuMem free goes through
// ncclCuMemFreeAddr, and ncclCudaFree's internal ncclCuMemFree call was already
// bound to the real inline when alloc.h was included above), so no intercept is
// needed for it.
#define ncclCuMemFreeAddr(...) RecordCuMemFreeAddr(__VA_ARGS__)
#define ncclCudaFree(...)      RecordCudaFree(__VA_ARGS__)

// Pull in the hipified copy of p2p.cc (cudaXxx -> hipXxx rewrites already
// applied by the hipify pass that runs as part of the main RCCL build).
// P2P_CC_PATH is defined by this target's CMakeLists.txt as a string
// literal pointing at ${PROJECT_BINARY_DIR}/hipify/src/transport/p2p.cc.
#include P2P_CC_PATH

// ===========================================================================
// p2p.cc link-satisfying stubs + this file's controllable alloc seams.
//
// Defined here (not in a shared fakes/ .cc) because they have no owning fakes
// file: allocTracker is an alloc.h data symbol p2p.cc alone references. The
// ncclCudaCallocAsync / ncclCudaMemcpyAsync emulators back the macro shims
// above. They must land after #include P2P_CC_PATH so the production types
// they mention (allocationTracker, the alloc.h templates, etc.) are already
// in scope. busIdToInt64 / getBusId are owned by src/misc/utils.cc, so they
// live in fakes/utils_fakes.cc, not here.
// ---------------------------------------------------------------------------

// allocTracker is an array of per-device counters in alloc.h; size it to the
// same MAX_ALLOC_TRACK_NGPU the header uses. Zero-initialised.
struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};

// Controllable seams: ncclCudaCallocAsync / ncclCudaMemcpyAsync. Substitutes
// for the header-only function templates in alloc.h -- the shim macros above
// route those call sites here, type-erased to (void*, nbytes), so the test
// binary never reaches real HIP runtime.
//
// Defaults behave like an honest emulator: heap-allocate zeroed memory and
// memcpy bytes between host pointers. ResetP2pFakes() frees any allocations
// the default hook handed out so individual tests don't have to. Tests that
// install their own hook also take responsibility for any memory they hand out.
//
// Scope: the shim macros above intercept every call site in the included
// p2p.cc, not just ipcRegisterBuffer's fresh-registration arm.
namespace {
std::vector<void*> g_fakeAllocations;

ncclResult_t DefaultFakeCudaCallocAsync(void** ptr, std::size_t nbytes,
                                        hipStream_t /*stream*/)
{
    if (ptr == nullptr) return ncclInvalidArgument;
    void* p = std::calloc(1, nbytes);
    if (p == nullptr && nbytes > 0) return ncclSystemError;
    g_fakeAllocations.push_back(p);
    *ptr = p;
    return ncclSuccess;
}

ncclResult_t DefaultFakeCudaMemcpyAsync(void* dst, void* src,
                                        std::size_t nbytes,
                                        hipStream_t /*stream*/)
{
    if (nbytes > 0 && (dst == nullptr || src == nullptr)) return ncclInvalidArgument;
    if (nbytes > 0) std::memcpy(dst, src, nbytes);
    return ncclSuccess;
}

// Sentinel handle the default ncclCuMemAlloc emulator hands back, so tests
// can assert the allocated handle is threaded through export/retain.
constexpr std::uintptr_t kFakeCuMemAllocHandleBits = 0xA110C0DE0000ull;

ncclResult_t DefaultFakeCuMemAlloc(void** ptr,
                                   hipMemGenericAllocationHandle_t* handlep,
                                   std::size_t nbytes)
{
    if (ptr == nullptr) return ncclInvalidArgument;
    void* p = std::calloc(1, nbytes ? nbytes : 1);
    if (p == nullptr) return ncclSystemError;
    g_fakeAllocations.push_back(p);
    *ptr = p;
    if (handlep) {
        std::memset(handlep, 0, sizeof(*handlep));
        std::memcpy(handlep, &kFakeCuMemAllocHandleBits, sizeof(std::uintptr_t));
    }
    return ncclSuccess;
}
}  // namespace

std::function<ncclResult_t(void**, std::size_t, hipStream_t)>
    g_fakeCudaCallocAsync = DefaultFakeCudaCallocAsync;
std::function<ncclResult_t(void*, void*, std::size_t, hipStream_t)>
    g_fakeCudaMemcpyAsync = DefaultFakeCudaMemcpyAsync;
std::function<ncclResult_t(void**, hipMemGenericAllocationHandle_t*, std::size_t)>
    g_fakeCuMemAlloc = DefaultFakeCuMemAlloc;

static void ResetP2pFakes()
{
    g_fakeCudaCallocAsync = DefaultFakeCudaCallocAsync;
    g_fakeCudaMemcpyAsync = DefaultFakeCudaMemcpyAsync;
    g_fakeCuMemAlloc      = DefaultFakeCuMemAlloc;
    ResetNcclFakes();  // restore the nccl* hooks owned by nccl_fakes.cc
    ResetHipFakes();   // restore the HIP hooks owned by hip_fakes.cc
    SetP2pCallocFailSize(0);  // disarm any ncclCalloc failure arming
    for (void* p : g_fakeAllocations) std::free(p);
    g_fakeAllocations.clear();
    g_freeCalls.clear();  // discard any recorded alloc.h free calls
}

// ===========================================================================
// Default fixture for every test in this file.
//
// Several tests install per-test hooks into the controllable seams declared
// above (e.g. g_strongStreamAcquire). ResetP2pFakes() puts
// every hook back to its default in TearDown so tests don't leak state into
// each other. Tests that don't currently install hooks still use this
// fixture -- it's the file-wide default so adding a hook to a test that
// previously didn't need one doesn't silently contaminate the next test.
// ===========================================================================
class P2pMicrotest : public ::testing::Test {
protected:
    void TearDown() override { ResetP2pFakes(); }
};

// RAII guard for the fakes-owned ncclCuMemHandleType global. Many tests drive
// this global to select which cuMem*-export sub-arm fires, then must restore
// it so the change does not leak into later tests in the same binary. A manual
// save/restore pair is unsafe: an ASSERT_* between the two lines returns from
// the test body and skips the restore, latching the global and cascading into
// unrelated cuMem-arm failures. This guard restores on scope exit no matter
// how the body leaves. Construct with the desired handle type; the previous
// value is captured and put back in the destructor.
struct ScopedCuMemHandleType {
    explicit ScopedCuMemHandleType(hipMemAllocationHandleType type)
        : saved_(ncclCuMemHandleType) { ncclCuMemHandleType = type; }
    ~ScopedCuMemHandleType() { ncclCuMemHandleType = saved_; }
    ScopedCuMemHandleType(const ScopedCuMemHandleType&)            = delete;
    ScopedCuMemHandleType& operator=(const ScopedCuMemHandleType&) = delete;
private:
    hipMemAllocationHandleType saved_;
};

// ===========================================================================
// Helpers: lightweight builders for the recurring input/output shapes.
//
// Goal: each test body should read as "build the state that makes this test
// different, call the function, assert". Anything that's identical between
// tests lives here.
// ===========================================================================

namespace {

// MakeHeapIpcInfo -- allocate a zeroed ncclIpcRegInfo the same way
// ipcRegisterBuffer's fresh-registration arm does (ncclCalloc == malloc +
// memset)
inline ncclIpcRegInfo* MakeHeapIpcInfo(int peerRank, uintptr_t rmtRegAddr,
                                       bool legacyIpcCap)
{
    auto* info = static_cast<ncclIpcRegInfo*>(
        std::calloc(1, sizeof(ncclIpcRegInfo)));
    info->peerRank             = peerRank;
    info->impInfo.rmtRegAddr   = reinterpret_cast<void*>(rmtRegAddr);
    info->impInfo.legacyIpcCap = legacyIpcCap;
    return info;
}

// RegRecordCleaner -- RAII guard that frees the allocations
// ipcRegisterBuffer makes *into* a ncclReg on the fresh-registration path:
//
//   - regRecord.ipcInfos[i]                       (per-peer ncclCalloc'd newInfo)
//   - regRecord.regIpcAddrs.hostPeerRmtAddrs      (lazily-ncclCalloc'd host table)
//
// regIpcAddrs.devPeerRmtAddrs is owned by g_fakeAllocations (the
// ncclCudaCallocAsync default registers it there), so this guard
// deliberately doesn't touch it.
struct RegRecordCleaner {
    ncclReg& reg;
    explicit RegRecordCleaner(ncclReg& r) : reg(r) {}
    ~RegRecordCleaner() {
        for (int i = 0; i < reg.ipcInfosSize; ++i) {
            if (reg.ipcInfos[i]) { std::free(reg.ipcInfos[i]); reg.ipcInfos[i] = nullptr; }
        }
        if (reg.regIpcAddrs.hostPeerRmtAddrs) {
            std::free(reg.regIpcAddrs.hostPeerRmtAddrs);
            reg.regIpcAddrs.hostPeerRmtAddrs = nullptr;
        }
    }
    RegRecordCleaner(const RegRecordCleaner&)            = delete;
    RegRecordCleaner& operator=(const RegRecordCleaner&) = delete;
};

// IpcInfosBacking -- provides stack storage for ncclReg::ipcInfos, which
// production now treats as a dynamically-(re)allocated ncclIpcRegInfo**
// sized to ipcInfosSize (it used to be a fixed inline array). By
// pre-attaching a NCCL_MAX_LOCAL_RANKS-sized zeroed array and setting
// ipcInfosSize to match, production's
//     if (ipcInfos == NULL || ipcInfosSize < ipcIndexSize) ncclRealloc(...)
// guard never fires (ipcIndexSize == localRanks <= NCCL_MAX_LOCAL_RANKS),
// so the array stays stack-owned -- no malloc to free, no per-test leak --
// while individual slots behave exactly like the real pointer array. Drop
// one of these next to every ncclReg a test drives ipcRegisterBuffer with;
// it must outlive the call.
struct IpcInfosBacking {
    std::array<ncclIpcRegInfo*, NCCL_MAX_LOCAL_RANKS> storage{};
    explicit IpcInfosBacking(ncclReg& reg) {
        reg.ipcInfos     = storage.data();
        reg.ipcInfosSize = NCCL_MAX_LOCAL_RANKS;
    }
    IpcInfosBacking(const IpcInfosBacking&)            = delete;
    IpcInfosBacking& operator=(const IpcInfosBacking&) = delete;
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
        if (comm_.nRanks <= peerRank) comm_.nRanks = peerRank + 1;
        if (comm_.localRanks <= peerLocalRank) comm_.localRanks = peerLocalRank + 1;
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

// ForceLegacyIpcCapable -- the HIP_VERSION >= 71260540 analogue of
// ForceLegacyCudaRegister. On that HIP the fresh-registration arm reads
// legacy-IPC capability from hipPointerGetAttribute (p2p.cc line 1152)
// rather than from ncclParamLegacyCudaRegister(). Tests that need the
// `else if (legacyIpcCap)` legacy-export arm install this hook (in addition
// to ForceLegacyCudaRegister, which drives the < 71260540 branch) so they
// pass regardless of the toolchain's HIP version. Returns hipSuccess and
// reports the buffer as legacy-IPC-capable.
auto ForceLegacyIpcCapable()
{
    return [](void* data, hipPointer_attribute attribute,
              hipDeviceptr_t) -> hipError_t {
        if (data && attribute == HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE)
            *static_cast<int*>(data) = 1;
        return hipSuccess;
    };
}

// TopoDirectP2p -- the g_ncclTopoCheckP2p hook every direct-P2P case in the
// canConnect tests needs: reports the pair as p2p-capable (p2p = 1) with no
// intermediate hop (inter = -1). Returned as a plain lambda so call sites
// compose: ScopedHook topo(g_ncclTopoCheckP2p, TopoDirectP2p()). Cases that
// need an intermediate hop (inter != -1) keep their own inline hook.
auto TopoDirectP2p()
{
    return [](int, int, int* p2p, int*, int* inter, int*, int*) -> ncclResult_t {
        if (p2p) *p2p = 1;
        if (inter) *inter = -1;
        return ncclSuccess;
    };
}

// ForceP2pUseCudaMemcpy -- the g_loadParam hook that the CE-memcpy tests
// install to make ncclParamP2pUseCudaMemcpy() report enabled. Non-capturing,
// so it composes inside the RUN_ISOLATED_TEST child lambdas:
//     ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
// The raw env string (no "NCCL_" prefix) matches the redirector's arg, as with
// ForceLegacyCudaRegister above.
auto ForceP2pUseCudaMemcpy()
{
    return [](const char* env, int64_t deft) -> int64_t {
        if (std::strcmp(env, "P2P_USE_CUDA_MEMCPY") == 0) return 1;
        return deft;
    };
}

auto GranularitySucceeds()
{
    return [](std::size_t* g, const hipMemAllocationProp*,
              hipMemAllocationGranularity_flags) -> hipError_t {
        if (g) *g = 4096;
        return hipSuccess;
    };
}

// CannedRmtRegAddr -- the ncclProxyCallBlocking lambda that every fresh-reg
// happy path needs: writes a fixed rmtRegAddr into the response buffer so
// the post-loop bookkeeping fires. It performs no request-struct assertions
// -- tests that want to pin down the request contents keep their own inline
// hook (e.g. FreshRegistrationLegacyIpcSucceedsFixture, the cuMem happy-path
// tests). Install with `ScopedHook proxy(g_proxyCallBlocking,
// CannedRmtRegAddr(kRmtRegAddr));`.
inline auto CannedRmtRegAddr(uintptr_t rmtRegAddr)
{
    return [rmtRegAddr](struct ncclComm*, struct ncclProxyConnector*, int,
                        void*, int, void* resp, int respSize) -> ncclResult_t {
        EXPECT_GE(static_cast<std::size_t>(respSize), sizeof(void*));
        if (resp && static_cast<std::size_t>(respSize) >= sizeof(void*))
            std::memcpy(resp, &rmtRegAddr, sizeof(void*));
        return ncclSuccess;
    };
}

// MarkProxyConnInitialized -- the ncclProxyConnect lambda that just marks the
// gproxyConn slot initialized and succeeds. The common happy-path connect
// hook; tests that assert the connect arguments keep their own inline hook.
inline auto MarkProxyConnInitialized()
{
    return [](struct ncclComm*, int, int, int,
              struct ncclProxyConnector* pc) -> ncclResult_t {
        pc->initialized = true;
        return ncclSuccess;
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
// FreshRegistrationMicrotest -- shared scaffolding for tests that drive the
// fresh-registration arm of ipcRegisterBuffer (the `else` arm of the per-peer
// loop, where ipcInfos[peerLocalRank] is NULL on entry).
//
// What this fixture owns:
//   - The scenario constants every fresh-reg test uses (peer rank, base
//     addr, buffer offset, ranks-in-comm).
//   - A pre-wired CommBuilder (rankToLocalRank table, localRanks set to
//     NCCL_MAX_LOCAL_RANKS, sharedRes pointer, gproxyConn array sized to
//     kNRanks).
//   - A ncclReg with begAddr/endAddr populated and a RegRecordCleaner
//     attached so per-peer ipcInfos / hostPeerRmtAddrs allocations are
//     freed automatically.
//
// What tests provide themselves:
//   - The specific seam hooks that make the test interesting. Use the
//     MakeDefault*Hook() helpers for the boilerplate happy-path memGet /
//     ipcGet hooks; install proxy/connect/acquire/etc. directly with
//     ScopedHook in the test body.
//   - InstallLegacyCudaRegisterHook() to force the legacy-IPC arm. Not
//     done by default because several tests (`NothingWorksFallthrough`,
//     `ProxyConnectFailurePropagates`, the cuMem-arm tests) deliberately
//     want g_loadParam at its default.
//
// The helper hooks return ScopedHook by value; C++17 guaranteed copy
// elision lets the non-movable, non-copyable ScopedHook propagate out of
// the prvalue return. Capture them with `auto x = MakeDefault...();`.
//
// Defined here (after the anonymous-namespace helpers block) because it
// names CommBuilder / RegRecordCleaner / ScopedHook / ForceLegacyCudaRegister
// from that namespace.
// ===========================================================================
class FreshRegistrationMicrotest : public P2pMicrotest {
protected:
    static constexpr int          kPeerRank      = 2;
    static constexpr int          kPeerLocalRank = 1;
    static constexpr uintptr_t    kBaseAddr      = 0x100000;
    static constexpr std::size_t  kBaseSize      = 0x4000;
    static constexpr uintptr_t    kBuffOffset    = 0x80;
    static constexpr uintptr_t    kBegOffset     = 0x20;
    static constexpr uintptr_t    kBegAddr       = kBaseAddr + kBegOffset;
    static constexpr int          kNRanks        = kPeerRank + 1;

    CommBuilder cb;
    ncclReg     regRecord{};
    IpcInfosBacking ipcInfosBacking{regRecord};
    // std::optional rather than raw members so RegRecordCleaner (which
    // captures &regRecord) and the ScopedHook (which mutates a global
    // std::function slot) are constructed in SetUp() *after* regRecord
    // has its initial state, and torn down in TearDown() in the right
    // order. Using std::optional here avoids the unique_ptr<ScopedHook<
    // long-sig...>> pattern the older fixture in this file has to use.
    std::optional<RegRecordCleaner> regCleanup;
    std::optional<ScopedHook<int64_t(const char*, int64_t)>> loadParam;
    std::optional<ScopedHook<hipError_t(void*, hipPointer_attribute,
                                        hipDeviceptr_t)>> pointerAttr;

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);

    void SetUp() override {
        cb.WithLocalRank(kPeerRank, kPeerLocalRank)
          .WithMaxLocalRanks()
          .WithSharedRes()
          .WithProxyConnArray(kNRanks);
        regRecord.begAddr = kBegAddr;
        regRecord.endAddr = kBegAddr + 0x1000;
        regCleanup.emplace(regRecord);
    }

    void TearDown() override {
        // Hooks first (restore the global slots), then the regRecord
        // cleaner (frees per-peer allocations the call may have made).
        loadParam.reset();
        pointerAttr.reset();
        regCleanup.reset();
        P2pMicrotest::TearDown();
    }

    // Force the legacy-IPC arm of the fresh-registration path. Tests that
    // need to reach the `else if (legacyIpcCap)` block call this in their
    // body before invoking ipcRegisterBuffer.
    void InstallLegacyCudaRegisterHook() {
        // Drive both HIP branches: ForceLegacyCudaRegister for
        // HIP_VERSION < 71260540 (param-gated) and ForceLegacyIpcCapable
        // for >= 71260540 (hipPointerGetAttribute-gated).
        loadParam.emplace(g_loadParam, ForceLegacyCudaRegister());
        pointerAttr.emplace(g_hipPointerGetAttribute, ForceLegacyIpcCapable());
    }

    // Standard happy-path hooks for the two HIP driver entry points the
    // fresh-reg arm hits before reaching the seam most tests actually
    // care about. Returned by value; capture with `auto`.
    auto MakeDefaultMemGetHook() {
        return ScopedHook(g_hipMemGetAddressRange,
            [](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t)
                -> hipError_t {
                if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
                if (psize) *psize = kBaseSize;
                return hipSuccess;
            });
    }
    auto MakeDefaultIpcGetHook() {
        return ScopedHook(g_hipIpcGetMemHandle,
            [](hipIpcMemHandle_t* h, void*) -> hipError_t {
                if (h) std::memset(h, 0x5A, sizeof(*h));
                return hipSuccess;
            });
    }
};

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
    IpcInfosBacking ipcInfosBacking{regRecord};
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
    IpcInfosBacking ipcInfosBacking{regRecord};
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

// Reuse-COLLECTIVE with the device peer-address table missing: pins down
// the post-NCCL#1861 behaviour that the post-loop strong-stream block is
// now gated solely on `if (needUpdate)`. On a pure-reuse path no peer is
// freshly registered, so needUpdate stays false and the block is skipped
// entirely -- even though devPeerRmtAddrs is NULL.
//
// Before the fix the outer guard was `devPeerRmtAddrs == NULL || needUpdate`,
// so a null dev table alone would enter the block; this test used to drive
// a strong-stream failure through that arm. That arm no longer exists, so
// the strong-stream seam must never be touched here. We install a hook that
// returns an error and *would* propagate if it were ever called; asserting
// acquire.calls == 0 (plus a successful, non-failure return) pins the
// skipped-block contract.
TEST_F(P2pMicrotest, IpcRegisterBuffer_CollectiveReuseSkipsStrongStreamWhenNoUpdate)
{
    constexpr int       kPeerRank      = 1;
    constexpr int       kPeerLocalRank = 1;
    constexpr uintptr_t kBegAddr       = 0x10000;
    constexpr uintptr_t kBuffOffset    = 0x100;
    constexpr uintptr_t kRmtRegAddr    = 0xA000;

    // sharedRes must be non-null: the strong-stream call site takes the
    // address of comm->sharedRes->hostStream. The block is skipped, so it
    // is never dereferenced, but keep it set defensively.
    CommBuilder cb;
    cb.WithLocalRank(kPeerRank, kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes();

    ReusableIpcInfo existing(kPeerRank, kPeerLocalRank, kRmtRegAddr,
                             /*legacyIpcCap=*/ false);
    ncclReg regRecord{};
    IpcInfosBacking ipcInfosBacking{regRecord};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x4000;
    existing.InstallInto(regRecord);
    // devPeerRmtAddrs intentionally left null. Under the old guard this
    // alone entered the strong-stream block; under the new needUpdate-only
    // guard the block stays skipped.

    ScopedHook acquire(g_strongStreamAcquire,
        [](struct ncclCudaGraph, struct ncclStrongStream*, bool,
           hipStream_t* stream) -> ncclResult_t {
            ADD_FAILURE() << "strong-stream block must not be entered on a "
                          << "pure-reuse path (needUpdate=false)";
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

    EXPECT_EQ(acquire.calls, 0);    // strong-stream block skipped
    EXPECT_EQ(r, ncclSuccess);      // no failure path taken
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    // Block skipped -> dev table never allocated -> COLLECTIVE returns the
    // (still-null) dev table.
    EXPECT_EQ(regRecord.regIpcAddrs.devPeerRmtAddrs, nullptr);
    EXPECT_EQ(out.peerRmtAddrs, nullptr);
    EXPECT_FALSE(isLegacyIpc);      // reused info's legacyIpcCap = false
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
    std::unique_ptr<IpcInfosBacking>   ipcInfosBacking;
    std::unique_ptr<RegRecordCleaner>  regCleanup;
    std::unique_ptr<ScopedHook<int64_t(const char*, int64_t)>> loadParam;
    std::unique_ptr<ScopedHook<hipError_t(void*, hipPointer_attribute, hipDeviceptr_t)>> pointerAttr;
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
        ipcInfosBacking = std::make_unique<IpcInfosBacking>(*regRecord);
        // ipcInfos[kPeerLocalRank] is NULL -> fresh-registration arm.
        regCleanup = std::make_unique<RegRecordCleaner>(*regRecord);

        // Force the legacy-IPC arm to be entered (see ForceLegacyCudaRegister
        // for why this is required under HIP_VERSION < 71260540).
        loadParam = std::make_unique<ScopedHook<int64_t(const char*, int64_t)>>(
            g_loadParam, ForceLegacyCudaRegister());
        // ...and the HIP_VERSION >= 71260540 analogue (legacy-IPC capable).
        pointerAttr = std::make_unique<ScopedHook<hipError_t(void*, hipPointer_attribute, hipDeviceptr_t)>>(
            g_hipPointerGetAttribute, ForceLegacyIpcCapable());

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
        pointerAttr.reset();
        loadParam.reset();
        regCleanup.reset();
        ipcInfosBacking.reset();
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
TEST_F(FreshRegistrationMicrotest, ProxyReturnsNullRmtAddrSkipsBookkeeping)
{
    InstallLegacyCudaRegisterHook();
    auto memGet = MakeDefaultMemGetHook();
    auto ipcGet = MakeDefaultIpcGetHook();
    ScopedHook connect(g_proxyConnect, MarkProxyConnInitialized());
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
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank], nullptr);  // fixture's regRecord
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
TEST_F(FreshRegistrationMicrotest, SkipsProxyConnectWhenAlreadyInitialized)
{
    constexpr uintptr_t kRmtRegAddr = 0xCAFE0000ull;

    // Key precondition: skip the connect.
    cb.comm().gproxyConn[kPeerRank].initialized = true;

    InstallLegacyCudaRegisterHook();
    auto memGet = MakeDefaultMemGetHook();
    auto ipcGet = MakeDefaultIpcGetHook();
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
        CannedRmtRegAddr(kRmtRegAddr));

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;
    bool isLegacyIpc = false;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, &isLegacyIpc);

    EXPECT_EQ(connect.calls, 0);    // the contract under test
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
TEST_F(FreshRegistrationMicrotest, FreesNewInfoOnPostLoopFailure)
{
    constexpr uintptr_t kRmtRegAddr = 0xCAFE0000ull;

    InstallLegacyCudaRegisterHook();
    auto memGet = MakeDefaultMemGetHook();
    auto ipcGet = MakeDefaultIpcGetHook();
    ScopedHook connect(g_proxyConnect, MarkProxyConnInitialized());
    ScopedHook proxy(g_proxyCallBlocking, CannedRmtRegAddr(kRmtRegAddr));
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

// Regression test for NVIDIA/nccl#1859 / AICOMRCCL-1100 ("user-buffer p2p +
// coll rmtAddr nullptr error"). The same buggy code was present in RCCL's
// p2p.cc; the fix (nccl#1861) is now merged.
//
// The bug lived in the post-loop copy gate. The buggy form was:
//     if (needUpdate && type == NCCL_IPC_COLLECTIVE) { ... calloc + memcpy }
// the fix drops the type check:
//     if (needUpdate) { ... calloc + memcpy }
//
// Repro at the application level, driven here as an actual two-call
// sequence against ONE ncclReg:
//
//   Call 1 -- fresh SENDRECV registration of a user buffer.
//     needUpdate becomes true (a peer was freshly registered). Under the
//     BUG the copy block is skipped because type != COLLECTIVE, so
//     devPeerRmtAddrs stays NULL. Under the FIX the block runs: the dev
//     table is allocated and populated from hostPeerRmtAddrs.
//
//   Call 2 -- COLLECTIVE reuse of the SAME buffer.
//     needUpdate is false (pure reuse), so the copy block is skipped
//     either way. The COLLECTIVE arm simply returns devPeerRmtAddrs.
//     Under the BUG that is NULL -> the device kernel dereferences a null
//     remote-address table and crashes. Under the FIX it is the table
//     populated back in call 1.
//
// The contract this test pins down: after the SENDRECV-then-COLLECTIVE
// sequence, COLLECTIVE returns a non-null dev table whose per-peer slot
// equals the remote address recorded during the fresh SENDRECV
// registration. The honest-emulator calloc/memcpy fakes (defaults) make
// the difference observable: zeros / null under the bug, kRmtRegAddr
// under the fix.
TEST_F(FreshRegistrationMicrotest, RegressionNcclIssue1859_SendrecvThenCollectivePopulatesDevTable)
{
    constexpr uintptr_t kRmtRegAddr = 0xCAFE0000ull;

    // Fresh-registration seam hooks (legacy-IPC arm). Same happy-path set
    // the other fresh-reg tests use.
    InstallLegacyCudaRegisterHook();
    auto memGet = MakeDefaultMemGetHook();
    auto ipcGet = MakeDefaultIpcGetHook();
    ScopedHook connect(g_proxyConnect, MarkProxyConnInitialized());
    ScopedHook proxy(g_proxyCallBlocking, CannedRmtRegAddr(kRmtRegAddr));

    // g_strongStreamAcquire / g_fakeCudaCallocAsync / g_fakeCudaMemcpyAsync
    // stay at their honest-emulator defaults so the copy block runs against
    // real heap memory + real memcpy and we can read the dev table back.

    int peerRanks[] = {kPeerRank};

    // --- Call 1: fresh SENDRECV registration ------------------------------
    // This is where the fix does its work: needUpdate=true triggers the
    // host->dev copy even though type is SENDRECV, not COLLECTIVE.
    IpcRegOutputs outSend;
    bool isLegacyIpcSend = false;
    auto rSend = CallIpcRegisterBuffer(cb, kUserbuff,
                                       /*buffSize=*/ 256,
                                       peerRanks, 1,
                                       NCCL_IPC_SENDRECV,
                                       &regRecord, outSend, &isLegacyIpcSend);
    ASSERT_EQ(rSend, ncclSuccess);
    ASSERT_EQ(outSend.regBufFlag, 1);

    // The crux of the fix: the fresh SENDRECV registration must have
    // allocated AND populated the dev table. Under the bug this pointer is
    // still NULL here (copy block was gated on type == COLLECTIVE).
    ASSERT_NE(regRecord.regIpcAddrs.devPeerRmtAddrs, nullptr)
        << "devPeerRmtAddrs was not allocated during the fresh SENDRECV "
           "registration -- this is the nccl#1859 bug: the host->dev copy "
           "block was gated on `type == NCCL_IPC_COLLECTIVE`.";
    EXPECT_EQ(regRecord.regIpcAddrs.devPeerRmtAddrs[kPeerLocalRank],
              kRmtRegAddr)
        << "dev table slot was not copied from the host table during the "
           "fresh SENDRECV registration.";
    // SENDRECV returns the raw host remote address, not the dev table.
    EXPECT_EQ(reinterpret_cast<uintptr_t>(outSend.peerRmtAddrs), kRmtRegAddr);

    // --- Call 2: COLLECTIVE reuse of the SAME buffer ----------------------
    // Pure reuse (ipcInfos already populated), so needUpdate stays false
    // and the copy block is skipped. COLLECTIVE returns the dev table that
    // call 1 populated. Under the bug this would be a NULL table -> crash.
    IpcRegOutputs outColl;
    bool isLegacyIpcColl = true;
    auto rColl = CallIpcRegisterBuffer(cb, kUserbuff,
                                       /*buffSize=*/ 256,
                                       peerRanks, 1,
                                       NCCL_IPC_COLLECTIVE,
                                       &regRecord, outColl, &isLegacyIpcColl);
    ASSERT_EQ(rColl, ncclSuccess);
    EXPECT_EQ(outColl.regBufFlag, 1);
    EXPECT_EQ(outColl.offsetOut,  kBuffOffset);

    // The contract under test: COLLECTIVE hands back the populated dev
    // table, not a null pointer.
    ASSERT_NE(outColl.peerRmtAddrs, nullptr)
        << "COLLECTIVE reuse returned a NULL peer-address table -- the "
           "nccl#1859 crash. The dev table should have been populated by "
           "the prior SENDRECV registration.";
    EXPECT_EQ(outColl.peerRmtAddrs, regRecord.regIpcAddrs.devPeerRmtAddrs);
    EXPECT_EQ(outColl.peerRmtAddrs[kPeerLocalRank], kRmtRegAddr);
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
    IpcInfosBacking ipcInfosBacking{regRecord};
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
TEST_F(FreshRegistrationMicrotest, ProxyConnectFailurePropagates)
{
    auto memGet = MakeDefaultMemGetHook();
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
TEST_F(FreshRegistrationMicrotest, LegacyIpcGetMemHandleFailurePropagates)
{
    // Pre-mark gproxyConn initialized so the (covered-elsewhere)
    // proxyConnect call is skipped -- shaves a hook off this test.
    cb.comm().gproxyConn[kPeerRank].initialized = true;
    // directMode=false so the `comm->directMode || !ncclParam...` guard
    // inside the legacy-export arm doesn't short-circuit to fail before
    // the hipIpcGetMemHandle call.
    cb.comm().directMode = 0;

    InstallLegacyCudaRegisterHook();
    auto memGet = MakeDefaultMemGetHook();
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
    IpcInfosBacking ipcInfosBacking{regRecord};
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
    IpcInfosBacking ipcInfosBacking{regRecord};
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
TEST_F(FreshRegistrationMicrotest, DirectModeShortCircuitsLegacyExport)
{
    cb.comm().gproxyConn[kPeerRank].initialized = true;  // skip proxyConnect
    cb.comm().directMode = 1;                            // the seam under test

    // Force legacyIpcCap=1 so control reaches the `else if (legacyIpcCap)`
    // arm. Without this it would take the `// nothing works` goto fail
    // one branch earlier, exercising a different failure surface.
    InstallLegacyCudaRegisterHook();
    auto memGet = MakeDefaultMemGetHook();
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
    ncclIpcRegInfo* info0 = MakeHeapIpcInfo(kPeer0Rank, kRmt0,
                                            /*legacyIpcCap=*/true);

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
    IpcInfosBacking ipcInfosBacking{regRecord};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    regRecord.ipcInfos[kPeer0LocalRank]    = info0;
    regRecord.regIpcAddrs.hostPeerRmtAddrs = hostTable;
    regRecord.regIpcAddrs.devPeerRmtAddrs  = devTable.data();
    RegRecordCleaner regCleanup(regRecord);

    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());
    ScopedHook pointerAttr(g_hipPointerGetAttribute, ForceLegacyIpcCapable());

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
    ScopedHook proxy(g_proxyCallBlocking, CannedRmtRegAddr(kRmt1Fresh));

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
    EXPECT_EQ(regRecord.ipcInfos[kPeer0LocalRank], info0);
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
}

// Reuse-COLLECTIVE with a missing dev table but no fresh registrations:
// pins down the post-NCCL#1861 behaviour that the strong-stream block is
// gated solely on `if (needUpdate)`. All peers are reuse, so needUpdate
// stays false and the entire block is skipped -- no calloc, no memcpy, and
// devPeerRmtAddrs is left null.
//
// Before the fix the outer guard was `devPeerRmtAddrs == NULL || needUpdate`,
// which on this state allocated a fresh dev table with calloc but skipped
// the memcpy (needUpdate false), leaving a zero-filled table -- the exact
// nccl#1859 bug the fix removes. This test now asserts neither seam fires.
TEST_F(P2pMicrotest, IpcRegisterBuffer_CollectiveReuseSkipsDevTableAllocWhenNoUpdate)
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
    IpcInfosBacking ipcInfosBacking{regRecord};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x2000;
    existing.InstallInto(regRecord);
    // devPeerRmtAddrs intentionally left null. Under the new needUpdate-only
    // guard the strong-stream block is skipped, so it stays null.

    // Neither seam may fire on a pure-reuse path.
    ScopedHook calloc(g_fakeCudaCallocAsync,
        [](void**, std::size_t, hipStream_t) -> ncclResult_t {
            ADD_FAILURE() << "calloc must not fire on pure-reuse path "
                          << "(needUpdate=false)";
            return ncclSystemError;
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
    EXPECT_EQ(calloc.calls,  0);
    EXPECT_EQ(memcpy_.calls, 0);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  kBuffOffset);
    // Block skipped -> dev table never allocated -> COLLECTIVE returns the
    // (still-null) dev table.
    EXPECT_EQ(regRecord.regIpcAddrs.devPeerRmtAddrs, nullptr);
    EXPECT_EQ(out.peerRmtAddrs, nullptr);
    EXPECT_FALSE(isLegacyIpc);
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
    IpcInfosBacking ipcInfosBacking{regRecord};
    regRecord.begAddr = kBegAddr;
    regRecord.endAddr = kBegAddr + 0x1000;
    // Both ipcInfos[] slots NULL -> both peers take the fresh-reg arm.
    RegRecordCleaner regCleanup(regRecord);

    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());
    ScopedHook pointerAttr(g_hipPointerGetAttribute, ForceLegacyIpcCapable());

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

// RetainSentinelHandle -- the hipMemRetainAllocationHandle hook that hands
// back the sentinel handle and succeeds. The common cuMem-arm entry seam;
// tests that want to pin the retain address keep their own inline hook (e.g.
// CuMemSameProcessSucceeds, which also asserts EXPECT_EQ(addr, kBaseAddr)).
inline auto RetainSentinelHandle()
{
    return [](hipMemGenericAllocationHandle_t* h, void*) -> hipError_t {
        if (h) *h = MakeSentinelHandle();
        return hipSuccess;
    };
}

// FailRetainAllocationHandle -- the hipMemRetainAllocationHandle hook that
// fails the cuMem-arm entry seam, driving the retry-as-legacy fallback.
inline auto FailRetainAllocationHandle()
{
    return [](hipMemGenericAllocationHandle_t*, void*) -> hipError_t {
        return hipErrorInvalidValue;
    };
}

// ExpectReleaseSentinelHandle -- the hipMemRelease hook that asserts it is
// handed the sentinel handle (the handle-leak guard contract) and succeeds.
inline auto ExpectReleaseSentinelHandle()
{
    return [](hipMemGenericAllocationHandle_t h) -> hipError_t {
        EXPECT_TRUE(HandleHasSentinel(h));
        return hipSuccess;
    };
}

}  // namespace

// cuMem* arm, sameProcess=true: Retain -> memcpy handle into ipcInfo ->
// proxy register -> Release. Lights up the same-process sub-arm without
// touching hipMemExportToShareableHandle or the POSIX_FD/fabric branches.
TEST_F(FreshRegistrationMicrotest, CuMemSameProcessSucceeds)
{
    constexpr uintptr_t kRmtRegAddr = 0xCAFE0000ull;

    // Skip the (covered-elsewhere) proxyConnect call; pre-mark the slot.
    cb.comm().gproxyConn[kPeerRank].initialized = true;
    // Drive the `if (proxyConn->sameProcess)` True arm.
    cb.comm().gproxyConn[kPeerRank].sameProcess = 1;

    // Enter the cuMem* arm.
    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    auto memGet = MakeDefaultMemGetHook();

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
    ScopedHook release(g_hipMemRelease, ExpectReleaseSentinelHandle());

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
TEST_F(FreshRegistrationMicrotest, CuMemPosixFdSucceeds)
{
    constexpr uintptr_t kRmtRegAddr    = 0xCAFE0000ull;
    constexpr int       kRmtImportedFd = 0x7e1e7;  // canned remote-fd value

    // ncclCuMemHandleType is a fakes-owned global; existing tests don't
    // touch it (its default is hipMemHandleTypePosixFileDescriptor, which
    // is what we need here). Pin it down explicitly so a future fakes
    // change doesn't silently turn this into a fabric-arm test.
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().gproxyConn[kPeerRank].sameProcess = 0;
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
    ScopedHook retain(g_hipMemRetainAllocationHandle, RetainSentinelHandle());

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

    ScopedHook release(g_hipMemRelease, ExpectReleaseSentinelHandle());

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

}

// cuMem* arm, sameProcess=false, fabric handle type, export failure:
// drives the `if (CUPFN(hipMemExportToShareableHandle(...)) != hipSuccess)`
// True arm in the fabric sub-branch. The contract: hipMemRelease must
// fire on the handle before `goto fail`, otherwise the handle leaks.
TEST_F(FreshRegistrationMicrotest, CuMemFabricExportFailureReleasesHandle)
{
    // Switch fakes-owned global to the non-POSIX_FD branch.
    ScopedCuMemHandleType handleType(hipMemHandleTypeWin32);  // anything != POSIX_FD

    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().gproxyConn[kPeerRank].sameProcess = 0;

    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t) -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook retain(g_hipMemRetainAllocationHandle, RetainSentinelHandle());
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
    ScopedHook release(g_hipMemRelease, ExpectReleaseSentinelHandle());
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
TEST_F(FreshRegistrationMicrotest, CuMemRetainFailureFallsBackToLegacyExport)
{
    constexpr uintptr_t kRmtRegAddr = 0xCAFE0000ull;

    cb.comm().gproxyConn[kPeerRank].initialized = true;
    // directMode=false so the `directMode || !legacyParam` short-circuit's
    // first operand is False; combined with ForceLegacyCudaRegister
    // (second operand also False), the whole guard takes its False arm
    // and control proceeds to the legacy hipIpcGetMemHandle fallback.
    cb.comm().directMode = 0;

    // Enter the cuMem arm.
    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    // Required for the inner fallback guard not to short-circuit.
    ScopedHook loadParam(g_loadParam, ForceLegacyCudaRegister());
    ScopedHook pointerAttr(g_hipPointerGetAttribute, ForceLegacyIpcCapable());

    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t) -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });

    // Retain *fails* -- the key seam for this test.
    ScopedHook retain(g_hipMemRetainAllocationHandle, FailRetainAllocationHandle());
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
TEST_F(FreshRegistrationMicrotest, CuMemRetainFailureDirectModeShortCircuits)
{
    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().directMode = 1;  // drives the guard's True arm

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
    ScopedHook retain(g_hipMemRetainAllocationHandle, FailRetainAllocationHandle());
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

// ===========================================================================
// Remaining-coverage tests: each one drives a single previously-unhit
// branch identified via llvm-cov (see the Coverage section in the README, using
// --name=ipcRegisterBuffer --show-branches=count). The branches
// 961:35, 975:13, 981:11 and 1018:9 in ipcRegisterBuffer are intentionally
// not covered: 961:35 is dead under our build (its precondition requires
// `legacyIpcCap` to be true while ForceLegacyCudaRegister is *off*, which
// is contradictory for HIP_VERSION < 71260540); 975:13 is a defensive
// nullptr check on a pointer that is unconditionally written one line
// earlier; 981:11 and 1018:9 are the False (assert-fail) sides of
// `assert(...)` macros, not reachable by a passing test.
// ===========================================================================

// Branch 959:20 False -- `else if (legacyIpcCap)` False arm: enters the
// trailing `// nothing works, just return` bare `goto fail`. Precondition
// (HIP_VERSION < 71260540 path): `ncclCuMemEnable() == 0` AND
// `ncclParamLegacyCudaRegister() == 0`, so neither the cuMem nor the
// legacy export arm is eligible. Both defaults (cuMemEnable returns 0,
// loadParam returns deftVal=0) hand us this state without any hooks --
// the test just refrains from installing ForceLegacyCudaRegister.
TEST_F(FreshRegistrationMicrotest, NothingWorksFallthrough)
{
    cb.comm().gproxyConn[kPeerRank].initialized = true;

    // No loadParam hook -> ncclParamLegacyCudaRegister() returns 0
    // (default). No cuMemEnable hook -> ncclCuMemEnable() returns 0
    // (default). legacyIpcCap therefore stays 0 across the prologue,
    // and the trailing `else { goto fail; }` fires.
    auto memGet = MakeDefaultMemGetHook();
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t*, void*) -> hipError_t {
            ADD_FAILURE() << "nothing-works fall-through must not reach hipIpcGetMemHandle";
            return hipErrorInvalidValue;
        });
    ScopedHook proxy(g_proxyCallBlocking,
        [](struct ncclComm*, struct ncclProxyConnector*, int,
           void*, int, void*, int) -> ncclResult_t {
            ADD_FAILURE() << "nothing-works fall-through must not reach proxy register";
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

    EXPECT_EQ(memGet.calls, 1);
    EXPECT_EQ(ipcGet.calls, 0);
    EXPECT_EQ(proxy.calls,  0);

    // Same quirk as the directMode short-circuits: bare `goto fail` so
    // ret stays ncclSuccess, fail: epilogue zeros outputs.
    EXPECT_EQ(r, ncclSuccess);
    out.ExpectZeroed();
    EXPECT_FALSE(isLegacyIpc);
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_FALSE(regRecord.state & IPC_REG_COMPLETE);
}

// Branch 964:15 False -- `if (isLegacyIpc) *isLegacyIpc = true` in the
// legacy-export arm with isLegacyIpc=nullptr. Mirror of
// FreshRegistrationLegacyIpcSucceeds but passes nullptr; pins down that
// the gated write doesn't deref a null pointer.
TEST_F(FreshRegistrationMicrotest, LegacyIpcNullIsLegacyIpcPointerIsSkipped)
{
    constexpr uintptr_t kRmtRegAddr = 0xCAFE0000ull;

    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().directMode = 0;

    InstallLegacyCudaRegisterHook();
    auto memGet = MakeDefaultMemGetHook();
    auto ipcGet = MakeDefaultIpcGetHook();
    ScopedHook proxy(g_proxyCallBlocking, CannedRmtRegAddr(kRmtRegAddr));

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;

    auto r = CallIpcRegisterBuffer(cb, kUserbuff,
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, /*isLegacyIpc=*/ nullptr);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    ASSERT_NE(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_TRUE(regRecord.ipcInfos[kPeerLocalRank]->impInfo.legacyIpcCap);
}

#if ROCM_VERSION >= 70000

// Branches 938:17 False and 935:17 False -- the two `if (isLegacyIpc)`
// writes inside the cuMem arm (success sub-branch line 938 sets it to
// false; retry-as-legacy sub-branch line 935 sets it to true) with
// isLegacyIpc=nullptr. One test covers both sub-arms via the
// sameProcess success path and the Retain-failure fallback path.
TEST_F(FreshRegistrationMicrotest, CuMemNullIsLegacyIpcPointerSuccessSkipped)
{
    constexpr uintptr_t kRmtRegAddr = 0xCAFE0000ull;

    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().gproxyConn[kPeerRank].sameProcess = 1;

    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    auto memGet = MakeDefaultMemGetHook();
    ScopedHook retain(g_hipMemRetainAllocationHandle, RetainSentinelHandle());
    ScopedHook release(g_hipMemRelease, ExpectReleaseSentinelHandle());
    ScopedHook proxy(g_proxyCallBlocking, CannedRmtRegAddr(kRmtRegAddr));

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;

    auto r = CallIpcRegisterBuffer(cb,
                                   reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, /*isLegacyIpc=*/ nullptr);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    ASSERT_NE(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_FALSE(regRecord.ipcInfos[kPeerLocalRank]->impInfo.legacyIpcCap);
}

// Branch 935:17 False -- isLegacyIpc=nullptr on the cuMem retry-as-legacy
// fallback arm. Slimmed copy of the existing
// CuMemFreshRegistrationRetainFailureFallsBackToLegacyExport test.
TEST_F(FreshRegistrationMicrotest, CuMemNullIsLegacyIpcPointerRetryArmSkipped)
{
    constexpr uintptr_t kRmtRegAddr = 0xCAFE0000ull;

    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().directMode = 0;

    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    InstallLegacyCudaRegisterHook();
    auto memGet = MakeDefaultMemGetHook();
    ScopedHook retain(g_hipMemRetainAllocationHandle, FailRetainAllocationHandle());
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t* h, void*) -> hipError_t {
            if (h) std::memset(h, 0x5A, sizeof(*h));
            return hipSuccess;
        });
    ScopedHook proxy(g_proxyCallBlocking, CannedRmtRegAddr(kRmtRegAddr));

    int peerRanks[] = {kPeerRank};
    IpcRegOutputs out;

    auto r = CallIpcRegisterBuffer(cb,
                                   reinterpret_cast<const void*>(kBegAddr + kBuffOffset),
                                   /*buffSize=*/ 256,
                                   peerRanks, 1,
                                   NCCL_IPC_COLLECTIVE,
                                   &regRecord, out, /*isLegacyIpc=*/ nullptr);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(ipcGet.calls, 1);  // confirm retry arm taken
    EXPECT_EQ(out.regBufFlag, 1);
    ASSERT_NE(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_TRUE(regRecord.ipcInfos[kPeerLocalRank]->impInfo.legacyIpcCap);
}

// Branch 950:21 False -- `hipMemExportToShareableHandle(...) != hipSuccess`
// False (export succeeds) on the fabric sub-arm. Sister to
// CuMemFreshRegistrationFabricExportFailureReleasesHandle (which drives
// the True arm). The contract: on success the function continues into
// hipMemRelease and the proxy register, returning the canned rmtRegAddr.
TEST_F(FreshRegistrationMicrotest, CuMemFabricExportSucceeds)
{
    constexpr uintptr_t kRmtRegAddr = 0xCAFE0000ull;

    ScopedCuMemHandleType handleType(hipMemHandleTypeWin32);  // anything != POSIX_FD

    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().gproxyConn[kPeerRank].sameProcess = 0;

    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    auto memGet = MakeDefaultMemGetHook();
    ScopedHook retain(g_hipMemRetainAllocationHandle, RetainSentinelHandle());
    // Fabric export *succeeds* -- the seam under test.
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [](void* shareableHandle, hipMemGenericAllocationHandle_t h,
           hipMemAllocationHandleType handleType,
           unsigned long long) -> hipError_t {
            EXPECT_TRUE(HandleHasSentinel(h));
            EXPECT_NE(handleType, hipMemHandleTypePosixFileDescriptor);
            // Production writes the exported handle into
            // ipcInfo.ipcDesc.cuDesc.handle; leave its contents
            // unspecified -- this test pins down control flow, not
            // the handle bytes. Touch the first byte so a tooling
            // check that ipcRegisterBuffer hands us a writable buffer
            // surfaces here.
            if (shareableHandle) {
                static_cast<char*>(shareableHandle)[0] = 0;
            }
            return hipSuccess;
        });
    ScopedHook release(g_hipMemRelease, ExpectReleaseSentinelHandle());
    ScopedHook proxy(g_proxyCallBlocking, CannedRmtRegAddr(kRmtRegAddr));

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
    EXPECT_EQ(release.calls, 1);
    EXPECT_EQ(proxy.calls,   1);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_FALSE(isLegacyIpc);  // cuMem-success arm cleared it
    ASSERT_NE(regRecord.ipcInfos[kPeerLocalRank], nullptr);
    EXPECT_FALSE(regRecord.ipcInfos[kPeerLocalRank]->impInfo.legacyIpcCap);

}

// Branch 932:37 True -- the second operand of
// `comm->directMode || !ncclParamLegacyCudaRegister()` evaluates True
// inside the cuMem retry-as-legacy path. Precondition: directMode=false
// (so the LHS doesn't short-circuit) AND ncclParamLegacyCudaRegister()
// returns 0 (so the RHS is True). Mirror of
// CuMemFreshRegistrationRetainFailureDirectModeShortCircuits but the
// short-circuit is driven by the param rather than directMode.
TEST_F(FreshRegistrationMicrotest, CuMemRetainFailureParamOffShortCircuits)
{
    cb.comm().gproxyConn[kPeerRank].initialized = true;
    cb.comm().directMode = 0;  // force the guard to evaluate the RHS

    ScopedHook cuMemEnable(g_cuMemEnable, [] { return 1; });
    // No loadParam hook -> ncclParamLegacyCudaRegister() returns 0
    // (default) -> RHS of `directMode || !ncclParamLegacyCudaRegister()`
    // is True -> guard short-circuits to goto fail.
    const void* const kUserbuff =
        reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    ScopedHook memGet(g_hipMemGetAddressRange,
        [&](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t) -> hipError_t {
            if (pbase) *pbase = reinterpret_cast<hipDeviceptr_t>(kBaseAddr);
            if (psize) *psize = kBaseSize;
            return hipSuccess;
        });
    ScopedHook retain(g_hipMemRetainAllocationHandle, FailRetainAllocationHandle());
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
    EXPECT_EQ(r, ncclSuccess);  // bare `goto fail`
    out.ExpectZeroed();
    EXPECT_FALSE(isLegacyIpc);
    EXPECT_EQ(regRecord.ipcInfos[kPeerLocalRank], nullptr);
}

#endif  // ROCM_VERSION >= 70000

// ===========================================================================
// useMemcpy priming / vtable-slot patching (initCeOperation).
//
// initCeOperation() is the lazy one-shot that reads NCCL_P2P_USE_CUDA_MEMCPY
// into the file-static `useMemcpy` and, when enabled, patches the CE
// proxyConnect / proxyProgress slots onto the global p2pTransport.send vtable.
// Only ncclP2pUsesMemcpy() and p2pCanConnect() reach it; every other
// setup/connect/free/proxy function merely *reads* the already-primed state.
//
// Two properties make this behaviour resist an ordinary in-process test:
//   1. initCeOperation() latches a `static int init` -- it runs its body
//      exactly once per process, so a second test can never observe the
//      "before priming" state or a different NCCL_P2P_USE_CUDA_MEMCPY value.
//   2. It mutates the *global* p2pTransport struct in place; that mutation
//      would leak into any later test in the same process.
// So each scenario runs in its own forked process (RUN_ISOLATED_TEST), where
// the latch and the vtable slots start fresh and the env var can be set
// independently. The param redirector routes ncclParamP2pUseCudaMemcpy()
// through g_loadParam, so the child sets the value via a g_loadParam hook
// rather than the real environment.
// ===========================================================================

class P2pMicrotestIsolated : public P2pMicrotest {};

TEST_F(P2pMicrotestIsolated, UsesMemcpy_ParamEnabled_ReportsTrue)
{
    RUN_ISOLATED_TEST("P2p_UsesMemcpy_ParamEnabled_ReportsTrue", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ASSERT_TRUE(ncclP2pUsesMemcpy());
    });
}

TEST_F(P2pMicrotestIsolated, UsesMemcpy_ParamDisabled_ReportsFalse)
{
    RUN_ISOLATED_TEST("P2p_UsesMemcpy_ParamDisabled_ReportsFalse", []() {
        // No hook: ncclParamP2pUseCudaMemcpy() sits at its default (0).
        ASSERT_FALSE(ncclP2pUsesMemcpy());
    });
}

TEST_F(P2pMicrotestIsolated, Prime_MemcpyEnabled_PatchesSendProxySlots)
{
    RUN_ISOLATED_TEST("P2p_Prime_MemcpyEnabled_PatchesSendProxySlots", []() {
        // Guard: before any priming call, the CE slots are unset on the
        // freshly-loaded p2pTransport (documents the ordering precondition).
        ASSERT_EQ(p2pTransport.send.proxyConnect, nullptr);
        ASSERT_EQ(p2pTransport.send.proxyProgress, nullptr);

        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());

        // Priming through the public surface patches the send-side CE slots.
        ncclP2pUsesMemcpy();

        EXPECT_NE(p2pTransport.send.proxyConnect, nullptr);
        EXPECT_NE(p2pTransport.send.proxyProgress, nullptr);
    });
}

TEST_F(P2pMicrotestIsolated, Prime_MemcpyDisabled_LeavesSendProxySlotsUnset)
{
    RUN_ISOLATED_TEST("P2p_Prime_MemcpyDisabled_LeavesSendProxySlotsUnset", []() {
        ASSERT_EQ(p2pTransport.send.proxyConnect, nullptr);
        ASSERT_EQ(p2pTransport.send.proxyProgress, nullptr);

        // No hook -> memcpy disabled -> priming must NOT patch the CE slots.
        ncclP2pUsesMemcpy();

        EXPECT_EQ(p2pTransport.send.proxyConnect, nullptr);
        EXPECT_EQ(p2pTransport.send.proxyProgress, nullptr);
    });
}

// ===========================================================================
// Transport eligibility (p2pCanConnect, reached through the
// p2pTransport.canConnect vtable slot).
//
// p2pCanConnect decides whether two peers may use the P2P transport, writing
// its verdict into *ret (1 = eligible, 0 = not). Its own contract is exactly
// that verdict plus the short-circuits it takes to reach it; the topology and
// device-access answers it consults are owned by the ncclTopoCheck* /
// cudaDeviceCanAccessPeer seams, so tests drive those seams and assert only on
// *ret and on which seams the function did or did not reach.
//
// Reachability notes for the fakes:
//   - ncclTopoCheckP2p / ncclTopoCheckNet are controllable seams
//     (g_ncclTopoCheckP2p / g_ncclTopoCheckNet, default "no p2p, no net").
//   - busIdToCudaDev() resolves a peer's busId to a local device index via
//     hipDeviceGetPCIBusId + busIdToInt64. The g_hipDeviceGetPCIBusId hook
//     below encodes the device index into the bus string so distinct peers
//     map to distinct cudaDev indices; BusIdForDev() mirrors that parse so a
//     test can set peerInfo.busId to the matching int64.
//   - useMemcpy sits at 0 for every non-isolated test in this process (the
//     first canConnect/usesMemcpy call latches it to the param default 0),
//     so the `if (useMemcpy)` intermediate-rank arm is exercised False here
//     and True in the isolated test at the end.
// ===========================================================================

// Encode a device index into a bus-id string and back, mirroring
// busIdToCudaDev's hipDeviceGetPCIBusId -> busIdToInt64 pipeline so a test can
// pin peerInfo.busId to the value that resolves to `dev`.
static void BusStrForDev(int dev, char* out, int len) {
    std::snprintf(out, len, "0000:00:%02x.0", dev & 0xff);
}
static int64_t BusIdForDev(int dev) {
    char s[32];
    BusStrForDev(dev, s, sizeof(s));
    int64_t id = 0;
    busIdToInt64(s, &id);
    return id;
}
// Hook that makes hipDeviceGetPCIBusId report a distinct bus string per
// device, so busIdToCudaDev can resolve BusIdForDev(d) back to device d.
static std::function<hipError_t(char*, int, int)> DeviceDistinctBusIds() {
    return [](char* buf, int len, int device) -> hipError_t {
        if (buf && len > 0) BusStrForDev(device, buf, len);
        return hipSuccess;
    };
}

// Fixture owning the comm + peerInfo backing the canConnect call. peerInfo[0]
// is "my" info (comm.rank == 0); the two candidates default to same-host,
// fine-grain-capable, and device-resolvable so each test only perturbs the
// one property it is about.
class P2pCanConnectMicrotest : public P2pMicrotest {
protected:
    void SetUp() override {
        comm_.rank = 0;
        comm_.peerInfo = myInfo_.data();
        // myInfo_[0].hostHash defaults to 0; candidates match it (same host).
        for (auto* p : {&info1_, &info2_}) {
            p->hasFineGrain = true;
            p->hostHash = 0;
        }
        info1_.rank = 0;
        info2_.rank = 1;
        info1_.busId = BusIdForDev(0);
        info2_.busId = BusIdForDev(1);
    }

    int Call() {
        int ret = -1;
        EXPECT_EQ(p2pTransport.canConnect(&ret, &comm_, nullptr, &info1_, &info2_),
                  ncclSuccess);
        return ret;
    }

    ncclComm comm_{};
    std::array<ncclPeerInfo, 2> myInfo_{};
    ncclPeerInfo info1_{};
    ncclPeerInfo info2_{};
};

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
TEST_F(P2pCanConnectMicrotest, FirstPeerLacksFineGrain_ReturnsCannotConnect)
{
    info1_.hasFineGrain = false;
    ScopedHook topo(g_ncclTopoCheckP2p,
                    [](int, int, int*, int*, int*, int*, int*) -> ncclResult_t {
                        ADD_FAILURE() << "topo check reached despite missing fine-grain";
                        return ncclSuccess;
                    });
    EXPECT_EQ(Call(), 0);
}

TEST_F(P2pCanConnectMicrotest, PeerLacksFineGrain_ReturnsCannotConnect)
{
    info2_.hasFineGrain = false;
    // No topo seam should even be consulted -- the fine-grain gate is first.
    ScopedHook topo(g_ncclTopoCheckP2p,
                    [](int, int, int*, int*, int*, int*, int*) -> ncclResult_t {
                        ADD_FAILURE() << "topo check reached despite missing fine-grain";
                        return ncclSuccess;
                    });
    EXPECT_EQ(Call(), 0);
}
#endif

TEST_F(P2pCanConnectMicrotest, TopoRejectsP2p_ReturnsCannotConnect)
{
    // Default g_ncclTopoCheckP2p reports p2p = 0 -> verdict is "cannot connect"
    // and the function returns before consulting the NET seam.
    ScopedHook net(g_ncclTopoCheckNet, [](int, int, int*) -> ncclResult_t {
        ADD_FAILURE() << "NET check reached after topo already rejected p2p";
        return ncclSuccess;
    });
    EXPECT_EQ(Call(), 0);
}

TEST_F(P2pCanConnectMicrotest, NetIsBetter_ReturnsCannotConnect)
{
    ScopedHook topo(g_ncclTopoCheckP2p, TopoDirectP2p());
    ScopedHook net(g_ncclTopoCheckNet, [](int, int, int* useNet) -> ncclResult_t {
        if (useNet) *useNet = 1;  // NET would work better
        return ncclSuccess;
    });
    EXPECT_EQ(Call(), 0);
}

TEST_F(P2pCanConnectMicrotest, CrossClique_SkipsNetCheckAndStaysEligible)
{
    // Cross-clique MNNVL: the peer is absent from this rank's topology, so the
    // NET comparison is deliberately skipped. Even a NET-is-better verdict must
    // not flip the result to cannot-connect (contrast NetIsBetter above, which
    // is the same topology minus the cross-clique flag). This pins the
    // `if (!isCrossClique)` gate that guards the whole ncclTopoCheckNet block.
    ScopedHook topo(g_ncclTopoCheckP2p,
                    [](int, int, int* p2p, int*, int* inter, int*, int* isCrossClique) -> ncclResult_t {
                        if (p2p)   *p2p   = 1;
                        if (inter) *inter = -1;
                        if (isCrossClique) *isCrossClique = 1;
                        return ncclSuccess;
                    });
    ScopedHook net(g_ncclTopoCheckNet, [](int, int, int*) -> ncclResult_t {
        ADD_FAILURE() << "NET check reached despite cross-clique short-circuit";
        return ncclSuccess;
    });
    EXPECT_EQ(Call(), 1);
}

TEST_F(P2pCanConnectMicrotest, IntermediateRankNoMemcpy_ReportsEligible)
{
    // topo reports p2p-capable via an intermediate hop; with useMemcpy == 0
    // (this process's latched value) the intermediate arm leaves the verdict
    // untouched and returns before the NET/device checks.
    ScopedHook topo(g_ncclTopoCheckP2p,
                    [](int, int, int* p2p, int*, int* inter, int*, int*) -> ncclResult_t {
                        if (p2p) *p2p = 1;
                        if (inter) *inter = 3;  // != -1
                        return ncclSuccess;
                    });
    ScopedHook net(g_ncclTopoCheckNet, [](int, int, int*) -> ncclResult_t {
        ADD_FAILURE() << "NET check reached despite intermediate-rank short-circuit";
        return ncclSuccess;
    });
    EXPECT_EQ(Call(), 1);
}

TEST_F(P2pCanConnectMicrotest, FirstPeerCrossHost_ShortCircuitsBeforeDeviceCheck)
{
    // "My" host hash differs from info1's -> the first half of the host-mismatch
    // test fires and the function returns before the device check.
    myInfo_[0].hostHash = 0x1234;
    info1_.hostHash = 0xABCD;
    info2_.hostHash = 0xABCD;
    ScopedHook topo(g_ncclTopoCheckP2p, TopoDirectP2p());
    ScopedHook canAccess(g_hipDeviceCanAccessPeer,
                         [](int*, int, int) -> hipError_t {
                             ADD_FAILURE() << "device access queried for a cross-host peer";
                             return hipErrorInvalidValue;
                         });
    EXPECT_EQ(Call(), 1);
}

TEST_F(P2pCanConnectMicrotest, CrossHostPeers_ShortCircuitsBeforeDeviceCheck)
{
    // Peer on a different host: p2pCanConnect returns as soon as it sees the
    // host mismatch, before any device-access query.
    info2_.hostHash = 0xBEEF;
    ScopedHook topo(g_ncclTopoCheckP2p, TopoDirectP2p());
    ScopedHook canAccess(g_hipDeviceCanAccessPeer,
                         [](int*, int, int) -> hipError_t {
                             ADD_FAILURE() << "device access queried for a cross-host peer";
                             return hipErrorInvalidValue;
                         });
    EXPECT_EQ(Call(), 1);
}

TEST_F(P2pCanConnectMicrotest, BusIdUnresolved_ReportsEligibleOnHip)
{
    // hipDeviceGetPCIBusId fails -> busIdToCudaDev yields -1. On HIP the
    // invisible-device arm returns success with the topo verdict intact.
    ScopedHook topo(g_ncclTopoCheckP2p, TopoDirectP2p());
    // Default g_hipDeviceGetPCIBusId returns error (g_hipDeviceGetPCIBusIdResult
    // is hipErrorInvalidValue), so both busIds resolve to -1.
    EXPECT_EQ(Call(), 1);
}

TEST_F(P2pCanConnectMicrotest, SecondBusIdUnresolved_ReportsEligibleOnHip)
{
    // info1 resolves to a device but info2's bus id does not -> the second
    // disjunct of the invisible-device gate fires. On HIP that reports
    // eligible.
    ScopedHook busid(g_hipDeviceGetPCIBusId,
                     [](char* buf, int len, int device) -> hipError_t {
                         if (device != 0) return hipErrorInvalidValue;  // only dev 0 resolvable
                         if (buf && len > 0) BusStrForDev(device, buf, len);
                         return hipSuccess;
                     });
    info1_.busId = BusIdForDev(0);   // resolves to dev 0
    info2_.busId = BusIdForDev(5);   // no device reports this bus -> -1
    ScopedHook topo(g_ncclTopoCheckP2p, TopoDirectP2p());
    EXPECT_EQ(Call(), 1);
}

TEST_F(P2pCanConnectMicrotest, SameDevicePeers_ReportsEligible)
{
    // Both peers resolve to the same local device -> p2p is assumed available
    // (multi-rank GPU) without querying cudaDeviceCanAccessPeer.
    info2_.busId = info1_.busId;  // same device index 0
    ScopedHook busid(g_hipDeviceGetPCIBusId, DeviceDistinctBusIds());
    ScopedHook topo(g_ncclTopoCheckP2p, TopoDirectP2p());
    ScopedHook canAccess(g_hipDeviceCanAccessPeer,
                         [](int*, int, int) -> hipError_t {
                             ADD_FAILURE() << "same-device path must not query peer access";
                             return hipErrorInvalidValue;
                         });
    EXPECT_EQ(Call(), 1);
}

TEST_F(P2pCanConnectMicrotest, DistinctDevicesCanAccessPeer_ReportsEligible)
{
    ScopedHook busid(g_hipDeviceGetPCIBusId, DeviceDistinctBusIds());
    ScopedHook topo(g_ncclTopoCheckP2p, TopoDirectP2p());
    ScopedHook canAccess(g_hipDeviceCanAccessPeer,
                         [](int* ok, int, int) -> hipError_t {
                             if (ok) *ok = 1;
                             return hipSuccess;
                         });
    EXPECT_EQ(Call(), 1);
}

TEST_F(P2pCanConnectMicrotest, DistinctDevicesCannotAccessPeer_ReturnsCannotConnect)
{
    ScopedHook busid(g_hipDeviceGetPCIBusId, DeviceDistinctBusIds());
    ScopedHook topo(g_ncclTopoCheckP2p, TopoDirectP2p());
    // Query succeeds but reports peer access is unavailable -> verdict 0.
    ScopedHook canAccess(g_hipDeviceCanAccessPeer,
                         [](int* ok, int, int) -> hipError_t {
                             if (ok) *ok = 0;
                             return hipSuccess;
                         });
    EXPECT_EQ(Call(), 0);
}

TEST_F(P2pCanConnectMicrotest, PeerAccessQueryFails_ReturnsCannotConnect)
{
    ScopedHook busid(g_hipDeviceGetPCIBusId, DeviceDistinctBusIds());
    ScopedHook topo(g_ncclTopoCheckP2p, TopoDirectP2p());
    // Query itself fails -> verdict 0.
    ScopedHook canAccess(g_hipDeviceCanAccessPeer,
                         [](int*, int, int) -> hipError_t { return hipErrorInvalidValue; });
    EXPECT_EQ(Call(), 0);
}

// The intermediate-rank arm's `if (useMemcpy)` True direction and the priming
// side effect both require the memcpy-enabled latch, so they run
// process-isolated (the latch is one-shot per process).
class P2pCanConnectMicrotestIsolated : public P2pMicrotest {};

TEST_F(P2pCanConnectMicrotestIsolated, IntermediateRankWithMemcpy_ReturnsCannotConnect)
{
    RUN_ISOLATED_TEST("P2p_CanConnect_IntermediateRankWithMemcpy_ReturnsCannotConnect", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ScopedHook topo(g_ncclTopoCheckP2p,
                        [](int, int, int* p2p, int*, int* inter, int*, int*) -> ncclResult_t {
                            if (p2p) *p2p = 1;
                            if (inter) *inter = 3;  // != -1
                            return ncclSuccess;
                        });
        auto commStorage = std::make_unique<ncclComm>();  // ~3.8 MB: heap, not stack-safe
        ncclComm& comm = *commStorage;
        std::array<ncclPeerInfo, 2> myInfo{};
        comm.rank = 0;
        comm.peerInfo = myInfo.data();
        ncclPeerInfo i1{}, i2{};
        i1.hasFineGrain = i2.hasFineGrain = true;
        int ret = -1;
        ASSERT_EQ(p2pTransport.canConnect(&ret, &comm, nullptr, &i1, &i2), ncclSuccess);
        // With memcpy enabled the intermediate-rank hop is disallowed.
        EXPECT_EQ(ret, 0);
        // And the call primed useMemcpy, observable through the public getter.
        EXPECT_TRUE(ncclP2pUsesMemcpy());
    });
}

// ===========================================================================
// Connection setup -> connect -> free round-trip (driven through the
// p2pTransport send/recv vtable slots).
//
// p2pSendSetup / p2pRecvSetup allocate a p2pResources, decide the resource
// type from the peer relationship, and fill the caller's ncclConnect with a
// p2pConnectInfo (read flag + rank). p2pSendConnect / p2pRecvConnect map the
// exchanged buffer and wire up the ncclConnInfo. p2pSendFree / p2pRecvFree
// release the resources and are null-safe.
//
// Observable-through-the-public-surface state these tests pin:
//   - send->transportResources becomes non-NULL and resources->type reflects
//     the same-PID (DIRECT) vs cross-PID + cuMem (CUMEM) vs cross-PID legacy
//     (IPC) decision.
//   - the ncclConnect is a well-formed p2pConnectInfo: info->read mirrors the
//     topology read verdict, info->rank is this rank, and the connector's
//     NCCL_P2P_READ / NCCL_P2P_WRITE flag matches.
//   - free is a no-op-success when resources are absent and closes each
//     retained legacy-IPC handle when they are present (mock, since the freed
//     memory is not observable).
//
// Ordering precondition: setup/connect/free only *read* the file-static
// useMemcpy; they never prime it. The fixture calls ncclP2pUsesMemcpy() in
// SetUp so useMemcpy is latched to its default (0, the non-memcpy arm) rather
// than left in an undefined pre-init state (the priming latch).
// ===========================================================================

namespace {

// View the p2pConnectInfo a setup call wrote into an ncclConnect. Mirrors the
// production reinterpret_cast; only the leading rank/read ints are read here.
struct ConnectInfoView {
    int rank;
    int read;
    // (p2pBuff + CE desc follow in the real struct; not needed for asserts.)
};
const ConnectInfoView* AsConnectInfo(const ncclConnect& c) {
    return reinterpret_cast<const ConnectInfoView*>(c.data);
}

}  // namespace

// res->type is compared against the real p2pType enumerators (P2P_DIRECT etc.),
// which p2p.cc declares at file scope and are visible through the #include
// above -- so a production reorder of the enum is caught rather than silently
// matching a hardcoded ordinal.

// Fixture: a comm whose peerInfo[0] is "me", plus a myInfo/peerInfo pair the
// setup call takes by pointer. A real backing buffer stands in for the
// exchanged device memory so p2pMap's pointer arithmetic lands on valid
// storage. The proxy seams are wired to succeed and to hand back that buffer
// as the setup response.
class P2pSetupMicrotest : public P2pMicrotest {
protected:
    void SetUp() override {
        // Latch useMemcpy to its default (0) through the public getter so the
        // non-memcpy arm is exercised deterministically (ordering precond).
        ncclP2pUsesMemcpy();

        myInfo_.rank    = 0;
        myInfo_.cudaDev = 0;
        myInfo_.hostHash = kHost;
        myInfo_.pidHash  = kPid;

        // comm.peerInfo[info->rank] is consulted by p2pMap; info->rank == 0,
        // so slot 0 must be "me" (same host+pid+dev) to take the same-process
        // direct-map path (no HIP IPC calls).
        peers_[0] = myInfo_;
        comm_.peerInfo = peers_.data();
        comm_.nRanks   = static_cast<int>(peers_.size());
        comm_.rank     = 0;

        // The candidate peer defaults to same-process (=> DIRECT); individual
        // tests cross its pidHash to select IPC/CUMEM.
        peer_.rank    = 1;
        peer_.cudaDev = 0;
        peer_.hostHash = kHost;
        peer_.pidHash  = kPid;

        backing_.assign(sizeof(ncclRecvMem) + 4096, 0);
    }

    // Wire the two proxy seams: connect marks the slot initialised; the
    // blocking call records the message type and publishes the backing buffer
    // as the p2pBuff response so p2pMap has something to map.
    void InstallHappyProxy() {
        connect_.emplace(g_proxyConnect, MarkProxyConnInitialized());
        proxy_.emplace(g_proxyCallBlocking,
            [this](struct ncclComm*, struct ncclProxyConnector*, int type,
                   void*, int, void* resp, int respSize) -> ncclResult_t {
                lastProxyMsg_ = type;
                if (resp && static_cast<std::size_t>(respSize) >= sizeof(ncclP2pBuff)) {
                    auto* b = static_cast<ncclP2pBuff*>(resp);
                    b->directPtr = backing_.data();
                    b->size      = backing_.size();
                }
                return ncclSuccess;
            });
    }

    // Report the link as XGMI so p2pSendSetup skips the non-XGMI HDP-register
    // block (which would otherwise dereference comm->topo and call
    // hipDeviceGetAttribute); this fixture leaves comm.topo NULL.
    void InstallXgmiLink() {
        link_.emplace(g_ncclTopoGetLinkType,
            [](int, int, bool* isXGMI, int) -> ncclResult_t {
                if (isXGMI) *isXGMI = true;
                return ncclSuccess;
            });
    }

    // Topology hook: p2p-capable, no intermediate hop, read flag = `read`.
    void InstallTopo(int read) {
        topo_.emplace(g_ncclTopoCheckP2p,
            [read](int, int, int* p2p, int* rd, int* inter, int*, int*) -> ncclResult_t {
                if (p2p)   *p2p   = 1;
                if (rd)    *rd    = read;
                if (inter) *inter = -1;
                return ncclSuccess;
            });
    }

    // Topology hook that reports an intermediate hop (indirect P2P). The setup
    // path then routes the connect through `inter` rather than the peer's own
    // rank, selecting the P2P_INTERMEDIATE resource type.
    void InstallTopoIntermediate(int inter) {
        topo_.emplace(g_ncclTopoCheckP2p,
            [inter](int, int, int* p2p, int* rd, int* i, int*, int*) -> ncclResult_t {
                if (p2p) *p2p = 1;
                if (rd)  *rd  = 0;
                if (i)   *i   = inter;
                return ncclSuccess;
            });
    }

    static constexpr uint64_t kHost = 0x1111;
    static constexpr uint64_t kPid  = 0x2222;

    ncclComm comm_{};
    ncclPeerInfo myInfo_{};
    ncclPeerInfo peer_{};
    std::array<ncclPeerInfo, 2> peers_{};
    std::vector<char> backing_;
    ncclConnect connect_info_{};
    ncclConnector send_{};
    int lastProxyMsg_ = -1;

    std::optional<ScopedHook<ncclResult_t(struct ncclComm*, int, int, int, struct ncclProxyConnector*)>> connect_;
    std::optional<ScopedHook<ncclResult_t(struct ncclComm*, struct ncclProxyConnector*, int, void*, int, void*, int)>> proxy_;
    std::optional<ScopedHook<ncclResult_t(int, int, bool*, int)>> link_;
    std::optional<ScopedHook<ncclResult_t(int, int, int*, int*, int*, int*, int*)>> topo_;
};

TEST_F(P2pSetupMicrotest, SendSetup_SameProcessPeer_SelectsDirectAndFillsConnectInfo)
{
    InstallTopo(/*read=*/0);
    InstallXgmiLink();
    InstallHappyProxy();

    ASSERT_EQ(p2pTransport.send.setup(&comm_, /*graph=*/nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, /*channelId=*/0, /*connIndex=*/0),
              ncclSuccess);

    // Resources were allocated and typed as the same-process direct path.
    auto* res = static_cast<p2pResources*>(send_.transportResources);
    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->type, P2P_DIRECT);

    // The connect info is well-formed: this rank, write (read=0) flag.
    const auto* info = AsConnectInfo(connect_info_);
    EXPECT_EQ(info->rank, myInfo_.rank);
    EXPECT_EQ(info->read, 0);
    EXPECT_TRUE(send_.conn.flags & NCCL_P2P_WRITE);
    EXPECT_FALSE(send_.conn.flags & NCCL_P2P_READ);

    // Setup drove the proxy with a Setup message.
    EXPECT_EQ(lastProxyMsg_, ncclProxyMsgSetup);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, SendSetup_ReadEnabled_SetsReadFlag)
{
    InstallTopo(/*read=*/1);
    InstallXgmiLink();
    InstallHappyProxy();

    ASSERT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSuccess);

    const auto* info = AsConnectInfo(connect_info_);
    EXPECT_EQ(info->read, 1);
    EXPECT_TRUE(send_.conn.flags & NCCL_P2P_READ);
    EXPECT_FALSE(send_.conn.flags & NCCL_P2P_WRITE);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, SendSetup_CrossProcessLegacyPeer_SelectsIpc)
{
    peer_.pidHash = kPid + 1;  // different process => not DIRECT
    // cuMemEnable stays 0 (default) => legacy IPC rather than CUMEM.
    InstallTopo(/*read=*/0);
    InstallXgmiLink();
    InstallHappyProxy();

    ASSERT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSuccess);

    auto* res = static_cast<p2pResources*>(send_.transportResources);
    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->type, P2P_IPC);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, SendSetup_CrossProcessCuMemPeer_SelectsCumem)
{
    peer_.pidHash = kPid + 1;  // different process => not DIRECT
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    InstallTopo(/*read=*/0);
    InstallXgmiLink();
    InstallHappyProxy();

    ASSERT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSuccess);

    auto* res = static_cast<p2pResources*>(send_.transportResources);
    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->type, P2P_CUMEM);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, RecvSetup_SameProcessPeer_SelectsDirectAndFillsConnectInfo)
{
    InstallTopo(/*read=*/0);
    InstallHappyProxy();
    // p2pRecvSetup has no non-XGMI HDP block, so no link hook is needed.

    ncclConnector recv{};
    ASSERT_EQ(p2pTransport.recv.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, 0),
              ncclSuccess);

    auto* res = static_cast<p2pResources*>(recv.transportResources);
    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->type, P2P_DIRECT);

    const auto* info = AsConnectInfo(connect_info_);
    EXPECT_EQ(info->rank, myInfo_.rank);
    EXPECT_EQ(info->read, 0);
    EXPECT_TRUE(recv.conn.flags & NCCL_P2P_WRITE);

    p2pTransport.recv.free(&comm_, &recv);
}

TEST_F(P2pSetupMicrotest, SendSetup_ReadEnableParamOverridesTopology_ForcesReadFlag)
{
    // The topology says "no read", but NCCL_P2P_READ_ENABLE is set to a
    // non-default (!= -2) value, so p2pGetInfo overrides the topology verdict
    // and the connect info comes out read-enabled.
    InstallTopo(/*read=*/0);
    InstallXgmiLink();
    InstallHappyProxy();
    ScopedHook readEnable(g_loadParam,
        [](const char* env, int64_t def) -> int64_t {
            if (env && std::strcmp(env, "P2P_READ_ENABLE") == 0) return 1;
            return def;
        });

    ASSERT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSuccess);

    const auto* info = AsConnectInfo(connect_info_);
    EXPECT_EQ(info->read, 1);
    EXPECT_TRUE(send_.conn.flags & NCCL_P2P_READ);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, SendSetup_DirectDisableParam_SelectsIpcForSameProcessPeer)
{
    // Same-process peer would normally select the DIRECT path, but
    // NCCL_P2P_DIRECT_DISABLE forces it off; with cuMem disabled the fallback
    // is legacy IPC.
    InstallTopo(/*read=*/0);
    InstallXgmiLink();
    InstallHappyProxy();
    ScopedHook directDisable(g_loadParam,
        [](const char* env, int64_t def) -> int64_t {
            if (env && std::strcmp(env, "P2P_DIRECT_DISABLE") == 0) return 1;
            return def;
        });

    ASSERT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSuccess);

    auto* res = static_cast<p2pResources*>(send_.transportResources);
    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->type, P2P_IPC);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, SendSetup_IntermediateHop_SelectsIntermediateAndRoutesConnectRank)
{
    // The topology reports an intermediate rank, so setup selects the indirect
    // P2P_INTERMEDIATE type and writes the intermediate rank (0 == me here)
    // into the connect info rather than my own rank.
    InstallTopoIntermediate(/*inter=*/0);
    InstallXgmiLink();
    InstallHappyProxy();

    ASSERT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSuccess);

    auto* res = static_cast<p2pResources*>(send_.transportResources);
    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->type, P2P_INTERMEDIATE);
    EXPECT_EQ(AsConnectInfo(connect_info_)->rank, 0);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, SendSetup_CollNetScatterConn_ForcesWriteDespiteReadTopology)
{
    // For CollNet (non-null graph) the scatter-reduce connection (connIndex 1)
    // must use write even when the topology enabled read: p2pSendSetup clears
    // info->read back to 0.
    InstallTopo(/*read=*/1);
    InstallXgmiLink();
    InstallHappyProxy();
    ncclTopoGraph graph{};  // non-null; only its address is tested, not read.

    ASSERT_EQ(p2pTransport.send.setup(&comm_, &graph, &myInfo_, &peer_,
                                      &connect_info_, &send_, /*channelId=*/0,
                                      /*connIndex=*/1),
              ncclSuccess);

    EXPECT_EQ(AsConnectInfo(connect_info_)->read, 0);
    EXPECT_TRUE(send_.conn.flags & NCCL_P2P_WRITE);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, SendSetup_LinkTypeQueryFails_ReturnsInternalError)
{
    // A failing ncclTopoGetLinkType aborts setup with an internal error before
    // any resource type is decided.
    InstallTopo(/*read=*/0);
    InstallHappyProxy();
    link_.emplace(g_ncclTopoGetLinkType,
        [](int, int, bool*, int) -> ncclResult_t { return ncclInternalError; });

    EXPECT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclInternalError);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, RecvSetup_CollNetScatterConn_ForcesWriteDespiteReadTopology)
{
    InstallTopo(/*read=*/1);
    InstallHappyProxy();
    ncclTopoGraph graph{};

    ncclConnector recv{};
    ASSERT_EQ(p2pTransport.recv.setup(&comm_, &graph, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, /*connIndex=*/1),
              ncclSuccess);

    EXPECT_EQ(AsConnectInfo(connect_info_)->read, 0);
    EXPECT_TRUE(recv.conn.flags & NCCL_P2P_WRITE);

    p2pTransport.recv.free(&comm_, &recv);
}

TEST_F(P2pSetupMicrotest, RecvSetup_IntermediateHop_SelectsIntermediateAndRoutesConnectRank)
{
    InstallTopoIntermediate(/*inter=*/0);
    InstallHappyProxy();

    ncclConnector recv{};
    ASSERT_EQ(p2pTransport.recv.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, 0),
              ncclSuccess);

    auto* res = static_cast<p2pResources*>(recv.transportResources);
    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->type, P2P_INTERMEDIATE);
    EXPECT_EQ(AsConnectInfo(connect_info_)->rank, 0);

    p2pTransport.recv.free(&comm_, &recv);
}

TEST_F(P2pSetupMicrotest, RecvSetup_CrossProcessCuMemPeer_SelectsCumem)
{
    peer_.pidHash = kPid + 1;  // different process => not DIRECT
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    InstallTopo(/*read=*/0);
    InstallHappyProxy();

    ncclConnector recv{};
    ASSERT_EQ(p2pTransport.recv.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, 0),
              ncclSuccess);

    auto* res = static_cast<p2pResources*>(recv.transportResources);
    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->type, P2P_CUMEM);

    p2pTransport.recv.free(&comm_, &recv);
}

TEST_F(P2pSetupMicrotest, SendSetupThenConnect_WritePath_WiresConnBuffers)
{
    InstallTopo(/*read=*/0);
    InstallXgmiLink();
    InstallHappyProxy();

    ASSERT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSuccess);
    ASSERT_EQ(p2pTransport.send.connect(&comm_, &connect_info_, comm_.nRanks,
                                        /*rank=*/0, &send_),
              ncclSuccess);

    // connect wired the SIMPLE buffer pointer and the proxyProgress slot
    // (NULL here, matching the un-primed CE path).
    EXPECT_NE(send_.conn.buffs[NCCL_PROTO_SIMPLE], nullptr);
    EXPECT_EQ(send_.proxyConn.proxyProgress, nullptr);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, SendSetupThenConnect_ReadPath_UsesLocalSimpleBuffer)
{
    InstallTopo(/*read=*/1);
    InstallXgmiLink();
    InstallHappyProxy();

    ASSERT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSuccess);
    ASSERT_EQ(p2pTransport.send.connect(&comm_, &connect_info_, comm_.nRanks,
                                        /*rank=*/0, &send_),
              ncclSuccess);

    // On the read path the SIMPLE buffer is the local ncclSendMem (sendDevMem),
    // not the remote-mapped region: it sits just past resources->sendDevMem.
    auto* res = static_cast<p2pResources*>(send_.transportResources);
    ASSERT_NE(res->sendDevMem, nullptr);
    EXPECT_EQ(send_.conn.buffs[NCCL_PROTO_SIMPLE],
              reinterpret_cast<char*>(res->sendDevMem + 1));

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, RecvSetupThenConnect_WritePath_WiresConnBuffers)
{
    InstallTopo(/*read=*/0);
    InstallHappyProxy();

    ncclConnector recv{};
    ASSERT_EQ(p2pTransport.recv.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, 0),
              ncclSuccess);
    ASSERT_EQ(p2pTransport.recv.connect(&comm_, &connect_info_, comm_.nRanks,
                                        /*rank=*/0, &recv),
              ncclSuccess);

    // recv connect wired the SIMPLE buffer and the local tail pointer.
    EXPECT_NE(recv.conn.buffs[NCCL_PROTO_SIMPLE], nullptr);
    EXPECT_NE(recv.conn.tail, nullptr);
    EXPECT_NE(recv.conn.head, nullptr);

    p2pTransport.recv.free(&comm_, &recv);
}

TEST_F(P2pSetupMicrotest, RecvSetupThenConnect_ReadPath_TakesRemoteSimpleBuffer)
{
    // On the read path recv setup sizes the buffer pool without the SIMPLE
    // slot and recv connect points buffs[SIMPLE] at the remote ncclSendMem
    // (remDevMem + 1) instead of the local pool. In this harness the mapped
    // and local regions share the exchanged backing buffer, so the observable
    // contract asserted here is that the read verdict propagates and the SIMPLE
    // buffer lands on the mapped remote region.
    InstallTopo(/*read=*/1);
    InstallHappyProxy();

    ncclConnector recv{};
    ASSERT_EQ(p2pTransport.recv.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, 0),
              ncclSuccess);
    EXPECT_EQ(AsConnectInfo(connect_info_)->read, 1);

    ASSERT_EQ(p2pTransport.recv.connect(&comm_, &connect_info_, comm_.nRanks,
                                        /*rank=*/0, &recv),
              ncclSuccess);

    EXPECT_EQ(recv.conn.buffs[NCCL_PROTO_SIMPLE],
              reinterpret_cast<char*>(
                  reinterpret_cast<ncclSendMem*>(backing_.data()) + 1));

    p2pTransport.recv.free(&comm_, &recv);
}

// recv connect's p2pMap takes the cross-process import arm when the connect
// info names a peer in a different process; a failing import there propagates
// out of connect.
TEST_F(P2pSetupMicrotest, RecvConnect_CrossProcessMapImportFails_Propagates)
{
    // Slot 1 is a cross-process peer so p2pMap(comm->peerInfo+rank=0,
    // comm->peerInfo+info->rank=1) takes the different-PID import branch.
    peers_[1] = myInfo_;
    peers_[1].rank    = 1;
    peers_[1].pidHash = kPid + 1;   // different process

    InstallTopo(/*read=*/0);
    InstallHappyProxy();

    ncclConnector recv{};
    ASSERT_EQ(p2pTransport.recv.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, 0),
              ncclSuccess);

    // Point the exchanged connect info at the cross-process peer and refuse
    // the legacy IPC import that p2pMap then attempts.
    reinterpret_cast<p2pConnectInfo*>(connect_info_.data)->rank = 1;
    ScopedHook open(g_hipIpcOpenMemHandle,
        [](void**, hipIpcMemHandle_t, unsigned int) -> hipError_t {
            return hipErrorInvalidValue;
        });

    EXPECT_NE(p2pTransport.recv.connect(&comm_, &connect_info_, comm_.nRanks,
                                        /*rank=*/0, &recv),
              ncclSuccess);

    p2pTransport.recv.free(&comm_, &recv);
}

// --- setup/connect error propagation -------------------------------------
// Each of these drives one failure seam on the setup/connect path and asserts
// the error is propagated (the resource allocation / proxy handshake that
// failed is the seam under test; setup's own contract here is "stop and
// return the error", not the downstream bookkeeping).

TEST_F(P2pSetupMicrotest, SendSetup_ResourceAllocFails_Propagates)
{
    InstallTopo(/*read=*/0);
    InstallXgmiLink();
    InstallHappyProxy();
    // Starve the first allocation setup makes: the p2pResources block.
    SetP2pCallocFailSize(sizeof(p2pResources));

    EXPECT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSystemError);
    // The alloc never succeeded, so there is nothing to free.
    EXPECT_EQ(send_.transportResources, nullptr);
}

TEST_F(P2pSetupMicrotest, SendSetup_TopologyQueryFails_Propagates)
{
    // p2pGetInfo delegates straight to ncclTopoCheckP2p; a failure there
    // aborts setup before any proxy handshake.
    InstallXgmiLink();
    InstallHappyProxy();
    topo_.emplace(g_ncclTopoCheckP2p,
        [](int, int, int*, int*, int*, int*, int*) -> ncclResult_t {
            return ncclSystemError;
        });

    EXPECT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSystemError);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, SendSetup_ProxyConnectFails_Propagates)
{
    InstallTopo(/*read=*/0);
    InstallXgmiLink();
    proxy_.emplace(g_proxyCallBlocking,
        [](struct ncclComm*, struct ncclProxyConnector*, int, void*, int,
           void*, int) -> ncclResult_t {
            ADD_FAILURE() << "proxy call must not run after connect failure";
            return ncclSystemError;
        });
    connect_.emplace(g_proxyConnect,
        [](struct ncclComm*, int, int, int,
           struct ncclProxyConnector*) -> ncclResult_t {
            return ncclSystemError;
        });

    EXPECT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSystemError);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, SendSetup_ProxySetupCallFails_Propagates)
{
    InstallTopo(/*read=*/0);
    InstallXgmiLink();
    connect_.emplace(g_proxyConnect, MarkProxyConnInitialized());
    proxy_.emplace(g_proxyCallBlocking,
        [](struct ncclComm*, struct ncclProxyConnector*, int, void*, int,
           void*, int) -> ncclResult_t { return ncclSystemError; });

    EXPECT_EQ(p2pTransport.send.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &send_, 0, 0),
              ncclSystemError);

    p2pTransport.send.free(&comm_, &send_);
}

TEST_F(P2pSetupMicrotest, RecvSetup_ResourceAllocFails_Propagates)
{
    InstallTopo(/*read=*/0);
    InstallHappyProxy();
    SetP2pCallocFailSize(sizeof(p2pResources));

    ncclConnector recv{};
    EXPECT_EQ(p2pTransport.recv.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, 0),
              ncclSystemError);
    EXPECT_EQ(recv.transportResources, nullptr);
}

TEST_F(P2pSetupMicrotest, RecvSetup_TopologyQueryFails_Propagates)
{
    InstallHappyProxy();
    topo_.emplace(g_ncclTopoCheckP2p,
        [](int, int, int*, int*, int*, int*, int*) -> ncclResult_t {
            return ncclSystemError;
        });

    ncclConnector recv{};
    EXPECT_EQ(p2pTransport.recv.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, 0),
              ncclSystemError);

    p2pTransport.recv.free(&comm_, &recv);
}

TEST_F(P2pSetupMicrotest, RecvSetup_ProxyConnectFails_Propagates)
{
    InstallTopo(/*read=*/0);
    proxy_.emplace(g_proxyCallBlocking,
        [](struct ncclComm*, struct ncclProxyConnector*, int, void*, int,
           void*, int) -> ncclResult_t {
            ADD_FAILURE() << "proxy call must not run after connect failure";
            return ncclSystemError;
        });
    connect_.emplace(g_proxyConnect,
        [](struct ncclComm*, int, int, int,
           struct ncclProxyConnector*) -> ncclResult_t {
            return ncclSystemError;
        });

    ncclConnector recv{};
    EXPECT_EQ(p2pTransport.recv.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, 0),
              ncclSystemError);

    p2pTransport.recv.free(&comm_, &recv);
}

TEST_F(P2pSetupMicrotest, RecvSetup_ProxySetupCallFails_Propagates)
{
    InstallTopo(/*read=*/0);
    connect_.emplace(g_proxyConnect, MarkProxyConnInitialized());
    proxy_.emplace(g_proxyCallBlocking,
        [](struct ncclComm*, struct ncclProxyConnector*, int, void*, int,
           void*, int) -> ncclResult_t { return ncclSystemError; });

    ncclConnector recv{};
    EXPECT_EQ(p2pTransport.recv.setup(&comm_, nullptr, &myInfo_, &peer_,
                                      &connect_info_, &recv, 0, 0),
              ncclSystemError);

    p2pTransport.recv.free(&comm_, &recv);
}

// --- free contract -------------------------------------------------------

class P2pFreeMicrotest : public P2pMicrotest {
protected:
    // A heap p2pResources allocated the way p2pSendSetup does (ncclCalloc ==
    // calloc), so production's free(resources) matches the allocation.
    p2pResources* MakeResources() {
        return static_cast<p2pResources*>(std::calloc(1, sizeof(p2pResources)));
    }
};

TEST_F(P2pFreeMicrotest, SendFree_NoResources_IsNoopSuccess)
{
    ncclConnector send{};  // transportResources == NULL
    // Also null-safe on the comm pointer.
    EXPECT_EQ(p2pTransport.send.free(nullptr, &send), ncclSuccess);
}

TEST_F(P2pFreeMicrotest, SendFree_LegacyIpcResources_ClosesEachRetainedHandle)
{
    int closes = 0;
    ScopedHook close(g_hipIpcCloseMemHandle,
        [&closes](void*) -> hipError_t { ++closes; return hipSuccess; });

    ncclConnector send{};
    auto* res = MakeResources();
    res->sendMemIpc     = reinterpret_cast<void*>(0x1000);
    res->recvMemIpc     = reinterpret_cast<void*>(0x2000);
    res->sendMemSameProc = 0;  // cross-process => legacy close, not free-addr
    res->recvMemSameProc = 0;
    send.transportResources = res;

    // cuMemEnable stays 0 (default) => the legacy cudaIpcCloseMemHandle arm.
    EXPECT_EQ(p2pTransport.send.free(nullptr, &send), ncclSuccess);
    EXPECT_EQ(closes, 2);  // one per retained handle
}

// RAII guard that latches the process-shutdown flag for the duration of a
// test. With the flag set, ncclCuMemFreeAddr / ncclCudaFree short-circuit to
// success before touching the (absent) HIP runtime, so the cuMem free arm can
// be exercised in-process without a real GPU allocation. The flag is a
// process-global static, so it must be cleared again on scope exit.
struct ShutdownFlagGuard {
    ShutdownFlagGuard()  { rcclShutdownFlag().store(true,  std::memory_order_release); }
    ~ShutdownFlagGuard() { rcclShutdownFlag().store(false, std::memory_order_release); }
};

#if ROCM_VERSION >= 70000
TEST_F(P2pFreeMicrotest, SendFree_CuMemResources_ReleasesEachRetainedHandle)
{
    // cuMem enabled selects the cuMem free arm rather than the legacy
    // cudaIpcCloseMemHandle path; the shutdown flag lets ncclCuMemFreeAddr /
    // ncclCudaFree return without reaching HIP.
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ShutdownFlagGuard shutdown;

    // Two frees with complementary same-proc flags so each handle's
    // ncclCuMemFreeAddr (same-proc) and ncclCudaFree (cross-proc) arm is hit.
    ncclConnector send{};
    auto* res = MakeResources();
    res->sendMemIpc      = reinterpret_cast<void*>(0x1000);
    res->recvMemIpc      = reinterpret_cast<void*>(0x2000);
    res->sendMemSameProc = 1;  // same-proc => ncclCuMemFreeAddr
    res->recvMemSameProc = 0;  // cross-proc => ncclCudaFree
    send.transportResources = res;
    g_freeCalls.clear();
    EXPECT_EQ(p2pTransport.send.free(nullptr, &send), ncclSuccess);
    // Each retained handle is released on its selected arm, in order: the
    // send handle same-proc (ncclCuMemFreeAddr), the recv handle cross-proc
    // (ncclCudaFree). Deleting either release call under test drops the
    // matching entry and fails this.
    EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
        {FreeKind::CuMemFreeAddr, reinterpret_cast<void*>(0x1000)},
        {FreeKind::CudaFree,      reinterpret_cast<void*>(0x2000)}}));

    ncclConnector send2{};
    auto* res2 = MakeResources();
    res2->sendMemIpc      = reinterpret_cast<void*>(0x1000);
    res2->recvMemIpc      = reinterpret_cast<void*>(0x2000);
    res2->sendMemSameProc = 0;  // cross-proc => ncclCudaFree
    res2->recvMemSameProc = 1;  // same-proc => ncclCuMemFreeAddr
    send2.transportResources = res2;
    g_freeCalls.clear();
    EXPECT_EQ(p2pTransport.send.free(nullptr, &send2), ncclSuccess);
    // Complementary flags swap the arms.
    EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
        {FreeKind::CudaFree,      reinterpret_cast<void*>(0x1000)},
        {FreeKind::CuMemFreeAddr, reinterpret_cast<void*>(0x2000)}}));
}

#endif  // ROCM_VERSION >= 70000

TEST_F(P2pFreeMicrotest, RecvFree_NoResources_IsNoopSuccess)
{
    ncclConnector recv{};
    EXPECT_EQ(p2pTransport.recv.free(nullptr, &recv), ncclSuccess);
}

#if ROCM_VERSION >= 70000
TEST_F(P2pFreeMicrotest, RecvFree_CuMemResources_ReleasesEachRetainedHandle)
{
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ShutdownFlagGuard shutdown;

    ncclConnector recv{};
    auto* res = MakeResources();
    res->sendMemIpc      = reinterpret_cast<void*>(0x3000);
    res->recvMemIpc      = reinterpret_cast<void*>(0x4000);
    res->sendMemSameProc = 1;  // same-proc => ncclCuMemFreeAddr
    res->recvMemSameProc = 0;  // cross-proc => ncclCudaFree
    recv.transportResources = res;
    g_freeCalls.clear();
    EXPECT_EQ(p2pTransport.recv.free(nullptr, &recv), ncclSuccess);
    EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
        {FreeKind::CuMemFreeAddr, reinterpret_cast<void*>(0x3000)},
        {FreeKind::CudaFree,      reinterpret_cast<void*>(0x4000)}}));

    ncclConnector recv2{};
    auto* res2 = MakeResources();
    res2->sendMemIpc      = reinterpret_cast<void*>(0x3000);
    res2->recvMemIpc      = reinterpret_cast<void*>(0x4000);
    res2->sendMemSameProc = 0;  // cross-proc => ncclCudaFree
    res2->recvMemSameProc = 1;  // same-proc => ncclCuMemFreeAddr
    recv2.transportResources = res2;
    g_freeCalls.clear();
    EXPECT_EQ(p2pTransport.recv.free(nullptr, &recv2), ncclSuccess);
    EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
        {FreeKind::CudaFree,      reinterpret_cast<void*>(0x3000)},
        {FreeKind::CuMemFreeAddr, reinterpret_cast<void*>(0x4000)}}));
}
#endif  // ROCM_VERSION >= 70000

TEST_F(P2pFreeMicrotest, RecvFree_LegacyIpcResources_ClosesEachRetainedHandle)
{
    int closes = 0;
    ScopedHook close(g_hipIpcCloseMemHandle,
        [&closes](void*) -> hipError_t { ++closes; return hipSuccess; });

    ncclConnector recv{};
    auto* res = MakeResources();
    res->sendMemIpc      = reinterpret_cast<void*>(0x3000);
    res->recvMemIpc      = reinterpret_cast<void*>(0x4000);
    res->sendMemSameProc = 0;
    res->recvMemSameProc = 0;
    recv.transportResources = res;

    EXPECT_EQ(p2pTransport.recv.free(nullptr, &recv), ncclSuccess);
    EXPECT_EQ(closes, 2);
}

// Legacy close failure propagates: when cudaIpcCloseMemHandle refuses on a
// retained handle, send/recv free surface the HIP error rather than silently
// succeeding.
TEST_F(P2pFreeMicrotest, SendFree_LegacyIpcCloseFails_Propagates)
{
    ScopedHook close(g_hipIpcCloseMemHandle,
        [](void*) -> hipError_t { return hipErrorInvalidValue; });

    ncclConnector send{};
    auto* res = MakeResources();
    res->sendMemIpc      = reinterpret_cast<void*>(0x1000);
    res->sendMemSameProc = 0;
    send.transportResources = res;

    EXPECT_NE(p2pTransport.send.free(nullptr, &send), ncclSuccess);
    // The forced close failure returns before production's free(resources),
    // so the test owns the block (else LeakSanitizer reports it under ASAN).
    std::free(res);
}

// The recv-handle close arm (reached only after the send-handle close
// succeeds) also propagates its failure.
TEST_F(P2pFreeMicrotest, SendFree_LegacyIpcRecvCloseFails_Propagates)
{
    void* const kRecv = reinterpret_cast<void*>(0x2000);
    ScopedHook close(g_hipIpcCloseMemHandle,
        [kRecv](void* p) -> hipError_t {
            return p == kRecv ? hipErrorInvalidValue : hipSuccess;
        });

    ncclConnector send{};
    auto* res = MakeResources();
    res->sendMemIpc      = reinterpret_cast<void*>(0x1000);
    res->recvMemIpc      = kRecv;
    res->sendMemSameProc = 0;
    res->recvMemSameProc = 0;
    send.transportResources = res;

    EXPECT_NE(p2pTransport.send.free(nullptr, &send), ncclSuccess);
    std::free(res);  // forced failure returns before free(resources)
}

TEST_F(P2pFreeMicrotest, RecvFree_LegacyIpcCloseFails_Propagates)
{
    ScopedHook close(g_hipIpcCloseMemHandle,
        [](void*) -> hipError_t { return hipErrorInvalidValue; });

    ncclConnector recv{};
    auto* res = MakeResources();
    res->sendMemIpc      = reinterpret_cast<void*>(0x3000);
    res->sendMemSameProc = 0;
    recv.transportResources = res;

    EXPECT_NE(p2pTransport.recv.free(nullptr, &recv), ncclSuccess);
    std::free(res);  // forced failure returns before free(resources)
}

TEST_F(P2pFreeMicrotest, RecvFree_LegacyIpcRecvCloseFails_Propagates)
{
    void* const kRecv = reinterpret_cast<void*>(0x4000);
    ScopedHook close(g_hipIpcCloseMemHandle,
        [kRecv](void* p) -> hipError_t {
            return p == kRecv ? hipErrorInvalidValue : hipSuccess;
        });

    ncclConnector recv{};
    auto* res = MakeResources();
    res->sendMemIpc      = reinterpret_cast<void*>(0x3000);
    res->recvMemIpc      = kRecv;
    res->sendMemSameProc = 0;
    res->recvMemSameProc = 0;
    recv.transportResources = res;

    EXPECT_NE(p2pTransport.recv.free(nullptr, &recv), ncclSuccess);
    std::free(res);  // forced failure returns before free(resources)
}

// ===========================================================================
// Shareable-buffer alloc/import: ncclP2pAllocateShareableBuffer and
// ncclP2pImportShareableBuffer. These two free functions pick between the
// cuMem* driver arm and the legacy CUDA-IPC arm off ncclCuMemEnable(), then
// populate the export descriptor / import the remote buffer. Neither leaves
// much observable public state, so the tests assert the descriptor / output
// pointer the caller relies on where possible and fall back to mock-style
// "the arm drove this seam" assertions for the manager-tracking bookkeeping.
//
// The cuMem arms need ROCM_VERSION >= 7 (the #else compiles to
// `return ncclInternalError`); the legacy arms are always present.
// ===========================================================================

class P2pShareableBufferMicrotest : public P2pMicrotest {
protected:
    ncclIpcDesc ipcDesc{};
    void*       ptr = nullptr;

    // Import-path state, hoisted out of the eleven import tests so each is a
    // single CallImport(). commStorage_ is heap-allocated (~3.8 MB, not
    // stack-safe) on first use; peerInfo_ backs comm.peerInfo, which the cuMem
    // tracking call reads as peerInfo[peer].cudaDev. devMem_ receives the
    // mapped/reserved pointer the caller relies on.
    std::unique_ptr<ncclComm>   commStorage_;
    std::array<ncclPeerInfo, 4> peerInfo_{};
    void*                       devMem_ = nullptr;

    ncclResult_t CallImport(int peer = 1, std::size_t size = 256)
    {
        commStorage_ = std::make_unique<ncclComm>();
        commStorage_->peerInfo = peerInfo_.data();
        return ncclP2pImportShareableBuffer(commStorage_.get(), peer, size,
                                            &ipcDesc, &devMem_);
    }
};

// Legacy-IPC arm (ncclCuMemEnable() == 0): the buffer is allocated through
// ncclCudaCalloc and its IPC handle captured via hipIpcGetMemHandle into
// ipcDesc.devIpc. Contract: *ptr is set to the allocation and the caller's
// descriptor carries the exported IPC handle.
TEST_F(P2pShareableBufferMicrotest,
       Allocate_CuMemDisabled_ExportsLegacyIpcHandle)
{
    // cuMemEnable stays 0 (default) -> legacy arm.
    int getCalls = 0;
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [&getCalls](hipIpcMemHandle_t* h, void* devPtr) -> hipError_t {
            ++getCalls;
            EXPECT_NE(devPtr, nullptr);   // the freshly-allocated buffer
            if (h) std::memset(h, 0x5A, sizeof(*h));
            return hipSuccess;
        });

    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/0,
                                            &ipcDesc, &ptr);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(getCalls, 1);
    EXPECT_NE(ptr, nullptr);
    // Handle bytes were written into the legacy-IPC descriptor member.
    unsigned char* bytes = reinterpret_cast<unsigned char*>(&ipcDesc.devIpc);
    EXPECT_EQ(bytes[0], 0x5A);
}

// Legacy-IPC arm, hipIpcGetMemHandle fails: the HIP error is propagated
// (CUDACHECK) rather than the caller seeing success with a half-populated
// descriptor. (Releasing the buffer is ncclCudaFree's contract, covered
// elsewhere; this test pins only the propagation the unit owns.)
TEST_F(P2pShareableBufferMicrotest,
       Allocate_LegacyIpcGetMemHandleFails_Propagates)
{
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t*, void*) -> hipError_t {
            return hipErrorInvalidValue;
        });

    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/0,
                                            &ipcDesc, &ptr);

    EXPECT_NE(r, ncclSuccess);       // the HIP error propagated
}

#if ROCM_VERSION >= 70000

// cuMem arm, non-POSIX handle type: the allocation's handle is exported via
// hipMemExportToShareableHandle into ipcDesc.cuDesc (the UDS/POSIX_FD memcpy
// branch is skipped). With refcount > 0 the handle is also stashed in
// ipcDesc.memHandle and retained refcount times.
TEST_F(P2pShareableBufferMicrotest,
       Allocate_CuMemEnabledNonPosix_ExportsHandleAndRetains)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypeWin32);  // anything != POSIX_FD

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });

    int exportCalls = 0;
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [&exportCalls](void* shareableHandle, hipMemGenericAllocationHandle_t h,
                       hipMemAllocationHandleType type,
                       unsigned long long) -> hipError_t {
            ++exportCalls;
            EXPECT_NE(type, hipMemHandleTypePosixFileDescriptor);
            // Write a recognisable exported descriptor.
            if (shareableHandle) std::memset(shareableHandle, 0x3C, 8);
            (void)h;
            return hipSuccess;
        });
    int retainCalls = 0;
    ScopedHook retain(g_hipMemRetainAllocationHandle,
        [&retainCalls](hipMemGenericAllocationHandle_t*, void*) -> hipError_t {
            ++retainCalls;
            return hipSuccess;
        });

    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/2,
                                            &ipcDesc, &ptr);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(exportCalls, 1);          // the non-POSIX export arm fired
    EXPECT_EQ(retainCalls, 2);          // one retain per refcount
    EXPECT_NE(ptr, nullptr);
    unsigned char* bytes = reinterpret_cast<unsigned char*>(&ipcDesc.cuDesc);
    EXPECT_EQ(bytes[0], 0x3C);          // exported descriptor landed in cuDesc
}

// cuMem arm, POSIX_FD handle type: the native handle is memcpy'd straight
// into ipcDesc.cuDesc.data for later UDS conversion; hipMemExportToShareable-
// Handle is NOT called. refcount 0 => no retain.
TEST_F(P2pShareableBufferMicrotest,
       Allocate_CuMemEnabledPosixFd_StoresNativeHandleWithoutExport)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
           unsigned long long) -> hipError_t {
            ADD_FAILURE() << "POSIX_FD arm must not export a shareable handle";
            return hipErrorInvalidValue;
        });

    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/0,
                                            &ipcDesc, &ptr);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_NE(ptr, nullptr);
    // The emulator's sentinel handle bits were copied into cuDesc.data.
    std::uintptr_t stored = 0;
    std::memcpy(&stored, &ipcDesc.cuDesc.data, sizeof(stored));
    EXPECT_EQ(stored, kFakeCuMemAllocHandleBits);
}

// cuMem arm, export-to-peer gating: when the caller supplies a manager, a
// real peer rank, and a non-persistent memtype, the freshly-allocated buffer
// is marked for export via ncclDynMemMarkExportToPeer.
TEST_F(P2pShareableBufferMicrotest,
       Allocate_CuMemEnabledWithPeerAndManager_MarksExportToPeer)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);  // skip export

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    int markCalls = 0;
    int seenPeer  = -99;
    ScopedHook mark(g_dynMemMarkExportToPeer,
        [&](struct ncclMemManager*, void*, int peerRank) -> ncclResult_t {
            ++markCalls;
            seenPeer = peerRank;
            return ncclSuccess;
        });

    ncclMemManager* fakeManager = reinterpret_cast<ncclMemManager*>(0x1234);
    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/0,
                                            &ipcDesc, &ptr, /*peerRank=*/3,
                                            fakeManager, ncclMemScratch);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(markCalls, 1);
    EXPECT_EQ(seenPeer, 3);
}

// cuMem arm, export-to-peer gating skipped: each operand of the
// `manager != nullptr && peerRank >= 0 && memtype != ncclMemPersist` guard
// short-circuits the mark-for-export call in turn. Parameterised over the
// three ways to miss the guard so all three operand False arms are driven.
struct SkipMarkExportCase {
    ncclMemManager* manager;
    int             peerRank;
    ncclMemType_t   memtype;
};
class P2pAllocateSkipMarkExport
    : public P2pShareableBufferMicrotest,
      public ::testing::WithParamInterface<SkipMarkExportCase> {};

TEST_P(P2pAllocateSkipMarkExport, CuMemEnabled_GuardMisses_SkipsMarkExportToPeer)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook mark(g_dynMemMarkExportToPeer,
        [](struct ncclMemManager*, void*, int) -> ncclResult_t {
            ADD_FAILURE() << "mark-for-export must not fire when the guard misses";
            return ncclSuccess;
        });

    const auto& p = GetParam();
    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/0,
                                            &ipcDesc, &ptr, p.peerRank,
                                            p.manager, p.memtype);

    EXPECT_EQ(r, ncclSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    GuardOperands, P2pAllocateSkipMarkExport,
    ::testing::Values(
        // No manager -> first operand False (short-circuits before the rest).
        SkipMarkExportCase{nullptr, 3, ncclMemScratch},
        // Manager but no real peer -> second operand False.
        SkipMarkExportCase{reinterpret_cast<ncclMemManager*>(0x1234), -1,
                           ncclMemScratch},
        // Manager + real peer but persistent memtype -> third operand False.
        SkipMarkExportCase{reinterpret_cast<ncclMemManager*>(0x1234), 3,
                           ncclMemPersist}));

// cuMem arm, allocation itself fails: ncclCuMemAlloc's error is propagated
// before any export/retain bookkeeping.
TEST_F(P2pShareableBufferMicrotest,
       Allocate_CuMemAllocFails_Propagates)
{
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook alloc(g_fakeCuMemAlloc,
        [](void**, hipMemGenericAllocationHandle_t*, std::size_t) -> ncclResult_t {
            return ncclSystemError;
        });
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
           unsigned long long) -> hipError_t {
            ADD_FAILURE() << "export must not fire after alloc failure";
            return hipErrorInvalidValue;
        });

    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/0,
                                            &ipcDesc, &ptr);

    EXPECT_EQ(r, ncclSystemError);
}

#endif  // ROCM_VERSION >= 70000

// ---------------------------------------------------------------------------
// ncclP2pImportShareableBuffer
// ---------------------------------------------------------------------------

// Legacy-IPC arm (ncclCuMemEnable() == 0): the remote handle is opened with
// hipIpcOpenMemHandle and the mapped device pointer handed back through
// *devMemPtr.
TEST_F(P2pShareableBufferMicrotest,
       Import_CuMemDisabled_OpensLegacyIpcHandle)
{
    void* const kMapped = reinterpret_cast<void*>(0xB0000);
    int openCalls = 0;
    ScopedHook open(g_hipIpcOpenMemHandle,
        [&](void** devPtr, hipIpcMemHandle_t, unsigned int flags) -> hipError_t {
            ++openCalls;
            EXPECT_EQ(flags, static_cast<unsigned>(hipIpcMemLazyEnablePeerAccess));
            if (devPtr) *devPtr = kMapped;
            return hipSuccess;
        });

    auto r = CallImport();

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(openCalls, 1);
    EXPECT_EQ(devMem_, kMapped);   // caller receives the mapped pointer
}

// Legacy-IPC arm, hipIpcOpenMemHandle fails: the error is propagated.
TEST_F(P2pShareableBufferMicrotest,
       Import_LegacyIpcOpenFails_Propagates)
{
    ScopedHook open(g_hipIpcOpenMemHandle,
        [](void**, hipIpcMemHandle_t, unsigned int) -> hipError_t {
            return hipErrorInvalidValue;
        });

    auto r = CallImport();

    EXPECT_NE(r, ncclSuccess);
}

#if ROCM_VERSION >= 70000

// cuMem arm, non-POSIX handle type: import -> address-reserve -> map ->
// set-access -> track. The reserved virtual address is handed back through
// *devMemPtr and the mapped buffer is recorded via ncclMemTrackImportFromPeer.
TEST_F(P2pShareableBufferMicrotest,
       Import_CuMemEnabledNonPosix_MapsAndTracksRemoteBuffer)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypeWin32);  // != POSIX_FD

    void* const kReserved = reinterpret_cast<void*>(0xC0000);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook gran(g_hipMemGetAllocationGranularity, GranularitySucceeds());
    int importCalls = 0, mapCalls = 0, accessCalls = 0, trackCalls = 0;
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [&](hipMemGenericAllocationHandle_t* h, void*,
            hipMemAllocationHandleType) -> hipError_t {
            ++importCalls;
            if (h) *h = nullptr;
            return hipSuccess;
        });
    ScopedHook reserve(g_hipMemAddressReserve,
        [&](void** p, std::size_t, std::size_t, void*,
            unsigned long long) -> hipError_t {
            if (p) *p = kReserved;
            return hipSuccess;
        });
    ScopedHook map(g_hipMemMap,
        [&](void*, std::size_t, std::size_t, hipMemGenericAllocationHandle_t,
            unsigned long long) -> hipError_t { ++mapCalls; return hipSuccess; });
    ScopedHook access(g_hipMemSetAccess,
        [&](void*, std::size_t, const hipMemAccessDesc*,
            std::size_t) -> hipError_t { ++accessCalls; return hipSuccess; });
    ScopedHook track(g_memTrackImportFromPeer,
        [&](struct ncclMemManager*, void* p, size_t, hipMemGenericAllocationHandle_t,
            hipMemAllocationHandleType, ncclMemType_t, int, int,
            void*) -> ncclResult_t {
            ++trackCalls;
            EXPECT_EQ(p, kReserved);   // the mapped address is what's tracked
            return ncclSuccess;
        });

    auto r = CallImport();

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(importCalls, 1);
    EXPECT_EQ(mapCalls, 1);
    EXPECT_EQ(accessCalls, 1);
    EXPECT_EQ(trackCalls, 1);
    EXPECT_EQ(devMem_, kReserved);
}

// cuMem arm, POSIX_FD handle type: the remote handle is converted to a local
// fd via ncclProxyClientGetFdBlocking, then imported from that fd (the plain
// hipMemImportFromShareableHandle(cuDesc) branch is skipped).
TEST_F(P2pShareableBufferMicrotest,
       Import_CuMemEnabledPosixFd_ConvertsHandleToFdBeforeImport)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    void* const kReserved = reinterpret_cast<void*>(0xD0000);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook gran(g_hipMemGetAllocationGranularity, GranularitySucceeds());
    int getFdCalls = 0;
    ScopedHook getFd(g_ncclProxyClientGetFdBlocking,
        [&](struct ncclComm*, int, void*, int* fd) -> ncclResult_t {
            ++getFdCalls;
            // Hand back a real, closable fd so the SYSCHECK(close(fd)) succeeds.
            if (fd) *fd = dup(STDERR_FILENO);
            return ncclSuccess;
        });
    int importCalls = 0;
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [&](hipMemGenericAllocationHandle_t* h, void*,
            hipMemAllocationHandleType) -> hipError_t {
            ++importCalls;
            if (h) *h = nullptr;
            return hipSuccess;
        });
    ScopedHook reserve(g_hipMemAddressReserve,
        [&](void** p, std::size_t, std::size_t, void*,
            unsigned long long) -> hipError_t {
            if (p) *p = kReserved;
            return hipSuccess;
        });
    ScopedHook map(g_hipMemMap,
        [](void*, std::size_t, std::size_t, hipMemGenericAllocationHandle_t,
           unsigned long long) -> hipError_t { return hipSuccess; });
    ScopedHook access(g_hipMemSetAccess,
        [](void*, std::size_t, const hipMemAccessDesc*, std::size_t) -> hipError_t {
            return hipSuccess;
        });
    ScopedHook track(g_memTrackImportFromPeer,
        [](struct ncclMemManager*, void*, size_t, hipMemGenericAllocationHandle_t,
           hipMemAllocationHandleType, ncclMemType_t, int, int, void*) -> ncclResult_t {
            return ncclSuccess;
        });

    auto r = CallImport();

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(getFdCalls, 1);     // the POSIX_FD conversion fired
    EXPECT_EQ(importCalls, 1);
    EXPECT_EQ(devMem_, kReserved);
}

// cuMem arm, import failure: hipMemImportFromShareableHandle's error is
// propagated before any reserve/map/track work.
TEST_F(P2pShareableBufferMicrotest,
       Import_CuMemImportFails_Propagates)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypeWin32);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook gran(g_hipMemGetAllocationGranularity, GranularitySucceeds());
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [](hipMemGenericAllocationHandle_t*, void*,
           hipMemAllocationHandleType) -> hipError_t {
            return hipErrorInvalidValue;
        });
    ScopedHook map(g_hipMemMap,
        [](void*, std::size_t, std::size_t, hipMemGenericAllocationHandle_t,
           unsigned long long) -> hipError_t {
            ADD_FAILURE() << "map must not run after import failure";
            return hipErrorInvalidValue;
        });

    auto r = CallImport();

    EXPECT_NE(r, ncclSuccess);
}

// cuMem arm, granularity query fails: the first driver call in the import
// sequence refuses, so control returns the HIP error before importing,
// reserving, or mapping anything.
TEST_F(P2pShareableBufferMicrotest,
       Import_CuMemGranularityQueryFails_Propagates)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypeWin32);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook gran(g_hipMemGetAllocationGranularity,
        [](std::size_t*, const hipMemAllocationProp*,
           hipMemAllocationGranularity_flags) -> hipError_t {
            return hipErrorInvalidValue;
        });
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [](hipMemGenericAllocationHandle_t*, void*,
           hipMemAllocationHandleType) -> hipError_t {
            ADD_FAILURE() << "import must not run after granularity failure";
            return hipErrorInvalidValue;
        });

    auto r = CallImport();

    EXPECT_NE(r, ncclSuccess);
}

// cuMem POSIX_FD arm, fd conversion fails: ncclProxyClientGetFdBlocking
// refuses, so control returns the error before importing from the fd.
TEST_F(P2pShareableBufferMicrotest,
       Import_CuMemPosixFdConversionFails_Propagates)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook gran(g_hipMemGetAllocationGranularity, GranularitySucceeds());
    ScopedHook getFd(g_ncclProxyClientGetFdBlocking,
        [](struct ncclComm*, int, void*, int*) -> ncclResult_t {
            return ncclSystemError;
        });
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [](hipMemGenericAllocationHandle_t*, void*,
           hipMemAllocationHandleType) -> hipError_t {
            ADD_FAILURE() << "import must not run after fd conversion failure";
            return hipErrorInvalidValue;
        });

    auto r = CallImport();

    EXPECT_NE(r, ncclSuccess);
}

// cuMem POSIX_FD arm, import-from-fd fails: the fd is converted, but
// hipMemImportFromShareableHandle refuses on the fd value, so the error is
// propagated before reserve/map.
TEST_F(P2pShareableBufferMicrotest,
       Import_CuMemPosixFdImportFails_Propagates)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook gran(g_hipMemGetAllocationGranularity, GranularitySucceeds());
    ScopedHook getFd(g_ncclProxyClientGetFdBlocking,
        [](struct ncclComm*, int, void*, int* fd) -> ncclResult_t {
            if (fd) *fd = dup(STDERR_FILENO);   // a real, closable fd
            return ncclSuccess;
        });
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [](hipMemGenericAllocationHandle_t*, void*,
           hipMemAllocationHandleType) -> hipError_t {
            return hipErrorInvalidValue;
        });

    auto r = CallImport();

    EXPECT_NE(r, ncclSuccess);
}

// cuMem POSIX_FD arm, closing the converted fd fails: the import succeeds but
// the SYSCHECK(close(fd)) refuses because the converted handle is not a valid
// fd, so the system error is propagated before mapping.
TEST_F(P2pShareableBufferMicrotest,
       Import_CuMemPosixFdCloseFails_Propagates)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook gran(g_hipMemGetAllocationGranularity, GranularitySucceeds());
    ScopedHook getFd(g_ncclProxyClientGetFdBlocking,
        [](struct ncclComm*, int, void*, int* fd) -> ncclResult_t {
            // Hand back a definitely-invalid fd so close() fails with EBADF.
            if (fd) *fd = 1 << 30;
            return ncclSuccess;
        });
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [](hipMemGenericAllocationHandle_t* h, void*,
           hipMemAllocationHandleType) -> hipError_t {
            if (h) *h = nullptr;
            return hipSuccess;
        });
    ScopedHook reserve(g_hipMemAddressReserve,
        [](void** p, std::size_t, std::size_t, void*,
           unsigned long long) -> hipError_t {
            ADD_FAILURE() << "reserve must not run after close failure";
            if (p) *p = nullptr;
            return hipSuccess;
        });

    auto r = CallImport();

    EXPECT_NE(r, ncclSuccess);
}

// cuMem arm, address-reserve / map / set-access each fail in turn: the import
// succeeds but a later mapping-stage driver call refuses, so the HIP error is
// propagated before the buffer is tracked. Parameterised over the three
// mapping stages so each CUCHECK error arm is driven.
enum class ImportMapStage { Reserve, Map, SetAccess };
class P2pImportMapStageFails
    : public P2pShareableBufferMicrotest,
      public ::testing::WithParamInterface<ImportMapStage> {};

TEST_P(P2pImportMapStageFails, CuMem_MappingStageRefuses_Propagates)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypeWin32);
    const auto stage = GetParam();

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook gran(g_hipMemGetAllocationGranularity, GranularitySucceeds());
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [](hipMemGenericAllocationHandle_t* h, void*,
           hipMemAllocationHandleType) -> hipError_t {
            if (h) *h = nullptr;
            return hipSuccess;
        });
    ScopedHook reserve(g_hipMemAddressReserve,
        [stage](void** p, std::size_t, std::size_t, void*,
                unsigned long long) -> hipError_t {
            if (stage == ImportMapStage::Reserve) return hipErrorInvalidValue;
            if (p) *p = reinterpret_cast<void*>(0xE0000);
            return hipSuccess;
        });
    ScopedHook map(g_hipMemMap,
        [stage](void*, std::size_t, std::size_t, hipMemGenericAllocationHandle_t,
                unsigned long long) -> hipError_t {
            return stage == ImportMapStage::Map ? hipErrorInvalidValue : hipSuccess;
        });
    ScopedHook access(g_hipMemSetAccess,
        [stage](void*, std::size_t, const hipMemAccessDesc*,
                std::size_t) -> hipError_t {
            return stage == ImportMapStage::SetAccess ? hipErrorInvalidValue
                                                      : hipSuccess;
        });
    ScopedHook track(g_memTrackImportFromPeer,
        [](struct ncclMemManager*, void*, size_t, hipMemGenericAllocationHandle_t,
           hipMemAllocationHandleType, ncclMemType_t, int, int,
           void*) -> ncclResult_t {
            ADD_FAILURE() << "buffer must not be tracked after a mapping-stage "
                             "failure";
            return ncclSuccess;
        });

    auto r = CallImport();

    EXPECT_NE(r, ncclSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    MappingStages, P2pImportMapStageFails,
    ::testing::Values(ImportMapStage::Reserve, ImportMapStage::Map,
                      ImportMapStage::SetAccess));

// cuMem arm, tracking fails: everything mapped successfully but the final
// ncclMemTrackImportFromPeer refuses; the error is propagated even though
// *devMemPtr was already written.
TEST_F(P2pShareableBufferMicrotest,
       Import_CuMemTrackImportFails_Propagates)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypeWin32);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook gran(g_hipMemGetAllocationGranularity, GranularitySucceeds());
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [](hipMemGenericAllocationHandle_t* h, void*,
           hipMemAllocationHandleType) -> hipError_t {
            if (h) *h = nullptr;
            return hipSuccess;
        });
    ScopedHook reserve(g_hipMemAddressReserve,
        [](void** p, std::size_t, std::size_t, void*,
           unsigned long long) -> hipError_t {
            if (p) *p = reinterpret_cast<void*>(0xE0000);
            return hipSuccess;
        });
    ScopedHook map(g_hipMemMap,
        [](void*, std::size_t, std::size_t, hipMemGenericAllocationHandle_t,
           unsigned long long) -> hipError_t { return hipSuccess; });
    ScopedHook access(g_hipMemSetAccess,
        [](void*, std::size_t, const hipMemAccessDesc*, std::size_t) -> hipError_t {
            return hipSuccess;
        });
    ScopedHook track(g_memTrackImportFromPeer,
        [](struct ncclMemManager*, void*, size_t, hipMemGenericAllocationHandle_t,
           hipMemAllocationHandleType, ncclMemType_t, int, int,
           void*) -> ncclResult_t { return ncclSystemError; });

    auto r = CallImport();

    EXPECT_EQ(r, ncclSystemError);
}

#endif  // ROCM_VERSION >= 70000

// Legacy-IPC alloc arm, buffer allocation fails: ncclCudaCalloc refuses, so
// the export handle is never queried and the error is propagated.
TEST_F(P2pShareableBufferMicrotest,
       Allocate_LegacyIpcBufferAllocFails_Propagates)
{
    // cuMemEnable stays 0 -> legacy arm.
    ScopedHook alloc(g_fakeCudaCallocAsync,
        [](void**, std::size_t, hipStream_t) -> ncclResult_t {
            return ncclSystemError;
        });
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t*, void*) -> hipError_t {
            ADD_FAILURE() << "export must not run after buffer alloc failure";
            return hipErrorInvalidValue;
        });

    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/0,
                                            &ipcDesc, &ptr);

    EXPECT_EQ(r, ncclSystemError);
}

#if ROCM_VERSION >= 70000

// cuMem alloc arm, export-to-peer marking fails: ncclDynMemMarkExportToPeer
// refuses, so the error is propagated before the handle is exported.
TEST_F(P2pShareableBufferMicrotest,
       Allocate_CuMemMarkExportToPeerFails_Propagates)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook mark(g_dynMemMarkExportToPeer,
        [](struct ncclMemManager*, void*, int) -> ncclResult_t {
            return ncclSystemError;
        });

    ncclMemManager* fakeManager = reinterpret_cast<ncclMemManager*>(0x1234);
    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/0,
                                            &ipcDesc, &ptr, /*peerRank=*/3,
                                            fakeManager, ncclMemScratch);

    EXPECT_EQ(r, ncclSystemError);
}

// cuMem alloc arm, non-POSIX export fails: hipMemExportToShareableHandle
// refuses, so the HIP error is propagated (no retain work runs).
TEST_F(P2pShareableBufferMicrotest,
       Allocate_CuMemExportHandleFails_Propagates)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypeWin32);  // non-POSIX

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [](void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
           unsigned long long) -> hipError_t { return hipErrorInvalidValue; });
    ScopedHook retain(g_hipMemRetainAllocationHandle,
        [](hipMemGenericAllocationHandle_t*, void*) -> hipError_t {
            ADD_FAILURE() << "retain must not run after export failure";
            return hipErrorInvalidValue;
        });

    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/2,
                                            &ipcDesc, &ptr);

    EXPECT_NE(r, ncclSuccess);
}

// cuMem alloc arm, handle retain fails: the export succeeds but a
// hipMemRetainAllocationHandle in the refcount loop refuses, propagating the
// HIP error.
TEST_F(P2pShareableBufferMicrotest,
       Allocate_CuMemRetainHandleFails_Propagates)
{
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);  // skip export

    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedHook retain(g_hipMemRetainAllocationHandle,
        [](hipMemGenericAllocationHandle_t*, void*) -> hipError_t {
            return hipErrorInvalidValue;
        });

    auto r = ncclP2pAllocateShareableBuffer(/*size=*/256, /*refcount=*/2,
                                            &ipcDesc, &ptr);

    EXPECT_NE(r, ncclSuccess);
}

#endif  // ROCM_VERSION >= 70000

// ===========================================================================
// Registration family: the public wrappers around ipcRegisterBuffer, the
// deferred graph-cleanup callback they enqueue, and the deregister wrapper.
//
//   ncclIpcLocalRegisterBuffer  -- validate args + record, delegate on valid.
//   ncclIpcGraphRegisterBuffer  -- delegate, then either enqueue a cleanup
//                                  callback (legacy vs plan queue) on success
//                                  or graph-deregister the record on failure.
//   cleanupIpc                  -- reached by draining the queue a successful
//                                  graph register populated (deregisters).
//   ncclIpcDeregBuffer          -- ship a deregister proxy message.
//
// Scenario list (each -> one test below):
//   1. Local: invalid args               -> no-op, outputs zeroed.
//   2. Local: record not locally valid   -> no-op, outputs zeroed.
//   3. Local: valid record               -> delegates; outputs are the reuse
//                                            arm's (offset + raw remote addr).
//   4. Graph: success, non-legacy        -> enqueues onto the plan queue and
//                                            bumps nCleanupQueueElts.
//   5. Graph: success, legacy IPC        -> enqueues onto legacyRegCleanupQueue,
//                                            leaves nCleanupQueueElts untouched.
//   6. Graph: register produced no reg    -> graph-deregisters the record.
//   7. Cleanup callback (via queue drain) -> graph-deregisters the record.
//   8. Deregister                         -> ships ncclProxyMsgDeregister with
//                                            the registration's impInfo payload.
//
// The reuse arm of ipcRegisterBuffer is the cheapest way to make a *successful*
// registration observable without the whole fresh-registration proxy dance:
// a record whose ipcInfos[peerIndex] is already populated returns regBufFlag=1
// with no driver/proxy traffic (see IpcRegisterBuffer_SendrecvReusesExistingIpcInfo).
// These tests reuse that to focus each assertion on the *wrapper's* own
// contract (arg gating, queue choice, dereg dispatch) rather than
// ipcRegisterBuffer's internals.
// ===========================================================================

namespace {

using CommCallbackQueue =
    ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>;

// RegFamilyReuseState -- a comm + reg record wired so ipcRegisterBuffer's
// reuse arm succeeds for a single peer. Owns the backing storage the wrappers
// and ipcRegisterBuffer read; must outlive the call.
struct RegFamilyReuseState {
    static constexpr int       kPeerRank      = 2;
    static constexpr int       kPeerLocalRank = 1;
    static constexpr uintptr_t kBegAddr       = 0x20000;
    static constexpr uintptr_t kBuffOffset    = 0x40;
    static constexpr uintptr_t kRmtRegAddr    = 0xB00000ull;

    CommBuilder cb;
    ncclReg regRecord{};
    IpcInfosBacking ipcInfosBacking{regRecord};
    ReusableIpcInfo reuse;
    std::array<ncclReg*, 1> cacheSlots{};

    const void* userbuff() const {
        return reinterpret_cast<const void*>(kBegAddr + kBuffOffset);
    }

    explicit RegFamilyReuseState(bool legacyIpcCap)
        : reuse(kPeerRank, kPeerLocalRank, kRmtRegAddr, legacyIpcCap)
    {
        cb.WithLocalRank(kPeerRank, kPeerLocalRank);
        regRecord.begAddr = kBegAddr;
        regRecord.endAddr = kBegAddr + 0x1000;
        reuse.InstallInto(regRecord);
        // ncclRegFind walks comm->regCache.slots; a single populated slot
        // whose [begAddr,endAddr) spans the buffer resolves to regRecord.
        cacheSlots[0] = &regRecord;
        cb.comm().regCache.slots      = cacheSlots.data();
        cb.comm().regCache.population = 1;
    }

    RegFamilyReuseState(const RegFamilyReuseState&)            = delete;
    RegFamilyReuseState& operator=(const RegFamilyReuseState&) = delete;
};

}  // namespace

class P2pRegisterFamilyMicrotest : public P2pMicrotest {};

// Scenario 1: each way the argument guard can fail (null comm, null userbuff,
// zero size, zero peers) makes the wrapper a no-op that never touches
// ncclRegFind / ipcRegisterBuffer and leaves the outputs zeroed.
TEST_F(P2pRegisterFamilyMicrotest, LocalRegister_InvalidArgs_IsNoopWithZeroedOutputs)
{
    RegFamilyReuseState st(/*legacyIpcCap=*/false);
    // Prove the guard short-circuits ahead of the delegate: a valid-record
    // hook that would otherwise steer into ipcRegisterBuffer must not run.
    ScopedHook valid(g_regLocalIsValid,
        [](struct ncclReg*, bool*) -> ncclResult_t {
            ADD_FAILURE() << "arg guard must short-circuit before ncclRegLocalIsValid";
            return ncclSystemError;
        });

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    ncclComm* const comm = &st.cb.comm();
    const void* const buff = st.userbuff();

    struct Case { ncclComm* comm; const void* buff; size_t size; int nPeers; };
    const Case cases[] = {
        {nullptr, buff,    256, 1},   // null comm
        {comm,    nullptr, 256, 1},   // null userbuff
        {comm,    buff,      0, 1},   // zero buffSize
        {comm,    buff,    256, 0},   // zero nPeers
    };
    for (const auto& c : cases) {
        IpcRegOutputs out;
        auto r = ncclIpcLocalRegisterBuffer(c.comm, c.buff, c.size, peerRanks,
                                            c.nPeers, NCCL_IPC_SENDRECV,
                                            &out.regBufFlag, &out.offsetOut,
                                            &out.peerRmtAddrs);
        EXPECT_EQ(r, ncclSuccess);
        out.ExpectZeroed();
    }
}

// Scenario 2: the record is found but reports not-locally-valid, so the
// wrapper skips the delegate and leaves the outputs zeroed.
TEST_F(P2pRegisterFamilyMicrotest, LocalRegister_RecordNotValid_IsNoopWithZeroedOutputs)
{
    RegFamilyReuseState st(/*legacyIpcCap=*/false);
    // Default g_regLocalIsValid already reports false; make it explicit and
    // assert ipcRegisterBuffer is never reached by failing loudly if the
    // reuse arm's remote address surfaces.
    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    IpcRegOutputs out;

    auto r = ncclIpcLocalRegisterBuffer(&st.cb.comm(), st.userbuff(),
                                        /*buffSize=*/256, peerRanks, /*nPeers=*/1,
                                        NCCL_IPC_SENDRECV, &out.regBufFlag,
                                        &out.offsetOut, &out.peerRmtAddrs);

    EXPECT_EQ(r, ncclSuccess);
    out.ExpectZeroed();
}

// Scenario 3: a valid record delegates to ipcRegisterBuffer; the wrapper
// returns that call's outputs -- the buffer offset and the peer's raw remote
// address from the reuse arm.
TEST_F(P2pRegisterFamilyMicrotest, LocalRegister_ValidRecord_DelegatesAndReturnsRemoteAddr)
{
    RegFamilyReuseState st(/*legacyIpcCap=*/true);
    ScopedHook valid(g_regLocalIsValid,
        [](struct ncclReg*, bool* isValid) -> ncclResult_t {
            if (isValid) *isValid = true;
            return ncclSuccess;
        });

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    IpcRegOutputs out;

    auto r = ncclIpcLocalRegisterBuffer(&st.cb.comm(), st.userbuff(),
                                        /*buffSize=*/256, peerRanks, /*nPeers=*/1,
                                        NCCL_IPC_SENDRECV, &out.regBufFlag,
                                        &out.offsetOut, &out.peerRmtAddrs);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    EXPECT_EQ(out.offsetOut,  RegFamilyReuseState::kBuffOffset);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(out.peerRmtAddrs),
              RegFamilyReuseState::kRmtRegAddr);
}

// Scenario 3b: a delegate failure propagates and the wrapper zeroes its
// outputs on the fail path (the sentinel-seeded outputs must not survive).
TEST_F(P2pRegisterFamilyMicrotest, LocalRegister_DelegateFails_PropagatesAndZeroesOutputs)
{
    RegFamilyReuseState st(/*legacyIpcCap=*/false);
    // ncclRegLocalIsValid failing is the cheapest way to make the delegate
    // arm return an error before ipcRegisterBuffer.
    ScopedHook valid(g_regLocalIsValid,
        [](struct ncclReg*, bool*) -> ncclResult_t { return ncclSystemError; });

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    IpcRegOutputs out;

    auto r = ncclIpcLocalRegisterBuffer(&st.cb.comm(), st.userbuff(),
                                        /*buffSize=*/256, peerRanks, /*nPeers=*/1,
                                        NCCL_IPC_SENDRECV, &out.regBufFlag,
                                        &out.offsetOut, &out.peerRmtAddrs);

    EXPECT_EQ(r, ncclSystemError);
    out.ExpectZeroed();
}

// GraphRegisterState -- drives ncclIpcGraphRegisterBuffer's success path. It
// wires ncclCuMemGetAddressRange (via the hipMemGetAddressRange seam) to a
// single-segment range and ncclCommGraphRegister to hand back a reuse-armed
// record, so the delegate returns regBufFlag=1 and the wrapper's own
// queue-bookkeeping is what's left to observe.
namespace {
struct GraphRegisterState {
    RegFamilyReuseState reuse;
    std::optional<ScopedHook<hipError_t(hipDeviceptr_t*, std::size_t*, hipDeviceptr_t)>> memGet;
    std::optional<ScopedHook<ncclResult_t(struct ncclComm*, void*, size_t, void**)>> graphReg;

    explicit GraphRegisterState(bool legacyIpcCap) : reuse(legacyIpcCap) {
        // ncclCuMemGetAddressRange loops until it spans the buffer; one
        // segment starting at userbuff and sized to the request covers it.
        memGet.emplace(
            g_hipMemGetAddressRange,
            [](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t dptr) -> hipError_t {
                if (pbase) *pbase = dptr;
                if (psize) *psize = 256;
                return hipSuccess;
            });
        // ncclCommGraphRegister hands back the reuse-armed record so the
        // delegate's reuse arm succeeds.
        ncclReg* rec = &reuse.regRecord;
        graphReg.emplace(
            g_commGraphRegister,
            [rec](struct ncclComm*, void*, size_t, void** handle) -> ncclResult_t {
                if (handle) *handle = rec;
                return ncclSuccess;
            });
    }
};
}  // namespace

#if ROCM_VERSION >= 70000
// Scenario 4: a successful non-legacy registration enqueues the cleanup
// callback onto the caller's plan queue and bumps the element count.
TEST_F(P2pRegisterFamilyMicrotest, GraphRegister_NonLegacySuccess_EnqueuesPlanCleanupAndCountsIt)
{
    GraphRegisterState st(/*legacyIpcCap=*/false);
    CommCallbackQueue cleanupQueue;
    ncclIntruQueueConstruct(&cleanupQueue);
    int nCleanupQueueElts = 0;

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    IpcRegOutputs out;

    auto r = ncclIpcGraphRegisterBuffer(&st.reuse.cb.comm(), st.reuse.userbuff(),
                                        /*buffSize=*/256, peerRanks, /*nPeers=*/1,
                                        NCCL_IPC_SENDRECV, &out.regBufFlag,
                                        &out.offsetOut, &out.peerRmtAddrs,
                                        &cleanupQueue, &nCleanupQueueElts);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    // The plan queue received exactly one callback; the legacy queue stayed empty.
    ASSERT_FALSE(ncclIntruQueueEmpty(&cleanupQueue));
    EXPECT_TRUE(ncclIntruQueueEmpty(&st.reuse.cb.comm().legacyRegCleanupQueue));
    EXPECT_EQ(nCleanupQueueElts, 1);

    // Drain and free the callback so it doesn't leak (its fn is cleanupIpc,
    // which frees the record; here we just release the malloc'd wrapper).
    std::free(ncclIntruQueueDequeue(&cleanupQueue));
}

// Scenario 4b: the same non-legacy success path is safe when the caller
// passes a null nCleanupQueueElts -- the callback still lands on the plan
// queue and the count-bump is simply skipped.
TEST_F(P2pRegisterFamilyMicrotest, GraphRegister_NonLegacySuccessNullCount_EnqueuesWithoutCounting)
{
    GraphRegisterState st(/*legacyIpcCap=*/false);
    CommCallbackQueue cleanupQueue;
    ncclIntruQueueConstruct(&cleanupQueue);

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    IpcRegOutputs out;

    auto r = ncclIpcGraphRegisterBuffer(&st.reuse.cb.comm(), st.reuse.userbuff(),
                                        /*buffSize=*/256, peerRanks, /*nPeers=*/1,
                                        NCCL_IPC_SENDRECV, &out.regBufFlag,
                                        &out.offsetOut, &out.peerRmtAddrs,
                                        &cleanupQueue, /*nCleanupQueueElts=*/nullptr);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    ASSERT_FALSE(ncclIntruQueueEmpty(&cleanupQueue));
    std::free(ncclIntruQueueDequeue(&cleanupQueue));
}

#endif  // ROCM_VERSION >= 70000
// Scenario 4c: the same argument guard as the local wrapper -- null comm,
// null userbuff, zero size, or zero peers make graph-register a no-op that
// never reaches the address-range / graph-register seams.
TEST_F(P2pRegisterFamilyMicrotest, GraphRegister_InvalidArgs_IsNoopWithZeroedOutputs)
{
    GraphRegisterState st(/*legacyIpcCap=*/false);
    CommCallbackQueue cleanupQueue;
    ncclIntruQueueConstruct(&cleanupQueue);
    int nCleanupQueueElts = 0;

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    ncclComm* const comm = &st.reuse.cb.comm();
    const void* const buff = st.reuse.userbuff();

    struct Case { ncclComm* comm; const void* buff; size_t size; int nPeers; };
    const Case cases[] = {
        {nullptr, buff,    256, 1},
        {comm,    nullptr, 256, 1},
        {comm,    buff,      0, 1},
        {comm,    buff,    256, 0},
    };
    for (const auto& c : cases) {
        IpcRegOutputs out;
        auto r = ncclIpcGraphRegisterBuffer(c.comm, c.buff, c.size, peerRanks,
                                            c.nPeers, NCCL_IPC_SENDRECV,
                                            &out.regBufFlag, &out.offsetOut,
                                            &out.peerRmtAddrs, &cleanupQueue,
                                            &nCleanupQueueElts);
        EXPECT_EQ(r, ncclSuccess);
        out.ExpectZeroed();
    }
    EXPECT_TRUE(ncclIntruQueueEmpty(&cleanupQueue));
    EXPECT_EQ(nCleanupQueueElts, 0);
}

#if ROCM_VERSION >= 70000
// Scenario 5: a successful legacy-IPC registration routes the cleanup callback
// onto comm->legacyRegCleanupQueue and leaves the plan-queue count untouched.
TEST_F(P2pRegisterFamilyMicrotest, GraphRegister_LegacySuccess_EnqueuesLegacyCleanupWithoutCounting)
{
    GraphRegisterState st(/*legacyIpcCap=*/true);
    CommCallbackQueue cleanupQueue;
    ncclIntruQueueConstruct(&cleanupQueue);
    int nCleanupQueueElts = 0;

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    IpcRegOutputs out;

    auto r = ncclIpcGraphRegisterBuffer(&st.reuse.cb.comm(), st.reuse.userbuff(),
                                        /*buffSize=*/256, peerRanks, /*nPeers=*/1,
                                        NCCL_IPC_SENDRECV, &out.regBufFlag,
                                        &out.offsetOut, &out.peerRmtAddrs,
                                        &cleanupQueue, &nCleanupQueueElts);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(out.regBufFlag, 1);
    ASSERT_FALSE(ncclIntruQueueEmpty(&st.reuse.cb.comm().legacyRegCleanupQueue));
    EXPECT_TRUE(ncclIntruQueueEmpty(&cleanupQueue));
    EXPECT_EQ(nCleanupQueueElts, 0);

    std::free(ncclIntruQueueDequeue(&st.reuse.cb.comm().legacyRegCleanupQueue));
}

// Scenario 6: when the delegate registers nothing (regBufFlag stays 0), the
// wrapper graph-deregisters the record it just created rather than enqueuing
// a cleanup callback.
TEST_F(P2pRegisterFamilyMicrotest, GraphRegister_NoRegistration_DeregistersRecord)
{
    // A fresh (not reuse-armed) record: ipcRegisterBuffer walks its
    // fresh-registration arm, and with every driver/proxy seam at its
    // fail-loud default it produces no registration (regBufFlag == 0)
    // without reaching them (the cuMem/legacy arms are gated off).
    CommBuilder cb;
    cb.WithLocalRank(RegFamilyReuseState::kPeerRank,
                     RegFamilyReuseState::kPeerLocalRank)
      .WithMaxLocalRanks()
      .WithSharedRes()
      .WithProxyConnArray(RegFamilyReuseState::kPeerRank + 1);
    cb.comm().gproxyConn[RegFamilyReuseState::kPeerRank].initialized = true;
    ncclReg regRecord{};
    IpcInfosBacking ipcInfosBacking{regRecord};
    regRecord.begAddr = RegFamilyReuseState::kBegAddr;
    regRecord.endAddr = RegFamilyReuseState::kBegAddr + 0x1000;

    ncclReg* rec = &regRecord;
    ScopedHook memGet(g_hipMemGetAddressRange,
        [](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t dptr) -> hipError_t {
            if (pbase) *pbase = dptr;
            if (psize) *psize = 256;
            return hipSuccess;
        });
    ScopedHook graphReg(g_commGraphRegister,
        [rec](struct ncclComm*, void*, size_t, void** handle) -> ncclResult_t {
            if (handle) *handle = rec;
            return ncclSuccess;
        });
    // The failure branch deregisters exactly the record graph-register handed back.
    int deregCalls = 0;
    const ncclReg* deregArg = nullptr;
    ScopedHook dereg(g_commGraphDeregister,
        [&](struct ncclComm*, struct ncclReg* reg) -> ncclResult_t {
            ++deregCalls;
            deregArg = reg;
            return ncclSuccess;
        });

    CommCallbackQueue cleanupQueue;
    ncclIntruQueueConstruct(&cleanupQueue);
    int nCleanupQueueElts = 0;

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    IpcRegOutputs out;

    auto r = ncclIpcGraphRegisterBuffer(&cb.comm(),
                                        reinterpret_cast<const void*>(
                                            RegFamilyReuseState::kBegAddr +
                                            RegFamilyReuseState::kBuffOffset),
                                        /*buffSize=*/256, peerRanks, /*nPeers=*/1,
                                        NCCL_IPC_SENDRECV, &out.regBufFlag,
                                        &out.offsetOut, &out.peerRmtAddrs,
                                        &cleanupQueue, &nCleanupQueueElts);

    EXPECT_EQ(r, ncclSuccess);
    out.ExpectZeroed();
    EXPECT_EQ(deregCalls, 1);
    EXPECT_EQ(deregArg, rec);
    EXPECT_TRUE(ncclIntruQueueEmpty(&cleanupQueue));
    EXPECT_EQ(nCleanupQueueElts, 0);
}

// Scenario 7: the cleanup callback a successful graph register enqueues,
// invoked as the queue drain would, graph-deregisters the record it holds.
TEST_F(P2pRegisterFamilyMicrotest, GraphCleanupCallback_WhenDrained_DeregistersHeldRecord)
{
    GraphRegisterState st(/*legacyIpcCap=*/false);
    CommCallbackQueue cleanupQueue;
    ncclIntruQueueConstruct(&cleanupQueue);
    int nCleanupQueueElts = 0;

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    IpcRegOutputs out;

    ASSERT_EQ(ncclIpcGraphRegisterBuffer(&st.reuse.cb.comm(), st.reuse.userbuff(),
                                         /*buffSize=*/256, peerRanks, /*nPeers=*/1,
                                         NCCL_IPC_SENDRECV, &out.regBufFlag,
                                         &out.offsetOut, &out.peerRmtAddrs,
                                         &cleanupQueue, &nCleanupQueueElts),
              ncclSuccess);
    ASSERT_EQ(out.regBufFlag, 1);

    // Observe the deregister the callback issues when the queue is drained.
    int deregCalls = 0;
    const ncclReg* deregArg = nullptr;
    ScopedHook dereg(g_commGraphDeregister,
        [&](struct ncclComm*, struct ncclReg* reg) -> ncclResult_t {
            ++deregCalls;
            deregArg = reg;
            return ncclSuccess;
        });

    struct ncclCommCallback* cb = ncclIntruQueueDequeue(&cleanupQueue);
    ASSERT_NE(cb, nullptr);
    EXPECT_EQ(cb->fn(&st.reuse.cb.comm(), cb), ncclSuccess);  // frees cb

    EXPECT_EQ(deregCalls, 1);
    EXPECT_EQ(deregArg, &st.reuse.regRecord);
}

// Scenario 7b: if the deregister the cleanup callback issues fails, the
// callback propagates that error out of the queue drain.
TEST_F(P2pRegisterFamilyMicrotest, GraphCleanupCallback_DeregisterFails_Propagates)
{
    GraphRegisterState st(/*legacyIpcCap=*/false);
    CommCallbackQueue cleanupQueue;
    ncclIntruQueueConstruct(&cleanupQueue);
    int nCleanupQueueElts = 0;

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    IpcRegOutputs out;

    ASSERT_EQ(ncclIpcGraphRegisterBuffer(&st.reuse.cb.comm(), st.reuse.userbuff(),
                                         /*buffSize=*/256, peerRanks, /*nPeers=*/1,
                                         NCCL_IPC_SENDRECV, &out.regBufFlag,
                                         &out.offsetOut, &out.peerRmtAddrs,
                                         &cleanupQueue, &nCleanupQueueElts),
              ncclSuccess);
    ASSERT_EQ(out.regBufFlag, 1);

    ScopedHook dereg(g_commGraphDeregister,
        [](struct ncclComm*, struct ncclReg*) -> ncclResult_t {
            return ncclSystemError;
        });

    struct ncclCommCallback* cb = ncclIntruQueueDequeue(&cleanupQueue);
    ASSERT_NE(cb, nullptr);
    // On the failure arm cleanupIpc's NCCLCHECK returns before free(obj)
    // (p2p.cc:1384), so the dequeued record is not reclaimed -- free it here to
    // keep LeakSanitizer quiet.
    EXPECT_EQ(cb->fn(&st.reuse.cb.comm(), cb), ncclSystemError);
    std::free(cb);
}

// Scenario 6b: a graph-register failure propagates and the wrapper zeroes
// its outputs on the fail path.
TEST_F(P2pRegisterFamilyMicrotest, GraphRegister_GraphRegisterFails_PropagatesAndZeroesOutputs)
{
    GraphRegisterState st(/*legacyIpcCap=*/false);
    // Override the success graph-register hook with a failing one.
    ScopedHook graphReg(g_commGraphRegister,
        [](struct ncclComm*, void*, size_t, void** handle) -> ncclResult_t {
            if (handle) *handle = nullptr;
            return ncclSystemError;
        });

    CommCallbackQueue cleanupQueue;
    ncclIntruQueueConstruct(&cleanupQueue);
    int nCleanupQueueElts = 0;

    int peerRanks[] = {RegFamilyReuseState::kPeerRank};
    IpcRegOutputs out;

    auto r = ncclIpcGraphRegisterBuffer(&st.reuse.cb.comm(), st.reuse.userbuff(),
                                        /*buffSize=*/256, peerRanks, /*nPeers=*/1,
                                        NCCL_IPC_SENDRECV, &out.regBufFlag,
                                        &out.offsetOut, &out.peerRmtAddrs,
                                        &cleanupQueue, &nCleanupQueueElts);

    EXPECT_EQ(r, ncclSystemError);
    out.ExpectZeroed();
    EXPECT_TRUE(ncclIntruQueueEmpty(&cleanupQueue));
    EXPECT_EQ(nCleanupQueueElts, 0);
}

#endif  // ROCM_VERSION >= 70000
// Scenario 8: deregister ships a blocking ncclProxyMsgDeregister carrying the
// registration's impInfo payload to the record's proxy connector.
TEST_F(P2pRegisterFamilyMicrotest, Deregister_ShipsDeregisterMessageWithImpInfoPayload)
{
    auto commStorage = std::make_unique<ncclComm>();  // ~3.8 MB: heap, not stack-safe
    ncclComm& comm = *commStorage;
    ncclProxyConnector proxyConn{};
    ncclIpcRegInfo regInfo{};
    regInfo.peerRank                = 3;
    regInfo.baseAddr                = reinterpret_cast<void*>(0x50000);
    regInfo.ipcProxyconn            = &proxyConn;
    regInfo.impInfo.rmtRegAddr      = reinterpret_cast<void*>(0xABCD0000ull);

    int   msgType     = -1;
    void* msgReq      = nullptr;
    int   msgReqSize  = -1;
    const ncclProxyConnector* msgConn = nullptr;
    ScopedHook proxy(g_proxyCallBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector* pc, int type,
            void* req, int reqSize, void*, int) -> ncclResult_t {
            msgConn    = pc;
            msgType    = type;
            msgReq     = req;
            msgReqSize = reqSize;
            return ncclSuccess;
        });

    auto r = ncclIpcDeregBuffer(&comm, &regInfo);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(msgConn, &proxyConn);
    EXPECT_EQ(msgType, ncclProxyMsgDeregister);
    EXPECT_EQ(msgReq, &regInfo.impInfo);
    EXPECT_EQ(msgReqSize, static_cast<int>(sizeof(struct ncclIpcImpInfo)));
}

// Scenario 8b: when the deregister proxy message fails, the error is
// propagated rather than swallowed.
TEST_F(P2pRegisterFamilyMicrotest, Deregister_ProxyMessageFails_Propagates)
{
    auto commStorage = std::make_unique<ncclComm>();  // ~3.8 MB: heap, not stack-safe
    ncclComm& comm = *commStorage;
    ncclProxyConnector proxyConn{};
    ncclIpcRegInfo regInfo{};
    regInfo.peerRank     = 3;
    regInfo.ipcProxyconn = &proxyConn;

    ScopedHook proxy(g_proxyCallBlocking,
        [](struct ncclComm*, struct ncclProxyConnector*, int, void*, int,
           void*, int) -> ncclResult_t { return ncclSystemError; });

    EXPECT_EQ(ncclIpcDeregBuffer(&comm, &regInfo), ncclSystemError);
}

// ===========================================================================
// Multi-segment registration: ipcHandleMultiSegmentRegistration walks a
// buffer whose mapped physical allocation spans several segments, retaining
// one handle per segment and populating a p2pIpcExpInfo array that grows
// (realloc) past its initial two-slot capacity as the segment count climbs.
//
// The unit's own contract:
//   - one segment per physical allocation range cuMemGetAddressRange reports,
//     with numSegments and totalMappedBufferSize summing them;
//   - the p2pIpcExpInfo array grows to hold every segment (the realloc arm);
//   - each retained allocation handle is released again -- on the success
//     path (final release loop) and on the cleanup path when the segment
//     count exceeds NCCL_P2P_MAX_PHYSICAL_SEGMENTS, so retains and releases
//     always balance.
//
// These drive the sameProcess + non-POSIX handle type combination so the
// export / batch-fd-query sub-arms stay out of scope: each segment simply
// retains a handle, memcpy's it into the ipcInfo, and is released. The
// NCCL_P2P_MAX_PHYSICAL_SEGMENTS constant is 8192 (src/transport/p2p.cc).
// ===========================================================================

namespace {

// SegmentRangeEmulator -- stands in for cuMemGetAddressRange
// (hipMemGetAddressRange). Reports fixed-size contiguous physical segments
// starting at each queried address, so a buffer of size N*segSize resolves to
// exactly N segments. Also counts retain/release calls so a test can assert
// they balance.
struct SegmentRangeEmulator {
    std::size_t segSize;
    int retainCalls  = 0;
    int releaseCalls = 0;

    std::optional<ScopedHook<hipError_t(hipDeviceptr_t*, std::size_t*, hipDeviceptr_t)>> memGet;
    std::optional<ScopedHook<hipError_t(hipMemGenericAllocationHandle_t*, void*)>> retain;
    std::optional<ScopedHook<hipError_t(hipMemGenericAllocationHandle_t)>> release;

    explicit SegmentRangeEmulator(std::size_t segSize_) : segSize(segSize_) {
        std::size_t s = segSize;
        memGet.emplace(
            g_hipMemGetAddressRange,
            [s](hipDeviceptr_t* pbase, std::size_t* psize, hipDeviceptr_t dptr) -> hipError_t {
                if (pbase) *pbase = dptr;      // segment starts at the query address
                if (psize) *psize = s;         // fixed physical span
                return hipSuccess;
            });
        retain.emplace(
            g_hipMemRetainAllocationHandle,
            [this](hipMemGenericAllocationHandle_t* h, void* addr) -> hipError_t {
                ++retainCalls;
                if (h) *h = reinterpret_cast<hipMemGenericAllocationHandle_t>(addr);
                return hipSuccess;
            });
        release.emplace(
            g_hipMemRelease,
            [this](hipMemGenericAllocationHandle_t) -> hipError_t {
                ++releaseCalls;
                return hipSuccess;
            });
    }
};

}  // namespace

class P2pMultiSegmentMicrotest : public P2pMicrotest {
protected:
    void SetUp() override {
        P2pMicrotest::SetUp();
        // Drive the sameProcess + non-POSIX handle-type combination so the
        // shareable-handle export and batch-fd-query sub-arms are skipped;
        // this isolates the segment-walk / grow / retain-release contract.
        saved_handle_type_ = ncclCuMemHandleType;
        ncclCuMemHandleType = hipMemHandleTypeWin32;  // anything != POSIX_FD
        proxyConn_.sameProcess = 1;
    }
    void TearDown() override {
        ncclCuMemHandleType = saved_handle_type_;
        P2pMicrotest::TearDown();
    }

    hipMemAllocationHandleType saved_handle_type_{};
    ncclComm            comm_{};
    ncclProxyConnector  proxyConn_{};

    // Every walk test issues the same call, differing only in the user-buffer
    // size; hoist the declare-and-call preamble behind this accessor and
    // destructure the outputs with a structured binding at the call site.
    struct WalkOutputs {
        ncclResult_t   r;
        size_t         totalMappedBufferSize;
        int            numSegments;
        p2pIpcExpInfo* ipcInfos;
    };
    WalkOutputs CallWalk(size_t userBuffSize)
    {
        const hipDeviceptr_t userBuff = reinterpret_cast<hipDeviceptr_t>(0x100000);
        WalkOutputs o{ncclSuccess, 0, 0, nullptr};
        o.r = ipcHandleMultiSegmentRegistration(userBuff, userBuffSize, &comm_,
                                                &proxyConn_, &o.totalMappedBufferSize,
                                                &o.numSegments, &o.ipcInfos);
        return o;
    }
};

// A buffer spanning more segments than the initial two-slot capacity walks
// the realloc arm: every segment is recorded, numSegments and
// totalMappedBufferSize sum them, and each retained handle is released again.
TEST_F(P2pMultiSegmentMicrotest, Walk_BufferSpansManySegments_GrowsArrayAndBalancesHandles)
{
    constexpr std::size_t kSegSize     = 0x1000;
    constexpr int         kNumSegments = 5;   // > initial capacity 2 -> realloc
    SegmentRangeEmulator seg(kSegSize);

    const size_t userBuffSize = kSegSize * kNumSegments;
    auto [r, totalMappedBufferSize, numSegments, ipcInfos] = CallWalk(userBuffSize);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(numSegments, kNumSegments);
    EXPECT_EQ(totalMappedBufferSize, userBuffSize);
    ASSERT_NE(ipcInfos, nullptr);
    // The array grew to hold every segment; each slot carries that segment's
    // physical size.
    for (int i = 0; i < numSegments; ++i) {
        EXPECT_EQ(ipcInfos[i].size, kSegSize);
    }
    // Every retained handle was released again on the success path.
    EXPECT_EQ(seg.retainCalls, kNumSegments);
    EXPECT_EQ(seg.releaseCalls, kNumSegments);

    std::free(ipcInfos);
}

// A buffer that resolves to more than NCCL_P2P_MAX_PHYSICAL_SEGMENTS segments
// fails, and the cleanup path releases every handle retained up to the point
// of failure -- retains and releases still balance and the ipcInfos array is
// freed (left null).
TEST_F(P2pMultiSegmentMicrotest, Walk_ExceedsMaxSegments_FailsAndReleasesRetainedHandles)
{
    constexpr int kMaxSegments = 8192;  // NCCL_P2P_MAX_PHYSICAL_SEGMENTS
    constexpr std::size_t kSegSize = 1;
    // One segment past the ceiling triggers the failure branch.
    const int kNumSegments = kMaxSegments + 1;
    SegmentRangeEmulator seg(kSegSize);

    const size_t userBuffSize = kSegSize * kNumSegments;
    auto [r, totalMappedBufferSize, numSegments, ipcInfos] = CallWalk(userBuffSize);

    EXPECT_EQ(r, ncclInternalError);
    // The array is freed on the cleanup path.
    EXPECT_EQ(ipcInfos, nullptr);
    // Every handle retained before the ceiling was hit is released again.
    EXPECT_EQ(seg.releaseCalls, seg.retainCalls);
    EXPECT_EQ(seg.retainCalls, kNumSegments);
}

// Cross-process registration with a POSIX-fd handle type exports every
// segment as a shareable fd, batch-queries the remote proxy for the imported
// fds, and stores each returned fd on its segment's ipcInfo.
TEST_F(P2pMultiSegmentMicrotest, Walk_PosixFdCrossProcess_ExportsFdsAndStoresImportedFds)
{
    ncclCuMemHandleType    = hipMemHandleTypePosixFileDescriptor;
    proxyConn_.sameProcess = 0;   // cross-process arm

    constexpr std::size_t kSegSize     = 0x1000;
    constexpr int         kNumSegments = 3;   // > initial capacity 2 -> realloc
    SegmentRangeEmulator seg(kSegSize);

    // Each export hands back a real, closeable fd (production close()s the
    // exported fd once the batch query has consumed it).
    int exportCalls = 0;
    std::vector<int> exportedFds;
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [&](void* shareableHandle, hipMemGenericAllocationHandle_t,
            hipMemAllocationHandleType type,
            unsigned long long) -> hipError_t {
            ++exportCalls;
            EXPECT_EQ(type, hipMemHandleTypePosixFileDescriptor);
            if (shareableHandle) {
                int fd = ::dup(0);
                *static_cast<int*>(shareableHandle) = fd;
                exportedFds.push_back(fd);
            }
            return hipSuccess;
        });

    // The batch query returns a recognisable imported fd per segment.
    constexpr int kImpFdBase = 1000;
    int   batchCalls    = 0;
    int   batchSegments = -1;
    ScopedHook batch(g_proxyClientBatchQueryFdBlocking,
        [&](struct ncclComm*, struct ncclProxyConnector*, int*, int* rmtFds,
            int numSegments) -> ncclResult_t {
            ++batchCalls;
            batchSegments = numSegments;
            for (int i = 0; i < numSegments; ++i) rmtFds[i] = kImpFdBase + i;
            return ncclSuccess;
        });

    const size_t userBuffSize = kSegSize * kNumSegments;
    auto [r, totalMappedBufferSize, numSegments, ipcInfos] = CallWalk(userBuffSize);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(numSegments, kNumSegments);
    EXPECT_EQ(exportCalls, kNumSegments);      // one export per segment
    EXPECT_EQ(batchCalls, 1);                  // a single batched fd query
    EXPECT_EQ(batchSegments, kNumSegments);
    ASSERT_NE(ipcInfos, nullptr);
    // Each segment carries the imported fd the proxy handed back.
    for (int i = 0; i < numSegments; ++i) {
        EXPECT_EQ(ipcInfos[i].impFd, kImpFdBase + i);
    }
    EXPECT_EQ(seg.retainCalls, kNumSegments);
    EXPECT_EQ(seg.releaseCalls, kNumSegments);
    // Production close()s each exported fd once the batch query has consumed it
    // (p2p.cc); a second close of each must fail with EBADF.
    EXPECT_EQ(static_cast<int>(exportedFds.size()), kNumSegments);
    for (int fd : exportedFds) {
        EXPECT_EQ(::close(fd), -1);
        EXPECT_EQ(errno, EBADF);
    }

    std::free(ipcInfos);
}

// Cross-process registration with a fabric (non-POSIX) handle type exports
// each segment's handle straight into that segment's ipcDesc; there is no
// fd batch-query, and every retained handle is still released.
TEST_F(P2pMultiSegmentMicrotest, Walk_FabricCrossProcess_ExportsHandlesIntoIpcDesc)
{
    // SetUp() already selected the fabric handle type; make it cross-process.
    proxyConn_.sameProcess = 0;

    constexpr std::size_t kSegSize     = 0x1000;
    constexpr int         kNumSegments = 3;
    SegmentRangeEmulator seg(kSegSize);

    int exportCalls = 0;
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [&exportCalls](void* shareableHandle, hipMemGenericAllocationHandle_t,
                       hipMemAllocationHandleType type,
                       unsigned long long) -> hipError_t {
            ++exportCalls;
            EXPECT_NE(type, hipMemHandleTypePosixFileDescriptor);
            if (shareableHandle) std::memset(shareableHandle, 0x5A, 4);
            return hipSuccess;
        });
    // The batch fd-query must not run on the non-POSIX arm.
    ScopedHook batch(g_proxyClientBatchQueryFdBlocking,
        [](struct ncclComm*, struct ncclProxyConnector*, int*, int*, int)
            -> ncclResult_t {
            ADD_FAILURE() << "non-POSIX arm must not batch-query fds";
            return ncclSystemError;
        });

    const size_t userBuffSize = kSegSize * kNumSegments;
    auto [r, totalMappedBufferSize, numSegments, ipcInfos] = CallWalk(userBuffSize);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(numSegments, kNumSegments);
    EXPECT_EQ(exportCalls, kNumSegments);
    ASSERT_NE(ipcInfos, nullptr);
    // The exported descriptor landed in each segment's cuDesc.handle.
    for (int i = 0; i < numSegments; ++i) {
        unsigned char* bytes =
            reinterpret_cast<unsigned char*>(&ipcInfos[i].ipcDesc.cuDesc.handle);
        EXPECT_EQ(bytes[0], 0x5A);
    }
    EXPECT_EQ(seg.retainCalls, kNumSegments);
    EXPECT_EQ(seg.releaseCalls, kNumSegments);

    std::free(ipcInfos);
}

// A POSIX-fd, cross-process registration whose batch fd-query fails takes the
// cleanup path: the exported fds are closed, every retained handle released,
// and the ipcInfos array freed.
TEST_F(P2pMultiSegmentMicrotest, Walk_PosixFdBatchQueryFails_ClosesFdsAndReleasesHandles)
{
    ncclCuMemHandleType    = hipMemHandleTypePosixFileDescriptor;
    proxyConn_.sameProcess = 0;

    constexpr std::size_t kSegSize     = 0x1000;
    constexpr int         kNumSegments = 3;
    SegmentRangeEmulator seg(kSegSize);

    std::vector<int> exportedFds;
    ScopedHook xport(g_hipMemExportToShareableHandle,
        [&](void* shareableHandle, hipMemGenericAllocationHandle_t,
            hipMemAllocationHandleType, unsigned long long) -> hipError_t {
            if (shareableHandle) {
                int fd = ::dup(0);
                *static_cast<int*>(shareableHandle) = fd;
                exportedFds.push_back(fd);
            }
            return hipSuccess;
        });
    ScopedHook batch(g_proxyClientBatchQueryFdBlocking,
        [](struct ncclComm*, struct ncclProxyConnector*, int*, int*, int)
            -> ncclResult_t {
            return ncclSystemError;   // force the cleanup path
        });

    const size_t userBuffSize = kSegSize * kNumSegments;
    auto [r, totalMappedBufferSize, numSegments, ipcInfos] = CallWalk(userBuffSize);

    EXPECT_EQ(r, ncclSystemError);
    EXPECT_EQ(ipcInfos, nullptr);          // freed on the cleanup path
    // All segments were walked before the query failed; each retained handle
    // is released again on cleanup.
    EXPECT_EQ(numSegments, kNumSegments);
    EXPECT_EQ(seg.releaseCalls, seg.retainCalls);
    EXPECT_EQ(seg.retainCalls, kNumSegments);
    // The cleanup path close()s each exported fd (p2p.cc); a second close of
    // each must fail with EBADF.
    EXPECT_EQ(static_cast<int>(exportedFds.size()), kNumSegments);
    for (int fd : exportedFds) {
        EXPECT_EQ(::close(fd), -1);
        EXPECT_EQ(errno, EBADF);
    }
}

// The very first bookkeeping allocation (the ipcInfos array) failing aborts
// the walk before any segment is retained, propagating the error and leaving
// no handles outstanding.
TEST_F(P2pMultiSegmentMicrotest, Walk_IpcInfosAllocFails_PropagatesWithoutRetaining)
{
    constexpr std::size_t kSegSize     = 0x1000;
    constexpr int         kNumSegments = 3;
    SegmentRangeEmulator seg(kSegSize);

    // capacity starts at 2; the ipcInfos array is the first ncclCalloc.
    SetP2pCallocFailSize(2 * sizeof(p2pIpcExpInfo));

    const size_t userBuffSize = kSegSize * kNumSegments;
    auto [r, totalMappedBufferSize, numSegments, ipcInfos] = CallWalk(userBuffSize);

    EXPECT_EQ(r, ncclSystemError);
    EXPECT_EQ(seg.retainCalls, 0);   // no segment was ever walked
}

// On the POSIX-fd path the exported-fd scratch array (expFds) is allocated
// between ipcInfos and segmentHandles; its failure aborts the walk before any
// segment is retained.
TEST_F(P2pMultiSegmentMicrotest, Walk_ExpFdsAllocFails_PropagatesWithoutRetaining)
{
    ncclCuMemHandleType    = hipMemHandleTypePosixFileDescriptor;
    proxyConn_.sameProcess = 0;

    constexpr std::size_t kSegSize     = 0x1000;
    constexpr int         kNumSegments = 3;
    SegmentRangeEmulator seg(kSegSize);

    // capacity(2) * sizeof(int) is the expFds array (POSIX-fd path only).
    SetP2pCallocFailSize(2 * sizeof(int));

    const size_t userBuffSize = kSegSize * kNumSegments;
    auto [r, totalMappedBufferSize, numSegments, ipcInfos] = CallWalk(userBuffSize);

    EXPECT_EQ(r, ncclSystemError);
    EXPECT_EQ(seg.retainCalls, 0);
}

// The segment-handles scratch array failing (the ipcInfos array already
// allocated) also aborts before the walk, still leaving no handle retained.
TEST_F(P2pMultiSegmentMicrotest, Walk_SegmentHandlesAllocFails_PropagatesWithoutRetaining)
{
    constexpr std::size_t kSegSize     = 0x1000;
    constexpr int         kNumSegments = 3;
    SegmentRangeEmulator seg(kSegSize);

    // Non-POSIX handle type here, so expFds is skipped and segmentHandles is
    // the second ncclCalloc: capacity(2) * sizeof(handle).
    SetP2pCallocFailSize(2 * sizeof(hipMemGenericAllocationHandle_t));

    const size_t userBuffSize = kSegSize * kNumSegments;
    auto [r, totalMappedBufferSize, numSegments, ipcInfos] = CallWalk(userBuffSize);

    EXPECT_EQ(r, ncclSystemError);
    EXPECT_EQ(seg.retainCalls, 0);
}

// ===========================================================================
// Proxy-side register / deregister, reached through the send vtable's
// proxyRegister / proxyDeregister slots (p2pProxyRegister / p2pProxyDeregister).
//
// These run on the proxy thread: proxyRegister receives a request buffer of
// one p2pIpcExpInfo per segment, imports/maps each segment's handle into a
// local VA, writes the mapped address back into respBuff, and records a
// proxyMemHandle (carrying an ncclIpcImpInfo) on the connection's
// proxyMemHandleQueue. proxyDeregister receives an ncclIpcImpInfo, finds the
// matching queued handle via p2pHandleCmp, releases the import, and frees the
// record.
//
// Observable state through the surface these functions own:
//   - respBuff carries the mapped register address (proxyRegister's output);
//   - *done is latched to 1;
//   - connection->proxyMemHandleQueue gains/loses the record;
//   - the recorded ncclIpcImpInfo reflects the request (rmtRegAddr, offset,
//     legacyIpcCap, numSegments).
// The import/map/release themselves are HIP driver seams with no observable
// state here, so those are asserted mock-style ("the arm drove this seam").
//
// The vtable slots are populated at static init of p2pTransport (not gated on
// initCeOperation's useMemcpy latch), so these run in-process without fork.
// ===========================================================================

namespace {

// Head of the connection's proxyMemHandleQueue, or nullptr if empty. The
// queue is intrusive; a bare zero-initialised connection is already in the
// empty/constructed state.
inline proxyMemHandle* QueueHead(ncclProxyConnection& conn) {
    return ncclIntruQueueEmpty(&conn.proxyMemHandleQueue)
               ? nullptr
               : ncclIntruQueueHead(&conn.proxyMemHandleQueue);
}

// Allocate a proxyMemHandle + ncclIpcImpInfo the way p2pProxyRegister does
// (ncclCalloc == calloc), so production's free() on the deregister path
// matches the allocation. Enqueues it on the connection.
inline proxyMemHandle* EnqueueRecordedHandle(ncclProxyConnection& conn,
                                             void* rmtRegAddr, uintptr_t offset,
                                             bool legacyIpcCap, int numSegments) {
    auto* mh = static_cast<proxyMemHandle*>(std::calloc(1, sizeof(proxyMemHandle)));
    auto* ii = static_cast<ncclIpcImpInfo*>(std::calloc(1, sizeof(ncclIpcImpInfo)));
    ii->rmtRegAddr   = rmtRegAddr;
    ii->offset       = offset;
    ii->legacyIpcCap = legacyIpcCap;
    ii->numSegments  = numSegments;
    mh->handle       = ii;
    ncclIntruQueueEnqueue(&conn.proxyMemHandleQueue, mh);
    return mh;
}

}  // namespace

class P2pProxyRegisterMicrotest : public P2pMicrotest {
protected:
    ncclProxyConnection conn_{};
    ncclProxyState      state_{};

    void SetUp() override {
        P2pMicrotest::SetUp();
        state_.tpRank  = 2;
        state_.cudaDev = 0;
    }

    // Drain any records the test did not consume, freeing them the way
    // production would, so nothing leaks between cases.
    void TearDown() override {
        while (!ncclIntruQueueEmpty(&conn_.proxyMemHandleQueue)) {
            auto* mh = ncclIntruQueueDequeue(&conn_.proxyMemHandleQueue);
            std::free(mh->handle);
            std::free(mh);
        }
        P2pMicrotest::TearDown();
    }
};

// Legacy-IPC register: a single-segment request imports the peer handle with
// cudaIpcOpenMemHandle, returns the mapped address (offset applied) in
// respBuff, and records a matching ncclIpcImpInfo on the queue.
TEST_F(P2pProxyRegisterMicrotest, ProxyRegister_LegacyIpc_ImportsHandleAndRecordsIt)
{
    constexpr uintptr_t kBase   = 0x8000;
    constexpr uintptr_t kOffset = 0x40;

    p2pIpcExpInfo req{};
    req.legacyIpcCap = true;
    req.size         = 0x1000;
    req.offset       = kOffset;

    int openCalls = 0;
    ScopedHook open(g_hipIpcOpenMemHandle,
        [&openCalls](void** devPtr, hipIpcMemHandle_t, unsigned int) -> hipError_t {
            ++openCalls;
            if (devPtr) *devPtr = reinterpret_cast<void*>(kBase);
            return hipSuccess;
        });

    void* resp = nullptr;
    int   done = 0;
    auto r = p2pTransport.send.proxyRegister(&conn_, &state_, &req,
                                             sizeof(p2pIpcExpInfo), &resp,
                                             sizeof(void*), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(openCalls, 1);                    // the legacy import arm fired
    EXPECT_EQ(done, 1);
    // The mapped address returned to the caller is the import + request offset.
    EXPECT_EQ(resp, reinterpret_cast<void*>(kBase + kOffset));

    // A record carrying the request's impInfo now sits on the queue.
    proxyMemHandle* head = QueueHead(conn_);
    ASSERT_NE(head, nullptr);
    auto* ii = static_cast<ncclIpcImpInfo*>(head->handle);
    EXPECT_EQ(ii->rmtRegAddr,   resp);
    EXPECT_EQ(ii->offset,       kOffset);
    EXPECT_TRUE(ii->legacyIpcCap);
    EXPECT_EQ(ii->numSegments,  1);
}

// Legacy-IPC register whose import fails takes the fail path: no address is
// returned (respBuff stays NULL), nothing is recorded, but *done is still
// latched so the proxy reply is sent.
TEST_F(P2pProxyRegisterMicrotest, ProxyRegister_LegacyIpcImportFails_RecordsNothing)
{
    p2pIpcExpInfo req{};
    req.legacyIpcCap = true;
    req.size         = 0x1000;

    ScopedHook open(g_hipIpcOpenMemHandle,
        [](void** devPtr, hipIpcMemHandle_t, unsigned int) -> hipError_t {
            if (devPtr) *devPtr = nullptr;
            return hipErrorInvalidValue;
        });

    void* resp = reinterpret_cast<void*>(0xDEAD);
    int   done = 0;
    auto r = p2pTransport.send.proxyRegister(&conn_, &state_, &req,
                                             sizeof(p2pIpcExpInfo), &resp,
                                             sizeof(void*), &done);

    EXPECT_NE(r, ncclSuccess);
    EXPECT_EQ(resp, nullptr);                   // no address handed back
    EXPECT_EQ(done, 1);                         // reply still latched
    EXPECT_EQ(QueueHead(conn_), nullptr);       // nothing recorded
}

#if ROCM_VERSION >= 70000

// Same-process cuMem register: a multi-segment request reserves a VA range,
// maps each segment's handle (copied in directly, no cross-process import),
// grants device access, and records the mapped range. numSegments on the
// record reflects the segment count.
TEST_F(P2pProxyRegisterMicrotest, ProxyRegister_SameProcessCuMem_MapsSegmentsAndRecords)
{
    conn_.sameProcess = 1;

    constexpr int      kNumSegments = 2;
    constexpr uintptr_t kBase       = 0x100000;
    constexpr uintptr_t kOffset     = 0x80;

    std::array<p2pIpcExpInfo, kNumSegments> req{};
    for (auto& seg : req) seg.size = 0x1000;
    req[0].offset = kOffset;

    ScopedHook reserve(g_hipMemAddressReserve,
        [](void** ptr, std::size_t, std::size_t, void*, unsigned long long) -> hipError_t {
            if (ptr) *ptr = reinterpret_cast<void*>(kBase);
            return hipSuccess;
        });
    int mapCalls = 0;
    ScopedHook map(g_hipMemMap,
        [&mapCalls](void*, std::size_t, std::size_t,
                    hipMemGenericAllocationHandle_t, unsigned long long) -> hipError_t {
            ++mapCalls;
            return hipSuccess;
        });
    int setAccessCalls = 0;
    ScopedHook access(g_hipMemSetAccess,
        [&setAccessCalls](void*, std::size_t, const hipMemAccessDesc*, std::size_t) -> hipError_t {
            ++setAccessCalls;
            return hipSuccess;
        });
    // Same-process copies the handle in directly; the cross-process import
    // seam must not run.
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [](hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType) -> hipError_t {
            ADD_FAILURE() << "same-process register must not import a shareable handle";
            return hipErrorInvalidValue;
        });

    void* resp = nullptr;
    int   done = 0;
    auto r = p2pTransport.send.proxyRegister(&conn_, &state_, req.data(),
                                             sizeof(p2pIpcExpInfo) * kNumSegments,
                                             &resp, sizeof(void*), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(mapCalls, kNumSegments);          // one map per segment
    EXPECT_EQ(setAccessCalls, 1);               // one access grant over the range
    EXPECT_EQ(done, 1);
    // Returned address is the reserved base plus the first segment's offset.
    EXPECT_EQ(resp, reinterpret_cast<void*>(kBase + kOffset));

    proxyMemHandle* head = QueueHead(conn_);
    ASSERT_NE(head, nullptr);
    auto* ii = static_cast<ncclIpcImpInfo*>(head->handle);
    EXPECT_EQ(ii->numSegments, kNumSegments);
    EXPECT_FALSE(ii->legacyIpcCap);
}

// Cross-process cuMem register with a POSIX-fd handle type: each segment's
// handle is imported from its shareable fd (then the fd closed), mapped, and
// the range recorded.
TEST_F(P2pProxyRegisterMicrotest, ProxyRegister_CrossProcessPosixFd_ImportsFromFdAndMaps)
{
    conn_.sameProcess = 0;
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    constexpr int kNumSegments = 2;
    std::array<p2pIpcExpInfo, kNumSegments> req{};
    for (auto& seg : req) {
        seg.size  = 0x1000;
        seg.impFd = ::dup(0);   // a real, closeable fd for production's close()
    }

    ScopedHook reserve(g_hipMemAddressReserve,
        [](void** ptr, std::size_t, std::size_t, void*, unsigned long long) -> hipError_t {
            if (ptr) *ptr = reinterpret_cast<void*>(0x200000);
            return hipSuccess;
        });
    int importCalls = 0;
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [&importCalls](hipMemGenericAllocationHandle_t* h, void*,
                       hipMemAllocationHandleType type) -> hipError_t {
            ++importCalls;
            EXPECT_EQ(type, hipMemHandleTypePosixFileDescriptor);
            if (h) *h = nullptr;
            return hipSuccess;
        });
    int mapCalls = 0;
    ScopedHook map(g_hipMemMap,
        [&mapCalls](void*, std::size_t, std::size_t,
                    hipMemGenericAllocationHandle_t, unsigned long long) -> hipError_t {
            ++mapCalls;
            return hipSuccess;
        });
    ScopedHook access(g_hipMemSetAccess,
        [](void*, std::size_t, const hipMemAccessDesc*, std::size_t) -> hipError_t {
            return hipSuccess;
        });

    void* resp = nullptr;
    int   done = 0;
    auto r = p2pTransport.send.proxyRegister(&conn_, &state_, req.data(),
                                             sizeof(p2pIpcExpInfo) * kNumSegments,
                                             &resp, sizeof(void*), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(importCalls, kNumSegments);       // one fd import per segment
    EXPECT_EQ(mapCalls, kNumSegments);
    EXPECT_EQ(done, 1);
    ASSERT_NE(QueueHead(conn_), nullptr);
}

// Cross-process cuMem register with a non-POSIX (fabric) handle type imports
// each segment's handle straight from its ipcDesc.cuDesc rather than an fd.
TEST_F(P2pProxyRegisterMicrotest, ProxyRegister_CrossProcessNonPosix_ImportsFromCuDesc)
{
    conn_.sameProcess = 0;
    ScopedCuMemHandleType handleType(hipMemHandleTypeWin32);  // anything != POSIX_FD

    p2pIpcExpInfo req{};
    req.size = 0x1000;

    ScopedHook reserve(g_hipMemAddressReserve,
        [](void** ptr, std::size_t, std::size_t, void*, unsigned long long) -> hipError_t {
            if (ptr) *ptr = reinterpret_cast<void*>(0x300000);
            return hipSuccess;
        });
    int importCalls = 0;
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [&importCalls](hipMemGenericAllocationHandle_t* h, void*,
                       hipMemAllocationHandleType type) -> hipError_t {
            ++importCalls;
            EXPECT_NE(type, hipMemHandleTypePosixFileDescriptor);
            if (h) *h = nullptr;
            return hipSuccess;
        });
    ScopedHook map(g_hipMemMap,
        [](void*, std::size_t, std::size_t,
           hipMemGenericAllocationHandle_t, unsigned long long) -> hipError_t {
            return hipSuccess;
        });
    ScopedHook access(g_hipMemSetAccess,
        [](void*, std::size_t, const hipMemAccessDesc*, std::size_t) -> hipError_t {
            return hipSuccess;
        });

    void* resp = nullptr;
    int   done = 0;
    auto r = p2pTransport.send.proxyRegister(&conn_, &state_, &req,
                                             sizeof(p2pIpcExpInfo), &resp,
                                             sizeof(void*), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(importCalls, 1);                   // imported from the cuDesc
    EXPECT_EQ(done, 1);
    ASSERT_NE(QueueHead(conn_), nullptr);
}

// A cuMem register whose per-segment map fails takes the fail-cleanup path:
// every segment mapped/imported before the failure is unmapped and released,
// no record is left on the queue, respBuff is NULL, and *done is still latched.
TEST_F(P2pProxyRegisterMicrotest, ProxyRegister_CuMemMapFails_ReleasesImportedHandlesAndRecordsNothing)
{
    conn_.sameProcess = 1;   // same-process: handle copied in, no import seam

    constexpr int kNumSegments = 2;
    std::array<p2pIpcExpInfo, kNumSegments> req{};
    for (auto& seg : req) seg.size = 0x1000;

    ScopedHook reserve(g_hipMemAddressReserve,
        [](void** ptr, std::size_t, std::size_t, void*, unsigned long long) -> hipError_t {
            if (ptr) *ptr = reinterpret_cast<void*>(0x400000);
            return hipSuccess;
        });
    // First segment maps, second fails -> cleanup releases the first.
    int mapCalls = 0;
    ScopedHook map(g_hipMemMap,
        [&mapCalls](void*, std::size_t, std::size_t,
                    hipMemGenericAllocationHandle_t, unsigned long long) -> hipError_t {
            return (++mapCalls == 1) ? hipSuccess : hipErrorInvalidValue;
        });
    int releaseCalls = 0;
    ScopedHook release(g_hipMemRelease,
        [&releaseCalls](hipMemGenericAllocationHandle_t) -> hipError_t {
            ++releaseCalls;
            return hipSuccess;
        });

    void* resp = reinterpret_cast<void*>(0xDEAD);
    int   done = 0;
    auto r = p2pTransport.send.proxyRegister(&conn_, &state_, req.data(),
                                             sizeof(p2pIpcExpInfo) * kNumSegments,
                                             &resp, sizeof(void*), &done);

    EXPECT_NE(r, ncclSuccess);
    EXPECT_EQ(resp, nullptr);                    // no address handed back
    EXPECT_EQ(done, 1);
    // Both segments were marked imported (handle copied in) before the map of
    // the second failed; cleanup releases each imported handle.
    EXPECT_EQ(releaseCalls, kNumSegments);
    EXPECT_EQ(QueueHead(conn_), nullptr);        // nothing recorded
}

// Cross-process POSIX-fd register whose fd close fails after import takes the
// fail-cleanup path: the imported handle is released and nothing is recorded.
TEST_F(P2pProxyRegisterMicrotest, ProxyRegister_PosixFdCloseFails_ReleasesAndRecordsNothing)
{
    conn_.sameProcess = 0;
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    p2pIpcExpInfo req{};
    req.size  = 0x1000;
    req.impFd = 1 << 30;   // a definitely-invalid fd => close() fails

    ScopedHook reserve(g_hipMemAddressReserve,
        [](void** ptr, std::size_t, std::size_t, void*, unsigned long long) -> hipError_t {
            if (ptr) *ptr = reinterpret_cast<void*>(0x500000);
            return hipSuccess;
        });
    ScopedHook import(g_hipMemImportFromShareableHandle,
        [](hipMemGenericAllocationHandle_t* h, void*,
           hipMemAllocationHandleType) -> hipError_t {
            if (h) *h = nullptr;
            return hipSuccess;
        });
    int releaseCalls = 0;
    ScopedHook release(g_hipMemRelease,
        [&releaseCalls](hipMemGenericAllocationHandle_t) -> hipError_t {
            ++releaseCalls;
            return hipSuccess;
        });

    void* resp = reinterpret_cast<void*>(0xDEAD);
    int   done = 0;
    auto r = p2pTransport.send.proxyRegister(&conn_, &state_, &req,
                                             sizeof(p2pIpcExpInfo), &resp,
                                             sizeof(void*), &done);

    EXPECT_NE(r, ncclSuccess);
    EXPECT_EQ(resp, nullptr);
    EXPECT_EQ(done, 1);
    // The close fails before the segment is marked imported, so cleanup has
    // nothing to release; the imported handle is released by the cleanup loop
    // only for segments flagged imported (none here).
    EXPECT_EQ(releaseCalls, 0);
    EXPECT_EQ(QueueHead(conn_), nullptr);    // nothing recorded
}

#endif  // ROCM_VERSION >= 70000

// A successful legacy import whose record bookkeeping allocation fails
// propagates the error: the mapped address is not recorded (the caller sees
// the failure rather than a half-built queue entry).
TEST_F(P2pProxyRegisterMicrotest, ProxyRegister_RecordAllocFails_Propagates)
{
    constexpr uintptr_t kBase = 0x8000;

    p2pIpcExpInfo req{};
    req.legacyIpcCap = true;
    req.size         = 0x1000;

    ScopedHook open(g_hipIpcOpenMemHandle,
        [](void** devPtr, hipIpcMemHandle_t, unsigned int) -> hipError_t {
            if (devPtr) *devPtr = reinterpret_cast<void*>(kBase);
            return hipSuccess;
        });
    // The record wrapper (proxyMemHandle) is the first ncclCalloc in the
    // exit-bookkeeping block; starve it.
    SetP2pCallocFailSize(sizeof(proxyMemHandle));

    void* resp = nullptr;
    int   done = 0;
    auto r = p2pTransport.send.proxyRegister(&conn_, &state_, &req,
                                             sizeof(p2pIpcExpInfo), &resp,
                                             sizeof(void*), &done);

    EXPECT_NE(r, ncclSuccess);
    EXPECT_EQ(QueueHead(conn_), nullptr);    // no record enqueued
    // Latent bug: on this bookkeeping-alloc-failure path the bare
    // NCCLCHECK(ncclCalloc(...)) returns before *done is latched, unlike the
    // fail: path which nulls regAddr and reaches *done = 1. done therefore
    // stays 0, which would leave the requesting rank blocked in
    // ncclProxyCallBlocking. Pin the current contract; latching done on this
    // path is a production follow-up.
    EXPECT_EQ(done, 0);
}

// The impInfo payload allocation (the second ncclCalloc in the bookkeeping
// block) failing likewise propagates without enqueuing a record.
TEST_F(P2pProxyRegisterMicrotest, ProxyRegister_ImpInfoAllocFails_Propagates)
{
    constexpr uintptr_t kBase = 0x8000;

    p2pIpcExpInfo req{};
    req.legacyIpcCap = true;
    req.size         = 0x1000;

    ScopedHook open(g_hipIpcOpenMemHandle,
        [](void** devPtr, hipIpcMemHandle_t, unsigned int) -> hipError_t {
            if (devPtr) *devPtr = reinterpret_cast<void*>(kBase);
            return hipSuccess;
        });
    // proxyMemHandle and ncclIpcImpInfo differ in size; starving the latter
    // reaches the second calloc while the first succeeds.
    ASSERT_NE(sizeof(proxyMemHandle), sizeof(ncclIpcImpInfo));
    SetP2pCallocFailSize(sizeof(ncclIpcImpInfo));

    void* resp = nullptr;
    int   done = 0;
    auto r = p2pTransport.send.proxyRegister(&conn_, &state_, &req,
                                             sizeof(p2pIpcExpInfo), &resp,
                                             sizeof(void*), &done);

    EXPECT_NE(r, ncclSuccess);
    EXPECT_EQ(QueueHead(conn_), nullptr);
    // See ProxyRegister_RecordAllocFails_Propagates: done is left unlatched on
    // this alloc-failure path, so it stays 0 (a latent peer-blocking bug).
    EXPECT_EQ(done, 0);
}

class P2pProxyDeregisterMicrotest : public P2pProxyRegisterMicrotest {};

// Deregister finds the queued record whose impInfo matches the request (by
// p2pHandleCmp), releases its legacy import, and removes it from the queue --
// leaving an unrelated record untouched.
TEST_F(P2pProxyDeregisterMicrotest, ProxyDeregister_MatchingRecord_ReleasesAndRemovesOnlyIt)
{
    void* const    kTargetAddr = reinterpret_cast<void*>(0x8040);
    constexpr uintptr_t kOffset = 0x40;

    // An unrelated record (different address) that must survive, plus the
    // target the request will match.
    proxyMemHandle* other  = EnqueueRecordedHandle(conn_, reinterpret_cast<void*>(0x9000),
                                                   0x10, /*legacy=*/true, /*segs=*/1);
    EnqueueRecordedHandle(conn_, kTargetAddr, kOffset, /*legacy=*/true, /*segs=*/1);

    // The request is an ncclIpcImpInfo identifying the target record.
    ncclIpcImpInfo reqInfo{};
    reqInfo.rmtRegAddr   = kTargetAddr;
    reqInfo.offset       = kOffset;
    reqInfo.legacyIpcCap = true;
    reqInfo.numSegments  = 1;

    int closeCalls = 0;
    void* closedPtr = nullptr;
    ScopedHook close(g_hipIpcCloseMemHandle,
        [&](void* devPtr) -> hipError_t {
            ++closeCalls;
            closedPtr = devPtr;
            return hipSuccess;
        });

    int  done = 0;
    auto r = p2pTransport.send.proxyDeregister(&conn_, &state_, &reqInfo,
                                               sizeof(ncclIpcImpInfo), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(done, 1);
    // The legacy import was released at (rmtRegAddr - offset).
    EXPECT_EQ(closeCalls, 1);
    EXPECT_EQ(closedPtr, reinterpret_cast<void*>(
                             reinterpret_cast<uintptr_t>(kTargetAddr) - kOffset));
    // Only the unrelated record remains on the queue (the target was removed);
    // TearDown frees it the way production would.
    proxyMemHandle* head = QueueHead(conn_);
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head, other);
    EXPECT_EQ(head->next, nullptr);   // exactly one record left
}

#if ROCM_VERSION >= 70000

// cuMem deregister: a non-legacy record is released via the cuMem free arm
// -- ncclCuMemFreeAddr for a same-process connection, ncclCudaFree for a
// cross-process one -- selected off connection->sameProcess. The shutdown
// flag lets those header-only frees short-circuit before touching HIP.
TEST_F(P2pProxyDeregisterMicrotest, ProxyDeregister_CuMemSameProcessRecord_ReleasesAndRemoves)
{
    conn_.sameProcess = 1;   // same-process => ncclCuMemFreeAddr arm
    ShutdownFlagGuard shutdown;

    void* const kAddr = reinterpret_cast<void*>(0x50040);
    EnqueueRecordedHandle(conn_, kAddr, /*offset=*/0x40,
                          /*legacy=*/false, /*segs=*/1);

    ncclIpcImpInfo reqInfo{};
    reqInfo.rmtRegAddr   = kAddr;
    reqInfo.offset       = 0x40;
    reqInfo.legacyIpcCap = false;
    reqInfo.numSegments  = 1;

    g_freeCalls.clear();
    int  done = 0;
    auto r = p2pTransport.send.proxyDeregister(&conn_, &state_, &reqInfo,
                                               sizeof(ncclIpcImpInfo), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(done, 1);
    EXPECT_EQ(QueueHead(conn_), nullptr);        // record removed
    // Same-process selects the ncclCuMemFreeAddr arm on (rmtRegAddr - offset).
    EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
        {FreeKind::CuMemFreeAddr, reinterpret_cast<void*>(0x50000)}}));
}

TEST_F(P2pProxyDeregisterMicrotest, ProxyDeregister_CuMemCrossProcessRecord_ReleasesAndRemoves)
{
    conn_.sameProcess = 0;   // cross-process => ncclCudaFree arm
    ShutdownFlagGuard shutdown;

    void* const kAddr = reinterpret_cast<void*>(0x60080);
    EnqueueRecordedHandle(conn_, kAddr, /*offset=*/0x80,
                          /*legacy=*/false, /*segs=*/1);

    ncclIpcImpInfo reqInfo{};
    reqInfo.rmtRegAddr   = kAddr;
    reqInfo.offset       = 0x80;
    reqInfo.legacyIpcCap = false;
    reqInfo.numSegments  = 1;

    g_freeCalls.clear();
    int  done = 0;
    auto r = p2pTransport.send.proxyDeregister(&conn_, &state_, &reqInfo,
                                               sizeof(ncclIpcImpInfo), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(done, 1);
    EXPECT_EQ(QueueHead(conn_), nullptr);
    // Cross-process selects the ncclCudaFree arm on (rmtRegAddr - offset).
    EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
        {FreeKind::CudaFree, reinterpret_cast<void*>(0x60000)}}));
}

#endif  // ROCM_VERSION >= 70000

// When releasing the found record's import fails, deregister propagates the
// error (fail arm) yet still latches *done so the proxy reply is sent, and the
// record has already been removed from the queue.
TEST_F(P2pProxyDeregisterMicrotest, ProxyDeregister_ReleaseFails_PropagatesButStillLatchesDone)
{
    void* const kAddr = reinterpret_cast<void*>(0x70040);
    EnqueueRecordedHandle(conn_, kAddr, /*offset=*/0x40,
                          /*legacy=*/true, /*segs=*/1);

    ncclIpcImpInfo reqInfo{};
    reqInfo.rmtRegAddr   = kAddr;
    reqInfo.offset       = 0x40;
    reqInfo.legacyIpcCap = true;
    reqInfo.numSegments  = 1;

    ScopedHook close(g_hipIpcCloseMemHandle,
        [](void*) -> hipError_t { return hipErrorInvalidValue; });

    int  done = 0;
    auto r = p2pTransport.send.proxyDeregister(&conn_, &state_, &reqInfo,
                                               sizeof(ncclIpcImpInfo), &done);

    EXPECT_NE(r, ncclSuccess);                   // release failure propagated
    EXPECT_EQ(done, 1);                          // reply still latched
    EXPECT_EQ(QueueHead(conn_), nullptr);        // record already removed
}

// Coverage gap (AICOMRCCL-2417): the no-match deregister path is not exercised
// here because it currently crashes. When a request matches no queued record,
// ncclIntruQueueDelete returns nullptr (src/include/utils.h) and
// p2pProxyDeregister dereferences it unguarded (src/transport/p2p.cc), so a
// fifth case that enqueues nothing segfaults rather than asserting. Add that
// case once the production null-guard from AICOMRCCL-2417 lands.

// ===========================================================================
// Proxy-thread buffer lifecycle: the four proxy-side vtable slots that a
// proxy connection drives to allocate and release its transport buffer, plus
// the free counterpart of the shareable-buffer allocator.
//
//   ncclP2pFreeShareableBuffer -- the release counterpart of
//                                 ncclP2pAllocateShareableBuffer.
//   proxySetup (send / recv)   -- allocate the connection's transport buffer
//                                 and stash it on connection->transportResources.
//   proxyFree  (send / recv)   -- release whatever proxySetup stashed.
//
// These run through the proxy vtable slots on p2pTransport, which are wired to
// the proxy statics at static-init time (unlike the CE proxyConnect/
// proxyProgress slots, which are only patched after memcpy priming and are
// covered in the isolated suite below).
//
// The non-memcpy arm is the one every non-CE connection takes: proxySetup
// routes through ncclP2pAllocateShareableBuffer (legacy-IPC, cuMemEnable == 0),
// storing the raw device pointer; proxyFree hands that pointer back to
// ncclCudaFree. The shutdown-flag guard lets ncclCudaFree return before
// touching the (absent) HIP runtime.
// ===========================================================================

class P2pProxyLifecycleMicrotest : public P2pMicrotest {
protected:
    ncclProxyConnection conn_{};
    ncclProxyState      state_{};

    void SetUp() override {
        P2pMicrotest::SetUp();
        // useMemcpy latches to its default (0) through the public getter so the
        // non-memcpy proxy arm is exercised deterministically (ordering
        // precondition: only p2pCanConnect / ncclP2pUsesMemcpy prime it).
        ncclP2pUsesMemcpy();
        state_.tpRank  = 2;
        state_.cudaDev = 0;
    }
};

// ncclP2pFreeShareableBuffer is currently an unconditional
// `{ return ncclSuccess; }` (p2p.cc): it never dereferences its ncclIpcDesc*
// argument. This test pins exactly that no-op contract, so if the release
// counterpart ever grows real behaviour (touching the descriptor, failing on a
// bad handle) this becomes the place it has to be re-specified. It deliberately
// does NOT set up a populated descriptor -- that would falsely suggest the
// descriptor contents matter to the result.
TEST_F(P2pProxyLifecycleMicrotest, FreeShareableBuffer_IsCurrentlyNoopSuccess)
{
    ncclIpcDesc ipcDesc{};
    EXPECT_EQ(ncclP2pFreeShareableBuffer(&ipcDesc), ncclSuccess);
}

// Send proxySetup, non-memcpy arm: a well-formed request allocates the
// transport buffer through the legacy-IPC allocator, publishes it (pointer +
// size) into the response ncclP2pBuff, stashes the raw pointer on
// connection->transportResources, and latches *done.
TEST_F(P2pProxyLifecycleMicrotest, SendProxySetup_ValidRequest_AllocatesBufferAndPublishesResponse)
{
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t*, void*) -> hipError_t { return hipSuccess; });

    ncclP2pRequest req{};
    req.size     = 0x800;
    req.refcount = 0;
    req.peerRank = 1;

    ncclP2pBuff resp{};
    int         done = 0;
    auto r = p2pTransport.send.proxySetup(&conn_, &state_, &req, sizeof(req),
                                          &resp, sizeof(resp), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(done, 1);
    EXPECT_NE(resp.directPtr, nullptr);                     // buffer published
    EXPECT_EQ(resp.size, static_cast<size_t>(0x800));       // request size echoed
    // The raw device pointer is stashed for the matching proxyFree.
    EXPECT_EQ(conn_.transportResources, resp.directPtr);

    // Release the stashed buffer the way the matching proxyFree would.
    ShutdownFlagGuard shutdown;
    EXPECT_EQ(p2pTransport.send.proxyFree(&conn_, &state_), ncclSuccess);
}

#if ROCM_VERSION >= 70000
// Send proxySetup, cuMem arm: with cuMemEnable the allocated buffer is copied
// into a heap p2pCuMemProxyInfo which becomes the stashed transportResources
// (rather than the raw device pointer), and *done is latched.
TEST_F(P2pProxyLifecycleMicrotest, SendProxySetup_CuMemEnabled_StashesCuMemProxyInfo)
{
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);  // skip export

    ncclP2pRequest req{};
    req.size     = 0x800;
    req.peerRank = 1;

    ncclP2pBuff resp{};
    int         done = 0;
    auto r = p2pTransport.send.proxySetup(&conn_, &state_, &req, sizeof(req),
                                          &resp, sizeof(resp), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(done, 1);
    EXPECT_NE(resp.directPtr, nullptr);
    // The stash is a distinct p2pCuMemProxyInfo carrying a copy of the buffer,
    // not the raw device pointer.
    ASSERT_NE(conn_.transportResources, nullptr);
    EXPECT_NE(conn_.transportResources, resp.directPtr);
    auto* info = static_cast<p2pCuMemProxyInfo*>(conn_.transportResources);
    EXPECT_EQ(info->p2pBuff.directPtr, resp.directPtr);

    ShutdownFlagGuard shutdown;
    EXPECT_EQ(p2pTransport.send.proxyFree(&conn_, &state_), ncclSuccess);
}

// Recv proxySetup, cuMem arm: mirrors the send cuMem stash path.
TEST_F(P2pProxyLifecycleMicrotest, RecvProxySetup_CuMemEnabled_StashesCuMemProxyInfo)
{
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ScopedCuMemHandleType handleType(hipMemHandleTypePosixFileDescriptor);

    ncclP2pRequest req{};
    req.size     = 0x400;
    req.peerRank = 1;

    ncclP2pBuff resp{};
    int         done = 0;
    auto r = p2pTransport.recv.proxySetup(&conn_, &state_, &req, sizeof(req),
                                          &resp, sizeof(resp), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(done, 1);
    ASSERT_NE(conn_.transportResources, nullptr);
    EXPECT_NE(conn_.transportResources, resp.directPtr);
    auto* info = static_cast<p2pCuMemProxyInfo*>(conn_.transportResources);
    EXPECT_EQ(info->p2pBuff.directPtr, resp.directPtr);

    ShutdownFlagGuard shutdown;
    EXPECT_EQ(p2pTransport.recv.proxyFree(&conn_, &state_), ncclSuccess);
}

#endif  // ROCM_VERSION >= 70000

// The two send-side size-check rejections return before any cuMem code, so
// they live outside the ROCM_VERSION >= 70000 guard -- like their recv twins
// below and GraphRegister_InvalidArgs above -- to keep the coverage on ROCm 6.

// Send proxySetup rejects a response buffer whose size does not match the
// expected ncclP2pBuff, returning an internal error before allocating.
TEST_F(P2pProxyLifecycleMicrotest, SendProxySetup_WrongResponseSize_ReturnsInternalError)
{
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t*, void*) -> hipError_t {
            ADD_FAILURE() << "must not allocate when the response size is wrong";
            return hipErrorInvalidValue;
        });

    ncclP2pRequest req{};
    req.size = 0x800;
    ncclP2pBuff resp{};
    int         done = 0;
    auto r = p2pTransport.send.proxySetup(&conn_, &state_, &req, sizeof(req),
                                          &resp, sizeof(resp) - 1, &done);

    EXPECT_EQ(r, ncclInternalError);
    EXPECT_EQ(done, 0);
    EXPECT_EQ(conn_.transportResources, nullptr);
}

// Send proxySetup rejects a request buffer whose size does not match the
// expected ncclP2pRequest, returning an internal error before allocating.
TEST_F(P2pProxyLifecycleMicrotest, SendProxySetup_WrongRequestSize_ReturnsInternalError)
{
    ncclP2pRequest req{};
    ncclP2pBuff    resp{};
    int            done = 0;
    auto r = p2pTransport.send.proxySetup(&conn_, &state_, &req, sizeof(req) - 1,
                                          &resp, sizeof(resp), &done);

    EXPECT_EQ(r, ncclInternalError);
    EXPECT_EQ(done, 0);
}

// Recv proxySetup, non-memcpy arm: mirrors the send path -- allocate, publish
// into the response, stash on transportResources, latch *done.
TEST_F(P2pProxyLifecycleMicrotest, RecvProxySetup_ValidRequest_AllocatesBufferAndPublishesResponse)
{
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t*, void*) -> hipError_t { return hipSuccess; });

    ncclP2pRequest req{};
    req.size     = 0x400;
    req.peerRank = 1;

    ncclP2pBuff resp{};
    int         done = 0;
    auto r = p2pTransport.recv.proxySetup(&conn_, &state_, &req, sizeof(req),
                                          &resp, sizeof(resp), &done);

    EXPECT_EQ(r, ncclSuccess);
    EXPECT_EQ(done, 1);
    EXPECT_NE(resp.directPtr, nullptr);
    EXPECT_EQ(resp.size, static_cast<size_t>(0x400));
    EXPECT_EQ(conn_.transportResources, resp.directPtr);

    ShutdownFlagGuard shutdown;
    EXPECT_EQ(p2pTransport.recv.proxyFree(&conn_, &state_), ncclSuccess);
}

// Send proxySetup, non-memcpy arm, buffer allocation fails: the allocator's
// error is propagated and *done is left unlatched.
TEST_F(P2pProxyLifecycleMicrotest, SendProxySetup_BufferAllocFails_Propagates)
{
    ScopedHook alloc(g_fakeCudaCallocAsync,
        [](void**, std::size_t, hipStream_t) -> ncclResult_t {
            return ncclSystemError;
        });

    ncclP2pRequest req{};
    req.size = 0x800;
    ncclP2pBuff resp{};
    int         done = 0;
    auto r = p2pTransport.send.proxySetup(&conn_, &state_, &req, sizeof(req),
                                          &resp, sizeof(resp), &done);

    EXPECT_EQ(r, ncclSystemError);
    EXPECT_EQ(done, 0);
}

// Recv proxySetup, buffer allocation fails: the allocator's error is
// propagated before the buffer is published or stashed.
TEST_F(P2pProxyLifecycleMicrotest, RecvProxySetup_BufferAllocFails_Propagates)
{
    ScopedHook alloc(g_fakeCudaCallocAsync,
        [](void**, std::size_t, hipStream_t) -> ncclResult_t {
            return ncclSystemError;
        });

    ncclP2pRequest req{};
    req.size = 0x400;
    ncclP2pBuff resp{};
    int         done = 0;
    auto r = p2pTransport.recv.proxySetup(&conn_, &state_, &req, sizeof(req),
                                          &resp, sizeof(resp), &done);

    EXPECT_EQ(r, ncclSystemError);
    EXPECT_EQ(done, 0);
    EXPECT_EQ(conn_.transportResources, nullptr);
}

// Recv proxySetup rejects a response buffer whose size does not match the
// expected ncclP2pBuff, returning an internal error before allocating.
TEST_F(P2pProxyLifecycleMicrotest, RecvProxySetup_WrongResponseSize_ReturnsInternalError)
{
    ScopedHook ipcGet(g_hipIpcGetMemHandle,
        [](hipIpcMemHandle_t*, void*) -> hipError_t {
            ADD_FAILURE() << "must not allocate when the response size is wrong";
            return hipErrorInvalidValue;
        });

    ncclP2pRequest req{};
    req.size = 0x400;
    ncclP2pBuff resp{};
    int         done = 0;
    auto r = p2pTransport.recv.proxySetup(&conn_, &state_, &req, sizeof(req),
                                          &resp, sizeof(resp) - 1, &done);

    EXPECT_EQ(r, ncclInternalError);
    EXPECT_EQ(done, 0);
    EXPECT_EQ(conn_.transportResources, nullptr);
}

// Recv proxySetup rejects a mismatched request size before allocating.
TEST_F(P2pProxyLifecycleMicrotest, RecvProxySetup_WrongRequestSize_ReturnsInternalError)
{
    ncclP2pRequest req{};
    ncclP2pBuff    resp{};
    int            done = 0;
    auto r = p2pTransport.recv.proxySetup(&conn_, &state_, &req, sizeof(req) - 1,
                                          &resp, sizeof(resp), &done);

    EXPECT_EQ(r, ncclInternalError);
    EXPECT_EQ(done, 0);
}

// Send proxyFree with no stashed resources and an empty mem-handle queue is a
// no-op that succeeds (nothing to release).
TEST_F(P2pProxyLifecycleMicrotest, SendProxyFree_NoResources_IsNoopSuccess)
{
    EXPECT_EQ(p2pTransport.send.proxyFree(&conn_, &state_), ncclSuccess);
}

// Send proxyFree, cuMem arm: a stashed p2pCuMemProxyInfo is released -- its
// shareable buffer through ncclP2pFreeShareableBuffer and its device pointer
// through ncclCudaFree -- then the info struct freed. The shutdown-flag guard
// lets ncclCudaFree short-circuit before HIP.
TEST_F(P2pProxyLifecycleMicrotest, SendProxyFree_CuMemResources_ReleasesBufferAndFrees)
{
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ShutdownFlagGuard shutdown;

    // Allocated the way p2pSendProxySetup's cuMem arm does (ncclCalloc).
    auto* info = static_cast<p2pCuMemProxyInfo*>(
        std::calloc(1, sizeof(p2pCuMemProxyInfo)));
    info->p2pBuff.directPtr = reinterpret_cast<void*>(0x5000);
    conn_.transportResources = info;

    g_freeCalls.clear();
    EXPECT_EQ(p2pTransport.send.proxyFree(&conn_, &state_), ncclSuccess);
    // The stashed directPtr is released through ncclCudaFree; deleting that
    // call in production drops this entry.
    EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
        {FreeKind::CudaFree, reinterpret_cast<void*>(0x5000)}}));
    // production free(proxyInfo) already released `info`.
}

// Send proxyFree, cuMem arm with no stashed info: the null-guard skips the
// release and the call is a no-op success.
TEST_F(P2pProxyLifecycleMicrotest, SendProxyFree_CuMemEnabledNoResources_IsNoopSuccess)
{
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    EXPECT_EQ(p2pTransport.send.proxyFree(&conn_, &state_), ncclSuccess);
}

// Recv proxyFree, cuMem arm with no stashed info: the null-guard skips the
// release and the call is a no-op success.
TEST_F(P2pProxyLifecycleMicrotest, RecvProxyFree_CuMemEnabledNoResources_IsNoopSuccess)
{
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    EXPECT_EQ(p2pTransport.recv.proxyFree(&conn_, &state_), ncclSuccess);
}

// Recv proxyFree, cuMem arm: mirrors the send cuMem release path.
TEST_F(P2pProxyLifecycleMicrotest, RecvProxyFree_CuMemResources_ReleasesBufferAndFrees)
{
    ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
    ShutdownFlagGuard shutdown;

    auto* info = static_cast<p2pCuMemProxyInfo*>(
        std::calloc(1, sizeof(p2pCuMemProxyInfo)));
    info->p2pBuff.directPtr = reinterpret_cast<void*>(0x6000);
    conn_.transportResources = info;

    g_freeCalls.clear();
    EXPECT_EQ(p2pTransport.recv.proxyFree(&conn_, &state_), ncclSuccess);
    EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
        {FreeKind::CudaFree, reinterpret_cast<void*>(0x6000)}}));
}

// Recv proxyFree, non-cuMem arm: the stashed raw device pointer is handed to
// ncclCudaFree (short-circuited by the shutdown guard) and the call succeeds.
TEST_F(P2pProxyLifecycleMicrotest, RecvProxyFree_DirectResources_ReleasesThroughCudaFree)
{
    ShutdownFlagGuard shutdown;
    conn_.transportResources = reinterpret_cast<void*>(0x7000);
    g_freeCalls.clear();
    EXPECT_EQ(p2pTransport.recv.proxyFree(&conn_, &state_), ncclSuccess);
    // The else arm hands transportResources straight to ncclCudaFree.
    EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
        {FreeKind::CudaFree, reinterpret_cast<void*>(0x7000)}}));
}

// Send proxyFree, non-cuMem arm: the stashed raw device pointer is handed to
// ncclCudaFree (short-circuited by the shutdown guard) and the call succeeds.
TEST_F(P2pProxyLifecycleMicrotest, SendProxyFree_DirectResources_ReleasesThroughCudaFree)
{
    ShutdownFlagGuard shutdown;
    conn_.transportResources = reinterpret_cast<void*>(0x9000);
    g_freeCalls.clear();
    EXPECT_EQ(p2pTransport.send.proxyFree(&conn_, &state_), ncclSuccess);
    EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
        {FreeKind::CudaFree, reinterpret_cast<void*>(0x9000)}}));
}

// Recv proxyFree also drains the mem-handle queue before releasing its buffer.
TEST_F(P2pProxyLifecycleMicrotest, RecvProxyFree_QueuedRegistration_DeregistersAndDrainsQueue)
{
    int closes = 0;
    ScopedHook close(g_hipIpcCloseMemHandle,
        [&closes](void*) -> hipError_t { ++closes; return hipSuccess; });

    EnqueueRecordedHandle(conn_, reinterpret_cast<void*>(0x8040), /*offset=*/0x40,
                          /*legacy=*/true, /*segs=*/1);

    EXPECT_EQ(p2pTransport.recv.proxyFree(&conn_, &state_), ncclSuccess);
    EXPECT_EQ(closes, 1);
    EXPECT_EQ(QueueHead(conn_), nullptr);
}

// Both proxyFree slots first drain the connection's proxyMemHandleQueue,
// deregistering each stale registration before releasing the transport buffer.
// A queued legacy record is imported-closed via hipIpcCloseMemHandle and the
// queue emptied.
TEST_F(P2pProxyLifecycleMicrotest, SendProxyFree_QueuedRegistration_DeregistersAndDrainsQueue)
{
    int closes = 0;
    ScopedHook close(g_hipIpcCloseMemHandle,
        [&closes](void*) -> hipError_t { ++closes; return hipSuccess; });

    EnqueueRecordedHandle(conn_, reinterpret_cast<void*>(0x8040), /*offset=*/0x40,
                          /*legacy=*/true, /*segs=*/1);

    // No transport buffer stashed -> the release stage is a no-op; the test
    // pins the queue-drain half of proxyFree's contract.
    EXPECT_EQ(p2pTransport.send.proxyFree(&conn_, &state_), ncclSuccess);
    EXPECT_EQ(closes, 1);                       // the stale record was released
    EXPECT_EQ(QueueHead(conn_), nullptr);       // queue drained
}

// ===========================================================================
// CE-memcpy proxy slots: p2pSendProxyConnect and p2pSendProxyProgress.
//
// These two slots are only wired onto p2pTransport.send after memcpy priming
// (initCeOperation patches them; see the Prime_* isolated tests above). So a
// test that drives them through the public vtable must prime first -- which
// latches a one-shot static and permanently mutates the global p2pTransport --
// and therefore runs process-isolated, exactly like the priming tests.
//
// proxyConnect finishes the CE connection: it records the receiver FIFO
// pointer from the request, creates the copy stream and per-step events, and
// arms the proxy-append pointer. proxyProgress is the copy pump that advances a
// CE operation from Ready through Progress to None.
// ===========================================================================

class P2pProxyCeMicrotestIsolated : public P2pMicrotest {};

// Prime memcpy so the CE proxyConnect slot is wired, then drive it: a
// request carrying the receiver FIFO pointer is recorded on the proxy info,
// the copy stream and per-step events are created, and the proxy-append
// pointer is armed.
TEST_F(P2pProxyCeMicrotestIsolated, ProxyConnect_ValidRequest_RecordsFifoAndArmsAppend)
{
    RUN_ISOLATED_TEST("P2p_ProxyConnect_ValidRequest_RecordsFifoAndArmsAppend", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ncclP2pUsesMemcpy();  // prime -> patches p2pTransport.send.proxyConnect
        ASSERT_NE(p2pTransport.send.proxyConnect, nullptr);

        // Stream + event creation must succeed for the happy path.
        g_hipStreamCreateResult = hipSuccess;
        g_hipEventCreateResult  = hipSuccess;

        ncclProxyConnection conn{};
        ncclProxyState      state{};
        auto* info = static_cast<p2pShmProxyInfo*>(
            std::calloc(1, sizeof(p2pShmProxyInfo)));
        conn.transportResources = info;

        char  fifoStorage = 0;
        char* fifo = &fifoStorage;
        int   done = 0;
        auto r = p2pTransport.send.proxyConnect(&conn, &state, &fifo, sizeof(void*),
                                                nullptr, 0, &done);

        EXPECT_EQ(r, ncclSuccess);
        EXPECT_EQ(info->recvFifo, fifo);                  // FIFO pointer recorded
        EXPECT_EQ(conn.proxyAppendPtr, &conn.proxyAppend); // append armed
        // The copy stream and every per-step event the CE proxy needs were
        // created and stashed on the proxy info (calloc'd to null above, so a
        // skipped creation leaves them null). Asserted as "a handle exists"
        // rather than a specific value, so the fake's sentinel is not pinned.
        EXPECT_NE(info->stream, nullptr);
        for (int i = 0; i < NCCL_STEPS; i++)
            EXPECT_NE(info->events[i], nullptr) << "event " << i << " not created";
        std::free(info);
    });
}

// proxyConnect rejects a request whose size is not a single pointer, returning
// an internal error before creating any stream or events.
TEST_F(P2pProxyCeMicrotestIsolated, ProxyConnect_WrongRequestSize_ReturnsInternalError)
{
    RUN_ISOLATED_TEST("P2p_ProxyConnect_WrongRequestSize_ReturnsInternalError", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ncclP2pUsesMemcpy();
        ASSERT_NE(p2pTransport.send.proxyConnect, nullptr);

        ncclProxyConnection conn{};
        ncclProxyState      state{};
        auto* info = static_cast<p2pShmProxyInfo*>(
            std::calloc(1, sizeof(p2pShmProxyInfo)));
        conn.transportResources = info;

        char  buf[8] = {};
        int   done = 0;
        auto r = p2pTransport.send.proxyConnect(&conn, &state, buf, sizeof(void*) + 1,
                                                nullptr, 0, &done);

        EXPECT_EQ(r, ncclInternalError);
        EXPECT_EQ(conn.proxyAppendPtr, nullptr);   // append not armed
        std::free(info);
    });
}

// Prime memcpy, then drive send proxySetup's CE arm: the response size selects
// the CE path, which allocates the device copy buffer and peer SHM segment,
// stashes a p2pShmProxyInfo on transportResources, and copies it into the
// response buffer.
TEST_F(P2pProxyCeMicrotestIsolated, SendProxySetup_MemcpyEnabled_AllocatesCeStateAndPublishes)
{
    RUN_ISOLATED_TEST("P2p_SendProxySetup_MemcpyEnabled_AllocatesCeStateAndPublishes", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ncclP2pUsesMemcpy();  // prime -> useMemcpy = 1

        // ncclCudaHostCalloc swaps the stream-capture mode via the async-ops
        // seam; enable it so the host allocation succeeds.
        g_hipAsyncOpsResult = hipSuccess;

        // The peer SHM segment is allocated through this seam; hand back real
        // backing storage for the host/device SHM pointers.
        static p2pShm shmHostStorage;
        static p2pShm shmDevStorage;
        ScopedHook shmAlloc(g_shmAllocateShareableBuffer,
            [](size_t, bool, void*, void** hptr, void** dptr) -> ncclResult_t {
                if (hptr) *hptr = &shmHostStorage;
                if (dptr) *dptr = &shmDevStorage;
                return ncclSuccess;
            });

        ncclProxyConnection conn{};
        ncclProxyState      state{};
        state.buffSizes[NCCL_PROTO_SIMPLE] = 0x1000;

        p2pShmProxyInfo resp{};
        int             done = 0;
        auto r = p2pTransport.send.proxySetup(&conn, &state, nullptr, 0,
                                              &resp, sizeof(resp), &done);

        EXPECT_EQ(r, ncclSuccess);
        EXPECT_EQ(done, 1);
        ASSERT_NE(conn.transportResources, nullptr);
        auto* info = static_cast<p2pShmProxyInfo*>(conn.transportResources);
        EXPECT_EQ(info->shm, &shmHostStorage);      // SHM segment recorded
        EXPECT_NE(info->ceDevBuff, nullptr);        // device copy buffer allocated
        // The proxy info was published into the response for the peer.
        EXPECT_EQ(resp.shm, info->shm);
    });
}

// Send proxySetup, CE arm, wrong response size: the memcpy path rejects a
// response buffer that is not sized as a p2pShmProxyInfo before allocating.
TEST_F(P2pProxyCeMicrotestIsolated, SendProxySetup_MemcpyEnabledWrongResponseSize_ReturnsInternalError)
{
    RUN_ISOLATED_TEST("P2p_SendProxySetup_MemcpyEnabledWrongResponseSize_ReturnsInternalError", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ncclP2pUsesMemcpy();

        ScopedHook shmAlloc(g_shmAllocateShareableBuffer,
            [](size_t, bool, void*, void**, void**) -> ncclResult_t {
                ADD_FAILURE() << "must not allocate when the response size is wrong";
                return ncclSystemError;
            });

        ncclProxyConnection conn{};
        ncclProxyState      state{};
        p2pShmProxyInfo     resp{};
        int                 done = 0;
        auto r = p2pTransport.send.proxySetup(&conn, &state, nullptr, 0,
                                              &resp, sizeof(resp) - 1, &done);

        EXPECT_EQ(r, ncclInternalError);
        EXPECT_EQ(done, 0);
    });
}

// Prime memcpy, then drive send proxyFree's CE arm: a stashed p2pShmProxyInfo
// is torn down -- the SHM segment closed, the host and device buffers freed,
// and the copy stream + per-step events destroyed -- and the info struct freed.
TEST_F(P2pProxyCeMicrotestIsolated, SendProxyFree_MemcpyResources_TearsDownCeState)
{
    RUN_ISOLATED_TEST("P2p_SendProxyFree_MemcpyResources_TearsDownCeState", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ncclP2pUsesMemcpy();  // prime -> useMemcpy = 1

        // The device-buffer free routes through ncclCudaFree; the shutdown
        // guard lets it short-circuit before reaching HIP.
        rcclShutdownFlag().store(true, std::memory_order_release);

        ncclProxyConnection conn{};
        ncclProxyState      state{};
        auto* info = static_cast<p2pShmProxyInfo*>(
            std::calloc(1, sizeof(p2pShmProxyInfo)));
        info->ceRecvMem = nullptr;   // ncclCudaHostFree(nullptr) is a no-op
        // Sentinel device buffer so the ncclCudaFree release is observable in
        // g_freeCalls (the shutdown guard short-circuits before HIP but the
        // recording shim still logs the pointer).
        info->ceDevBuff = reinterpret_cast<char*>(0x8000);
        conn.transportResources = info;

        g_freeCalls.clear();
        auto r = p2pTransport.send.proxyFree(&conn, &state);

        EXPECT_EQ(r, ncclSuccess);
        // The CE device buffer is released through ncclCudaFree; deleting that
        // release in production drops this entry.
        EXPECT_EQ(g_freeCalls, (std::vector<FreeCall>{
            {FreeKind::CudaFree, reinterpret_cast<void*>(0x8000)}}));
        rcclShutdownFlag().store(false, std::memory_order_release);
        // production free(proxyInfo) already released `info`.
    });
}

// proxyConnect, stream-create failure: the CE connect propagates the HIP error
// from stream creation before creating any events or arming the append pointer.
TEST_F(P2pProxyCeMicrotestIsolated, ProxyConnect_StreamCreateFails_Propagates)
{
    RUN_ISOLATED_TEST("P2p_ProxyConnect_StreamCreateFails_Propagates", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ncclP2pUsesMemcpy();
        ASSERT_NE(p2pTransport.send.proxyConnect, nullptr);

        g_hipStreamCreateResult = hipErrorInvalidValue;  // stream creation fails

        ncclProxyConnection conn{};
        ncclProxyState      state{};
        auto* info = static_cast<p2pShmProxyInfo*>(
            std::calloc(1, sizeof(p2pShmProxyInfo)));
        conn.transportResources = info;

        char  fifoStorage = 0;
        char* fifo = &fifoStorage;
        int   done = 0;
        auto r = p2pTransport.send.proxyConnect(&conn, &state, &fifo, sizeof(void*),
                                                nullptr, 0, &done);

        EXPECT_NE(r, ncclSuccess);
        EXPECT_EQ(conn.proxyAppendPtr, nullptr);   // append not armed
        std::free(info);
    });
}

// proxyProgress, SIMPLE protocol, GPU tail not yet ready: the copy is skipped
// (the recvTail gate is False) so the op stays in Progress with nothing
// transmitted or done.
TEST_F(P2pProxyCeMicrotestIsolated, ProxyProgress_SimpleProtocolTailNotReady_SkipsCopyAndStaysInProgress)
{
    RUN_ISOLATED_TEST("P2p_ProxyProgress_SimpleProtocolTailNotReady_SkipsCopyAndStaysInProgress", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ncclP2pUsesMemcpy();
        ASSERT_NE(p2pTransport.send.proxyProgress, nullptr);

        ncclProxyState state{};
        state.buffSizes[NCCL_PROTO_SIMPLE] = 8 * NCCL_STEPS;

        ncclRecvMem ceRecvMem{};
        ceRecvMem.tail = 0;   // GPU has produced nothing -> copy gate stays False
        p2pShm shm{};

        p2pShmProxyInfo info{};
        info.ceRecvMem = &ceRecvMem;
        info.shm       = &shm;

        ncclProxyConnection conn{};
        conn.transportResources = &info;

        ncclProxyArgs args{};
        args.state      = ncclProxyOpReady;
        args.nsubs      = 1;
        args.protocol   = NCCL_PROTO_SIMPLE;
        args.chunkSteps = 1;
        args.sliceSteps = 1;
        args.subs[0].connection = &conn;
        args.subs[0].nsteps     = 1;

        auto r = p2pTransport.send.proxyProgress(&state, &args);

        EXPECT_EQ(r, ncclSuccess);
        EXPECT_EQ(args.done, 0);                        // nothing completed
        EXPECT_EQ(args.state, ncclProxyOpProgress);     // op still in flight
        EXPECT_EQ(args.subs[0].transmitted, 0u);        // no copy issued
    });
}

// proxyProgress, resumed in-progress with a sub already fully transmitted: the
// Ready-init is skipped (state is already Progress) and the transmit gate is
// False (nothing left to send), so the pump only drains the outstanding
// completion and finishes the op.
TEST_F(P2pProxyCeMicrotestIsolated, ProxyProgress_ResumedWithTransmittedSub_DrainsCompletionAndFinishes)
{
    RUN_ISOLATED_TEST("P2p_ProxyProgress_ResumedWithTransmittedSub_DrainsCompletionAndFinishes", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ncclP2pUsesMemcpy();
        ASSERT_NE(p2pTransport.send.proxyProgress, nullptr);

        g_hipAsyncOpsResult = hipSuccess;   // event query reports complete

        ncclProxyState state{};
        state.buffSizes[NCCL_PROTO_SIMPLE] = 8 * NCCL_STEPS;

        ncclRecvMem ceRecvMem{};
        p2pShm shm{};
        p2pShmProxyInfo info{};
        info.ceRecvMem = &ceRecvMem;
        info.shm       = &shm;

        ncclProxyConnection conn{};
        conn.transportResources = &info;

        ncclProxyArgs args{};
        args.state      = ncclProxyOpProgress;   // resumed, not Ready
        args.nsubs      = 1;
        args.protocol   = NCCL_PROTO_SIMPLE;
        args.chunkSteps = 1;
        args.sliceSteps = 1;
        // A sub already fully transmitted with one completion still pending.
        args.subs[0].connection  = &conn;
        args.subs[0].nsteps      = 1;
        args.subs[0].base        = 0;
        args.subs[0].transmitted = 1;   // == nsteps -> transmit gate False
        args.subs[0].done        = 0;   // one completion left to drain

        auto r = p2pTransport.send.proxyProgress(&state, &args);

        EXPECT_EQ(r, ncclSuccess);
        EXPECT_EQ(args.done, 1);                    // completion drained
        EXPECT_EQ(args.state, ncclProxyOpNone);     // op finished
    });
}

// proxyProgress, non-SIMPLE protocol: a Ready operation is initialised (its
// subs re-based) and moved to Progress; because only SIMPLE uses the copy
// engine, each sub is immediately marked done and the op completes to None in
// a single pump.
TEST_F(P2pProxyCeMicrotestIsolated, ProxyProgress_NonSimpleProtocol_CompletesWithoutCopy)
{
    RUN_ISOLATED_TEST("P2p_ProxyProgress_NonSimpleProtocol_CompletesWithoutCopy", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ncclP2pUsesMemcpy();  // prime -> patches p2pTransport.send.proxyProgress
        ASSERT_NE(p2pTransport.send.proxyProgress, nullptr);

        ncclProxyState state{};
        state.buffSizes[NCCL_PROTO_LL] = 8 * NCCL_STEPS;

        p2pShmProxyInfo info{};
        info.step = 0;

        ncclProxyConnection conn{};
        conn.transportResources = &info;

        ncclProxyArgs args{};
        args.state      = ncclProxyOpReady;
        args.nsubs      = 1;
        args.protocol   = NCCL_PROTO_LL;   // != SIMPLE -> no copy engine
        args.chunkSteps = 1;
        args.subs[0].connection = &conn;
        args.subs[0].nsteps     = 4;

        auto r = p2pTransport.send.proxyProgress(&state, &args);

        EXPECT_EQ(r, ncclSuccess);
        EXPECT_EQ(args.done, 1);                        // the sub completed
        EXPECT_EQ(args.state, ncclProxyOpNone);         // op finished
        EXPECT_EQ(info.step, args.subs[0].base + args.subs[0].nsteps);
    });
}

// proxyProgress, SIMPLE protocol: a Ready operation drives the copy engine --
// with the GPU tail ahead of the transmit cursor the pump issues the async
// copy + event record, the event query reports complete, and the single-step
// op advances to done and completes to None.
TEST_F(P2pProxyCeMicrotestIsolated, ProxyProgress_SimpleProtocol_CopiesAndCompletes)
{
    RUN_ISOLATED_TEST("P2p_ProxyProgress_SimpleProtocol_CopiesAndCompletes", []() {
        ScopedHook loadParam(g_loadParam, ForceP2pUseCudaMemcpy());
        ncclP2pUsesMemcpy();
        ASSERT_NE(p2pTransport.send.proxyProgress, nullptr);

        // Async copy, event record and event query all succeed so the single
        // step transmits and then completes in one pump.
        g_hipAsyncOpsResult = hipSuccess;
        // Model the async publish-ordering: the query for a slot only reports
        // the copy complete once the proxy has recorded that slot's event. This
        // makes the completion the test observes depend on the event actually
        // being recorded, not just on the shared async-ops result.
        g_hipEventQueryRequiresRecord = true;

        ncclProxyState state{};
        state.buffSizes[NCCL_PROTO_SIMPLE] = 8 * NCCL_STEPS;

        ncclRecvMem ceRecvMem{};
        ceRecvMem.tail = 64;   // GPU has produced well past the transmit cursor
        // The slot the proxy will transmit carries a payload; production must
        // move exactly these bytes from the device buffer into the recv FIFO.
        constexpr int kPayloadBytes = 8;   // == stepSize (buffSizes[SIMPLE] / NCCL_STEPS)
        ceRecvMem.connFifo[0].size = kPayloadBytes;
        p2pShm shm{};

        p2pShmProxyInfo info{};
        info.step      = 0;
        info.ceRecvMem = &ceRecvMem;
        info.shm       = &shm;
        char recvFifo[64] = {};
        char ceDevBuff[64] = {};
        // Seed the source with a recognisable payload and leave the destination
        // zeroed, so a real copy is the only thing that can make them match.
        const char payload[kPayloadBytes] = {'C', 'E', 'c', 'o', 'p', 'y', '!', '\n'};
        std::memcpy(ceDevBuff, payload, kPayloadBytes);
        info.recvFifo = recvFifo;
        info.ceDevBuff = ceDevBuff;
        // A distinct event handle for the slot the proxy will transmit, so the
        // record->query dependency has something to key on.
        info.events[0] = reinterpret_cast<hipEvent_t>(0x1);

        // Make the async-copy seam actually move bytes (the default fake only
        // returns a status). We assert on the bytes that land, not on the fact
        // that any particular primitive was called, so the implementation stays
        // free to move the payload however it likes.
        ScopedHook copyHook(g_hipMemcpyAsync,
            [](void* dst, const void* src, size_t n, hipMemcpyKind, hipStream_t) {
                if (dst && src && dst != src) std::memcpy(dst, src, n);
                return hipSuccess;
            });

        ncclProxyConnection conn{};
        conn.transportResources = &info;

        ncclProxyArgs args{};
        args.state      = ncclProxyOpReady;
        args.nsubs      = 1;
        args.protocol   = NCCL_PROTO_SIMPLE;
        args.chunkSteps = 1;
        args.sliceSteps = 1;
        args.subs[0].connection = &conn;
        args.subs[0].nsteps     = 1;   // a single step to transmit and complete

        auto r = p2pTransport.send.proxyProgress(&state, &args);

        EXPECT_EQ(r, ncclSuccess);
        EXPECT_EQ(args.done, 1);                        // the sub completed
        EXPECT_EQ(args.state, ncclProxyOpNone);         // op finished
        EXPECT_EQ(shm.recvMem.tail, args.subs[0].base + args.subs[0].done);
        // The observable job of the copy engine: the payload produced by the
        // GPU is now present in the recv FIFO. This fails if the proxy skips
        // the transfer, regardless of which primitive performs it.
        EXPECT_EQ(0, std::memcmp(recvFifo, payload, kPayloadBytes));
    });
}
