/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file SymmetricWindowMPITests.cpp
 * @brief Tests for relaxed symmetric buffer registration (one-buffer path) and
 *        for asymmetric buffer sizes during window registration
 *
 * Validates that symmetric kernels work correctly across different window
 * registration patterns:
 * - Both send and recv buffers registered (baseline)
 * - Only one buffer registered (relaxed / one-buffer path)
 * - No buffers registered (non-symmetric fallback)
 * - In-place operations with a single window
 *
 * REQUIRED Environment Variables:
 *   NCCL_CUMEM_ENABLE=1          Enables cuMem API for symmetric support
 *   NCCL_DEBUG=INFO              Enables debug logging to observe kernel path
 *   HSA_NO_SCRATCH_RECLAIM=1     Required for multi-GPU RCCL tests
 *
 * Run examples:
 *   mpirun -np 2 --bind-to none ./rccl-UnitTestsMPI --gtest_filter=SymWin_*
 *   mpirun -np 8 --bind-to none -x NCCL_DEBUG=INFO \
 *     ./rccl-UnitTestsMPI --gtest_filter=SymWin_AllReduce.*
 *   mpirun -np 8 --bind-to none -x NCCL_CUMEM_ENABLE=1 \
 *     ./rccl-UnitTestsMPI --gtest_filter=SymWin_Asym*
 */

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "rccl_float8.h"
#include "nccl_device.h"
#include <cstdint>
#include <cstdlib>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace {
    constexpr size_t DEFAULT_COUNT = 256 * 1024;
    constexpr int MIN_RANKS = 2;

    template <typename Fp8T>
    void FillFp8ReduceScatterInput(std::vector<Fp8T>& hostSend, int rank) {
        for (size_t j = 0; j < hostSend.size(); j++) {
            // Keep values in a compact signed range while varying by rank/index.
            const int base = static_cast<int>(j % 61) - 30;
            hostSend[j] = Fp8T(static_cast<float>(base + rank));
        }
    }

    template <typename Fp8T>
    float ExpectedFp8ReduceScatterSum(int rank, int nRanks, size_t count, size_t i) {
        const size_t globalIdx = static_cast<size_t>(rank) * count + i;
        const int base = static_cast<int>(globalIdx % 61) - 30;
        float acc = 0.0f;
        for (int r = 0; r < nRanks; r++) {
            acc += static_cast<float>(Fp8T(static_cast<float>(base + r)));
        }
        return static_cast<float>(Fp8T(acc));
    }

    template <typename Fp8T>
    void FillFp8AllReduceInput(std::vector<Fp8T>& hostSend, int rank) {
        for (size_t j = 0; j < hostSend.size(); j++) {
            // Match reduce-scatter's compact signed range to exercise the same regime.
            const int base = static_cast<int>(j % 61) - 30;
            hostSend[j] = Fp8T(static_cast<float>(base + rank));
        }
    }

    template <typename Fp8T>
    float ExpectedFp8AllReduceSum(int nRanks, size_t i) {
        const int base = static_cast<int>(i % 61) - 30;
        float acc = 0.0f;
        for (int r = 0; r < nRanks; r++) {
            acc += static_cast<float>(Fp8T(static_cast<float>(base + r)));
        }
        return static_cast<float>(Fp8T(acc));
    }

    // Aligned to the granularity ncclMemAlloc rounds up to, so that per-rank
    // multiples of this chunk stay distinct instead of collapsing onto one size.
    // Answers per rank; the fixture agrees on one value across ranks.
    size_t localAsymChunkBytes()
    {
        constexpr size_t requested = 2 * 1024 * 1024;

        int dev = 0;
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
        if (hipGetDevice(&dev) == hipSuccess) prop.location.id = dev;

        size_t granularity = 0;
        if (hipMemGetAllocationGranularity(&granularity, &prop,
                                           hipMemAllocationGranularityRecommended) != hipSuccess ||
            granularity == 0) {
            granularity = requested;
        }

        return ((requested + granularity - 1) / granularity) * granularity;
    }

    size_t& asymChunkCache()
    {
        static size_t cached = 0;
        return cached;
    }

    size_t asymChunkBytes()
    {
        size_t& cached = asymChunkCache();
        if (cached == 0) cached = localAsymChunkBytes();
        return cached;
    }

    enum class SizePattern {
        Ascending,
        Descending,
        SingleLarger,
        ExtremeRatio
    };

    size_t asymBytes(SizePattern pattern, int rank, int nRanks)
    {
        const size_t chunk = asymChunkBytes();
        switch (pattern) {
            case SizePattern::Ascending:    return chunk * (rank + 1);
            case SizePattern::Descending:   return chunk * (nRanks - rank);
            case SizePattern::SingleLarger: return rank == 0 ? chunk * nRanks : chunk;
            case SizePattern::ExtremeRatio: return rank == 0 ? chunk : chunk * 8;
        }
        return chunk;
    }

    const char* patternName(SizePattern pattern)
    {
        switch (pattern) {
            case SizePattern::Ascending:    return "Ascending";
            case SizePattern::Descending:   return "Descending";
            case SizePattern::SingleLarger: return "SingleLarger";
            case SizePattern::ExtremeRatio: return "ExtremeRatio";
        }
        return "Unknown";
    }

    uint64_t allreduceMin(uint64_t value)
    {
        uint64_t result = value;
        MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
        return result;
    }

    uint64_t allreduceMax(uint64_t value)
    {
        uint64_t result = value;
        MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
        return result;
    }
}

__global__ void samplePeerFloatsKernel(const float* peer, size_t count, float* out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    out[0] = peer[0];
    out[1] = peer[count / 2];
    out[2] = peer[count - 1];
}

// ============================================================================
// Base class for symmetric window tests
// ============================================================================

class SymmetricWindowTestBase : public MPITestBase
{
protected:
    struct NcclBufInfo {
        void* ptr = nullptr;
        size_t size = 0;
    };

    struct WinInfo {
        ncclWindow_t win = nullptr;
        ncclComm_t comm = nullptr;
    };

    std::vector<NcclBufInfo> allocatedBufs_;
    std::vector<WinInfo> registeredWins_;

    // Ranks a window is load/store reachable from. A team never spans a node and
    // is what flat-VA slots are sized over (ncclDevrMemory::lsaMaxSize), so the
    // peer-pointer invariants below hold per team rather than per job.
    MPI_Comm lsaComm_ = MPI_COMM_NULL;
    int lsaSize_ = 1;
    int lsaRank_ = 0;
    int lsaBase_ = 0; // world rank of this team's rank 0
    int nLsaTeams_ = 1;
    bool deviceApiSupport_ = false;

    void SetUp() override
    {
        MPITestBase::SetUp();
    }

    void TearDown() override
    {
        for (auto it = registeredWins_.rbegin(); it != registeredWins_.rend(); ++it) {
            if (it->win && it->comm) {
                ncclCommWindowDeregister(it->comm, it->win);
            }
        }
        registeredWins_.clear();

        for (auto it = allocatedBufs_.rbegin(); it != allocatedBufs_.rend(); ++it) {
            if (it->ptr) {
                ncclMemFree(it->ptr);
            }
        }
        allocatedBufs_.clear();

        if (lsaComm_ != MPI_COMM_NULL) {
            MPI_Comm_free(&lsaComm_);
            lsaComm_ = MPI_COMM_NULL;
        }

        MPITestBase::TearDown();
    }

    void* allocNcclBuf(size_t size)
    {
        void* ptr = nullptr;
        ncclResult_t res = ncclMemAlloc(&ptr, size);
        if (res != ncclSuccess || ptr == nullptr) return nullptr;
        allocatedBufs_.push_back({ptr, size});
        return ptr;
    }

    ncclWindow_t registerWindow(ncclComm_t comm, void* buf, size_t size,
                                int flags = NCCL_WIN_COLL_SYMMETRIC)
    {
        ncclWindow_t win = nullptr;
        ncclResult_t res = ncclCommWindowRegister(comm, buf, size, &win, flags);
        if (res != ncclSuccess) return nullptr;
        registeredWins_.push_back({win, comm});
        return win;
    }

    // No node-count gate: every check below is a property the whole communicator
    // agrees on, so no rank is left alone in a collective.
    bool setupForSymmetric(int minRanks = MIN_RANKS)
    {
        const char* cuMemEnv = std::getenv("NCCL_CUMEM_ENABLE");
        if (!cuMemEnv || std::string(cuMemEnv) != "1") return false;

        if (!validateTestPrerequisites(minRanks)) return false;
        if (createTestCommunicator() != ncclSuccess) return false;

        ncclComm_t comm = getActiveCommunicator();

        // ncclCommWindowRegister returns success and a null window only when both
        // supports are absent; host RMA alone still hands back a real window.
        ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
        if (ncclCommQueryProperties(comm, &props) != ncclSuccess) return false;
        if (!props.deviceApiSupport && !props.hostRmaSupport) return false;
        deviceApiSupport_ = props.deviceApiSupport;
        nLsaTeams_ = props.nLsaTeams;

        if (!setupLsaComm(comm)) return false;

        // Verify ncclMemAlloc works (proxy check for VMM/symmetric support)
        void* testBuf = nullptr;
        ncclResult_t res = ncclMemAlloc(&testBuf, 4096);
        if (res != ncclSuccess || testBuf == nullptr) return false;
        ncclMemFree(testBuf);

        return true;
    }

    // Teams are disjoint blocks of consecutive world ranks, so the team leader's
    // world rank is a valid split color.
    bool setupLsaComm(ncclComm_t comm)
    {
        ncclTeam_t lsa = ncclTeamLsa(comm);
        if (lsa.nRanks <= 0) return false; // device runtime failed to initialize

        lsaSize_ = lsa.nRanks;
        lsaRank_ = lsa.rank;
        lsaBase_ = ncclTeamRankToWorld(comm, lsa, 0);

        int worldRank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

        if (lsaComm_ != MPI_COMM_NULL) MPI_Comm_free(&lsaComm_);
        return MPI_Comm_split(MPI_COMM_WORLD, lsaBase_, worldRank, &lsaComm_) == MPI_SUCCESS;
    }

    uint64_t lsaTeamMin(uint64_t value) const
    {
        uint64_t result = value;
        MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_UINT64_T, MPI_MIN, lsaComm_);
        return result;
    }

    uint64_t lsaTeamMax(uint64_t value) const
    {
        uint64_t result = value;
        MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_UINT64_T, MPI_MAX, lsaComm_);
        return result;
    }

    bool isLsaPeer(int worldRank) const
    {
        return worldRank >= lsaBase_ && worldRank < lsaBase_ + lsaSize_;
    }

    template<typename T>
    void initSendBuffer(void* buffer, size_t count, int rank)
    {
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(buffer, count,
            [rank](size_t) { return static_cast<T>(static_cast<float>(rank + 1)); }));
    }

    template<typename T>
    bool checkAllReduceResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool checkReduceScatterResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool checkAllGatherResult(void* buffer, size_t countPerRank, int nRanks)
    {
        return verifyBufferData<T>(buffer, countPerRank * nRanks,
            [countPerRank](size_t i) {
                int srcRank = i / countPerRank;
                return static_cast<T>(static_cast<float>(srcRank + 1));
            });
    }

    bool setupForAsymmetric(int minRanks = MIN_RANKS)
    {
        if (!setupForSymmetric(minRanks)) return false;

        if (!deviceApiSupport_) return false;

        asymChunkCache() = static_cast<size_t>(allreduceMax(localAsymChunkBytes()));

        TEST_INFO("Rank %d: %d LSA team(s), team size %d, team rank %d, team leader %d, "
                  "chunk %zu bytes", MPIEnvironment::world_rank, nLsaTeams_, lsaSize_,
                  lsaRank_, lsaBase_, asymChunkBytes());
        return true;
    }

    // Element count that is inside every rank's window.
    template<typename T>
    size_t commonCount(size_t localBytes)
    {
        return static_cast<size_t>(allreduceMin(localBytes)) / sizeof(T);
    }

    // allocBytes of 0 allocates exactly what is registered; a larger, rank-uniform
    // value isolates a differing registered size from a differing allocation.
    void registerAsymmetricOnly(SizePattern pattern, size_t allocBytes = 0)
    {
        ncclComm_t comm = getActiveCommunicator();
        int rank, nRanks;
        ncclCommUserRank(comm, &rank);
        ncclCommCount(comm, &nRanks);

        const size_t bufSize   = asymBytes(pattern, rank, nRanks);
        const size_t allocSize = allocBytes != 0 ? allocBytes : bufSize;

        void* buf = allocNcclBuf(allocSize);
        ASSERT_MPI_NE(buf, nullptr);

        ncclWindow_t win = registerWindow(comm, buf, bufSize);
        ASSERT_MPI_NE(win, nullptr);

        const uint64_t minBytes = allreduceMin(bufSize);
        const uint64_t maxBytes = allreduceMax(bufSize);
        ASSERT_MPI_GT(maxBytes, minBytes);

        const uint64_t teamMin = lsaTeamMin(bufSize);
        const uint64_t teamMax = lsaTeamMax(bufSize);
        if (teamMin == teamMax) {
            TEST_WARN("Rank %d: whole LSA team registered %llu bytes; sizes differ only "
                      "between teams here", rank, static_cast<unsigned long long>(teamMin));
        }

        TEST_INFO("Rank %d: %s registered %zu of %zu allocated bytes (job min=%llu "
                  "max=%llu, team min=%llu max=%llu)", rank, patternName(pattern), bufSize,
                  allocSize,
                  static_cast<unsigned long long>(minBytes),
                  static_cast<unsigned long long>(maxBytes),
                  static_cast<unsigned long long>(teamMin),
                  static_cast<unsigned long long>(teamMax));
    }

    // AllReduce over the range every rank's window covers. Bytes registered past
    // that range are fenced with a sentinel that must survive.
    void runAsymAllReduce(SizePattern pattern, bool inPlace, bool registerRecv,
                          bool equalAllocations, int iterations)
    {
        using T = float;
        constexpr T kSentinel = static_cast<T>(-12345.0);

        ncclComm_t comm = getActiveCommunicator();
        hipStream_t stream = getActiveStream();
        int rank, nRanks;
        ncclCommUserRank(comm, &rank);
        ncclCommCount(comm, &nRanks);

        const size_t regSize = asymBytes(pattern, rank, nRanks);
        // Smallest rank-uniform size that fits every rank's registration.
        const size_t allocSize =
            equalAllocations ? static_cast<size_t>(allreduceMax(regSize)) : regSize;

        void* sendBuf = allocNcclBuf(allocSize);
        ASSERT_MPI_NE(sendBuf, nullptr);
        void* recvBuf = inPlace ? sendBuf : allocNcclBuf(allocSize);
        ASSERT_MPI_NE(recvBuf, nullptr);

        ASSERT_MPI_NE(registerWindow(comm, sendBuf, regSize), nullptr);
        if (!inPlace && registerRecv) {
            ASSERT_MPI_NE(registerWindow(comm, recvBuf, regSize), nullptr);
        }

        const size_t count = commonCount<T>(regSize);
        ASSERT_MPI_GT(count, 0u);

        // Zero on the rank owning the job minimum, so the fill and check below
        // report through a local flag rather than diverging on a collective.
        const size_t surplusCount = regSize / sizeof(T) - count;
        T* const surplus = static_cast<T*>(recvBuf) + count;

        hipError_t fillStatus = hipSuccess;
        if (surplusCount > 0) {
            fillStatus = initializeBufferWithPattern<T>(surplus, surplusCount,
                [](size_t) { return kSentinel; });
        }
        ASSERT_MPI_EQ(hipSuccess, fillStatus);

        for (int i = 0; i < iterations; i++) {
            initSendBuffer<T>(sendBuf, count, rank);

            ASSERT_MPI_EQ(ncclSuccess,
                ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

            ASSERT_MPI_TRUE(checkAllReduceResult<T>(recvBuf, count, nRanks));
        }

        bool surplusIntact = true;
        if (surplusCount > 0) {
            surplusIntact = verifyBufferData<T>(surplus, surplusCount,
                [](size_t) { return kSentinel; });
        }
        ASSERT_MPI_TRUE(surplusIntact);

        TEST_INFO("Rank %d: %d x %s AllReduce over %zu of %zu registered (%zu allocated) "
                  "bytes, %zu surplus bytes untouched", rank, iterations,
                  patternName(pattern), count * sizeof(T), regSize, allocSize,
                  surplusCount * sizeof(T));
    }

    // Peers outside the LSA team resolve to nullptr and are skipped, so the
    // result holds only reachable ranks, in increasing world-rank order.
    bool collectPeerBasePointers(ncclWindow_t win, int nRanks,
                                 std::vector<int>& outPeers,
                                 std::vector<uintptr_t>& outPtrs)
    {
        outPeers.clear();
        outPtrs.clear();
        for (int peer = 0; peer < nRanks; peer++) {
            void* ptr = nullptr;
            if (ncclGetPeerDevicePointer(win, 0, peer, &ptr) != ncclSuccess) return false;
            if (ptr == nullptr) continue;
            outPeers.push_back(peer);
            outPtrs.push_back(reinterpret_cast<uintptr_t>(ptr));
        }
        return true;
    }

    // 0 when the pointers are not evenly strided (IPC-backed rather than flat VA).
    static size_t uniformStride(const std::vector<uintptr_t>& ptrs)
    {
        if (ptrs.size() < 2) return 0;
        const uintptr_t stride = ptrs[1] - ptrs[0];
        if (stride == 0) return 0;
        for (size_t i = 2; i < ptrs.size(); i++) {
            if (ptrs[i] - ptrs[i - 1] != stride) return 0;
        }
        return static_cast<size_t>(stride);
    }
};

// ============================================================================
// AllReduce with symmetric windows
// ============================================================================

class SymWin_AllReduce : public SymmetricWindowTestBase {};

TEST_F(SymWin_AllReduce, BothWindows_OutOfPlace)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t count = DEFAULT_COUNT;
    const size_t bufSize = count * sizeof(T);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    void* sendBuf = allocNcclBuf(bufSize);
    void* recvBuf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    ncclWindow_t sendWin = registerWindow(comm, sendBuf, bufSize);
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, bufSize);
    ASSERT_MPI_NE(sendWin, nullptr);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendBuf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllReduceResult<T>(recvBuf, count, nRanks));
    TEST_INFO("Rank %d: BothWindows_OutOfPlace passed", rank);
}

TEST_F(SymWin_AllReduce, OnlySendWindow_OutOfPlace)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t count = DEFAULT_COUNT;
    const size_t bufSize = count * sizeof(T);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    void* sendBuf = allocNcclBuf(bufSize);
    void* recvBuf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    // Register ONLY send buffer
    ncclWindow_t sendWin = registerWindow(comm, sendBuf, bufSize);
    ASSERT_MPI_NE(sendWin, nullptr);
    // recvBuf intentionally NOT registered

    initSendBuffer<T>(sendBuf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllReduceResult<T>(recvBuf, count, nRanks));
    TEST_INFO("Rank %d: OnlySendWindow_OutOfPlace passed (relaxed path)", rank);
}

TEST_F(SymWin_AllReduce, OnlyRecvWindow_OutOfPlace)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t count = DEFAULT_COUNT;
    const size_t bufSize = count * sizeof(T);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    void* sendBuf = allocNcclBuf(bufSize);
    void* recvBuf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    // Register ONLY recv buffer
    // sendBuf intentionally NOT registered
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, bufSize);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendBuf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllReduceResult<T>(recvBuf, count, nRanks));
    TEST_INFO("Rank %d: OnlyRecvWindow_OutOfPlace passed (relaxed path)", rank);
}

TEST_F(SymWin_AllReduce, NoWindows_OutOfPlace)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t count = DEFAULT_COUNT;
    const size_t bufSize = count * sizeof(T);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    void* sendBuf = allocNcclBuf(bufSize);
    void* recvBuf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    // No window registration — non-symmetric fallback path
    initSendBuffer<T>(sendBuf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllReduceResult<T>(recvBuf, count, nRanks));
    TEST_INFO("Rank %d: NoWindows_OutOfPlace passed (non-symmetric fallback)", rank);
}

TEST_F(SymWin_AllReduce, SingleWindow_InPlace)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t count = DEFAULT_COUNT;
    const size_t bufSize = count * sizeof(T);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    void* buf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(buf, nullptr);

    // Single window registration, in-place operation — primary one-buffer use case
    ncclWindow_t win = registerWindow(comm, buf, bufSize);
    ASSERT_MPI_NE(win, nullptr);

    initSendBuffer<T>(buf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllReduceResult<T>(buf, count, nRanks));
    TEST_INFO("Rank %d: SingleWindow_InPlace passed (one-buffer path)", rank);
}

TEST_F(SymWin_AllReduce, BothWindows_Fp8Sum)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    auto runCase = [&](ncclDataType_t dtype, const char* dtypeName, auto typeTag) {
        using Fp8T = decltype(typeTag);
        const std::vector<size_t> counts = {1, 1024, size_t{1} << 16};

        for (size_t count : counts) {
            SCOPED_TRACE(::testing::Message() << "dtype=" << dtypeName << " count=" << count);

            const size_t bufSize = count * sizeof(Fp8T);
            void* sendBuf = allocNcclBuf(bufSize);
            void* recvBuf = allocNcclBuf(bufSize);
            ASSERT_MPI_NE(sendBuf, nullptr);
            ASSERT_MPI_NE(recvBuf, nullptr);

            ncclWindow_t sendWin = registerWindow(comm, sendBuf, bufSize);
            ncclWindow_t recvWin = registerWindow(comm, recvBuf, bufSize);
            ASSERT_MPI_NE(sendWin, nullptr);
            ASSERT_MPI_NE(recvWin, nullptr);

            std::vector<Fp8T> hostSend(count);
            FillFp8AllReduceInput(hostSend, rank);
            std::vector<Fp8T> hostRecv(count, Fp8T(0.0f));
            ASSERT_MPI_EQ(hipSuccess, hipMemcpy(sendBuf, hostSend.data(), bufSize, hipMemcpyHostToDevice));
            ASSERT_MPI_EQ(hipSuccess, hipMemcpy(recvBuf, hostRecv.data(), bufSize, hipMemcpyHostToDevice));

            ASSERT_MPI_EQ(ncclSuccess,
                ncclAllReduce(sendBuf, recvBuf, count, dtype, ncclSum, comm, stream));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

            ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hostRecv.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
            size_t localErrors = 0;
            for (size_t i = 0; i < count; i++) {
                const float expected = ExpectedFp8AllReduceSum<Fp8T>(nRanks, i);
                if (expected != static_cast<float>(hostRecv[i])) {
                    if (localErrors == 0) {
                        TEST_INFO("rank=%d dtype=%s first mismatch at i=%zu: expected=%f got=%f",
                            rank, dtypeName, i, expected, static_cast<float>(hostRecv[i]));
                    }
                    localErrors++;
                }
            }
            ASSERT_MPI_EQ(localErrors, size_t{0});
        }
    };

    runCase(ncclFloat8e4m3, "fp8_e4m3", rccl_float8{});
    runCase(ncclFloat8e5m2, "fp8_e5m2", rccl_bfloat8{});
}

// ============================================================================
// ReduceScatter with symmetric windows
// ============================================================================

class SymWin_ReduceScatter : public SymmetricWindowTestBase {};

TEST_F(SymWin_ReduceScatter, BothWindows)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * nRanks * sizeof(T);
    const size_t recvSize = countPerRank * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    ncclWindow_t sendWin = registerWindow(comm, sendBuf, sendSize);
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, recvSize);
    ASSERT_MPI_NE(sendWin, nullptr);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank * nRanks, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclReduceScatter(sendBuf, recvBuf, countPerRank, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkReduceScatterResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: ReduceScatter BothWindows passed", rank);
}

TEST_F(SymWin_ReduceScatter, OnlySendWindow)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * nRanks * sizeof(T);
    const size_t recvSize = countPerRank * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    // Only send buffer registered
    ncclWindow_t sendWin = registerWindow(comm, sendBuf, sendSize);
    ASSERT_MPI_NE(sendWin, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank * nRanks, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclReduceScatter(sendBuf, recvBuf, countPerRank, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkReduceScatterResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: ReduceScatter OnlySendWindow passed (relaxed path)", rank);
}

TEST_F(SymWin_ReduceScatter, NoWindows)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * nRanks * sizeof(T);
    const size_t recvSize = countPerRank * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank * nRanks, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclReduceScatter(sendBuf, recvBuf, countPerRank, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkReduceScatterResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: ReduceScatter NoWindows passed (non-symmetric fallback)", rank);
}

TEST_F(SymWin_ReduceScatter, BothWindows_Fp8Sum)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    auto runCase = [&](ncclDataType_t dtype, const char* dtypeName, auto typeTag) {
        using Fp8T = decltype(typeTag);
        const std::vector<size_t> counts = {1, 1024, size_t{1} << 16};

        for (size_t countPerRank : counts) {
            SCOPED_TRACE(::testing::Message() << "dtype=" << dtypeName << " count=" << countPerRank);

            const size_t sendElems = countPerRank * nRanks;
            const size_t sendSize = sendElems * sizeof(Fp8T);
            const size_t recvSize = countPerRank * sizeof(Fp8T);

            void* sendBuf = allocNcclBuf(sendSize);
            void* recvBuf = allocNcclBuf(recvSize);
            ASSERT_MPI_NE(sendBuf, nullptr);
            ASSERT_MPI_NE(recvBuf, nullptr);

            ncclWindow_t sendWin = registerWindow(comm, sendBuf, sendSize);
            ncclWindow_t recvWin = registerWindow(comm, recvBuf, recvSize);
            ASSERT_MPI_NE(sendWin, nullptr);
            ASSERT_MPI_NE(recvWin, nullptr);

            std::vector<Fp8T> hostSend(sendElems);
            FillFp8ReduceScatterInput(hostSend, rank);
            std::vector<Fp8T> hostRecv(countPerRank, Fp8T(0.0f));
            ASSERT_MPI_EQ(hipSuccess, hipMemcpy(sendBuf, hostSend.data(), sendSize, hipMemcpyHostToDevice));
            ASSERT_MPI_EQ(hipSuccess, hipMemcpy(recvBuf, hostRecv.data(), recvSize, hipMemcpyHostToDevice));

            ASSERT_MPI_EQ(ncclSuccess,
                ncclReduceScatter(sendBuf, recvBuf, countPerRank, dtype, ncclSum, comm, stream));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

            ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hostRecv.data(), recvBuf, recvSize, hipMemcpyDeviceToHost));
            size_t localErrors = 0;
            for (size_t i = 0; i < countPerRank; i++) {
                const float expected = ExpectedFp8ReduceScatterSum<Fp8T>(rank, nRanks, countPerRank, i);
                if (expected != static_cast<float>(hostRecv[i])) {
                    if (localErrors == 0) {
                        TEST_INFO("rank=%d dtype=%s first mismatch at i=%zu: expected=%f got=%f",
                            rank, dtypeName, i, expected, static_cast<float>(hostRecv[i]));
                    }
                    localErrors++;
                }
            }
            ASSERT_MPI_EQ(localErrors, size_t{0});
        }
    };

    runCase(ncclFloat8e4m3, "fp8_e4m3", rccl_float8{});
    runCase(ncclFloat8e5m2, "fp8_e5m2", rccl_bfloat8{});
}

// ============================================================================
// AllGather with symmetric windows
// ============================================================================

class SymWin_AllGather : public SymmetricWindowTestBase {};

TEST_F(SymWin_AllGather, BothWindows)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * sizeof(T);
    const size_t recvSize = countPerRank * nRanks * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    ncclWindow_t sendWin = registerWindow(comm, sendBuf, sendSize);
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, recvSize);
    ASSERT_MPI_NE(sendWin, nullptr);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllGather(sendBuf, recvBuf, countPerRank, ncclFloat, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllGatherResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: AllGather BothWindows passed", rank);
}

TEST_F(SymWin_AllGather, OnlyRecvWindow)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * sizeof(T);
    const size_t recvSize = countPerRank * nRanks * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    // Only recv buffer registered
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, recvSize);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllGather(sendBuf, recvBuf, countPerRank, ncclFloat, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllGatherResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: AllGather OnlyRecvWindow passed (relaxed path)", rank);
}

TEST_F(SymWin_AllGather, NoWindows)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * sizeof(T);
    const size_t recvSize = countPerRank * nRanks * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllGather(sendBuf, recvBuf, countPerRank, ncclFloat, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllGatherResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: AllGather NoWindows passed (non-symmetric fallback)", rank);
}

// ============================================================================
// Registration with asymmetric (per-rank) window sizes
// ============================================================================

class SymWin_AsymRegister : public SymmetricWindowTestBase {};

TEST_F(SymWin_AsymRegister, AscendingSizes)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    registerAsymmetricOnly(SizePattern::Ascending);
}

TEST_F(SymWin_AsymRegister, DescendingSizes)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    registerAsymmetricOnly(SizePattern::Descending);
}

TEST_F(SymWin_AsymRegister, SingleRankLarger)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    registerAsymmetricOnly(SizePattern::SingleLarger);
}

TEST_F(SymWin_AsymRegister, ExtremeSizeRatio)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    registerAsymmetricOnly(SizePattern::ExtremeRatio);
}

TEST_F(SymWin_AsymRegister, SubRangeOfEqualAllocations)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    int nRanks = 0;
    ncclCommCount(getActiveCommunicator(), &nRanks);

    // Equal allocations isolate a differing registered size from a differing
    // backing allocation.
    registerAsymmetricOnly(SizePattern::Ascending, asymChunkBytes() * nRanks);
}

// ============================================================================
// Collectives on asymmetric windows, restricted to the commonly valid range
// ============================================================================

class SymWin_AsymCollective : public SymmetricWindowTestBase {};

TEST_F(SymWin_AsymCollective, AllReduce_OutOfPlace)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    runAsymAllReduce(SizePattern::Ascending, /*inPlace=*/false, /*registerRecv=*/true,
                     /*equalAllocations=*/false, /*iterations=*/1);
}

TEST_F(SymWin_AsymCollective, AllReduce_InPlace_SingleWindow)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    runAsymAllReduce(SizePattern::Descending, /*inPlace=*/true, /*registerRecv=*/false,
                     /*equalAllocations=*/false, /*iterations=*/1);
}

TEST_F(SymWin_AsymCollective, AllReduce_OnlySendWindow)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    runAsymAllReduce(SizePattern::SingleLarger, /*inPlace=*/false, /*registerRecv=*/false,
                     /*equalAllocations=*/false, /*iterations=*/1);
}

TEST_F(SymWin_AsymCollective, AllReduce_SubRangeOfEqualAllocations)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    runAsymAllReduce(SizePattern::Ascending, /*inPlace=*/false, /*registerRecv=*/true,
                     /*equalAllocations=*/true, /*iterations=*/1);
}

TEST_F(SymWin_AsymCollective, AllGather_PaddedWindows)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // AllGather needs one count for all ranks, so rank-dependent padding is what
    // makes the window sizes differ while the used range stays valid everywhere.
    const size_t chunk    = asymChunkBytes();
    const size_t padding  = chunk * rank;
    const size_t sendSize = chunk + padding;
    const size_t recvSize = chunk * nRanks + padding;

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    ncclWindow_t sendWin = registerWindow(comm, sendBuf, sendSize);
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, recvSize);
    ASSERT_MPI_NE(sendWin, nullptr);
    ASSERT_MPI_NE(recvWin, nullptr);

    const size_t countPerRank = chunk / sizeof(T);
    initSendBuffer<T>(sendBuf, countPerRank, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllGather(sendBuf, recvBuf, countPerRank, ncclFloat, comm, stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_MPI_TRUE(checkAllGatherResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: AllGather with %zu/%zu byte windows passed", rank, sendSize, recvSize);
}

TEST_F(SymWin_AsymCollective, ReduceScatter_PaddedWindows)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t chunk    = asymChunkBytes();
    const size_t padding  = chunk * rank;
    const size_t sendSize = chunk * nRanks + padding;
    const size_t recvSize = chunk + padding;

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    ncclWindow_t sendWin = registerWindow(comm, sendBuf, sendSize);
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, recvSize);
    ASSERT_MPI_NE(sendWin, nullptr);
    ASSERT_MPI_NE(recvWin, nullptr);

    const size_t countPerRank = chunk / sizeof(T);
    initSendBuffer<T>(sendBuf, countPerRank * nRanks, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclReduceScatter(sendBuf, recvBuf, countPerRank, ncclFloat, ncclSum, comm, stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_MPI_TRUE(checkReduceScatterResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: ReduceScatter with %zu/%zu byte windows passed", rank, sendSize, recvSize);
}

TEST_F(SymWin_AsymCollective, AllReduce_RepeatedIterations)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    // Catches per-op state derived from the first operation only.
    runAsymAllReduce(SizePattern::ExtremeRatio, /*inPlace=*/true, /*registerRecv=*/false,
                     /*equalAllocations=*/false, /*iterations=*/3);
}

// ============================================================================
// Peer access into asymmetric windows, scoped to the LSA team
// ============================================================================

class SymWin_AsymLsa : public SymmetricWindowTestBase {};

TEST_F(SymWin_AsymLsa, PeerContent_WithinCommonRange)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t bufSize = asymBytes(SizePattern::Ascending, rank, nRanks);

    void* buf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(buf, nullptr);

    ncclWindow_t win = registerWindow(comm, buf, bufSize);
    ASSERT_MPI_NE(win, nullptr);

    const size_t localCount  = bufSize / sizeof(T);
    const size_t commonElems = commonCount<T>(bufSize);
    ASSERT_MPI_GT(commonElems, 0u);

    initSendBuffer<T>(buf, localCount, rank);

    float* dSamples = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&dSamples, 3 * sizeof(float)));
    auto samplesCleanup = makeScopeGuard([&]() { if (dSamples) (void)hipFree(dSamples); });

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> peers;
    std::vector<uintptr_t> peerPtrs;
    ASSERT_MPI_TRUE(collectPeerBasePointers(win, nRanks, peers, peerPtrs));
    ASSERT_MPI_GT(peers.size(), 0u);

    // A pointer resolving outside the team would promise load/store access across
    // nodes; a missing one would mean a team member became unreachable.
    ASSERT_MPI_EQ(peers.size(), static_cast<size_t>(lsaSize_));
    bool peersWithinTeam = true;
    for (int peer : peers) {
        if (!isLsaPeer(peer)) {
            TEST_WARN("Rank %d: peer %d resolved but is outside LSA team [%d,%d)", rank,
                      peer, lsaBase_, lsaBase_ + lsaSize_);
            peersWithinTeam = false;
        }
    }
    ASSERT_MPI_TRUE(peersWithinTeam);

    // Failures are accumulated so the per-peer loop issues no collectives itself.
    bool allOk = true;
    for (size_t i = 0; i < peers.size(); i++) {
        const int peer = peers[i];
        samplePeerFloatsKernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<const float*>(peerPtrs[i]), commonElems, dSamples);
        if (hipStreamSynchronize(stream) != hipSuccess) {
            TEST_WARN("Rank %d: peer %d sample kernel failed", rank, peer);
            allOk = false;
            continue;
        }

        float samples[3] = {0.f, 0.f, 0.f};
        if (hipMemcpy(samples, dSamples, sizeof(samples), hipMemcpyDeviceToHost) != hipSuccess) {
            TEST_WARN("Rank %d: peer %d sample copy failed", rank, peer);
            allOk = false;
            continue;
        }

        const float expected = static_cast<float>(peer + 1);
        for (float sample : samples) {
            if (sample != expected) {
                TEST_WARN("Rank %d: peer %d sample %f != expected %f", rank, peer,
                          sample, expected);
                allOk = false;
            }
        }
    }
    ASSERT_MPI_TRUE(allOk);

    MPI_Barrier(MPI_COMM_WORLD);
    TEST_INFO("Rank %d: verified %zu peer windows in LSA team of %d over %zu common bytes",
              rank, peers.size(), lsaSize_, commonElems * sizeof(T));
}

TEST_F(SymWin_AsymLsa, PeerPointerStride_IndependentOfRankSizes)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // Asymmetric first, so the window after it must clear the team maximum
    // rather than this rank's own size.
    const size_t asymSize = asymBytes(SizePattern::SingleLarger, rank, nRanks);
    void* asymBuf = allocNcclBuf(asymSize);
    ASSERT_MPI_NE(asymBuf, nullptr);
    ncclWindow_t asymWin = registerWindow(comm, asymBuf, asymSize);
    ASSERT_MPI_NE(asymWin, nullptr);

    std::vector<int> asymPeers;
    std::vector<uintptr_t> asymPtrs;
    ASSERT_MPI_TRUE(collectPeerBasePointers(asymWin, nRanks, asymPeers, asymPtrs));
    const size_t asymStride = uniformStride(asymPtrs);

    // A stride needs two reachable ranks and a flat-VA mapping; the logged team
    // size tells a single-member team apart from an IPC-backed window.
    TEST_INFO("Rank %d: LSA team size %d, stride %zu", rank, lsaSize_, asymStride);
    if (auto reason = mpiCoordinatedSkipReason(asymStride == 0,
            "Needs an LSA team of 2+ ranks with a flat-VA symmetric mapping");
        !reason.empty()) {
        GTEST_SKIP() << reason;
    }

    // Equal-size baseline the asymmetric sizes must not have moved.
    const size_t symSize = asymChunkBytes();
    void* symBuf = allocNcclBuf(symSize);
    ASSERT_MPI_NE(symBuf, nullptr);
    ncclWindow_t symWin = registerWindow(comm, symBuf, symSize);
    ASSERT_MPI_NE(symWin, nullptr);

    std::vector<int> symPeers;
    std::vector<uintptr_t> symPtrs;
    ASSERT_MPI_TRUE(collectPeerBasePointers(symWin, nRanks, symPeers, symPtrs));
    const size_t symStride = uniformStride(symPtrs);

    // The stride is the per-rank flat-VA reservation, fixed at init.
    ASSERT_MPI_EQ(symStride, asymStride);

    // Window slots are sized from the LSA team maximum, not from what this rank
    // registered, so ranks holding one chunk must still clear chunk * nRanks.
    ASSERT_MPI_EQ(symPeers[0], asymPeers[0]);
    const size_t asymTeamMax = static_cast<size_t>(lsaTeamMax(asymSize));
    const size_t symTeamMax  = static_cast<size_t>(lsaTeamMax(symSize));

    const uintptr_t asymSlot = asymPtrs[0];
    const uintptr_t symSlot  = symPtrs[0];
    const size_t slotGap     = asymSlot < symSlot ? symSlot - asymSlot : asymSlot - symSlot;
    const size_t requiredGap = asymSlot < symSlot ? asymTeamMax : symTeamMax;

    ASSERT_MPI_TRUE(slotGap >= requiredGap);

    TEST_INFO("Rank %d: peer stride %zu unchanged with asymmetric sizes; slots %zu apart "
              "for a team maximum of %zu (own registration %zu)", rank, asymStride,
              slotGap, requiredGap, asymSize);
}

TEST_F(SymWin_AsymLsa, PointerOffsetAtLocalWindowEnd_Rejected)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t bufSize = asymBytes(SizePattern::Ascending, rank, nRanks);

    void* buf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(buf, nullptr);

    ncclWindow_t win = registerWindow(comm, buf, bufSize);
    ASSERT_MPI_NE(win, nullptr);

    void* ptr = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclGetPeerDevicePointer(win, bufSize - 1, rank, &ptr));
    ASSERT_MPI_NE(ptr, nullptr);

    ptr = nullptr;
    ASSERT_MPI_EQ(ncclInvalidArgument, ncclGetPeerDevicePointer(win, bufSize, rank, &ptr));

    // The bound is the local window size only: an offset past a smaller peer's
    // end still resolves. Ascending puts the smallest window on lsaBase_.
    const size_t peerSize = asymBytes(SizePattern::Ascending, lsaBase_, nRanks);
    bool peerOverrunResolves = true;
    if (bufSize > peerSize) {
        void* peerPtr = nullptr;
        peerOverrunResolves =
            ncclGetPeerDevicePointer(win, peerSize, lsaBase_, &peerPtr) == ncclSuccess &&
            peerPtr != nullptr;
    }
    ASSERT_MPI_TRUE(peerOverrunResolves);

    TEST_INFO("Rank %d: offset bound of a %zu byte window enforced locally; peer %d "
              "registered %zu and is not bounds-checked here", rank, bufSize, lsaBase_,
              peerSize);
}

// ============================================================================
// Window lifecycle tests
// ============================================================================

class SymWin_WindowLifecycle : public SymmetricWindowTestBase {};

TEST_F(SymWin_WindowLifecycle, RegisterDeregister_Basic)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank;
    ncclCommUserRank(comm, &rank);

    const size_t bufSize = 1024 * 1024;
    void* buf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(buf, nullptr);

    ncclWindow_t win = registerWindow(comm, buf, bufSize);
    ASSERT_MPI_NE(win, nullptr);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowDeregister(comm, win));

    // Null out tracked entry so TearDown skips double-deregister
    registeredWins_.back().win = nullptr;

    TEST_INFO("Rank %d: RegisterDeregister_Basic passed", rank);
}

TEST_F(SymWin_WindowLifecycle, MultipleWindows)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank;
    ncclCommUserRank(comm, &rank);

    const int numWindows = 4;
    const size_t bufSize = 256 * 1024;
    std::vector<ncclWindow_t> wins(numWindows);

    for (int i = 0; i < numWindows; i++) {
        void* buf = allocNcclBuf(bufSize);
        ASSERT_MPI_NE(buf, nullptr);

        wins[i] = registerWindow(comm, buf, bufSize);
        ASSERT_MPI_NE(wins[i], nullptr);
    }

    // Verify all windows are unique
    for (int i = 0; i < numWindows; i++) {
        for (int j = i + 1; j < numWindows; j++) {
            ASSERT_MPI_NE(wins[i], wins[j]);
        }
    }

    TEST_INFO("Rank %d: MultipleWindows passed (%d windows)", rank, numWindows);
}

TEST_F(SymWin_WindowLifecycle, RepeatedRegisterDeregister)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank;
    ncclCommUserRank(comm, &rank);

    const size_t bufSize = 512 * 1024;
    void* buf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(buf, nullptr);

    const int iterations = 3;
    for (int i = 0; i < iterations; i++) {
        ncclWindow_t win = nullptr;
        ASSERT_MPI_EQ(ncclSuccess,
            ncclCommWindowRegister(comm, buf, bufSize, &win, NCCL_WIN_COLL_SYMMETRIC));
        ASSERT_MPI_NE(win, nullptr);

        ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowDeregister(comm, win));
    }

    TEST_INFO("Rank %d: RepeatedRegisterDeregister passed (%d iterations)",
              rank, iterations);
}

TEST_F(SymWin_WindowLifecycle, MultipleAsymmetricWindows)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const SizePattern patterns[] = {SizePattern::Ascending,
                                    SizePattern::Descending,
                                    SizePattern::SingleLarger,
                                    SizePattern::ExtremeRatio};
    const int numWindows = static_cast<int>(sizeof(patterns) / sizeof(patterns[0]));

    std::vector<ncclWindow_t> wins(numWindows);
    for (int i = 0; i < numWindows; i++) {
        const size_t bufSize = asymBytes(patterns[i], rank, nRanks);

        void* buf = allocNcclBuf(bufSize);
        ASSERT_MPI_NE(buf, nullptr);

        wins[i] = registerWindow(comm, buf, bufSize);
        ASSERT_MPI_NE(wins[i], nullptr);

        ASSERT_MPI_GT(allreduceMax(bufSize), allreduceMin(bufSize));
    }

    for (int i = 0; i < numWindows; i++) {
        for (int j = i + 1; j < numWindows; j++) {
            ASSERT_MPI_NE(wins[i], wins[j]);
        }
    }

    TEST_INFO("Rank %d: MultipleAsymmetricWindows passed (%d windows)", rank, numWindows);
}

TEST_F(SymWin_WindowLifecycle, RepeatedRegisterDeregister_AsymmetricSizes)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t bufSize = asymBytes(SizePattern::Ascending, rank, nRanks);
    void* buf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(buf, nullptr);

    const int iterations = 3;
    for (int i = 0; i < iterations; i++) {
        ncclWindow_t win = nullptr;
        ASSERT_MPI_EQ(ncclSuccess,
            ncclCommWindowRegister(comm, buf, bufSize, &win, NCCL_WIN_COLL_SYMMETRIC));
        ASSERT_MPI_NE(win, nullptr);

        ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowDeregister(comm, win));
    }

    TEST_INFO("Rank %d: %d asymmetric register/deregister cycles at %zu bytes passed",
              rank, iterations, bufSize);
}

TEST_F(SymWin_WindowLifecycle, ReregisterWithDifferentAsymmetricPattern)
{
    if (!setupForAsymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // One allocation large enough for either pattern, so only the registered size
    // changes between the two registrations.
    void* buf = allocNcclBuf(asymChunkBytes() * nRanks);
    ASSERT_MPI_NE(buf, nullptr);

    const size_t firstSize = asymBytes(SizePattern::Ascending, rank, nRanks);
    ncclWindow_t firstWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(comm, buf, firstSize, &firstWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_NE(firstWin, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowDeregister(comm, firstWin));

    // Which rank registers the most flips between the two patterns.
    const size_t secondSize = asymBytes(SizePattern::Descending, rank, nRanks);
    ncclWindow_t secondWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
        ncclCommWindowRegister(comm, buf, secondSize, &secondWin, NCCL_WIN_COLL_SYMMETRIC));
    ASSERT_MPI_NE(secondWin, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowDeregister(comm, secondWin));

    TEST_INFO("Rank %d: re-registered %zu bytes after %zu bytes", rank, secondSize, firstSize);
}

#endif // MPI_TESTS_ENABLED
