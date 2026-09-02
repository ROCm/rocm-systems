/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-GPU end-to-end dispatch and combine for rccl_ep.
//
// The single-GPU tests in device/RcclEpTests.cpp cover the wave primitives and
// the window layout arithmetic. Everything below needs peer memory and a
// communicator, so it runs one process per rank under MPI, in the same shape as
// SymmetricWindowMPITests.cpp. Dispatch and combine move data over LSA peer memory,
// so the eight ranks have to share a node.
//
// Build and run:
//   ./install.sh --debug -t --enable-mpi-tests --rccl-ep-tests
//   NCCL_CUMEM_ENABLE=1 mpirun -np 8 --bind-to none \
//     ./build/debug/test/rccl-UnitTestsMPI --gtest_filter=RcclEpDispatchCombineTest.*
//
// What is asserted, and why the expected value is computable on the host: dispatch
// stages a token once per DESTINATION RANK rather than once per expert, and the
// top-k weights ride along as payload without being applied by the reduction.
// So handing combine exactly what dispatch produced -- an identity expert --
// returns each token scaled by the number of distinct ranks that own its top-k
// experts. That single equality exercises the routing plan, the peer stores, the
// receive-side concatenation and the reduction together, and it is computable on
// the host from topk_idx alone.

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;

// rccl_ep is reached through its flat C ABI; it is header-only otherwise and is
// not hipified, so nothing here includes its device headers.
extern "C" {
int ep_unique_id_size();
int ep_get_unique_id(char* out);
void* ep_create(int rank, int num_ranks, const char* unique_id, int max_tokens_per_rank, int hidden,
                int device);
int ep_configure(void* handle, int num_experts, int num_topk);
void ep_destroy(void* handle);
int ep_plan(void* handle, uintptr_t topk_idx, int num_tokens, uintptr_t slot, uintptr_t send_list,
            uintptr_t sendc, uintptr_t stream);
int ep_dispatch(void* handle, uintptr_t x, uintptr_t x_sf, uintptr_t topk_idx, uintptr_t topk_w,
                int num_tokens, uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                uintptr_t out_x, uintptr_t out_sf, uintptr_t out_topk, uintptr_t out_tw,
                uintptr_t out_src, uintptr_t stream);
int ep_combine(void* handle, uintptr_t y, uintptr_t in_w, uintptr_t row_map, uintptr_t recv_topk,
               uintptr_t recv_src, int num_recv, uintptr_t topk_idx, int num_tokens,
               uintptr_t bias0, uintptr_t bias1, int grouped, uintptr_t out, uintptr_t out_w,
               int num_sms, uintptr_t stream);
}

namespace RcclUnitTesting
{

namespace {

// The MoE shape these kernels are built for. Anything much smaller measures the fixed
// per-call cost (launch plus the device-side barrier) rather than the transfer: at
// 256 x 1024 that floor is ~62 us against ~5 us of payload.
constexpr int kRanks         = 8;     // one node's worth; rccl_ep dispatch is intranode
constexpr int kTokens        = 4096;
constexpr int kHidden        = 7168;  // multiple of 8: ep_create rejects odd hidden
constexpr int kTopk          = 6;
constexpr int kExpertsPerRank = 32;   // 256 experts across eight ranks
constexpr int kNumSms        = 32;    // past saturation; 64 measures slower, not faster
constexpr int kWarmup        = 10;
constexpr int kIters         = 20;   // each timed separately; the minimum is reported

// Values of the form 1 + m/128 land in [1, 2), where bf16 spacing is exactly 1/128, so
// the payload survives the round trip bit-for-bit. m encodes the source rank, so a row
// delivered from the wrong rank is visible rather than matching by coincidence. Tokens
// 16 apart on the same rank do alias.
float TokenValue(int rank, int token)
{
    return 1.0f + static_cast<float>((rank * 16 + (token % 16)) % 128) / 128.0f;
}

// Experts are chosen by OWNER rather than by index, so the number of distinct ranks a
// token reaches actually varies. With a flat index expression that count is constant at
// every interesting rank count -- always 4 at eight ranks -- which would make the
// assertion below indistinguishable from a plain "scaled by num_topk" check and would
// never exercise combine's per-rank grouping.
int ExpertFor(int rank, int token, int k, int numExperts, int nRanks)
{
    const int perRank = numExperts / nRanks;
    // Every third token deliberately puts two of its experts on one rank.
    const int owner = ((token % 3) == 0 && k == 1)
                          ? ((token * 5) + rank) % nRanks
                          : ((token * 5) + (k * 7) + rank) % nRanks;
    return owner * perRank + ((token + k) % perRank);
}

class RcclEpDispatchCombineTest : public MPITestBase
{
protected:
    void* handle_ = nullptr;
    int   rank_ = 0, nRanks_ = 0, device_ = 0, numExperts_ = 0;

    void TearDown() override
    {
        if (handle_ != nullptr) {
            ep_destroy(handle_);
            handle_ = nullptr;
        }
        MPITestBase::TearDown();
    }

    // Mirrors SymmetricWindowMPITests' setupForSymmetric: refuse rather than fail on a
    // runtime without symmetric memory. ep_configure is the real probe -- it is what
    // allocates and registers the window.
    bool SetupEp()
    {
        if (!validateTestPrerequisites(kRanks, kNoProcessLimit, kNoPowerOfTwoRequired,
                                       kRequireSingleNode, kRequireSingleNode)) {
            return false;
        }
        const char* cuMemEnv = std::getenv("NCCL_CUMEM_ENABLE");
        if (cuMemEnv == nullptr || std::string(cuMemEnv) != "1") return false;

        rank_   = MPIEnvironment::world_rank;
        nRanks_ = MPIEnvironment::world_size;

        // MPIEnvironment has already bound this rank to its local device; adopt that
        // instead of re-deriving one from the global rank. This is a per-rank condition
        // and the collectives below are not, so disagreement is resolved before the first.
        int localOk = (hipGetDevice(&device_) == hipSuccess) ? 1 : 0;
        int deviceOk = 0;
        if (MPI_Allreduce(&localOk, &deviceOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS) return false;
        if (deviceOk == 0) return false;

        // A multiple of the rank count, so the owner of an expert is simply
        // index / experts_per_rank.
        numExperts_ = nRanks_ * kExpertsPerRank;

        // Every rank must reach the same decision. A rank-0-only early return here would
        // leave the others blocked in the collective below, hanging the job instead of
        // failing it, so the outcome is agreed with an Allreduce.
        std::vector<char> uid(static_cast<size_t>(ep_unique_id_size()), 0);
        int ok = 1;
        if (rank_ == 0 && ep_get_unique_id(uid.data()) != 0) ok = 0;
        if (MPI_Bcast(uid.data(), static_cast<int>(uid.size()), MPI_BYTE, 0, MPI_COMM_WORLD) !=
            MPI_SUCCESS) {
            ok = 0;
        }
        int allOk = 0;
        if (MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS) return false;
        if (allOk == 0) return false;

        handle_ = ep_create(rank_, nRanks_, uid.data(), kTokens, kHidden, device_);
        if (handle_ == nullptr) return false;
        return ep_configure(handle_, numExperts_, kTopk) == 0;
    }

    int DistinctDestinations(const std::vector<int32_t>& topk, int token) const
    {
        const int expertsPerRank = numExperts_ / nRanks_;
        std::set<int> owners;
        for (int k = 0; k < kTopk; ++k) {
            owners.insert(topk[static_cast<size_t>(token) * kTopk + k] / expertsPerRank);
        }
        return static_cast<int>(owners.size());
    }
};

// Plain hipMalloc: only rccl_ep's own window has to be symmetric, not the payload.
struct EpBuffers
{
    void* x = nullptr;        // [T, H]      bf16 (or fp8 when dispatching fp8)
    void* xSf = nullptr;      // [T, H/128]  float, the fp8 input scales
    void* topkIdx = nullptr;  // [T, K]      int32
    void* topkW = nullptr;    // [T, K]      float
    void* slot = nullptr;     // [R, T]      int32
    void* sendList = nullptr; // [R, T]      int32
    void* sendc = nullptr;    // [R]         int32
    void* outX = nullptr;     // [cap, H]
    void* outSf = nullptr;    // [cap, H/128] float, fp8 only
    void* outTopk = nullptr;  // [cap, K]    int32
    void* outTw = nullptr;    // [cap, K]    float
    void* outSrc = nullptr;   // [cap]       int32
    void* combined = nullptr; // [T, H]      bf16

    void Free()
    {
        for (void* p : {x, xSf, topkIdx, topkW, slot, sendList, sendc, outX, outSf, outTopk, outTw,
                        outSrc, combined}) {
            if (p != nullptr) (void)hipFree(p);
        }
    }
};

double GigabytesPerSecond(size_t bytes, float milliseconds)
{
    if (milliseconds <= 0.0f) return 0.0;
    return static_cast<double>(bytes) / (static_cast<double>(milliseconds) * 1.0e6);
}

// A received row costs its payload plus the routing metadata that travels with it; both
// cross the link, so both count toward the achieved rate.
size_t DispatchBytesPerRow(bool useFp8, size_t hiddenSf)
{
    const size_t payload =
        useFp8 ? (kHidden + hiddenSf * sizeof(float)) : (kHidden * sizeof(__hip_bfloat16));
    return payload + kTopk * sizeof(int32_t) + kTopk * sizeof(float);
}

// Combine carries the payload back with its weights, but not the expert indices.
size_t CombineBytesPerRow() { return kHidden * sizeof(__hip_bfloat16) + kTopk * sizeof(float); }

} // namespace

TEST_F(RcclEpDispatchCombineTest, DispatchCombineRoundTrip)
{
    if (!SetupEp()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }

    const int    cap       = nRanks_ * kTokens;
    const size_t hiddenSf  = (kHidden + 127) / 128;
    hipStream_t  stream    = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    std::vector<__hip_bfloat16> hX(static_cast<size_t>(kTokens) * kHidden);
    std::vector<int32_t>        hTopk(static_cast<size_t>(kTokens) * kTopk);
    std::vector<float>          hTopkW(static_cast<size_t>(kTokens) * kTopk, 1.0f / kTopk);
    for (int t = 0; t < kTokens; ++t) {
        const __hip_bfloat16 v = __float2bfloat16(TokenValue(rank_, t));
        std::fill_n(hX.begin() + static_cast<size_t>(t) * kHidden, kHidden, v);
        for (int k = 0; k < kTopk; ++k) {
            hTopk[static_cast<size_t>(t) * kTopk + k] = ExpertFor(rank_, t, k, numExperts_, nRanks_);
        }
    }

    EpBuffers b;
    SCOPE_EXIT(b.Free());
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.x,        hX.size() * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.topkIdx,  hTopk.size() * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.topkW,    hTopkW.size() * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.slot,     static_cast<size_t>(nRanks_) * kTokens * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.sendList, static_cast<size_t>(nRanks_) * kTokens * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.sendc,    static_cast<size_t>(nRanks_) * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outX,     static_cast<size_t>(cap) * kHidden * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outSf,    static_cast<size_t>(cap) * hiddenSf * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outTopk,  static_cast<size_t>(cap) * kTopk * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outTw,    static_cast<size_t>(cap) * kTopk * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outSrc,   static_cast<size_t>(cap) * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.combined, static_cast<size_t>(kTokens) * kHidden * sizeof(__hip_bfloat16)));

    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(b.x, hX.data(), hX.size() * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(b.topkIdx, hTopk.data(), hTopk.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(b.topkW, hTopkW.data(), hTopkW.size() * sizeof(float), hipMemcpyHostToDevice));

    ASSERT_MPI_EQ(0, ep_plan(handle_, (uintptr_t)b.topkIdx, kTokens, (uintptr_t)b.slot,
                             (uintptr_t)b.sendList, (uintptr_t)b.sendc, (uintptr_t)stream));

    const int nRecv = ep_dispatch(handle_, (uintptr_t)b.x, 0, (uintptr_t)b.topkIdx,
                                  (uintptr_t)b.topkW, kTokens, (uintptr_t)b.sendList,
                                  (uintptr_t)b.sendc, /*use_fp8=*/0, kNumSms, (uintptr_t)b.outX, 0,
                                  (uintptr_t)b.outTopk, (uintptr_t)b.outTw, (uintptr_t)b.outSrc,
                                  (uintptr_t)stream);
    ASSERT_MPI_TRUE(nRecv >= 0);
    ASSERT_MPI_TRUE(nRecv <= cap);
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Poison the destination first: 0xFF bytes read back as NaN in bf16, so a token
    // combine never writes fails the comparison instead of matching whatever the
    // allocator happened to leave there. Zero-filling would not do -- rank 0 token 0
    // legitimately expects 0.
    ASSERT_MPI_EQ(hipSuccess, hipMemset(b.combined, 0xFF,
                                        static_cast<size_t>(kTokens) * kHidden * sizeof(__hip_bfloat16)));

    // Identity expert: combine is handed exactly what dispatch produced.
    ASSERT_MPI_EQ(0, ep_combine(handle_, (uintptr_t)b.outX, 0, /*row_map=*/0, (uintptr_t)b.outTopk,
                                (uintptr_t)b.outSrc, nRecv, (uintptr_t)b.topkIdx, kTokens, 0, 0,
                                /*grouped=*/1, (uintptr_t)b.combined, 0, kNumSms, (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    std::vector<__hip_bfloat16> hOut(static_cast<size_t>(kTokens) * kHidden);
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(hOut.data(), b.combined, hOut.size() * sizeof(__hip_bfloat16),
                                        hipMemcpyDeviceToHost));

    // A uniform spread would be satisfied by any implementation that scaled by a constant.
    {
        std::set<int> spread;
        for (int t = 0; t < kTokens; ++t) spread.insert(DistinctDestinations(hTopk, t));
        ASSERT_MPI_GT(spread.size(), 1u);
    }

    int mismatches = 0;
    for (int t = 0; t < kTokens && mismatches < 8; ++t) {
        const float stored = __bfloat162float(__float2bfloat16(TokenValue(rank_, t)));
        const float expected = stored * static_cast<float>(DistinctDestinations(hTopk, t));
        // bf16 keeps ~8 significant bits, so scale the tolerance with the value.
        const float tol = 0.03f * std::max(1.0f, expected);
        for (int j = 0; j < kHidden; j += 97) {  // stride: every element of a row is identical
            const float got = __bfloat162float(hOut[static_cast<size_t>(t) * kHidden + j]);
            // Negated rather than `> tol`: the poison above reads back as NaN, and every
            // comparison against NaN is false, so `> tol` would let an unwritten row pass.
            if (!(std::abs(got - expected) <= tol)) {
                ADD_FAILURE() << "rank " << rank_ << " token " << t << " element " << j
                              << ": got " << got << ", expected " << expected << " (= value "
                              << stored << " x " << DistinctDestinations(hTopk, t)
                              << " destination ranks)";
                ++mismatches;
                break;
            }
        }
    }

}

TEST_F(RcclEpDispatchCombineTest, DispatchBf16AndFp8Bandwidth)
{
    if (!SetupEp()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }

    const int    cap      = nRanks_ * kTokens;
    const size_t hiddenSf = (kHidden + 127) / 128;
    hipStream_t  stream   = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    std::vector<__hip_bfloat16> hX(static_cast<size_t>(kTokens) * kHidden, __float2bfloat16(1.0f));
    std::vector<int32_t>        hTopk(static_cast<size_t>(kTokens) * kTopk);
    std::vector<float>          hTopkW(static_cast<size_t>(kTokens) * kTopk, 1.0f / kTopk);
    for (int t = 0; t < kTokens; ++t) {
        for (int k = 0; k < kTopk; ++k) {
            hTopk[static_cast<size_t>(t) * kTopk + k] = ExpertFor(rank_, t, k, numExperts_, nRanks_);
        }
    }

    EpBuffers b;
    SCOPE_EXIT(b.Free());
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.x,        hX.size() * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.topkIdx,  hTopk.size() * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.topkW,    hTopkW.size() * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.slot,     static_cast<size_t>(nRanks_) * kTokens * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.sendList, static_cast<size_t>(nRanks_) * kTokens * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.sendc,    static_cast<size_t>(nRanks_) * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outX,     static_cast<size_t>(cap) * kHidden * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outSf,    static_cast<size_t>(cap) * hiddenSf * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outTopk,  static_cast<size_t>(cap) * kTopk * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outTw,    static_cast<size_t>(cap) * kTopk * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.outSrc,   static_cast<size_t>(cap) * sizeof(int32_t)));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.combined, static_cast<size_t>(kTokens) * kHidden * sizeof(__hip_bfloat16)));

    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(b.x, hX.data(), hX.size() * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemcpy(b.topkIdx, hTopk.data(), hTopk.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(b.topkW, hTopkW.data(), hTopkW.size() * sizeof(float), hipMemcpyHostToDevice));

    // The fp8 phase must be given real scales: ep_dispatch casts x_sf straight through
    // and device/dispatch.h indexes it unconditionally, so a null here is a device-side
    // null dereference rather than a skipped copy.
    const std::vector<float> hXsf(static_cast<size_t>(kTokens) * hiddenSf, 1.0f);
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&b.xSf, hXsf.size() * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMemcpy(b.xSf, hXsf.data(), hXsf.size() * sizeof(float), hipMemcpyHostToDevice));

    ASSERT_MPI_EQ(0, ep_plan(handle_, (uintptr_t)b.topkIdx, kTokens, (uintptr_t)b.slot,
                             (uintptr_t)b.sendList, (uintptr_t)b.sendc, (uintptr_t)stream));

    hipEvent_t start, stop;
    ASSERT_MPI_EQ(hipSuccess, hipEventCreate(&start));
    ASSERT_MPI_EQ(hipSuccess, hipEventCreate(&stop));
    SCOPE_EXIT((void)hipEventDestroy(start); (void)hipEventDestroy(stop));

    struct Phase { const char* name; int useFp8; };
    for (const Phase& phase : {Phase{"dispatch bf16", 0}, Phase{"dispatch fp8", 1}}) {
        SCOPED_TRACE(phase.name);
        int nRecv = -1;
        for (int i = 0; i < kWarmup; ++i) {
            nRecv = ep_dispatch(handle_, (uintptr_t)b.x, phase.useFp8 ? (uintptr_t)b.xSf : 0,
                                (uintptr_t)b.topkIdx,
                                (uintptr_t)b.topkW, kTokens, (uintptr_t)b.sendList,
                                (uintptr_t)b.sendc, phase.useFp8, kNumSms, (uintptr_t)b.outX,
                                phase.useFp8 ? (uintptr_t)b.outSf : 0, (uintptr_t)b.outTopk,
                                (uintptr_t)b.outTw, (uintptr_t)b.outSrc, (uintptr_t)stream);
            ASSERT_MPI_TRUE(nRecv >= 0);
        }
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

        // Each iteration is timed on its own, behind a barrier. Without the barrier the
        // in-kernel peer wait absorbs the spread in host launch times and the sample
        // measures rank skew rather than the transfer. Errors are accumulated instead of
        // asserted in place: every rank must reach the same number of barriers, so a rank
        // returning early here would hang the rest.
        std::vector<float> samples;
        samples.reserve(kIters);
        bool timingOk = true;
        for (int i = 0; i < kIters; ++i) {
            timingOk &= (hipStreamSynchronize(stream) == hipSuccess);
            MPI_Barrier(MPI_COMM_WORLD);
            timingOk &= (hipEventRecord(start, stream) == hipSuccess);
            (void)ep_dispatch(handle_, (uintptr_t)b.x, phase.useFp8 ? (uintptr_t)b.xSf : 0,
                              (uintptr_t)b.topkIdx, (uintptr_t)b.topkW,
                              kTokens, (uintptr_t)b.sendList, (uintptr_t)b.sendc, phase.useFp8,
                              kNumSms, (uintptr_t)b.outX, phase.useFp8 ? (uintptr_t)b.outSf : 0,
                              (uintptr_t)b.outTopk, (uintptr_t)b.outTw, (uintptr_t)b.outSrc,
                              (uintptr_t)stream);
            timingOk &= (hipEventRecord(stop, stream) == hipSuccess);
            timingOk &= (hipEventSynchronize(stop) == hipSuccess);
            float ms = 0.0f;
            timingOk &= (hipEventElapsedTime(&ms, start, stop) == hipSuccess);
            samples.push_back(ms);
        }
        ASSERT_MPI_TRUE(timingOk);
        std::sort(samples.begin(), samples.end());

        const size_t bytes =
            static_cast<size_t>(nRecv) * DispatchBytesPerRow(phase.useFp8 != 0, hiddenSf);
        if (rank_ == 0) {
            // MPI tests only build under --debug, and this is wall clock for the whole call
            // rather than an isolated kernel time, so treat it as a regression signal and
            // not as achieved bandwidth.
            printf("[ rccl_ep  ] %-14s %6d rows, %7.1f MB | min %8.3f us %6.1f GB/s"
                   " | med %8.3f us %6.1f GB/s (debug build)\n",
                   phase.name, nRecv, static_cast<double>(bytes) / 1.0e6,
                   samples.front() * 1000.0, GigabytesPerSecond(bytes, samples.front()),
                   samples[samples.size() / 2] * 1000.0,
                   GigabytesPerSecond(bytes, samples[samples.size() / 2]));
            fflush(stdout);
        }

        if (phase.useFp8 == 0) {
            ASSERT_MPI_EQ(0, ep_combine(handle_, (uintptr_t)b.outX, 0, 0, (uintptr_t)b.outTopk,
                                        (uintptr_t)b.outSrc, nRecv, (uintptr_t)b.topkIdx, kTokens, 0, 0, 1,
                                        (uintptr_t)b.combined, 0, kNumSms, (uintptr_t)stream));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

            std::vector<float> cSamples;
            cSamples.reserve(kIters);
            bool combineOk = true;
            for (int i = 0; i < kIters; ++i) {
                combineOk &= (hipStreamSynchronize(stream) == hipSuccess);
                MPI_Barrier(MPI_COMM_WORLD);
                combineOk &= (hipEventRecord(start, stream) == hipSuccess);
                (void)ep_combine(handle_, (uintptr_t)b.outX, 0, 0, (uintptr_t)b.outTopk,
                                 (uintptr_t)b.outSrc, nRecv, (uintptr_t)b.topkIdx, kTokens, 0, 0, 1,
                                 (uintptr_t)b.combined, 0, kNumSms, (uintptr_t)stream);
                combineOk &= (hipEventRecord(stop, stream) == hipSuccess);
                combineOk &= (hipEventSynchronize(stop) == hipSuccess);
                float cms = 0.0f;
                combineOk &= (hipEventElapsedTime(&cms, start, stop) == hipSuccess);
                cSamples.push_back(cms);
            }
            ASSERT_MPI_TRUE(combineOk);
            std::sort(cSamples.begin(), cSamples.end());

            const size_t cbytes = static_cast<size_t>(nRecv) * CombineBytesPerRow();
            if (rank_ == 0) {
                printf("[ rccl_ep  ] %-14s %6d rows, %7.1f MB | min %8.3f us %6.1f GB/s"
                       " | med %8.3f us %6.1f GB/s (debug build)\n",
                       "combine bf16", nRecv, static_cast<double>(cbytes) / 1.0e6,
                       cSamples.front() * 1000.0, GigabytesPerSecond(cbytes, cSamples.front()),
                       cSamples[cSamples.size() / 2] * 1000.0,
                       GigabytesPerSecond(cbytes, cSamples[cSamples.size() / 2]));
                fflush(stdout);
            }
        }
    }
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED
