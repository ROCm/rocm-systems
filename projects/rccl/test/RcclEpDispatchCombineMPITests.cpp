/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-GPU end-to-end dispatch and combine for rccl_ep.
//
// The single-GPU tests in device/RcclEpTests.cpp cover the wave primitives, the combine reduce and the window layout
// arithmetic. Everything below needs peer memory and a communicator, so it runs one process per rank under MPI, in the
// same shape as SymmetricWindowMPITests.cpp. Dispatch and combine move data over LSA peer memory, so the eight ranks
// have to share a node.
//
// Build and run:
//   ./install.sh --debug -t --enable-mpi-tests --enable-rccl-ep-tests
//   NCCL_CUMEM_ENABLE=1 mpirun -np 8 --bind-to none \
//     ./build/debug/test/rccl-UnitTestsMPI --gtest_filter=RcclEpDispatchCombineTest.<one test>
//
// One test per process: under cuMem, device memory can accumulate across the communicators one process creates until
// comm init fails. Every test is self-contained for that reason. The round trips use the full MoE shape; tests of
// argument checks and idle ranks use kSmallTokens x kSmallHidden or smaller.
//
// What is asserted, and why the expected values are computable on the host. Every payload element is a function of its
// source rank, token and column (PayloadValue), exact in bf16, so a received row is checked against its source column
// by column, and the rows must arrive in source-rank order and token order within a source. Dispatch stages a token
// once per DESTINATION RANK rather than once per expert, and the top-k weights ride along as payload without being
// applied by the reduction. So handing combine exactly what dispatch produced -- an identity expert -- returns each
// token scaled by the number of distinct ranks that own its top-k experts. Combine sums in fp32 and rounds to bf16
// once, so that value is exact and the check is bit for bit. The combine that takes one row per (token, slot) is
// checked against the host's fp32 sum in ascending k with one rounding, on rows chosen so that any other order shows
// (ExpertValue).
//
// The tests after the bandwidth one cover the rest of the C ABI the Python layer drives: fp8 payloads and scales, the
// expanded layout and the combines that read it (also at num_topk 12, with more than 8 rows to a token), ranks that
// receive or send nothing, malformed routing, the cross-rank checks of the architecture and the wave size, a window
// registration RCCL declines, configure and destroy from another current device, the argument guards, and the internode
// direct mode: the round trip, fp8 rows, the per-slot combine of the expanded layout, combines with weights that only
// some ranks pass, and the bind that its dispatch and combine need. Internode runs on one host: ep_create_nodes'
// node_size splits the eight ranks into nodes by index, and the inter-node legs go over RCCL send/recv whatever
// transport RCCL picks between them.

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <set>
#include <string>
#include <thread>
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
void* ep_create_nodes(int rank, int num_ranks, const char* unique_id, int max_tokens_per_rank,
                      int hidden, int device, int node_size);
int ep_configure(void* handle, int num_experts, int num_topk);
void ep_destroy(void* handle);
size_t ep_window_bytes(void* handle);
int ep_barrier(void* handle, uintptr_t stream);
int ep_plan(void* handle, uintptr_t topk_idx, int num_tokens, uintptr_t slot, uintptr_t send_list,
            uintptr_t sendc, uintptr_t stream);
int ep_dispatch(void* handle, uintptr_t x, uintptr_t x_sf, uintptr_t topk_idx, uintptr_t topk_w,
                int num_tokens, uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                uintptr_t out_x, uintptr_t out_sf, uintptr_t out_topk, uintptr_t out_tw,
                uintptr_t out_src, uintptr_t stream);
int ep_recv_counts(void* handle, uintptr_t out_counts, uintptr_t stream);
int ep_expert_counts(void* handle, uintptr_t recv_topk, int num_recv, uintptr_t out_counts, uintptr_t stream);
int ep_expand_build(void* handle, uintptr_t recv_topk, int num_recv, int expert_alignment, uintptr_t counts,
                    uintptr_t offsets, uintptr_t psum, uintptr_t hist, uintptr_t row_map, uintptr_t stream);
int ep_expand_scatter(void* handle, int use_fp8, uintptr_t src_x, uintptr_t src_sf, uintptr_t src_w,
                      uintptr_t row_map, int num_recv, uintptr_t out_x, uintptr_t out_sf, int sf_row_stride,
                      int sf_col_stride, uintptr_t out_w, int zero_pad, int expert_alignment, uintptr_t counts,
                      uintptr_t offsets, int num_sms, uintptr_t stream);
int ep_combine(void* handle, uintptr_t y, uintptr_t in_w, uintptr_t row_map, uintptr_t recv_topk,
               uintptr_t recv_src, int num_recv, uintptr_t topk_idx, int num_tokens,
               uintptr_t bias0, uintptr_t bias1, int grouped, uintptr_t out, uintptr_t out_w,
               int num_sms, uintptr_t stream);
int ep_inter_counts(void* handle, uintptr_t sendc, int32_t* out, uintptr_t stream);
int ep_inter_bind(void* handle, const int32_t* sendc, const int32_t* recvc, uintptr_t send_list,
                  int num_tokens);
}

namespace RcclUnitTesting
{

namespace {

// The MoE shape these kernels are built for. Anything much smaller measures the fixed
// per-call cost (launch plus the device-side barrier) rather than the transfer: at
// 256 x 1024 that floor is ~62 us against ~5 us of payload.
constexpr int kRanks         = 8;     // one node's worth; the internode tests split it by kNodeSize
constexpr int kTokens        = 4096;
constexpr int kHidden        = 7168;  // multiple of 8: ep_create rejects odd hidden
constexpr int kTopk          = 6;
constexpr int kExpertsPerRank = 32;   // 256 experts across eight ranks
constexpr int kNumSms        = 32;    // past saturation; 64 measures slower, not faster
constexpr int kWarmup        = 10;
constexpr int kIters         = 20;   // each timed separately; the minimum is reported
// The shape of the tests that measure nothing: fast, well inside memory, and a hidden that is a
// multiple of 16, so fp8 runs too (16 scale blocks of 128 per row).
constexpr int kSmallTokens   = 1024;
constexpr int kSmallHidden   = 2048;

constexpr int kNodeSize = 4;       // internode tests: two nodes of four ranks

// 2^e, exactly.
__host__ __device__ inline float Pow2(int e)
{
    float p = 1.0f;
    for (; e > 0; --e) p *= 2.0f;
    for (; e < 0; ++e) p *= 0.5f;
    return p;
}

// Element `col` of token `token`'s payload on rank `rank`: (1 + m/128) x 2^e with e in [-3, 3],
// exact in bf16. m encodes the source rank and the token, and the column moves both m and the
// binade, so a row from the wrong rank or token, or a column out of place, reads back wrong.
// Tokens 1792 apart on one rank alias.
__host__ __device__ inline float PayloadValue(int rank, int token, int col)
{
    const int m = (rank * 16 + token % 16 + 5 * (col % 23)) % 128;
    return (1.0f + static_cast<float>(m) / 128.0f) * Pow2((col + token / 16) % 7 - 3);
}

// An expert's output for slot k of a token, for the combines that take one row per (token,
// slot); exact in bf16. Slot 3 is the exact negative of slot 0, and both sit near 2^20, far above
// the rest: a sum in ascending k rounds slots 1 and 2 against slot 0 before slot 3 cancels it, so
// the fp32 result depends on the order of the adds, and the bf16 one on there being a single
// rounding at the end. The other slots' sign and binade vary with the slot and the column.
__host__ __device__ inline float ExpertValue(int rank, int token, int k, int col)
{
    const bool negated = k == 3;
    const int  kk      = negated ? 0 : k;
    const int  m       = (rank * 16 + token % 16 + 3 * kk + 5 * (col % 23)) % 128;
    int        e       = kk - 8;
    switch (kk) {
    case 0: e = 20 + col % 2; break;
    case 1: e = col % 3 - 1; break;
    case 2: e = 1 + col % 2; break;
    case 4: e = -2; break;
    case 5: e = 2 - col % 2; break;
    default: break;
    }
    float v = (1.0f + static_cast<float>(m) / 128.0f) * Pow2(e);
    if (kk != 0 && (token + kk + col) % 3 == 0) v = -v;
    return negated ? -v : v;
}

// Top-k weights: (4 (k + 1) + (rank + token) % 4) / 128, different per slot and per token, exact
// in fp32.
float WeightFor(int rank, int token, int k)
{
    return static_cast<float>(4 * (k + 1) + (rank + token) % 4) / 128.0f;
}

// The fp8 tests' payload byte and per-128 scale. The library moves fp8 bytes without decoding
// them, so any value will do; a scale of the form (1 + j/8) x 2^e keeps byte x scale exact in
// fp32 (Fp8Decoded).
__host__ __device__ inline uint8_t Fp8Byte(int rank, int token, int col, int salt)
{
    return static_cast<uint8_t>((rank * 131 + token * 29 + col * 7 + (col >> 7) * 3 + salt * 61) & 0xFF);
}

__host__ __device__ inline float Fp8Scale(int rank, int token, int block, int salt)
{
    return (1.0f + static_cast<float>((rank * 5 + token * 3 + block + salt) % 8) / 8.0f) *
           Pow2((token + block + salt) % 9 - 4);
}

// The fp8 tests' identity expert: a byte decoded as byte x its block's scale, which is exact in
// fp32 and the same whichever fp8 format the architecture uses, rounded to bf16 once.
__host__ __device__ inline __hip_bfloat16 Fp8DecodedBf16(uint8_t b, float scale)
{
    return __float2bfloat16(static_cast<float>(b) * scale);
}

float Fp8Decoded(uint8_t b, float scale) { return __bfloat162float(Fp8DecodedBf16(b, scale)); }

// The expanded test's expert scales the rows of local expert e by 2^(e % 5 - 2), exactly.
__host__ __device__ inline float ExpertScale(int localExpert) { return Pow2(localExpert % 5 - 2); }

float Bf16Round(float f) { return __bfloat162float(__float2bfloat16(f)); }

float Bf16Bits(uint16_t bits)
{
    const uint32_t u = static_cast<uint32_t>(bits) << 16;
    float          f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
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

// Whether a top-k id names an expert; every path ignores one that does not.
bool ValidExpert(int e, int numExperts) { return e >= 0 && e < numExperts; }

// ExpertFor's routing with malformed entries mixed in: ids at and past the last expert, -1 and
// other negative ids, tokens with no valid id at all, and an expert repeated within a token.
int MalformedExpert(int rank, int token, int k, int numExperts, int nRanks)
{
    if (token % 97 == 0) return -1;                                       // sent nowhere
    if (k == 1 && token % 13 == 0) return numExperts;                     // one past the last
    if (k == 2 && token % 5 == 0) return numExperts + 7;                  // further past it
    if (k == 3 && token % 7 == 0) return -1;
    if (k == 4 && token % 11 == 0) return -2 - token % 3;                 // negative, not -1
    if (k == 5 && token % 4 == 0) return ExpertFor(rank, token, 0, numExperts, nRanks);  // repeats slot 0
    return ExpertFor(rank, token, k, numExperts, nRanks);
}

// Every token picks experts 0..K-1, all owned by rank 0: rank 0 receives every token of
// every rank and the other ranks receive nothing.
int HotExpert(int /*rank*/, int /*token*/, int k, int /*numExperts*/, int /*nRanks*/) { return k; }

// num_topk 12 over at least 12 experts per rank, for sums of more than 8 rows. Three tokens in four
// are crowded: 12, 10 or 9 of their slots on one rank, `home`, each on a distinct expert of it, the
// rest -1 or on other ranks; slots 0 to 3, whose ExpertValue rows make the order of a sum visible,
// are always home slots. The fourth token is spread over the ranks with 12, 9 or 8 valid slots.
// Every rank is home to every kind of crowded token from every sender.
int CrowdedExpert(int rank, int token, int k, int numExperts, int nRanks)
{
    const int perRank = numExperts / nRanks;
    const int home    = (token / 4 + rank) % nRanks;
    const int local   = (token + k) % perRank;  // distinct over 12 slots
    const int away    = (home + 1 + (token + k) % std::max(nRanks - 1, 1)) % nRanks;  // never home
    switch (token % 4) {
    case 0: return home * perRank + local;  // 12 slots at home
    case 1:                                 // 10 at home, the last slot elsewhere
        if (k == 5) return -1;
        return (k == 11 ? away : home) * perRank + local;
    case 2:  // 9 at home, with two slots elsewhere among them
        if (k == 4) return -1;
        return ((k == 7 || k == 10) ? away : home) * perRank + local;
    default: {  // spread over the ranks
        const int valid = (token / 4) % 3 == 0 ? 12 : (token / 4) % 3 == 1 ? 9 : 8;
        return k < valid ? ((token + 3 * k + rank) % nRanks) * perRank + local : -1;
    }
    }
}

using ExpertFn = int (*)(int rank, int token, int k, int numExperts, int nRanks);

inline unsigned GridFor(size_t n)
{
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>((n + 255) / 256, 8192)));
}

// out[r * ncols + c] = rows[r * hidden + cols[c]], bf16 bits.
__global__ void GatherColumns(const uint16_t* rows, size_t numRows, int hidden, const int* cols, int ncols,
                              uint16_t* out)
{
    const size_t n = numRows * static_cast<size_t>(ncols);
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        out[i] = rows[(i / ncols) * hidden + cols[i % ncols]];
    }
}

// x = PayloadValue of this rank's tokens.
__global__ void FillPayload(__hip_bfloat16* x, int rank, int numTokens, int hidden)
{
    const size_t n = static_cast<size_t>(numTokens) * hidden;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        x[i] = __float2bfloat16(PayloadValue(rank, static_cast<int>(i / hidden), static_cast<int>(i % hidden)));
    }
}

// Row (i, k) of a combine input with one row per (received token, slot): ExpertValue of received
// token i's source rank and token, from its src_token_global_idx (rank * T + token).
__global__ void FillExpertRows(const int32_t* src, int nRecv, int K, int T, int hidden, __hip_bfloat16* y)
{
    const size_t n = static_cast<size_t>(nRecv) * K * hidden;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = i / hidden;
        const int    g   = src[row / K];
        y[i] = __float2bfloat16(ExpertValue(g / T, g % T, static_cast<int>(row % K), static_cast<int>(i % hidden)));
    }
}

// The fp8 tests' payload bytes [numTokens, hidden] and scales [numTokens, hidden / 128].
__global__ void FillFp8(uint8_t* x, float* sf, int rank, int numTokens, int hidden, int salt)
{
    const int    nsf = (hidden + 127) / 128;
    const size_t nx = static_cast<size_t>(numTokens) * hidden, ns = static_cast<size_t>(numTokens) * nsf;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < nx;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        x[i] = Fp8Byte(rank, static_cast<int>(i / hidden), static_cast<int>(i % hidden), salt);
        if (i < ns) sf[i] = Fp8Scale(rank, static_cast<int>(i / nsf), static_cast<int>(i % nsf), salt);
    }
}

// The fp8 tests' identity expert over `rows` rows; the scale of row r, block j is at
// sf[r * sfRow + j * sfCol].
__global__ void Fp8Decode(const uint8_t* x, const float* sf, int rows, int hidden, int sfRow, int sfCol,
                          __hip_bfloat16* y)
{
    const size_t n = static_cast<size_t>(rows) * hidden;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t r = i / hidden;
        const int    j = static_cast<int>(i % hidden) / 128;
        y[i] = Fp8DecodedBf16(x[i], sf[r * sfRow + static_cast<size_t>(j) * sfCol]);
    }
}

// The expanded test's expert: row r of local expert rowExpert[r] times ExpertScale; padding rows
// (rowExpert -1) are copied as they are.
__global__ void ScaleByExpert(const __hip_bfloat16* x, const int32_t* rowExpert, int rows, int hidden,
                              __hip_bfloat16* y)
{
    const size_t n = static_cast<size_t>(rows) * hidden;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int e = rowExpert[i / hidden];
        y[i] = e < 0 ? x[i] : __float2bfloat16(__bfloat162float(x[i]) * ExpertScale(e));
    }
}

// The expanded rows' other expert: the row of pair p = i * K + k (rowPair, -1 for a padding row)
// becomes ExpertValue of received token i's source and slot k, as FillExpertRows' row p does.
// Padding rows are left as they are.
__global__ void FillExpandedExpertRows(const int* rowPair, const int32_t* src, int rows, int K, int T, int hidden,
                                       __hip_bfloat16* y)
{
    const size_t n = static_cast<size_t>(rows) * hidden;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int p = rowPair[i / hidden];
        if (p < 0) continue;
        const int g = src[p / K];
        y[i] = __float2bfloat16(ExpertValue(g / T, g % T, p % K, static_cast<int>(i % hidden)));
    }
}

// What the expanded round trip's expert does to the expanded rows.
enum class ExpandedExpert
{
    kScaleByExpert,  // bf16 rows times ExpertScale of their expert (ScaleByExpert); fp8 rows decoded
    kExpertValue,    // ExpertValue rows whatever the payload (FillExpandedExpertRows), so the order shows
};

// Sets an environment variable (unsets it for a null value) until the end of the scope, then
// restores what was there.
class ScopedEnv
{
public:
    ScopedEnv(const char* name, const char* value) : name_(name)
    {
        const char* old = std::getenv(name);
        had_ = old != nullptr;
        if (had_) old_ = old;
        Set(value);
    }
    ~ScopedEnv() { Set(had_ ? old_.c_str() : nullptr); }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    void Set(const char* value)
    {
        if (value != nullptr) {
            setenv(name_, value, 1);
        } else {
            unsetenv(name_);
        }
    }
    const char* name_;
    bool        had_ = false;
    std::string old_;
};

// A device allocation released on scope exit. Never zero bytes: a rank that receives no
// rows still passes a valid pointer.
struct DevBuf
{
    void* p = nullptr;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf()
    {
        if (p != nullptr) (void)hipFree(p);
    }
    bool Alloc(size_t bytes)
    {
        if (p != nullptr) return true;
        return hipMalloc(&p, std::max<size_t>(bytes, 256)) == hipSuccess;
    }
    uintptr_t u() const { return reinterpret_cast<uintptr_t>(p); }
};

template <typename T>
bool Download(const DevBuf& b, size_t n, std::vector<T>& out)
{
    out.resize(n);
    return n == 0 || hipMemcpy(out.data(), b.p, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess;
}

template <typename T>
bool Upload(DevBuf& b, const std::vector<T>& in)
{
    return b.Alloc(in.size() * sizeof(T)) &&
           (in.empty() || hipMemcpy(b.p, in.data(), in.size() * sizeof(T), hipMemcpyHostToDevice) == hipSuccess);
}

// The columns SampleColumns reads: both ends of a row, both sides of a 16-byte vector boundary at
// either end and in the middle, and a spread across the row, so a row that is only partly written
// or has columns out of place reads back wrong without downloading every row.
std::vector<int> SampledColumns(int hidden)
{
    std::vector<int> c = {0, 1, 7, 8, 9, hidden / 2 - 1, hidden / 2, hidden - 9, hidden - 8, hidden - 1};
    for (int j = 1; j < 16; ++j) c.push_back(j * (hidden / 16) + (j * 5) % 8);
    c.erase(std::remove_if(c.begin(), c.end(), [hidden](int v) { return v < 0 || v >= hidden; }), c.end());
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    return c;
}

// Element (r, cols[c]) of a bf16 [numRows, hidden] buffer as out[r * cols.size() + c], with cols
// SampledColumns(hidden).
bool SampleColumns(const DevBuf& rows, int numRows, std::vector<float>& out, int hidden)
{
    const std::vector<int> cols = SampledColumns(hidden);
    const size_t           n    = static_cast<size_t>(numRows) * cols.size();
    out.assign(n, 0.0f);
    if (n == 0) return true;
    DevBuf dCols, dOut;
    if (!Upload(dCols, cols) || !dOut.Alloc(n * sizeof(uint16_t))) return false;
    GatherColumns<<<GridFor(n), 256>>>(static_cast<const uint16_t*>(rows.p), static_cast<size_t>(numRows), hidden,
                                       static_cast<const int*>(dCols.p), static_cast<int>(cols.size()),
                                       static_cast<uint16_t*>(dOut.p));
    std::vector<uint16_t> bits;
    if (hipGetLastError() != hipSuccess || !Download(dOut, n, bits)) return false;
    for (size_t i = 0; i < n; ++i) out[i] = Bf16Bits(bits[i]);
    return true;
}

// One dispatch's inputs, plan and outputs, kept together so a later combine or replay can
// use the same plan.
struct Dispatch
{
    std::vector<int32_t> topk;          // [T, K]
    DevBuf x, topkIdx, topkW;
    DevBuf slot, sendList, sendc, recvc, counts;
    std::vector<int32_t> sendcH, recvcH;  // rows sent to (internode) and received from each rank
    std::vector<int32_t> expertCounts;  // received (token, slot) pairs per local expert
    int nRecv = 0;
    int cap   = 0;
    DevBuf outX, outTopk, outTw, outSrc;
};

class RcclEpDispatchCombineTest : public MPITestBase
{
protected:
    void* handle_ = nullptr;
    int   rank_ = 0, nRanks_ = 0, device_ = 0, numExperts_ = 0, topk_ = kTopk;
    int   tokens_ = kTokens, hidden_ = kHidden;  // the handle's num_max_tokens_per_rank and hidden
    int   nodeSize_ = 0;   // 0 for an intranode handle

    void TearDown() override
    {
        if (handle_ != nullptr) {
            ep_destroy(handle_);
            handle_ = nullptr;
        }
        MPITestBase::TearDown();
    }

    // ENVIRONMENT ONLY. A runtime without symmetric memory, or a run that is not eight
    // ranks on one node, is a reason to skip. rccl_ep itself failing is not: the
    // ep_create/ep_configure calls below are checked separately, so a null handle or a
    // rejected configure fails the test instead of turning into a green skip.
    bool EnvReady()
    {
        if (!validateTestPrerequisites(kRanks, kNoProcessLimit, kNoPowerOfTwoRequired,
                                       kRequireSingleNode, kRequireSingleNode)) {
            return false;
        }
        const char* cuMemEnv = std::getenv("NCCL_CUMEM_ENABLE");
        if (cuMemEnv == nullptr || std::string(cuMemEnv) != "1") return false;

        rank_   = MPIEnvironment::world_rank;
        nRanks_ = MPIEnvironment::world_size;

        // MPIEnvironment already bound this rank to its local device; adopt that rather
        // than re-deriving from the global rank. Per-rank, so agree before the collectives.
        int localOk = (hipGetDevice(&device_) == hipSuccess) ? 1 : 0;
        int deviceOk = 0;
        if (MPI_Allreduce(&localOk, &deviceOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS) {
            return false;
        }
        return deviceOk != 0;
    }

    // The library calls, all of them red on failure. Every rank calls in unconditionally
    // and the verdict is taken afterwards: ASSERT_MPI_* is itself an allreduce, so
    // asserting on rank 0's unique id BEFORE the Bcast would leave the other seven
    // blocked in it. (A rank that fails *inside* ep_configure can still leave the rest
    // blocked; a test-side assertion cannot reach that case.)
    //
    // `num_experts` must be a multiple of the rank count, so the owner of an expert is
    // simply index / experts_per_rank.
    void CreateEp(int numExperts, int numTopk, int tokens = kTokens, int hidden = kHidden);

    int DistinctDestinations(const std::vector<int32_t>& topk, int token) const
    {
        const int expertsPerRank = numExperts_ / nRanks_;
        std::set<int> owners;
        for (int k = 0; k < topk_; ++k) {
            const int e = topk[static_cast<size_t>(token) * topk_ + k];
            if (ValidExpert(e, numExperts_)) owners.insert(e / expertsPerRank);
        }
        return static_cast<int>(owners.size());
    }

    // Rank 0 makes a unique id and every rank gets a copy.
    void BroadcastUid(std::vector<char>* uid)
    {
        uid->assign(static_cast<size_t>(ep_unique_id_size()), 0);
        int idOk = (rank_ != 0 || ep_get_unique_id(uid->data()) == 0) ? 1 : 0;
        ASSERT_MPI_EQ(MPI_SUCCESS, MPI_Bcast(uid->data(), static_cast<int>(uid->size()), MPI_BYTE, 0,
                                             MPI_COMM_WORLD));
        ASSERT_MPI_EQ(1, idOk);
    }

    // Bounded ep_create: a missed num_ranks rejection would wait in init for ranks that never join.
    void CreateBounded(const std::vector<char>& uid, int rank, int numRanks, int maxTokens, int hidden,
                       const char* caseName, void** out)
    {
        *out = nullptr;
        std::promise<void*> result;
        std::future<void*> done = result.get_future();
        std::thread worker([&result, uid, rank, numRanks, maxTokens, hidden, dev = device_]() {
            result.set_value(ep_create(rank, numRanks, uid.data(), maxTokens, hidden, dev));
        });
        if (done.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
            std::fprintf(stderr, "[rank %d] ep_create timed out after 60s: %s; aborting the MPI job\n",
                         rank_, caseName);
            std::fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            std::_Exit(EXIT_FAILURE);
        }
        worker.join();
        *out = done.get();
    }

    // Plan and dispatch, check the plan, every received row and the receive counts against the
    // routing, combine with an identity expert and check every token came back scaled by the
    // number of distinct ranks that own its top-k experts, then combine one ExpertValue row per
    // (token, slot) and check the origin summed them in ascending k with one rounding. Reads
    // numExperts_ and topk_, so it can be re-run after a reconfigure at a different shape.
    void RoundTrip();

    // ep_create_nodes for nRanks_ / nodeSize nodes.
    void CreateEpNodes(int numExperts, int numTopk, int nodeSize, int tokens = kTokens, int hidden = kHidden);
    // ep_create (ep_create_nodes for a nodeSize) on every rank with one shared unique id, without
    // configuring; records tokens_, hidden_ and nodeSize_.
    void CreateHandle(int tokens, int hidden, int nodeSize = 0);
    void DestroyEp();
    int  ExpertsPerRank() const { return numExperts_ / nRanks_; }

    // Routing (expertFor), weights (WeightFor) and payloads (PayloadValue) for this rank's
    // tokens_ tokens.
    void MakeInputs(Dispatch& d, ExpertFn expertFor);
    void AllocOutputs(Dispatch& d, int cap);
    // ep_plan and ep_dispatch, then the receive counts the Python layer reads after a dispatch:
    // the rows from each source rank (ep_recv_counts) and the pairs per local expert
    // (ep_expert_counts).
    void DispatchBarrier(Dispatch& d, hipStream_t stream);
    // On an internode handle: ep_plan, the count exchange (ep_inter_counts), Bind and ep_dispatch,
    // then the pairs per local expert (ep_expert_counts).
    void DispatchInternode(Dispatch& d, hipStream_t stream);
    void Bind(Dispatch& d);
    // Grouped combine of `y` (received rows) into `out`.
    void Combine(Dispatch& d, const DevBuf& y, DevBuf& out, hipStream_t stream);

    // What this rank receives, from the routing of every senderStride-th rank: nRecv tokens, the
    // (token, slot) pairs per local expert and, when fromRank is given, the tokens from each
    // source rank. Ids that name no expert count nowhere.
    void ExpectedReceive(ExpertFn expertFor, int& nRecv, std::vector<int32_t>& expertCounts,
                         int senderStride = 1, std::vector<int32_t>* fromRank = nullptr) const;
    // The plan (slot, send list, send counts) against this rank's routing.
    void CheckPlan(const Dispatch& d);
    // Every received row against the routing alone: the sources (outSrc) must be exactly the
    // expected sequence -- senders in rank order (every senderStride-th rank), each one's tokens
    // with an expert here in token order, nothing else -- and each row carries its source token's
    // top-k ids rebased to this rank's experts (-1 for another rank's), its weights and, when outX
    // is given, its payload (PayloadValue) in every sampled column.
    void CheckReceivedBuffers(const DevBuf* outX, const DevBuf& outTopk, const DevBuf& outTw, const DevBuf& outSrc,
                              int nRecv, ExpertFn expertFor, int senderStride = 1);
    void CheckReceivedRows(const Dispatch& d, ExpertFn expertFor)
    {
        CheckReceivedBuffers(&d.outX, d.outTopk, d.outTw, d.outSrc, d.nRecv, expertFor);
    }
    // Sampled bf16 row `row` holds PayloadValue(srcRank, srcTok) in every sampled column.
    bool RowHolds(const std::vector<float>& sampled, size_t row, int srcRank, int srcTok) const;
    // Tokens of a combine's sampled output (SampleColumns, numTokens rows) that are not this rank's
    // payload x the number of distinct ranks owning the token's experts (topk): what an identity
    // expert returns. That product is exact in fp32 and combine rounds once, so the check is bit
    // for bit.
    int MismatchesScaled(const std::vector<float>& got, const std::vector<int32_t>& topk, int numTokens);
    // Each token comes back as value x distinct destination ranks (see the file header).
    void CheckScaledByDestinations(const Dispatch& d, const DevBuf& out, hipStream_t stream, int numTokens = -1);
    // ExpandedLayoutGroupsByExpert's round trip at the handle's shape, routed by expertFor: a bf16 dispatch
    // and an fp8 replay of its plan, the expanded layout and its rows checked against the routing, then
    // `expert` applied to the rows and both combines through the row map checked against the host's sums,
    // every element. With ExpandedExpert::kExpertValue it also checks that the rows make the order visible.
    void ExpandedRoundTrip(ExpertFn expertFor, ExpandedExpert expert);
    // On an internode handle: dispatch, check the receive counts and rows, combine and check the
    // result.
    void InternodeRoundTrip();
};

// CreateEp and RoundTrip have to be void to use the ASSERT_MPI_* macros, and an
// assertion inside a subroutine only returns from that subroutine -- both the FAIL()
// arm and the GTEST_SKIP() arm. A caller that ignores that runs on with a null handle,
// so every call to one of them goes through this.
#define ASSERT_EP_STEP(call)                          \
    do {                                              \
        call;                                         \
        if (HasFatalFailure() || IsSkipped()) return; \
    } while (0)

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

void RcclEpDispatchCombineTest::CreateEp(int numExperts, int numTopk, int tokens, int hidden)
{
    numExperts_ = numExperts;
    topk_       = numTopk;
    ASSERT_EP_STEP(CreateHandle(tokens, hidden));
    ASSERT_MPI_EQ(0, ep_configure(handle_, numExperts_, topk_));
}

void RcclEpDispatchCombineTest::RoundTrip()
{
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));
    const int    T        = tokens_, H = hidden_, K = topk_;
    const size_t outBytes = static_cast<size_t>(T) * H * sizeof(__hip_bfloat16);

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(DispatchBarrier(d, stream));
    ASSERT_EP_STEP(CheckPlan(d));
    ASSERT_EP_STEP(CheckReceivedRows(d, ExpertFor));
    {
        int                  wantRecv = 0;
        std::vector<int32_t> wantCounts, wantFrom;
        ExpectedReceive(ExpertFor, wantRecv, wantCounts, 1, &wantFrom);
        ASSERT_MPI_TRUE(d.recvcH == wantFrom && d.expertCounts == wantCounts);
    }

    // A uniform spread would be satisfied by any implementation that scaled by a constant.
    {
        std::set<int> spread;
        for (int t = 0; t < T; ++t) spread.insert(DistinctDestinations(d.topk, t));
        ASSERT_MPI_GT(spread.size(), 1u);
    }

    // Identity expert: combine is handed exactly what dispatch produced. The destination is poisoned
    // first: 0xFF bytes read back as NaN in bf16, so a token combine never writes fails the
    // comparison instead of matching whatever the allocator happened to leave there.
    DevBuf out;
    ASSERT_MPI_TRUE(out.Alloc(outBytes));
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, outBytes, stream));
    ASSERT_MPI_EQ(0, ep_combine(handle_, d.outX.u(), 0, /*row_map=*/0, d.outTopk.u(), d.outSrc.u(), d.nRecv,
                                d.topkIdx.u(), T, 0, 0, /*grouped=*/1, out.u(), 0, kNumSms, (uintptr_t)stream));
    ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream));

    // One row per (received token, slot), holding ExpertValue of the token's source: the origin sums
    // a token's slots in ascending k in fp32 and rounds once, which the host repeats.
    DevBuf       y;
    const size_t yElems = static_cast<size_t>(d.nRecv) * K * H;
    ASSERT_MPI_TRUE(y.Alloc(yElems * sizeof(__hip_bfloat16)));
    if (yElems > 0) {
        FillExpertRows<<<GridFor(yElems), 256, 0, stream>>>(static_cast<const int32_t*>(d.outSrc.p), d.nRecv, K, T, H,
                                                             static_cast<__hip_bfloat16*>(y.p));
    }
    ASSERT_MPI_EQ(hipSuccess, hipGetLastError());
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, outBytes, stream));
    ASSERT_MPI_EQ(0, ep_combine(handle_, y.u(), 0, 0, d.outTopk.u(), d.outSrc.u(), d.nRecv, d.topkIdx.u(), T, 0, 0,
                                /*grouped=*/0, out.u(), 0, kNumSms, (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    std::vector<float> got;
    ASSERT_MPI_TRUE(SampleColumns(out, T, got, H));
    const std::vector<int> cols = SampledColumns(H);
    const size_t           ns   = cols.size();
    int mismatches = 0, orderSensitive = 0;
    for (int t = 0; t < T; ++t) {
        for (size_t c = 0; c < ns; ++c) {
            float up = 0.0f, down = 0.0f;
            for (int k = 0; k < K; ++k) {
                if (ValidExpert(d.topk[static_cast<size_t>(t) * K + k], numExperts_)) {
                    up += ExpertValue(rank_, t, k, cols[c]);
                }
            }
            for (int k = K - 1; k >= 0; --k) {
                if (ValidExpert(d.topk[static_cast<size_t>(t) * K + k], numExperts_)) {
                    down += ExpertValue(rank_, t, k, cols[c]);
                }
            }
            const float want = Bf16Round(up);
            orderSensitive += Bf16Round(down) != want ? 1 : 0;
            const float g = got[static_cast<size_t>(t) * ns + c];
            if (!(g == want)) {
                if (mismatches < 8) {
                    ADD_FAILURE() << "rank " << rank_ << " token " << t << " column " << cols[c]
                                  << ": the per-slot combine gave " << g << ", the ascending-k sum is " << want;
                }
                ++mismatches;
            }
        }
    }
    ASSERT_MPI_EQ(0, mismatches);
    // The rows do make the order visible: summed the other way round, some tokens come out different.
    // (Slot 3 cancels slot 0 only when there are at least four slots.)
    ASSERT_MPI_TRUE(K <= 3 || orderSensitive > 0);
}

void RcclEpDispatchCombineTest::CreateEpNodes(int numExperts, int numTopk, int nodeSize, int tokens, int hidden)
{
    numExperts_ = numExperts;
    topk_       = numTopk;
    ASSERT_EP_STEP(CreateHandle(tokens, hidden, nodeSize));
    ASSERT_MPI_EQ(0, ep_configure(handle_, numExperts_, topk_));
}

void RcclEpDispatchCombineTest::CreateHandle(int tokens, int hidden, int nodeSize)
{
    tokens_   = tokens;
    hidden_   = hidden;
    nodeSize_ = nodeSize;
    std::vector<char> uid;
    ASSERT_EP_STEP(BroadcastUid(&uid));
    handle_ = nodeSize > 0 ? ep_create_nodes(rank_, nRanks_, uid.data(), tokens, hidden, device_, nodeSize)
                           : ep_create(rank_, nRanks_, uid.data(), tokens, hidden, device_);
    ASSERT_MPI_TRUE(handle_ != nullptr);
}

void RcclEpDispatchCombineTest::DestroyEp()
{
    if (handle_ != nullptr) ep_destroy(handle_);
    handle_   = nullptr;
    nodeSize_ = 0;
}

void RcclEpDispatchCombineTest::MakeInputs(Dispatch& d, ExpertFn expertFor)
{
    const int    T  = tokens_, H = hidden_;
    const size_t tk = static_cast<size_t>(T) * topk_;
    const size_t nx = static_cast<size_t>(T) * H;
    std::vector<float> hW(tk);
    d.topk.assign(tk, 0);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < topk_; ++k) {
            d.topk[static_cast<size_t>(t) * topk_ + k] = expertFor(rank_, t, k, numExperts_, nRanks_);
            hW[static_cast<size_t>(t) * topk_ + k]     = WeightFor(rank_, t, k);
        }
    }
    ASSERT_MPI_TRUE(d.x.Alloc(nx * sizeof(__hip_bfloat16)) && Upload(d.topkIdx, d.topk) && Upload(d.topkW, hW));
    FillPayload<<<GridFor(nx), 256>>>(static_cast<__hip_bfloat16*>(d.x.p), rank_, T, H);
    ASSERT_MPI_EQ(hipSuccess, hipGetLastError());
    ASSERT_MPI_EQ(hipSuccess, hipDeviceSynchronize());
}

void RcclEpDispatchCombineTest::AllocOutputs(Dispatch& d, int cap)
{
    d.cap = cap;
    ASSERT_MPI_TRUE(d.outX.Alloc(static_cast<size_t>(cap) * hidden_ * sizeof(__hip_bfloat16)) &&
                    d.outTopk.Alloc(static_cast<size_t>(cap) * topk_ * sizeof(int32_t)) &&
                    d.outTw.Alloc(static_cast<size_t>(cap) * topk_ * sizeof(float)) &&
                    d.outSrc.Alloc(static_cast<size_t>(cap) * sizeof(int32_t)));
}

void RcclEpDispatchCombineTest::DispatchBarrier(Dispatch& d, hipStream_t stream)
{
    const int    epr = ExpertsPerRank();
    const size_t rt  = static_cast<size_t>(nRanks_) * tokens_;
    ASSERT_MPI_TRUE(d.slot.Alloc(rt * sizeof(int32_t)) && d.sendList.Alloc(rt * sizeof(int32_t)) &&
                    d.sendc.Alloc(nRanks_ * sizeof(int32_t)) && d.recvc.Alloc(nRanks_ * sizeof(int32_t)) &&
                    d.counts.Alloc(epr * sizeof(int32_t)));
    ASSERT_EP_STEP(AllocOutputs(d, nRanks_ * tokens_));
    ASSERT_MPI_EQ(0, ep_plan(handle_, d.topkIdx.u(), tokens_, d.slot.u(), d.sendList.u(), d.sendc.u(),
                             (uintptr_t)stream));
    d.nRecv = ep_dispatch(handle_, d.x.u(), 0, d.topkIdx.u(), d.topkW.u(), tokens_, d.sendList.u(),
                          d.sendc.u(), 0, kNumSms, d.outX.u(), 0, d.outTopk.u(), d.outTw.u(),
                          d.outSrc.u(), (uintptr_t)stream);
    ASSERT_MPI_TRUE(d.nRecv >= 0 && d.nRecv <= d.cap);
    ASSERT_MPI_EQ(0, ep_recv_counts(handle_, d.recvc.u(), (uintptr_t)stream));
    ASSERT_MPI_EQ(0, ep_expert_counts(handle_, d.outTopk.u(), d.nRecv, d.counts.u(), (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    ASSERT_MPI_TRUE(Download(d.recvc, nRanks_, d.recvcH) && Download(d.counts, epr, d.expertCounts));
}

void RcclEpDispatchCombineTest::DispatchInternode(Dispatch& d, hipStream_t stream)
{
    const int    T   = tokens_;
    const int    epr = ExpertsPerRank();
    const size_t rt  = static_cast<size_t>(nRanks_) * T;
    ASSERT_MPI_TRUE(d.slot.Alloc(rt * sizeof(int32_t)) && d.sendList.Alloc(rt * sizeof(int32_t)) &&
                    d.sendc.Alloc(nRanks_ * sizeof(int32_t)) && d.counts.Alloc(epr * sizeof(int32_t)));
    ASSERT_MPI_EQ(0, ep_plan(handle_, d.topkIdx.u(), T, d.slot.u(), d.sendList.u(), d.sendc.u(),
                             (uintptr_t)stream));
    std::vector<int32_t> out(static_cast<size_t>(2 * nRanks_), 0);
    ASSERT_MPI_EQ(0, ep_inter_counts(handle_, d.sendc.u(), out.data(), (uintptr_t)stream));
    d.sendcH.assign(out.begin(), out.begin() + nRanks_);
    d.recvcH.assign(out.begin() + nRanks_, out.end());
    int total = 0;
    for (int r = 0; r < nRanks_; ++r) total += d.recvcH[r];
    ASSERT_EP_STEP(AllocOutputs(d, std::max(total, 1)));
    ASSERT_EP_STEP(Bind(d));
    d.nRecv = ep_dispatch(handle_, d.x.u(), 0, d.topkIdx.u(), d.topkW.u(), T, d.sendList.u(), d.sendc.u(), 0,
                          kNumSms, d.outX.u(), 0, d.outTopk.u(), d.outTw.u(), d.outSrc.u(), (uintptr_t)stream);
    ASSERT_MPI_EQ(total, d.nRecv);
    ASSERT_MPI_EQ(0, ep_expert_counts(handle_, d.outTopk.u(), d.nRecv, d.counts.u(), (uintptr_t)stream));
    ASSERT_MPI_TRUE(Download(d.counts, epr, d.expertCounts));
}

void RcclEpDispatchCombineTest::Bind(Dispatch& d)
{
    if (nodeSize_ == 0) return;
    ASSERT_MPI_EQ(0, ep_inter_bind(handle_, d.sendcH.data(), d.recvcH.data(), d.sendList.u(), tokens_));
}

void RcclEpDispatchCombineTest::Combine(Dispatch& d, const DevBuf& y, DevBuf& out, hipStream_t stream)
{
    const size_t bytes = static_cast<size_t>(tokens_) * hidden_ * sizeof(__hip_bfloat16);
    ASSERT_MPI_TRUE(out.Alloc(bytes));
    // 0xFF reads back as NaN in bf16, so a token that combine never writes cannot match.
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, bytes, stream));
    ASSERT_EP_STEP(Bind(d));
    ASSERT_MPI_EQ(0, ep_combine(handle_, y.u(), 0, 0, d.outTopk.u(), d.outSrc.u(), d.nRecv, d.topkIdx.u(), tokens_, 0,
                                0, /*grouped=*/1, out.u(), 0, kNumSms, (uintptr_t)stream));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
}

void RcclEpDispatchCombineTest::ExpectedReceive(ExpertFn expertFor, int& nRecv,
                                                std::vector<int32_t>& expertCounts,
                                                int senderStride, std::vector<int32_t>* fromRank) const
{
    const int epr = ExpertsPerRank();
    nRecv = 0;
    expertCounts.assign(epr, 0);
    if (fromRank != nullptr) fromRank->assign(nRanks_, 0);
    for (int r = 0; r < nRanks_; r += senderStride) {
        for (int t = 0; t < tokens_; ++t) {
            bool mine = false;
            for (int k = 0; k < topk_; ++k) {
                const int e = expertFor(r, t, k, numExperts_, nRanks_);
                if (!ValidExpert(e, numExperts_) || e / epr != rank_) continue;
                mine = true;
                ++expertCounts[e % epr];
            }
            nRecv += mine ? 1 : 0;
            if (mine && fromRank != nullptr) ++(*fromRank)[r];
        }
    }
}

void RcclEpDispatchCombineTest::CheckPlan(const Dispatch& d)
{
    const int T = tokens_, R = nRanks_, K = topk_, epr = ExpertsPerRank();
    std::vector<int32_t> slot, list, sendc;
    ASSERT_MPI_TRUE(Download(d.slot, static_cast<size_t>(R) * T, slot) &&
                    Download(d.sendList, static_cast<size_t>(R) * T, list) && Download(d.sendc, R, sendc));
    int bad = 0;
    for (int dst = 0; dst < R; ++dst) {
        int pos = 0;
        for (int t = 0; t < T; ++t) {
            bool sent = false;
            for (int k = 0; k < K; ++k) {
                const int e = d.topk[static_cast<size_t>(t) * K + k];
                sent |= ValidExpert(e, numExperts_) && e / epr == dst;
            }
            const size_t row = static_cast<size_t>(dst) * T;
            if (slot[row + t] != (sent ? pos : -1) || (sent && list[row + pos] != t)) {
                if (bad < 8) {
                    ADD_FAILURE() << "rank " << rank_ << ": plan for destination " << dst << " token " << t
                                  << ": slot " << slot[row + t] << ", expected " << (sent ? pos : -1);
                }
                ++bad;
            }
            pos += sent ? 1 : 0;
        }
        if (sendc[dst] != pos) {
            if (bad < 8) {
                ADD_FAILURE() << "rank " << rank_ << ": sends " << sendc[dst] << " tokens to " << dst << ", expected "
                              << pos;
            }
            ++bad;
        }
    }
    ASSERT_MPI_EQ(0, bad);
}

void RcclEpDispatchCombineTest::CheckReceivedBuffers(const DevBuf* outX, const DevBuf& outTopk, const DevBuf& outTw,
                                                     const DevBuf& outSrc, int nRecv, ExpertFn expertFor,
                                                     int senderStride)
{
    const int T = tokens_, K = topk_, epr = ExpertsPerRank();
    std::vector<int32_t> want;
    for (int r = 0; r < nRanks_; r += senderStride) {
        for (int t = 0; t < T; ++t) {
            bool mine = false;
            for (int k = 0; k < K; ++k) {
                const int e = expertFor(r, t, k, numExperts_, nRanks_);
                mine |= ValidExpert(e, numExperts_) && e / epr == rank_;
            }
            if (mine) want.push_back(r * T + t);
        }
    }
    ASSERT_MPI_EQ(static_cast<int>(want.size()), nRecv);
    std::vector<int32_t> src, ids;
    std::vector<float>   w, rows;
    ASSERT_MPI_TRUE(Download(outSrc, nRecv, src) && Download(outTopk, static_cast<size_t>(nRecv) * K, ids) &&
                    Download(outTw, static_cast<size_t>(nRecv) * K, w) &&
                    (outX == nullptr || SampleColumns(*outX, nRecv, rows, hidden_)));
    int bad = 0;
    for (int i = 0; i < nRecv; ++i) {
        if (src[i] != want[i]) {
            if (bad < 8) {
                ADD_FAILURE() << "rank " << rank_ << " row " << i << ": source " << src[i] << ", expected " << want[i];
            }
            ++bad;
            continue;
        }
        const int r = src[i] / T, t = src[i] % T;
        for (int k = 0; k < K; ++k) {
            const int    e      = expertFor(r, t, k, numExperts_, nRanks_);
            const int    wantId = ValidExpert(e, numExperts_) && e / epr == rank_ ? e % epr : -1;
            const size_t at     = static_cast<size_t>(i) * K + k;
            if (ids[at] != wantId || w[at] != WeightFor(r, t, k)) {
                if (bad < 8) {
                    ADD_FAILURE() << "rank " << rank_ << " row " << i << " (source " << src[i] << ") slot " << k
                                  << ": id " << ids[at] << " weight " << w[at] << ", expected " << wantId << " and "
                                  << WeightFor(r, t, k);
                }
                ++bad;
            }
        }
        if (outX != nullptr && !RowHolds(rows, i, r, t)) {
            if (bad < 8) ADD_FAILURE() << "rank " << rank_ << " row " << i << " (source " << src[i] << "): payload";
            ++bad;
        }
    }
    ASSERT_MPI_EQ(0, bad);
}

bool RcclEpDispatchCombineTest::RowHolds(const std::vector<float>& sampled, size_t row, int srcRank, int srcTok) const
{
    const std::vector<int> cols = SampledColumns(hidden_);
    for (size_t c = 0; c < cols.size(); ++c) {
        if (!(sampled[row * cols.size() + c] == PayloadValue(srcRank, srcTok, cols[c]))) return false;
    }
    return true;
}

int RcclEpDispatchCombineTest::MismatchesScaled(const std::vector<float>& got, const std::vector<int32_t>& topk,
                                                int numTokens)
{
    const std::vector<int> cols = SampledColumns(hidden_);
    const size_t           ns   = cols.size();
    int                    bad  = 0;
    for (int t = 0; t < numTokens; ++t) {
        const int dests = DistinctDestinations(topk, t);
        for (size_t c = 0; c < ns; ++c) {
            const float x    = PayloadValue(rank_, t, cols[c]);
            const float want = Bf16Round(x * static_cast<float>(dests));
            const float g    = got[static_cast<size_t>(t) * ns + c];
            // Negated comparison: the 0xFF poison reads back as NaN, and every comparison against NaN
            // is false, so an unwritten token fails.
            if (!(g == want)) {
                if (bad < 8) {
                    ADD_FAILURE() << "rank " << rank_ << " token " << t << " column " << cols[c] << ": got " << g
                                  << ", expected " << want << " (= " << x << " x " << dests << " destination ranks)";
                }
                ++bad;
                break;
            }
        }
    }
    return bad;
}

void RcclEpDispatchCombineTest::CheckScaledByDestinations(const Dispatch& d, const DevBuf& out, hipStream_t stream,
                                                          int numTokens)
{
    const int n = numTokens < 0 ? tokens_ : numTokens;
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    std::vector<float> got;
    ASSERT_MPI_TRUE(SampleColumns(out, n, got, hidden_));
    ASSERT_MPI_EQ(0, MismatchesScaled(got, d.topk, n));
}

void RcclEpDispatchCombineTest::InternodeRoundTrip()
{
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(DispatchInternode(d, stream));
    int                  wantRecv = 0;
    std::vector<int32_t> wantCounts;
    ExpectedReceive(ExpertFor, wantRecv, wantCounts);
    ASSERT_MPI_EQ(wantRecv, d.nRecv);
    ASSERT_MPI_TRUE(d.expertCounts == wantCounts);
    // Remote sources' rows land in the same place a local push puts them, so the receive order is
    // the intranode one.
    ASSERT_EP_STEP(CheckReceivedRows(d, ExpertFor));
    DevBuf out;
    ASSERT_EP_STEP(Combine(d, d.outX, out, stream));
    ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream));
}

// ExpertFor's routing with each odd rank's experts moved to the even rank below it: the odd ranks
// own experts that no token picks, so on both nodes some ranks receive rows and some receive none.
int EvenRankExpert(int rank, int token, int k, int numExperts, int nRanks)
{
    const int perRank = numExperts / nRanks;
    const int e       = ExpertFor(rank, token, k, numExperts, nRanks);
    return ((e / perRank) & ~1) * perRank + e % perRank;
}

// Slots of `rank`'s first numTokens tokens whose returned weight (w, [numTokens, K]) is not the slot's
// own top-k weight (WeightFor), or 0 for a slot that names no expert. The first eight are reported.
int WeightMismatches(const std::vector<float>& w, const std::vector<int32_t>& topk, int rank, int numTokens, int K,
                     int numExperts)
{
    int bad = 0;
    for (int t = 0; t < numTokens; ++t) {
        for (int k = 0; k < K; ++k) {
            const size_t at   = static_cast<size_t>(t) * K + k;
            const float  want = ValidExpert(topk[at], numExperts) ? WeightFor(rank, t, k) : 0.0f;
            if (!(w[at] == want)) {
                if (bad < 8) {
                    ADD_FAILURE() << "rank " << rank << " token " << t << " slot " << k << ": weight " << w[at]
                                  << ", expected " << want;
                }
                ++bad;
            }
        }
    }
    return bad;
}

void RcclEpDispatchCombineTest::ExpandedRoundTrip(ExpertFn expertFor, ExpandedExpert expert)
{
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));
    constexpr int   kAlignment = 48;  // not a power of two
    const uintptr_t s          = reinterpret_cast<uintptr_t>(stream);
    const int       T = tokens_, H = hidden_, K = topk_, R = nRanks_, epr = ExpertsPerRank();
    const int       nsf      = (H + 127) / 128;
    const size_t    cap      = static_cast<size_t>(R) * T;
    const size_t    tokBytes = static_cast<size_t>(T) * H * sizeof(__hip_bfloat16);
    const bool      byValue  = expert == ExpandedExpert::kExpertValue;

    // A bf16 dispatch, then the same plan replayed with an fp8 payload and its scales: the same rows,
    // in the same order.
    Dispatch d, f;
    ASSERT_EP_STEP(MakeInputs(d, expertFor));
    ASSERT_EP_STEP(DispatchBarrier(d, stream));
    const int n = d.nRecv;
    ASSERT_EP_STEP(AllocOutputs(f, static_cast<int>(cap)));
    DevBuf x8, sf8, rsf8;
    ASSERT_MPI_TRUE(x8.Alloc(static_cast<size_t>(T) * H) && sf8.Alloc(static_cast<size_t>(T) * nsf * sizeof(float)) &&
                    rsf8.Alloc(cap * nsf * sizeof(float)));
    FillFp8<<<GridFor(static_cast<size_t>(T) * H), 256>>>(static_cast<uint8_t*>(x8.p), static_cast<float*>(sf8.p),
                                                           rank_, T, H, 0);
    ASSERT_MPI_EQ(hipSuccess, hipGetLastError());
    ASSERT_MPI_EQ(hipSuccess, hipDeviceSynchronize());
    f.nRecv = ep_dispatch(handle_, x8.u(), sf8.u(), d.topkIdx.u(), d.topkW.u(), T, d.sendList.u(), d.sendc.u(),
                          /*use_fp8=*/1, kNumSms, f.outX.u(), rsf8.u(), f.outTopk.u(), f.outTw.u(), f.outSrc.u(), s);
    ASSERT_MPI_EQ(n, f.nRecv);
    ASSERT_EP_STEP(CheckReceivedBuffers(nullptr, f.outTopk, f.outTw, f.outSrc, n, expertFor));

    // The layout the host expects from the received local ids: per local expert, its pairs in
    // (token, slot) order from a base that is a multiple of the alignment; the rows after an expert's
    // pairs, up to the next base, are padding.
    std::vector<int32_t> ids;
    ASSERT_MPI_TRUE(Download(d.outTopk, static_cast<size_t>(n) * K, ids));
    std::vector<int32_t> wantCounts(static_cast<size_t>(epr), 0), wantOffsets(static_cast<size_t>(epr), 0),
        wantPsum(static_cast<size_t>(epr), 0), wantMap(static_cast<size_t>(n) * K, -1);
    for (int32_t e : ids) {
        if (e >= 0) ++wantCounts[e];
    }
    int total = 0;
    for (int e = 0; e < epr; ++e) {
        wantOffsets[e] = total;
        wantPsum[e]    = total + wantCounts[e];
        total += (wantCounts[e] + kAlignment - 1) / kAlignment * kAlignment;
    }
    std::vector<int32_t> rowExpert(static_cast<size_t>(std::max(total, 1)), -1);
    std::vector<int>     rowPair(static_cast<size_t>(std::max(total, 1)), -1);
    {
        std::vector<int32_t> next(wantOffsets);
        for (size_t p = 0; p < ids.size(); ++p) {
            if (ids[p] < 0) continue;
            wantMap[p]               = next[ids[p]]++;
            rowExpert[wantMap[p]]    = ids[p];
            rowPair[wantMap[p]]      = static_cast<int>(p);
        }
    }
    DevBuf counts, offsets, psum, hist, rowMap, rowExp, rowPairs, out;
    ASSERT_MPI_TRUE(counts.Alloc(epr * sizeof(int32_t)) && offsets.Alloc(epr * sizeof(int32_t)) &&
                    psum.Alloc(epr * sizeof(int32_t)) &&
                    hist.Alloc(static_cast<size_t>(std::max(n, 1)) * epr * sizeof(int32_t)) &&
                    rowMap.Alloc(static_cast<size_t>(n) * K * sizeof(int32_t)) && Upload(rowExp, rowExpert) &&
                    Upload(rowPairs, rowPair) && out.Alloc(tokBytes));

    for (int fp8 = 0; fp8 <= 1; ++fp8) {
        SCOPED_TRACE(fp8 ? "fp8" : "bf16");
        const Dispatch& src      = fp8 ? f : d;
        const size_t    rowBytes = fp8 ? static_cast<size_t>(H) : static_cast<size_t>(H) * sizeof(__hip_bfloat16);
        const int rows = ep_expand_build(handle_, src.outTopk.u(), n, kAlignment, counts.u(), offsets.u(), psum.u(),
                                         hist.u(), rowMap.u(), s);
        ASSERT_MPI_EQ(total, rows);
        std::vector<int32_t> gotCounts, gotOffsets, gotPsum, gotMap;
        ASSERT_MPI_TRUE(Download(counts, epr, gotCounts) && Download(offsets, epr, gotOffsets) &&
                        Download(psum, epr, gotPsum) && Download(rowMap, static_cast<size_t>(n) * K, gotMap));
        ASSERT_MPI_TRUE(gotCounts == wantCounts && gotOffsets == wantOffsets && gotPsum == wantPsum &&
                        gotMap == wantMap);

        // Poisoned, so padding the scatter does not zero, or a row it never writes, cannot pass.
        DevBuf ex, esf, ew;
        ASSERT_MPI_TRUE(ex.Alloc(static_cast<size_t>(total) * rowBytes) &&
                        esf.Alloc(static_cast<size_t>(total) * nsf * sizeof(float)) &&
                        ew.Alloc(static_cast<size_t>(total) * sizeof(float)));
        ASSERT_MPI_EQ(hipSuccess, hipMemset(ex.p, 0xFF, static_cast<size_t>(total) * rowBytes));
        ASSERT_MPI_EQ(hipSuccess, hipMemset(esf.p, 0xFF, static_cast<size_t>(total) * nsf * sizeof(float)));
        ASSERT_MPI_EQ(hipSuccess, hipMemset(ew.p, 0xFF, static_cast<size_t>(total) * sizeof(float)));
        ASSERT_MPI_EQ(0, ep_expand_scatter(handle_, fp8, src.outX.u(), fp8 ? rsf8.u() : 0, src.outTw.u(), rowMap.u(), n,
                                           ex.u(), fp8 ? esf.u() : 0, fp8 ? 1 : 0, fp8 ? rows : 0, ew.u(),
                                           /*zero_pad=*/1, kAlignment, counts.u(), offsets.u(), kNumSms, s));
        // Every row is a pair's payload (and scales) and weight, or zeros.
        std::vector<uint8_t> hx, hrx;
        std::vector<float>   hw, hrw, hsf, hrsf;
        ASSERT_MPI_TRUE(Download(ex, static_cast<size_t>(total) * rowBytes, hx) &&
                        Download(src.outX, static_cast<size_t>(n) * rowBytes, hrx) && Download(ew, total, hw) &&
                        Download(src.outTw, static_cast<size_t>(n) * K, hrw) &&
                        (!fp8 || (Download(esf, static_cast<size_t>(total) * nsf, hsf) &&
                                  Download(rsf8, static_cast<size_t>(n) * nsf, hrsf))));
        int bad = 0;
        for (int r = 0; r < total && bad < 8; ++r) {
            const int      p   = rowPair[r];
            const uint8_t* row = hx.data() + static_cast<size_t>(r) * rowBytes;
            bool           ok;
            if (p >= 0) {
                const int i = p / K;
                ok = std::memcmp(row, hrx.data() + static_cast<size_t>(i) * rowBytes, rowBytes) == 0 && hw[r] == hrw[p];
                for (int j = 0; fp8 && j < nsf; ++j) {
                    ok = ok && hsf[static_cast<size_t>(j) * total + r] == hrsf[static_cast<size_t>(i) * nsf + j];
                }
            } else {
                ok = std::all_of(row, row + rowBytes, [](uint8_t b) { return b == 0; }) && hw[r] == 0.0f;
                for (int j = 0; fp8 && j < nsf; ++j) ok = ok && hsf[static_cast<size_t>(j) * total + r] == 0.0f;
            }
            if (!ok) {
                ADD_FAILURE() << "rank " << rank_ << " expanded row " << r << " ("
                              << (p >= 0 ? "pair " + std::to_string(p) : std::string("padding")) << ") is wrong";
                ++bad;
            }
        }
        ASSERT_MPI_EQ(0, bad);

        // The expert, then the combines through the row map.
        DevBuf y;
        ASSERT_MPI_TRUE(y.Alloc(static_cast<size_t>(total) * H * sizeof(__hip_bfloat16)));
        if (byValue) {
            // The padding rows keep the 0xFF poison, NaN in bf16, so a combine that read one could not pass.
            ASSERT_MPI_EQ(hipSuccess,
                          hipMemsetAsync(y.p, 0xFF, static_cast<size_t>(total) * H * sizeof(__hip_bfloat16), stream));
        }
        if (total > 0 && byValue) {
            FillExpandedExpertRows<<<GridFor(static_cast<size_t>(total) * H), 256, 0, stream>>>(
                static_cast<const int*>(rowPairs.p), static_cast<const int32_t*>(src.outSrc.p), total, K, T, H,
                static_cast<__hip_bfloat16*>(y.p));
        } else if (total > 0 && fp8) {
            Fp8Decode<<<GridFor(static_cast<size_t>(total) * H), 256, 0, stream>>>(
                static_cast<const uint8_t*>(ex.p), static_cast<const float*>(esf.p), total, H, 1, total,
                static_cast<__hip_bfloat16*>(y.p));
        } else if (total > 0) {
            ScaleByExpert<<<GridFor(static_cast<size_t>(total) * H), 256, 0, stream>>>(
                static_cast<const __hip_bfloat16*>(ex.p), static_cast<const int32_t*>(rowExp.p), total, H,
                static_cast<__hip_bfloat16*>(y.p));
        }
        ASSERT_MPI_EQ(hipSuccess, hipGetLastError());
        // What slot k of this rank's token t comes back as, element c: its row after the expert.
        auto value = [&](int t, int k, int c) {
            if (byValue) return ExpertValue(rank_, t, k, c);
            const int le = d.topk[static_cast<size_t>(t) * K + k] % epr;
            return fp8 ? Fp8Decoded(Fp8Byte(rank_, t, c, 0), Fp8Scale(rank_, t, c / 128, 0))
                       : PayloadValue(rank_, t, c) * ExpertScale(le);
        };
        std::vector<int>   owner(static_cast<size_t>(K));
        std::vector<char>  last(static_cast<size_t>(K));
        std::vector<float> v(static_cast<size_t>(K));
        for (int grouped = 1; grouped >= 0; --grouped) {
            SCOPED_TRACE(grouped ? "grouped combine" : "one row per slot");
            ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, tokBytes, stream));
            ASSERT_MPI_EQ(0, ep_combine(handle_, y.u(), 0, rowMap.u(), src.outTopk.u(), src.outSrc.u(), n,
                                        d.topkIdx.u(), T, 0, 0, grouped, out.u(), 0, kNumSms, s));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
            std::vector<uint16_t> got;
            ASSERT_MPI_TRUE(Download(out, static_cast<size_t>(T) * H, got));
            // The token's slot values v summed as the combine must: one row per slot, every slot in
            // ascending k; grouped, each owner's rows summed in ascending k and rounded once, added at the
            // owner's last slot. Then rounded once. With ascending false, every one of those orders is
            // reversed instead.
            auto sum = [&](bool ascending) {
                float acc = 0.0f;
                for (int j = 0; j < K; ++j) {
                    const int k = ascending ? j : K - 1 - j;
                    if (owner[k] < 0 || (grouped && !last[k])) continue;
                    if (!grouped) {
                        acc += v[k];
                        continue;
                    }
                    float partial = 0.0f;
                    for (int j2 = 0; j2 <= k; ++j2) {
                        const int k2 = ascending ? j2 : k - j2;
                        if (owner[k2] == owner[k]) partial += v[k2];
                    }
                    acc += Bf16Round(partial);
                }
                return Bf16Round(acc);
            };
            int orderSensitive = 0;
            for (int t = 0; t < T && bad < 8; ++t) {
                // Each slot's owner, and whether it is the owner's last slot, where a grouped combine
                // folds the owner's partial.
                for (int k = 0; k < K; ++k) {
                    const int e = d.topk[static_cast<size_t>(t) * K + k];
                    owner[k]    = ValidExpert(e, numExperts_) ? e / epr : -1;
                }
                for (int k = 0; k < K; ++k) {
                    last[k] = owner[k] >= 0;
                    for (int k2 = k + 1; k2 < K; ++k2) last[k] = last[k] && owner[k2] != owner[k];
                }
                for (int c = 0; c < H; ++c) {
                    for (int k = 0; k < K; ++k) v[k] = owner[k] >= 0 ? value(t, k, c) : 0.0f;
                    const float want = sum(true);
                    orderSensitive += byValue && sum(false) != want ? 1 : 0;
                    const float g    = Bf16Bits(got[static_cast<size_t>(t) * H + c]);
                    if (!(g == want)) {
                        ADD_FAILURE() << "rank " << rank_ << " token " << t << " element " << c << ": got " << g
                                      << ", expected " << want;
                        ++bad;
                        break;
                    }
                }
            }
            ASSERT_MPI_EQ(0, bad);
            // ExpertValue rows do make the order visible: summed the other way round, some elements differ.
            ASSERT_MPI_TRUE(!byValue || orderSensitive > 0);
        }
    }
}

} // namespace

TEST_F(RcclEpDispatchCombineTest, DispatchCombineRoundTrip)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));
    ASSERT_EP_STEP(RoundTrip());
}

// ep_configure is re-enterable: called again at a different shape it tears the window
// down, reallocates and re-registers it, rebuilds the devComm and refills the peer
// views. The Python layer takes that path whenever a caller changes num_experts or
// num_topk between steps.
//
// The second shape changes BOTH: num_experts alone would leave the window the same
// size, so a reconfigure that quietly kept the old allocation would still pass. topk
// scales `y` and `cw`, so the round trip below -- whose per-slot combine fills one y row
// per (slot, k) -- can only hold if the new layout is the one in force. Running the round
// trip at each shape is what proves the first configure was live too, rather than only the
// last one.
TEST_F(RcclEpDispatchCombineTest, ReconfigureBetweenShapes)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }

    ASSERT_EP_STEP(CreateEp(nRanks_ * (kExpertsPerRank / 2), kTopk / 2));
    ASSERT_EP_STEP(RoundTrip());

    numExperts_ = nRanks_ * kExpertsPerRank;
    topk_       = kTopk;
    ASSERT_MPI_EQ(0, ep_configure(handle_, numExperts_, topk_));
    ASSERT_EP_STEP(RoundTrip());

    // Repeating the shape it already has is the early-return branch, and it must not
    // free and rebuild a window the caller is still using -- the round trip after it
    // reads the same window.
    ASSERT_MPI_EQ(0, ep_configure(handle_, numExperts_, topk_));
    ASSERT_EP_STEP(RoundTrip());
}

// Each case breaks one ep_create argument. A missed rejection shows as a non-null handle or a timeout.
TEST_F(RcclEpDispatchCombineTest, CreateRejectsOutOfRangeArguments)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }

    constexpr int kEpMaxRanks = 16;  // rccl_ep's kMaxRanks. This file stays off rccl_ep's device headers.
    struct Case { const char* name; int numRanks; int maxTokens; int hidden; };
    const Case rejected[] = {
        {"num_ranks kMaxRanks + 1",    kEpMaxRanks + 1, kTokens, kHidden},
        {"max_tokens_per_rank 0",      nRanks_, 0,       kHidden},
        {"max_tokens_per_rank -1",     nRanks_, -1,      kHidden},
        {"hidden 7 (not a multiple)",  nRanks_, kTokens, 7},
        {"hidden 12 (remainder 4)",    nRanks_, kTokens, 12},
        {"hidden 0",                   nRanks_, kTokens, 0},
        {"hidden -8",                  nRanks_, kTokens, -8},
    };
    // A rejected create never reaches init, so all rejected cases share one id.
    std::vector<char> uid;
    ASSERT_EP_STEP(BroadcastUid(&uid));
    for (const Case& c : rejected) {
        SCOPED_TRACE(c.name);
        void* h        = nullptr;
        ASSERT_EP_STEP(CreateBounded(uid, rank_, c.numRanks, c.maxTokens, c.hidden, c.name, &h));
        if (h != nullptr) ep_destroy(h);
        ASSERT_MPI_TRUE(h == nullptr);
    }

    {
        // Valid 1-rank communicator: each rank makes its own, as rank 0 of 1.
        std::vector<char> localUid(static_cast<size_t>(ep_unique_id_size()), 0);
        ASSERT_MPI_EQ(0, ep_get_unique_id(localUid.data()));
        void* solo = nullptr;
        ASSERT_EP_STEP(CreateBounded(localUid, 0, 1, kTokens, kHidden, "valid single-rank shape", &solo));
        ASSERT_MPI_TRUE(solo != nullptr);
        ep_destroy(solo);
    }

    void* h        = nullptr;
    ASSERT_EP_STEP(BroadcastUid(&uid));
    ASSERT_EP_STEP(CreateBounded(uid, rank_, nRanks_, /*maxTokens=*/1, /*hidden=*/8,
                                "valid minimum shape", &h));
    handle_ = h;  // TearDown destroys it
    ASSERT_MPI_TRUE(handle_ != nullptr);
}

TEST_F(RcclEpDispatchCombineTest, DispatchBf16AndFp8Bandwidth)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));

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
            ASSERT_MPI_TRUE(nRecv >= 0 && nRecv <= cap);
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

// Every token routed to rank 0: rank 0 receives the whole window and the other ranks
// receive nothing. Dispatch and combine must still complete on every rank and return the
// right rows; a rank with no rows must not leave its peers waiting.
TEST_F(RcclEpDispatchCombineTest, RanksThatReceiveNothing)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, HotExpert));
    ASSERT_EP_STEP(DispatchBarrier(d, stream));
    ASSERT_MPI_EQ(rank_ == 0 ? nRanks_ * tokens_ : 0, d.nRecv);
    ASSERT_EP_STEP(CheckReceivedRows(d, HotExpert));
    DevBuf out;
    ASSERT_EP_STEP(Combine(d, d.outX, out, stream));
    ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream));
}

// Odd ranks have no tokens, like an idle data-parallel rank, and pass a null pointer for every
// token-sized buffer, as an empty tensor's data pointer is. They still join every call and
// receive their share, with both payload types through one plan.
TEST_F(RcclEpDispatchCombineTest, RanksThatSendNothing)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk, kSmallTokens, kSmallHidden));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));
    const uintptr_t s    = reinterpret_cast<uintptr_t>(stream);
    const bool      idle = (rank_ % 2) == 1;
    auto tokenBuf = [idle](const DevBuf& b) { return idle ? uintptr_t{0} : b.u(); };
    const int       T    = tokens_, H = hidden_;
    const int       nTok = idle ? 0 : T;
    const size_t    cap  = static_cast<size_t>(nRanks_) * T;
    const size_t    nsf  = (H + 127) / 128;

    Dispatch             d;
    DevBuf               xSf, outSf, combined;
    int                  expRecv = 0;
    std::vector<int32_t> expCounts;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(AllocOutputs(d, static_cast<int>(cap)));
    ASSERT_MPI_TRUE(xSf.Alloc(static_cast<size_t>(T) * nsf * sizeof(float)) && outSf.Alloc(cap * nsf * sizeof(float)) &&
                    d.slot.Alloc(cap * sizeof(int32_t)) && d.sendList.Alloc(cap * sizeof(int32_t)) &&
                    d.sendc.Alloc(nRanks_ * sizeof(int32_t)) &&
                    combined.Alloc(static_cast<size_t>(T) * H * sizeof(__hip_bfloat16)));
    // The fp8 pass reuses the bf16 payload's bytes with zero scales; this test only moves them
    // (Fp8DispatchIsExact checks fp8 payloads).
    ASSERT_MPI_EQ(hipSuccess, hipMemset(xSf.p, 0, static_cast<size_t>(T) * nsf * sizeof(float)));
    ExpectedReceive(ExpertFor, expRecv, expCounts, /*senderStride=*/2);
    ASSERT_MPI_EQ(0, ep_plan(handle_, tokenBuf(d.topkIdx), nTok, tokenBuf(d.slot), tokenBuf(d.sendList), d.sendc.u(),
                             s));
    for (int useFp8 = 0; useFp8 <= 1; ++useFp8) {
        SCOPED_TRACE(useFp8 ? "fp8" : "bf16");
        d.nRecv = ep_dispatch(handle_, tokenBuf(d.x), useFp8 ? tokenBuf(xSf) : 0, tokenBuf(d.topkIdx),
                              tokenBuf(d.topkW), nTok, tokenBuf(d.sendList), d.sendc.u(), useFp8, kNumSms, d.outX.u(),
                              useFp8 ? outSf.u() : 0, d.outTopk.u(), d.outTw.u(), d.outSrc.u(), s);
        ASSERT_MPI_EQ(expRecv, d.nRecv);
        // Every row from the even ranks, in order, with its metadata (and its payload, for bf16).
        ASSERT_EP_STEP(CheckReceivedBuffers(useFp8 ? nullptr : &d.outX, d.outTopk, d.outTw, d.outSrc, d.nRecv,
                                            ExpertFor, /*senderStride=*/2));
        if (useFp8 == 0) {
            // Idle ranks combine nothing into a null output; the others get every token back.
            ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(combined.p, 0xFF,
                                                     static_cast<size_t>(T) * H * sizeof(__hip_bfloat16), stream));
            ASSERT_MPI_EQ(0, ep_combine(handle_, d.outX.u(), 0, 0, d.outTopk.u(), d.outSrc.u(), d.nRecv,
                                        tokenBuf(d.topkIdx), nTok, 0, 0, /*grouped=*/1, tokenBuf(combined), 0, kNumSms,
                                        s));
            ASSERT_EP_STEP(CheckScaledByDestinations(d, combined, stream, nTok));
        }
    }
}

// Malformed routing: top-k ids at and past the last expert, -1 and other negative ids, tokens
// with no valid id, and an expert repeated within a token (MalformedExpert). Every path ignores
// an id that names no expert. Dispatch delivers a repeated expert in each of its slots, so a
// token still comes back once per distinct destination rank.
TEST_F(RcclEpDispatchCombineTest, MalformedRoutingIsIgnored)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));
    const int    T     = tokens_;
    const int    K     = topk_;
    const size_t bytes = static_cast<size_t>(T) * hidden_ * sizeof(__hip_bfloat16);

    int                  wantRecv = 0;
    std::vector<int32_t> wantPairs;
    ExpectedReceive(MalformedExpert, wantRecv, wantPairs);
    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, MalformedExpert));
    ASSERT_EP_STEP(DispatchBarrier(d, stream));
    ASSERT_MPI_EQ(wantRecv, d.nRecv);
    ASSERT_MPI_TRUE(d.expertCounts == wantPairs);
    ASSERT_EP_STEP(CheckReceivedRows(d, MalformedExpert));
    DevBuf outA;
    ASSERT_EP_STEP(Combine(d, d.outX, outA, stream));
    ASSERT_EP_STEP(CheckScaledByDestinations(d, outA, stream));
    // With the weights carried back: each valid slot's own weight, 0 for a slot that names no expert.
    DevBuf outB, outW;
    ASSERT_MPI_TRUE(outB.Alloc(bytes) && outW.Alloc(static_cast<size_t>(T) * K * sizeof(float)));
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(outB.p, 0xFF, bytes, stream));
    ASSERT_MPI_EQ(0, ep_combine(handle_, d.outX.u(), d.outTw.u(), 0, d.outTopk.u(), d.outSrc.u(), d.nRecv,
                                d.topkIdx.u(), T, 0, 0, /*grouped=*/1, outB.u(), outW.u(), kNumSms,
                                (uintptr_t)stream));
    ASSERT_EP_STEP(CheckScaledByDestinations(d, outB, stream));
    std::vector<float> w;
    ASSERT_MPI_TRUE(Download(outW, static_cast<size_t>(T) * K, w));
    int bad = 0;
    for (int t = 0; t < T && bad < 8; ++t) {
        for (int k = 0; k < K; ++k) {
            const float want =
                ValidExpert(d.topk[static_cast<size_t>(t) * K + k], numExperts_) ? WeightFor(rank_, t, k) : 0.0f;
            if (w[static_cast<size_t>(t) * K + k] != want) {
                ADD_FAILURE() << "rank " << rank_ << " token " << t << " slot " << k << ": weight "
                              << w[static_cast<size_t>(t) * K + k] << ", expected " << want;
                ++bad;
            }
        }
    }
    ASSERT_MPI_EQ(0, bad);
}

// A mixed-architecture group fails ep_configure on every rank, before any payload moves, instead of
// exchanging fp8 bytes the ranks would decode differently. RCCL_EP_FAKE_ARCH stands in for another
// architecture on one rank (include/ep_arch.h): set on every rank with RCCL_EP_FAKE_ARCH_RANK
// naming that rank, and set on that rank alone. Intranode the check travels over the window;
// internode it is an all-gather over the global communicator. Each rejection releases what the
// configure took and leaves the handle as it was: the same configure without the fake then
// succeeds.
TEST_F(RcclEpDispatchCombineTest, MixedArchitectureFailsConfigure)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    constexpr int kT = 256, kH = 1024;
    struct Case
    {
        const char* what;
        int         nodeSize;   // 0: intranode
        int         fakeRank;
        bool        everyRank;  // RCCL_EP_FAKE_ARCH on every rank and RCCL_EP_FAKE_ARCH_RANK, or on fakeRank only
    };
    const Case cases[] = {
        {"intranode", 0, 3, true},
        {"internode direct", kNodeSize, 5, false},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.what);
        ASSERT_EP_STEP(CreateHandle(kT, kH, c.nodeSize));
        int rc = 0;
        {
            const std::string fakeRank = std::to_string(c.fakeRank);
            ScopedEnv fake("RCCL_EP_FAKE_ARCH", (c.everyRank || rank_ == c.fakeRank) ? "gfx000" : nullptr);
            ScopedEnv which("RCCL_EP_FAKE_ARCH_RANK", c.everyRank ? fakeRank.c_str() : nullptr);
            rc = ep_configure(handle_, nRanks_ * kExpertsPerRank, kTopk);
        }
        ASSERT_MPI_EQ(-1, rc);
        ASSERT_MPI_EQ(0, ep_configure(handle_, nRanks_ * kExpertsPerRank, kTopk));
        DestroyEp();
    }
}

// RCCL declines a window registration when window support is off: it returns success with a null window. ep_configure
// must fail on every rank then, and leave the handle unconfigured but sound: the same configure again is a second
// attempt that fails the same way, not the early return a live window takes; a data-path call is refused;
// ep_window_bytes still answers; ep_destroy releases it. RCCL reads NCCL_WIN_ENABLE once, at the first communicator
// init, which is ep_create's (one test per process, so nothing has read it before), so the setting is held across
// ep_create and ep_configure. What it read then stays in force for the rest of the process, so a test after this one in
// the same process could not configure at all.
TEST_F(RcclEpDispatchCombineTest, DeclinedWindowFailsConfigure)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    constexpr int kT = 256, kH = 1024;
    ScopedEnv noWindows("NCCL_WIN_ENABLE", "0");
    ASSERT_EP_STEP(CreateHandle(kT, kH));
    ASSERT_MPI_EQ(-1, ep_configure(handle_, nRanks_ * kExpertsPerRank, kTopk));
    ASSERT_MPI_EQ(-1, ep_configure(handle_, nRanks_ * kExpertsPerRank, kTopk));
    ASSERT_MPI_EQ(-1, ep_barrier(handle_, 0));
    // It reads only the handle; what it reports is the size the configure computed, not a window held.
    (void)ep_window_bytes(handle_);
    DestroyEp();
}

// The wave64 check is agreed across the group: one rank without a wave64 device fails ep_create on
// every rank, and each releases its communicator rather than leaving the others waiting in their next
// collective. RCCL_EP_FAKE_WAVE_SIZE stands in for a wave32 device, named by
// RCCL_EP_FAKE_WAVE_SIZE_RANK on every rank or set on that rank alone. A create without it then works.
TEST_F(RcclEpDispatchCombineTest, NonWave64RankFailsCreateOnEveryRank)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    constexpr int kT = 256, kH = 1024;
    struct Case
    {
        const char* what;
        int         fakeRank;
        bool        everyRank;  // the hook on every rank with RCCL_EP_FAKE_WAVE_SIZE_RANK, or on fakeRank only
    };
    const Case cases[] = {
        {"named by RCCL_EP_FAKE_WAVE_SIZE_RANK", 3, true},
        {"set on that rank alone", 5, false},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.what);
        std::vector<char> uid;
        ASSERT_EP_STEP(BroadcastUid(&uid));
        void* h = nullptr;
        {
            const std::string fakeRank = std::to_string(c.fakeRank);
            ScopedEnv fake("RCCL_EP_FAKE_WAVE_SIZE", (c.everyRank || rank_ == c.fakeRank) ? "32" : nullptr);
            ScopedEnv which("RCCL_EP_FAKE_WAVE_SIZE_RANK", c.everyRank ? fakeRank.c_str() : nullptr);
            h = ep_create(rank_, nRanks_, uid.data(), kT, kH, device_);
        }
        if (h != nullptr) ep_destroy(h);
        ASSERT_MPI_TRUE(h == nullptr);
    }
    ASSERT_EP_STEP(CreateHandle(kT, kH));
}

// ep_configure allocates, registers and launches on the handle's device whichever device the calling thread
// has made current since ep_create, and leaves the caller's current: configure from the next rank's device,
// then a round trip on the handle's device to show the window is sound. The destroy from the next rank's device
// is a restore check only: the RCCL teardown calls ep_destroy makes select their device and restore the
// caller's themselves, so it passes with or without ep_destroy's own device scope.
TEST_F(RcclEpDispatchCombineTest, ConfigureAndDestroyKeepTheCallersDevice)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    int count = 0, fewest = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGetDeviceCount(&count));
    ASSERT_MPI_EQ(MPI_SUCCESS, MPI_Allreduce(&count, &fewest, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (fewest < 2) {
        GTEST_SKIP() << "needs a second visible device on every rank";
    }
    constexpr int kT = 256, kH = 1024;
    const int     other   = (device_ + 1) % count;
    int           current = -1;
    numExperts_           = nRanks_ * kExpertsPerRank;
    topk_                 = kTopk;
    ASSERT_EP_STEP(CreateHandle(kT, kH));

    ASSERT_MPI_EQ(hipSuccess, hipSetDevice(other));
    ASSERT_MPI_EQ(0, ep_configure(handle_, numExperts_, topk_));
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&current));
    ASSERT_MPI_EQ(other, current);

    ASSERT_MPI_EQ(hipSuccess, hipSetDevice(device_));
    ASSERT_EP_STEP(RoundTrip());

    ASSERT_MPI_EQ(hipSuccess, hipSetDevice(other));
    ep_destroy(handle_);
    handle_ = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&current));
    ASSERT_MPI_EQ(other, current);
    ASSERT_MPI_EQ(hipSuccess, hipSetDevice(device_));
}

// The ep_create arguments CreateRejectsOutOfRangeArguments leaves out are rejected locally too, before
// ncclCommInitRank, so each rank returns a null handle on its own and none is left waiting in the init;
// nothing here needs symmetric memory or a particular rank count. Every entry point rejects a null handle.
TEST_F(RcclEpDispatchCombineTest, CreateAndNullHandleRejectBadArguments)
{
    int device = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGetDevice(&device));
    // No call reaches ncclCommInitRank, so the id's bytes do not matter.
    std::vector<char> uid(static_cast<size_t>(ep_unique_id_size()), 0);
    struct Case
    {
        const char* what;
        int rank, ranks, tokens, hidden, nodeSize;
    };
    const Case cases[] = {
        {"node_size not dividing num_ranks", 0, 8, kTokens, kHidden, 3},
        {"num_ranks 0", 0, 0, kTokens, kHidden, 0},
        {"rank outside num_ranks", 8, 8, kTokens, kHidden, 8},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.what);
        void* h = c.nodeSize == c.ranks
                      ? ep_create(c.rank, c.ranks, uid.data(), c.tokens, c.hidden, device)
                      : ep_create_nodes(c.rank, c.ranks, uid.data(), c.tokens, c.hidden, device, c.nodeSize);
        ASSERT_MPI_TRUE(h == nullptr);
    }
    ASSERT_MPI_TRUE(ep_create(0, 8, nullptr, kTokens, kHidden, device) == nullptr);

    ASSERT_MPI_EQ(-1, ep_configure(nullptr, 64, 4));
    ASSERT_MPI_EQ(-1, ep_barrier(nullptr, 0));
    ASSERT_MPI_EQ(-1, ep_plan(nullptr, 0, 0, 0, 0, 0, 0));
    ASSERT_MPI_EQ(-1, ep_dispatch(nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    ASSERT_MPI_EQ(-1, ep_recv_counts(nullptr, 0, 0));
    ASSERT_MPI_EQ(-1, ep_expert_counts(nullptr, 0, 0, 0, 0));
    ASSERT_MPI_EQ(-1, ep_expand_build(nullptr, 0, 0, 1, 0, 0, 0, 0, 0, 0));
    ASSERT_MPI_EQ(-1, ep_expand_scatter(nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0));
    ASSERT_MPI_EQ(-1, ep_combine(nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0));
    ASSERT_MPI_EQ(size_t{0}, ep_window_bytes(nullptr));
    ep_destroy(nullptr);
}

// On a configured handle, each data-path entry point rejects a bad argument on every rank alike,
// before it launches anything, so no rank is left in a barrier; and a rejected call leaves the
// handle as it was, which the round trip at the end shows. hidden is a multiple of 8 but not of
// 16, so bf16 runs and fp8 is refused; the window is small.
TEST_F(RcclEpDispatchCombineTest, DataPathRejectsBadArguments)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    constexpr int kT = 256, kH = 1032;
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk, kT, kH));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));
    const uintptr_t s   = reinterpret_cast<uintptr_t>(stream);
    const int       R   = nRanks_;
    const int       K   = topk_;
    const int       epr = ExpertsPerRank();
    const size_t    cap = static_cast<size_t>(R) * kT;

    Dispatch d;
    DevBuf   xSf, outSf, out;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(AllocOutputs(d, static_cast<int>(cap)));
    ASSERT_MPI_TRUE(d.slot.Alloc(cap * sizeof(int32_t)) && d.sendList.Alloc(cap * sizeof(int32_t)) &&
                    d.sendc.Alloc(R * sizeof(int32_t)) && d.counts.Alloc(epr * sizeof(int32_t)) &&
                    xSf.Alloc(static_cast<size_t>(kT) * kH * sizeof(float)) &&
                    outSf.Alloc(cap * kH * sizeof(float)) &&
                    out.Alloc(static_cast<size_t>(kT) * kH * sizeof(__hip_bfloat16)));
    std::vector<int32_t> host(static_cast<size_t>(2 * R), 0);

    struct Rejected
    {
        const char* what;
        int         rc;
    };
    std::vector<Rejected> rejected;
    auto rejects = [&](const char* what, int rc) { rejected.push_back({what, rc}); };
    // The valid dispatch, with one argument replaced per call.
    auto dispatch = [&](uintptr_t x, int n, uintptr_t sendc, int fp8, int sms, uintptr_t outX, uintptr_t outTw) {
        return ep_dispatch(handle_, x, fp8 ? xSf.u() : 0, d.topkIdx.u(), d.topkW.u(), n, d.sendList.u(), sendc, fp8,
                           sms, outX, fp8 ? outSf.u() : 0, d.outTopk.u(), outTw, d.outSrc.u(), s);
    };
    auto combine = [&](int nRecv, int n, uintptr_t o) {
        return ep_combine(handle_, d.outX.u(), 0, 0, d.outTopk.u(), d.outSrc.u(), nRecv, d.topkIdx.u(), n, 0, 0,
                          /*grouped=*/1, o, 0, kNumSms, s);
    };
    rejects("ep_plan past capacity",
            ep_plan(handle_, d.topkIdx.u(), kT + 1, d.slot.u(), d.sendList.u(), d.sendc.u(), s));
    rejects("ep_plan with negative tokens",
            ep_plan(handle_, d.topkIdx.u(), -1, d.slot.u(), d.sendList.u(), d.sendc.u(), s));
    rejects("ep_plan with a null topk_idx", ep_plan(handle_, 0, kT, d.slot.u(), d.sendList.u(), d.sendc.u(), s));
    rejects("dispatch past capacity", dispatch(d.x.u(), kT + 1, d.sendc.u(), 0, kNumSms, d.outX.u(), d.outTw.u()));
    rejects("fp8 dispatch with hidden % 16", dispatch(d.x.u(), kT, d.sendc.u(), 1, kNumSms, d.outX.u(), d.outTw.u()));
    rejects("dispatch with a null x", dispatch(0, kT, d.sendc.u(), 0, kNumSms, d.outX.u(), d.outTw.u()));
    rejects("dispatch with a null sendc", dispatch(d.x.u(), kT, 0, 0, kNumSms, d.outX.u(), d.outTw.u()));
    rejects("dispatch with a null out_x", dispatch(d.x.u(), kT, d.sendc.u(), 0, kNumSms, 0, d.outTw.u()));
    rejects("dispatch with a null out_tw", dispatch(d.x.u(), kT, d.sendc.u(), 0, kNumSms, d.outX.u(), 0));
    rejects("dispatch with a negative num_sms", dispatch(d.x.u(), kT, d.sendc.u(), 0, -1, d.outX.u(), d.outTw.u()));
    rejects("combine past capacity", combine(0, kT + 1, out.u()));
    rejects("combine with num_recv past the window", combine(static_cast<int>(cap) + 1, kT, out.u()));
    rejects("combine with a null out", combine(0, kT, 0));
    rejects("fp8 ep_expand_scatter with hidden % 16",
            ep_expand_scatter(handle_, 1, d.outX.u(), outSf.u(), d.outTw.u(), d.slot.u(), 0, out.u(), outSf.u(), 1, 1,
                              0, 0, 1, 0, 0, 0, s));
    rejects("ep_expand_build with alignment 0",
            ep_expand_build(handle_, d.outTopk.u(), 0, 0, d.counts.u(), d.counts.u(), d.counts.u(), 0, 0, s));
    rejects("ep_inter_counts on an intranode handle", ep_inter_counts(handle_, d.sendc.u(), host.data(), s));
    rejects("ep_configure with num_topk past kMaxTopk", ep_configure(handle_, numExperts_, 17));
    rejects("ep_configure with experts not a multiple of the ranks", ep_configure(handle_, numExperts_ + 1, K));
    for (const Rejected& r : rejected) {
        SCOPED_TRACE(r.what);
        ASSERT_MPI_EQ(-1, r.rc);
    }

    // Nothing above changed the handle: the configured window carries a round trip as before.
    ASSERT_MPI_EQ(0, ep_plan(handle_, d.topkIdx.u(), kT, d.slot.u(), d.sendList.u(), d.sendc.u(), s));
    d.nRecv = dispatch(d.x.u(), kT, d.sendc.u(), 0, kNumSms, d.outX.u(), d.outTw.u());
    int                  wantRecv = 0;
    std::vector<int32_t> wantCounts;
    ExpectedReceive(ExpertFor, wantRecv, wantCounts);
    ASSERT_MPI_EQ(wantRecv, d.nRecv);
    ASSERT_MPI_EQ(hipSuccess,
                  hipMemsetAsync(out.p, 0xFF, static_cast<size_t>(kT) * kH * sizeof(__hip_bfloat16), stream));
    ASSERT_MPI_EQ(0, combine(d.nRecv, kT, out.u()));
    ASSERT_EP_STEP(CheckScaledByDestinations(d, out, stream));
}

// Internode, direct mode: one flat domain; peers on the other node get their rows through
// RCCL send/recv, with the counts exchanged by ep_inter_counts.
TEST_F(RcclEpDispatchCombineTest, InternodeDirectRoundTrip)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEpNodes(nRanks_ * kExpertsPerRank, kTopk, kNodeSize));
    ASSERT_EP_STEP(InternodeRoundTrip());
}

// Internode, the rest of the direct data path, at the small shape. One bf16 dispatch supplies the
// plan and the received rows of the first two parts. An fp8 dispatch through that plan is checked
// as Fp8DispatchIsExact checks the intranode one: every payload byte and per-128 scale of every
// row, then the decoded grouped combine, every element bit for bit. The fp8 rows reach a remote
// peer through outbox views laid out for one byte per element. Then the expanded layout, combined
// one row per (token, slot) through the row map, with the expanded weights passed in and read back
// at the origin: each token's K rows of hidden and their weights cross between the nodes and are
// unpacked into the origin's window, and every element must be the host's ascending-k sum of the
// ExpertValue rows with one rounding, every weight the slot's own. Last, a dispatch that reaches
// only the even ranks and a grouped combine in which the odd ranks, which received nothing, pass no
// weights while the even ranks pass theirs: whether to pass weights is each rank's own choice, so
// the weight region travels either way, the transfers still pair up, and every rank that asks for
// the weights gets them.
TEST_F(RcclEpDispatchCombineTest, InternodeFp8AndWeightedCombines)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEpNodes(nRanks_ * kExpertsPerRank, kTopk, kNodeSize, kSmallTokens, kSmallHidden));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));
    constexpr int   kAlignment = 16;
    constexpr int   kSalt      = 1;
    const uintptr_t s          = reinterpret_cast<uintptr_t>(stream);
    const int       T = tokens_, H = hidden_, K = topk_, epr = ExpertsPerRank();
    const int       nsf      = (H + 127) / 128;
    const size_t    tokBytes = static_cast<size_t>(T) * H * sizeof(__hip_bfloat16);
    const size_t    wBytes   = static_cast<size_t>(T) * K * sizeof(float);

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(DispatchInternode(d, stream));
    ASSERT_EP_STEP(CheckReceivedRows(d, ExpertFor));
    const int    n   = d.nRecv;
    const size_t cap = static_cast<size_t>(d.cap);
    DevBuf       out, outW;
    ASSERT_MPI_TRUE(out.Alloc(tokBytes) && outW.Alloc(wBytes));
    std::vector<uint16_t> got;
    std::vector<float>    gotW;

    // fp8 through the same plan, bound again as a cached dispatch is. Poisoned, so a byte or a scale
    // the epilogue never writes cannot match.
    Dispatch f;
    DevBuf   x8, sf8, outSf, y8;
    ASSERT_EP_STEP(AllocOutputs(f, d.cap));
    ASSERT_MPI_TRUE(x8.Alloc(static_cast<size_t>(T) * H) && sf8.Alloc(static_cast<size_t>(T) * nsf * sizeof(float)) &&
                    outSf.Alloc(cap * nsf * sizeof(float)) && y8.Alloc(cap * H * sizeof(__hip_bfloat16)));
    FillFp8<<<GridFor(static_cast<size_t>(T) * H), 256>>>(static_cast<uint8_t*>(x8.p), static_cast<float*>(sf8.p),
                                                           rank_, T, H, kSalt);
    ASSERT_MPI_EQ(hipSuccess, hipGetLastError());
    ASSERT_MPI_EQ(hipSuccess, hipDeviceSynchronize());
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(f.outX.p, 0xFF, cap * H, stream));
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(outSf.p, 0xFF, cap * nsf * sizeof(float), stream));
    ASSERT_EP_STEP(Bind(d));
    f.nRecv = ep_dispatch(handle_, x8.u(), sf8.u(), d.topkIdx.u(), d.topkW.u(), T, d.sendList.u(), d.sendc.u(),
                          /*use_fp8=*/1, kNumSms, f.outX.u(), outSf.u(), f.outTopk.u(), f.outTw.u(), f.outSrc.u(), s);
    ASSERT_MPI_EQ(n, f.nRecv);
    ASSERT_EP_STEP(CheckReceivedBuffers(nullptr, f.outTopk, f.outTw, f.outSrc, n, ExpertFor));
    {
        std::vector<int32_t> src;
        std::vector<uint8_t> rx;
        std::vector<float>   rsf;
        ASSERT_MPI_TRUE(Download(f.outSrc, n, src) && Download(f.outX, static_cast<size_t>(n) * H, rx) &&
                        Download(outSf, static_cast<size_t>(n) * nsf, rsf));
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            const int      r   = src[i] / T, t = src[i] % T;
            const uint8_t* row = rx.data() + static_cast<size_t>(i) * H;
            const float*   rs  = rsf.data() + static_cast<size_t>(i) * nsf;
            int            c = 0, j = 0;
            while (c < H && row[c] == Fp8Byte(r, t, c, kSalt)) ++c;
            while (j < nsf && rs[j] == Fp8Scale(r, t, j, kSalt)) ++j;
            if (c < H || j < nsf) {
                if (bad < 8) {
                    ADD_FAILURE() << "rank " << rank_ << " row " << i << " (source " << src[i] << "): "
                                  << (c < H ? "byte " + std::to_string(c) : "scale " + std::to_string(j)) << " differs";
                }
                ++bad;
            }
        }
        ASSERT_MPI_EQ(0, bad);
    }
    // The identity expert decodes each row; every element of every token comes back as its decoded
    // value x the number of distinct ranks that own the token's experts.
    if (n > 0) {
        Fp8Decode<<<GridFor(static_cast<size_t>(n) * H), 256, 0, stream>>>(
            static_cast<const uint8_t*>(f.outX.p), static_cast<const float*>(outSf.p), n, H, nsf, 1,
            static_cast<__hip_bfloat16*>(y8.p));
    }
    ASSERT_MPI_EQ(hipSuccess, hipGetLastError());
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, tokBytes, stream));
    ASSERT_EP_STEP(Bind(d));
    ASSERT_MPI_EQ(0, ep_combine(handle_, y8.u(), 0, 0, f.outTopk.u(), f.outSrc.u(), n, d.topkIdx.u(), T, 0, 0,
                                /*grouped=*/1, out.u(), 0, kNumSms, s));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    ASSERT_MPI_TRUE(Download(out, static_cast<size_t>(T) * H, got));
    {
        int bad = 0;
        for (int t = 0; t < T && bad < 8; ++t) {
            const float dests = static_cast<float>(DistinctDestinations(d.topk, t));
            for (int c = 0; c < H; ++c) {
                const float want =
                    Bf16Round(Fp8Decoded(Fp8Byte(rank_, t, c, kSalt), Fp8Scale(rank_, t, c / 128, kSalt)) * dests);
                const float g = Bf16Bits(got[static_cast<size_t>(t) * H + c]);
                if (!(g == want)) {
                    ADD_FAILURE() << "rank " << rank_ << " fp8 token " << t << " element " << c << ": got " << g
                                  << ", expected " << want;
                    ++bad;
                    break;
                }
            }
        }
        ASSERT_MPI_EQ(0, bad);
    }

    // The expanded layout the host expects, as in ExpandedLayoutGroupsByExpert: per local expert, its
    // pairs in (token, slot) order from a base that is a multiple of the alignment.
    std::vector<int32_t> ids;
    ASSERT_MPI_TRUE(Download(d.outTopk, static_cast<size_t>(n) * K, ids));
    std::vector<int32_t> base(static_cast<size_t>(epr), 0), wantMap(static_cast<size_t>(n) * K, -1);
    for (int32_t id : ids) {
        if (id >= 0) ++base[id];
    }
    int total = 0;
    for (int le = 0; le < epr; ++le) {
        const int pairs = base[le];
        base[le]        = total;
        total += (pairs + kAlignment - 1) / kAlignment * kAlignment;
    }
    std::vector<int32_t> rowPair(static_cast<size_t>(std::max(total, 1)), -1);
    for (size_t p = 0; p < ids.size(); ++p) {
        if (ids[p] < 0) continue;
        wantMap[p]          = base[ids[p]]++;
        rowPair[wantMap[p]] = static_cast<int32_t>(p);
    }
    DevBuf counts, offsets, psum, hist, rowMap, rowPairBuf, ex, ew, y;
    ASSERT_MPI_TRUE(counts.Alloc(epr * sizeof(int32_t)) && offsets.Alloc(epr * sizeof(int32_t)) &&
                    psum.Alloc(epr * sizeof(int32_t)) &&
                    hist.Alloc(static_cast<size_t>(std::max(n, 1)) * epr * sizeof(int32_t)) &&
                    rowMap.Alloc(static_cast<size_t>(n) * K * sizeof(int32_t)) && Upload(rowPairBuf, rowPair) &&
                    ex.Alloc(static_cast<size_t>(total) * H * sizeof(__hip_bfloat16)) &&
                    ew.Alloc(static_cast<size_t>(total) * sizeof(float)) &&
                    y.Alloc(static_cast<size_t>(total) * H * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(total, ep_expand_build(handle_, d.outTopk.u(), n, kAlignment, counts.u(), offsets.u(), psum.u(),
                                         hist.u(), rowMap.u(), s));
    std::vector<int32_t> gotMap;
    ASSERT_MPI_TRUE(Download(rowMap, static_cast<size_t>(n) * K, gotMap) && gotMap == wantMap);
    ASSERT_MPI_EQ(0, ep_expand_scatter(handle_, 0, d.outX.u(), 0, d.outTw.u(), rowMap.u(), n, ex.u(), 0, 0, 0, ew.u(),
                                       /*zero_pad=*/1, kAlignment, counts.u(), offsets.u(), kNumSms, s));
    // The expert ignores the scattered rows (ExpandedLayoutGroupsByExpert checks them): its row for
    // pair (i, k) is ExpertValue of token i's source and slot k, so a token's K rows all differ and the
    // order the origin adds them in shows. The weights are the scatter's, one per row.
    if (total > 0) {
        FillExpandedExpertRows<<<GridFor(static_cast<size_t>(total) * H), 256, 0, stream>>>(
            static_cast<const int32_t*>(rowPairBuf.p), static_cast<const int32_t*>(d.outSrc.p), total, K, T, H,
            static_cast<__hip_bfloat16*>(y.p));
    }
    ASSERT_MPI_EQ(hipSuccess, hipGetLastError());
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, tokBytes, stream));
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(outW.p, 0xFF, wBytes, stream));
    ASSERT_EP_STEP(Bind(d));
    ASSERT_MPI_EQ(0, ep_combine(handle_, y.u(), ew.u(), rowMap.u(), d.outTopk.u(), d.outSrc.u(), n, d.topkIdx.u(), T,
                                0, 0, /*grouped=*/0, out.u(), outW.u(), kNumSms, s));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
    ASSERT_MPI_TRUE(Download(out, static_cast<size_t>(T) * H, got) &&
                    Download(outW, static_cast<size_t>(T) * K, gotW));
    {
        int bad = 0;
        for (int t = 0; t < T && bad < 8; ++t) {
            for (int c = 0; c < H; ++c) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    if (ValidExpert(d.topk[static_cast<size_t>(t) * K + k], numExperts_)) {
                        acc += ExpertValue(rank_, t, k, c);
                    }
                }
                const float want = Bf16Round(acc);
                const float g    = Bf16Bits(got[static_cast<size_t>(t) * H + c]);
                if (!(g == want)) {
                    ADD_FAILURE() << "rank " << rank_ << " expanded token " << t << " element " << c << ": got " << g
                                  << ", the ascending-k sum is " << want;
                    ++bad;
                    break;
                }
            }
        }
        ASSERT_MPI_EQ(0, bad);
    }
    ASSERT_MPI_EQ(0, WeightMismatches(gotW, d.topk, rank_, T, K, numExperts_));

    // Only the even ranks receive. The odd ranks, with no rows, have no weights to pass and pass none.
    // Ranks 0, 1, 4 and 5 ask for the weights back: on each node, of the ranks that pass weights and
    // of those that pass none, one asks and one does not.
    Dispatch even;
    ASSERT_EP_STEP(MakeInputs(even, EvenRankExpert));
    ASSERT_EP_STEP(DispatchInternode(even, stream));
    ASSERT_MPI_TRUE((even.nRecv == 0) == (rank_ % 2 == 1));
    ASSERT_EP_STEP(CheckReceivedRows(even, EvenRankExpert));
    const bool passW = even.nRecv > 0;
    const bool askW  = rank_ % kNodeSize < 2;
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, tokBytes, stream));
    ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(outW.p, 0xFF, wBytes, stream));
    ASSERT_EP_STEP(Bind(even));
    ASSERT_MPI_EQ(0, ep_combine(handle_, even.outX.u(), passW ? even.outTw.u() : 0, 0, even.outTopk.u(),
                                even.outSrc.u(), even.nRecv, even.topkIdx.u(), T, 0, 0, /*grouped=*/1, out.u(),
                                askW ? outW.u() : 0, kNumSms, s));
    ASSERT_EP_STEP(CheckScaledByDestinations(even, out, stream));
    ASSERT_MPI_TRUE(Download(outW, static_cast<size_t>(T) * K, gotW));
    ASSERT_MPI_EQ(0, askW ? WeightMismatches(gotW, even.topk, rank_, T, K, numExperts_) : 0);
}

// An internode handle moves rows at the counts ep_inter_bind recorded, so until a bind has succeeded
// ep_dispatch and ep_combine refuse, on every rank alike and before any transfer. The bind refuses a
// missing argument, a send count past the num_tokens it is given (a plan sends a token to a
// destination at most once, and the combine reads that destination's send list that far) and a
// receive count past num_max_tokens_per_rank (one source's region of the window); ep_inter_counts
// refuses a missing sendc. A refused bind binds nothing, a bind at both bounds is accepted, and the
// round trip at the end binds its own counts on the same handle.
TEST_F(RcclEpDispatchCombineTest, InternodeBindIsRequiredAndBounded)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEpNodes(nRanks_ * kExpertsPerRank, kTopk, kNodeSize, kSmallTokens, kSmallHidden));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));
    const uintptr_t s    = reinterpret_cast<uintptr_t>(stream);
    const int       R    = nRanks_, T = tokens_;
    const int       half = T / 2;                    // the num_tokens of the bounded binds
    const int       peer = (rank_ + kNodeSize) % R;  // this rank's counterpart on the other node
    const size_t    cap  = static_cast<size_t>(R) * T;

    Dispatch d;
    DevBuf   out;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(AllocOutputs(d, static_cast<int>(cap)));
    ASSERT_MPI_TRUE(d.slot.Alloc(cap * sizeof(int32_t)) && d.sendList.Alloc(cap * sizeof(int32_t)) &&
                    d.sendc.Alloc(R * sizeof(int32_t)) &&
                    out.Alloc(static_cast<size_t>(T) * hidden_ * sizeof(__hip_bfloat16)));
    ASSERT_MPI_EQ(0, ep_plan(handle_, d.topkIdx.u(), T, d.slot.u(), d.sendList.u(), d.sendc.u(), s));

    struct Rejected
    {
        const char* what;
        int         rc;
    };
    std::vector<Rejected> rejected;
    auto rejects = [&](const char* what, int rc) { rejected.push_back({what, rc}); };
    // Valid calls but for the missing bind.
    auto dispatch = [&]() {
        return ep_dispatch(handle_, d.x.u(), 0, d.topkIdx.u(), d.topkW.u(), T, d.sendList.u(), d.sendc.u(), 0, kNumSms,
                           d.outX.u(), 0, d.outTopk.u(), d.outTw.u(), d.outSrc.u(), s);
    };
    auto combine = [&]() {
        return ep_combine(handle_, d.outX.u(), 0, 0, d.outTopk.u(), d.outSrc.u(), 0, d.topkIdx.u(), T, 0, 0,
                          /*grouped=*/1, out.u(), 0, kNumSms, s);
    };
    std::vector<int32_t> host(static_cast<size_t>(2 * R), 0);
    int32_t* const       sendc = host.data();
    int32_t* const       recvc = host.data() + R;
    rejects("ep_dispatch before any ep_inter_bind", dispatch());
    rejects("ep_combine before any ep_inter_bind", combine());
    rejects("ep_inter_counts with a null sendc", ep_inter_counts(handle_, 0, host.data(), s));
    rejects("ep_inter_bind with a null sendc", ep_inter_bind(handle_, nullptr, recvc, 0, 0));
    sendc[peer] = half + 1;
    rejects("ep_inter_bind with a send count past its num_tokens",
            ep_inter_bind(handle_, sendc, recvc, d.sendList.u(), half));
    sendc[peer] = half;
    recvc[peer] = T + 1;
    rejects("ep_inter_bind with a receive count past num_max_tokens_per_rank",
            ep_inter_bind(handle_, sendc, recvc, d.sendList.u(), half));
    for (const Rejected& r : rejected) {
        SCOPED_TRACE(r.what);
        ASSERT_MPI_EQ(-1, r.rc);
    }
    // Only once every bind above is known to be refused: had one been accepted, this dispatch would
    // post transfers at its counts, which no peer matches, and hang instead of failing.
    ASSERT_MPI_EQ(-1, dispatch());

    // Both counts at their bounds. They describe no dispatch, so nothing runs on them.
    recvc[peer] = T;
    ASSERT_MPI_EQ(0, ep_inter_bind(handle_, sendc, recvc, d.sendList.u(), half));
    ASSERT_EP_STEP(InternodeRoundTrip());
}

// FP8 dispatch, exactly: every received row's payload bytes and fp32 per-128 scales, its top-k ids
// and weights, and the order of the rows, against the routing and the inputs; then a bf16 combine
// of an identity expert that decodes each row as byte x scale (Fp8Decoded, exact in fp32 and the
// same for either fp8 format), which must give back every element of every token, bit for bit. An
// uncached dispatch, then a cached replay of another payload and other scales through the same plan.
TEST_F(RcclEpDispatchCombineTest, Fp8DispatchIsExact)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk, kSmallTokens, kSmallHidden));
    hipStream_t stream = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream));
    SCOPE_EXIT(if (stream != nullptr) (void)hipStreamDestroy(stream));
    const uintptr_t s   = reinterpret_cast<uintptr_t>(stream);
    const int       T   = tokens_, H = hidden_, R = nRanks_;
    const int       nsf = (H + 127) / 128;
    const size_t    cap = static_cast<size_t>(R) * T;
    const size_t    tokBytes = static_cast<size_t>(T) * H * sizeof(__hip_bfloat16);

    Dispatch d;
    ASSERT_EP_STEP(MakeInputs(d, ExpertFor));
    ASSERT_EP_STEP(AllocOutputs(d, static_cast<int>(cap)));
    DevBuf x8[2], sf[2], outSf, y, out;
    ASSERT_MPI_TRUE(d.slot.Alloc(cap * sizeof(int32_t)) && d.sendList.Alloc(cap * sizeof(int32_t)) &&
                    d.sendc.Alloc(R * sizeof(int32_t)) && outSf.Alloc(cap * nsf * sizeof(float)) &&
                    y.Alloc(cap * H * sizeof(__hip_bfloat16)) && out.Alloc(tokBytes));
    for (int p = 0; p < 2; ++p) {
        ASSERT_MPI_TRUE(x8[p].Alloc(static_cast<size_t>(T) * H) &&
                        sf[p].Alloc(static_cast<size_t>(T) * nsf * sizeof(float)));
        FillFp8<<<GridFor(static_cast<size_t>(T) * H), 256>>>(static_cast<uint8_t*>(x8[p].p),
                                                               static_cast<float*>(sf[p].p), rank_, T, H, p);
    }
    ASSERT_MPI_EQ(hipSuccess, hipGetLastError());
    ASSERT_MPI_EQ(hipSuccess, hipDeviceSynchronize());

    int                  wantRecv = 0;
    std::vector<int32_t> wantCounts;
    ExpectedReceive(ExpertFor, wantRecv, wantCounts);
    ASSERT_MPI_EQ(0, ep_plan(handle_, d.topkIdx.u(), T, d.slot.u(), d.sendList.u(), d.sendc.u(), s));
    for (int pass = 0; pass < 2; ++pass) {
        SCOPED_TRACE(pass ? "cached replay" : "uncached dispatch");
        // Poisoned, so a byte or a scale the epilogue never writes cannot match.
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(d.outX.p, 0xFF, cap * H, stream));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(outSf.p, 0xFF, cap * nsf * sizeof(float), stream));
        d.nRecv = ep_dispatch(handle_, x8[pass].u(), sf[pass].u(), d.topkIdx.u(), d.topkW.u(), T, d.sendList.u(),
                              d.sendc.u(), /*use_fp8=*/1, kNumSms, d.outX.u(), outSf.u(), d.outTopk.u(), d.outTw.u(),
                              d.outSrc.u(), s);
        ASSERT_MPI_EQ(wantRecv, d.nRecv);

        // The order and the metadata, then every payload byte and scale of every received row.
        ASSERT_EP_STEP(CheckReceivedBuffers(nullptr, d.outTopk, d.outTw, d.outSrc, d.nRecv, ExpertFor));
        std::vector<int32_t> src;
        std::vector<uint8_t> rx;
        std::vector<float>   rsf;
        ASSERT_MPI_TRUE(Download(d.outSrc, d.nRecv, src) && Download(d.outX, static_cast<size_t>(d.nRecv) * H, rx) &&
                        Download(outSf, static_cast<size_t>(d.nRecv) * nsf, rsf));
        int bad = 0;
        for (int i = 0; i < d.nRecv && bad < 8; ++i) {
            const int r = src[i] / T, t = src[i] % T;
            for (int c = 0; c < H; ++c) {
                if (rx[static_cast<size_t>(i) * H + c] != Fp8Byte(r, t, c, pass)) {
                    ADD_FAILURE() << "rank " << rank_ << " row " << i << " (source " << src[i] << ") byte " << c << ": "
                                  << int(rx[static_cast<size_t>(i) * H + c]) << ", expected "
                                  << int(Fp8Byte(r, t, c, pass));
                    ++bad;
                    break;
                }
            }
            for (int j = 0; j < nsf; ++j) {
                if (rsf[static_cast<size_t>(i) * nsf + j] != Fp8Scale(r, t, j, pass)) {
                    ADD_FAILURE() << "rank " << rank_ << " row " << i << " (source " << src[i] << ") scale " << j
                                  << ": " << rsf[static_cast<size_t>(i) * nsf + j] << ", expected "
                                  << Fp8Scale(r, t, j, pass);
                    ++bad;
                    break;
                }
            }
        }
        ASSERT_MPI_EQ(0, bad);

        // The identity expert, then a grouped bf16 combine: every element of every token comes back as
        // its decoded value x the number of distinct ranks that own the token's experts.
        if (d.nRecv > 0) {
            Fp8Decode<<<GridFor(static_cast<size_t>(d.nRecv) * H), 256, 0, stream>>>(
                static_cast<const uint8_t*>(d.outX.p), static_cast<const float*>(outSf.p), d.nRecv, H, nsf, 1,
                static_cast<__hip_bfloat16*>(y.p));
        }
        ASSERT_MPI_EQ(hipSuccess, hipGetLastError());
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(out.p, 0xFF, tokBytes, stream));
        ASSERT_MPI_EQ(0, ep_combine(handle_, y.u(), 0, 0, d.outTopk.u(), d.outSrc.u(), d.nRecv, d.topkIdx.u(), T, 0, 0,
                                    /*grouped=*/1, out.u(), 0, kNumSms, s));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));
        std::vector<uint16_t> got;
        ASSERT_MPI_TRUE(Download(out, static_cast<size_t>(T) * H, got));
        for (int t = 0; t < T && bad < 8; ++t) {
            const float dests = static_cast<float>(DistinctDestinations(d.topk, t));
            for (int c = 0; c < H; ++c) {
                const float want = Bf16Round(Fp8Decoded(Fp8Byte(rank_, t, c, pass), Fp8Scale(rank_, t, c / 128, pass)) *
                                             dests);
                const float g    = Bf16Bits(got[static_cast<size_t>(t) * H + c]);
                if (!(g == want)) {
                    ADD_FAILURE() << "rank " << rank_ << " token " << t << " element " << c << ": got " << g
                                  << ", expected " << want;
                    ++bad;
                    break;
                }
            }
        }
        ASSERT_MPI_EQ(0, bad);
    }
}

// The expanded layout (do_expand): ep_expand_build and ep_expand_scatter give every received (token, slot) pair its own
// row, grouped by local expert in receive order, each expert's block starting at a multiple of the alignment and its
// padding rows zeroed. Checked against the routing for bf16 rows and for fp8 rows with the column-major scale layout
// (sf_row_stride 1, sf_col_stride = rows), then combined through the row map, grouped (each owner sums its rows of a
// token in ascending k) and one row per slot: the bf16 rows after an expert that scales each row by a power of two of
// its expert (ExpertScale), the fp8 rows after the fp8 decode, each against the host's sums in the contract's order,
// every element.
TEST_F(RcclEpDispatchCombineTest, ExpandedLayoutGroupsByExpert)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    ASSERT_EP_STEP(CreateEp(nRanks_ * kExpertsPerRank, kTopk, kSmallTokens, kSmallHidden));
    ASSERT_EP_STEP(ExpandedRoundTrip(ExpertFor, ExpandedExpert::kScaleByExpert));
}

// num_topk past 8. ep_configure takes up to kMaxTopk = 16 slots and every combine sums a token's rows
// in ascending k with one rounding however many there are, but the tests above route 6 slots or
// fewer. Here num_topk is 12 over 16 experts per rank, routed by CrowdedExpert, through
// ExpandedLayoutGroupsByExpert's round trip, so both combines that read the expanded layout sum more than 8 rows of a
// token: the grouped combine on the sender, for a token with 9 to 12 slots on that rank, and the combine of one row per
// slot at the origin, for a token with more than 8 valid slots. The expert writes ExpertValue rows, so a sum in any
// other order reads back wrong, and every element is checked against the host's sum. (The origin's grouped sum has one
// row per owner rank, so with eight ranks it never has more than 8.)
TEST_F(RcclEpDispatchCombineTest, TopkAboveEightReducesInOrder)
{
    if (!EnvReady()) {
        GTEST_SKIP() << "rccl_ep needs NCCL_CUMEM_ENABLE=1 and 8 ranks on a single node";
    }
    constexpr int kT = 256, kH = 1024, kManyTopk = 12, kManyExpertsPerRank = 16;
    ASSERT_EP_STEP(CreateEp(nRanks_ * kManyExpertsPerRank, kManyTopk, kT, kH));
    // Both sums really take more than 8 rows on every rank: some tokens it receives have more than 8
    // slots here, and some tokens of its own have more than 8 valid slots.
    int manyHere = 0, manyMine = 0;
    for (int r = 0; r < nRanks_; ++r) {
        for (int t = 0; t < tokens_; ++t) {
            int here = 0, valid = 0;
            for (int k = 0; k < topk_; ++k) {
                const int e = CrowdedExpert(r, t, k, numExperts_, nRanks_);
                valid += ValidExpert(e, numExperts_) ? 1 : 0;
                here += ValidExpert(e, numExperts_) && e / ExpertsPerRank() == rank_ ? 1 : 0;
            }
            manyHere += here > 8 ? 1 : 0;
            manyMine += r == rank_ && valid > 8 ? 1 : 0;
        }
    }
    ASSERT_MPI_GT(manyHere, 0);
    ASSERT_MPI_GT(manyMine, 0);
    ASSERT_EP_STEP(ExpandedRoundTrip(CrowdedExpert, ExpandedExpert::kExpertValue));
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED
