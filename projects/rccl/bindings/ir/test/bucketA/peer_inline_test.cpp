/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * peer_inline_test.cpp
 *
 * Multi-GPU test for ncclGetPeerPointer using the INLINE path:
 * the function body is supplied by <nccl_device.h>, which `#include`s
 * the impl headers that define ncclGetPeerPointer as
 * `NCCL_DEVICE_INLINE` (= __device__ __forceinline__). The body inlines
 * directly into our kernel at the consumer's compile.
 *
 * Test plan
 * ---------
 *   For every visible GPU:
 *     1. Allocate a synthetic ncclWindow_vidmem on the device.
 *     2. Launch a 16-thread kernel. Each thread reads a per-thread test
 *        case (lsaRank, worldRank, stride4G, offset, ncclTeam, peer)
 *        from device memory, materializes its own window with those
 *        fields, and calls
 *            ncclGetPeerPointer(&w, offset, tm, peer)
 *        recording the returned pointer in an output array.
 *     3. Host re-derives the expected pointer arithmetically and
 *        compares each thread's result.
 *
 * The test is a "kernel multi-GPU" test in the literal sense: it
 * exercises the function on every device the runtime exposes and
 * verifies portable behaviour. We don't dereference the returned
 * pointers (the windows are synthetic — their bytes don't exist) — the
 * point is to validate the address arithmetic the device API performs,
 * since that is exactly what the bitcode counterpart implements.
 *
 * Sister file `peer_bitcode_test.cpp` runs the same test cases against
 * the bitcode-linked thunk `ncclGetPeerPointerTeam`. Both must produce
 * byte-identical output arrays; the Q of whether their device ISA is
 * byte-identical is investigated by `compare_peer.sh`.
 ************************************************************************/
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <nccl.h>          /* typedef ncclWindow_vidmem* ncclWindow_t      */
#include <nccl_device.h>   /* ncclGetPeerPointer (inline), ncclTeam, ...    */

/* --------------------------- Per-thread test case --------------------- */
struct TestCase {
  int      lsaRank;
  int      worldRank;
  uint32_t stride4G;
  size_t   offset;
  int      tm_nRanks;
  int      tm_rank;
  int      tm_stride;
  int      peer;
};

/* Pure-host re-derivation of the kernel-side arithmetic.
 * Mirrors ncclGetPeerPointer + nccl::utility::add4G:
 *   i        = lsaRank + (peer - tm.rank) * tm.stride
 *   delta4G  = i * stride4G
 *   shifted  = base with its high 32 bits incremented by delta4G
 *   result   = shifted + offset
 */
static uintptr_t host_expected(uintptr_t base, const TestCase& c) {
  int      i       = c.lsaRank + (c.peer - c.tm_rank) * c.tm_stride;
  uint32_t delta4G = (uint32_t)((int32_t)i * (int32_t)c.stride4G);
  uint32_t lo      = (uint32_t)(base & 0xFFFFFFFFu);
  uint32_t hi      = (uint32_t)(base >> 32) + delta4G;
  uintptr_t shift  = ((uintptr_t)hi << 32) | lo;
  return shift + c.offset;
}

/* The kernel under test. Each thread builds its own ncclWindow_vidmem
 * from the shared `base` pointer plus its per-thread test case, then
 * calls the inline ncclGetPeerPointer overload that takes a team. */
__global__ void k_peer_inline(char* base, const TestCase* cases, int N,
                              void** out)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  const TestCase c = cases[t];

  ncclWindow_vidmem w{};
  w.winHost     = nullptr;
  w.lsaFlatBase = base;
  w.lsaRank     = c.lsaRank;
  w.worldRank   = c.worldRank;
  w.stride4G    = c.stride4G;
  w.mcOffset4K  = 0;

  ncclTeam tm{ c.tm_nRanks, c.tm_rank, c.tm_stride };
  out[t] = ncclGetPeerPointer(&w, c.offset, tm, c.peer);
}

#define HIP_CHECK(stmt) do {                                          \
    hipError_t _e = (stmt);                                           \
    if (_e != hipSuccess) {                                           \
      std::fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n",        \
                   (int)_e, hipGetErrorName(_e),                      \
                   __FILE__, __LINE__, hipGetErrorString(_e));        \
      std::exit(2);                                                   \
    }                                                                 \
  } while (0)

/* The 16 test cases. Designed to cover, on a single launch:
 *   - identity (peer == tm.rank): result == base + offset
 *   - +1 / -1 / +N peer relative to tm.rank
 *   - tm.stride > 1 sub-team layout
 *   - non-zero lsaRank shift
 *   - large offset
 *   - large stride4G
 *   - mixed combinations
 * Same array is uploaded to every GPU. */
static const TestCase kCases[] = {
  /* lsaRank, worldRank, stride4G, offset,    nRanks, rank, stride, peer */
  {  0,  0,    0,        0,         1,  0,  1,  0 },  /*  0 identity        */
  {  0,  0,    1,       64,         2,  0,  1,  1 },  /*  1 +1 peer         */
  {  0,  0,    1,       64,         2,  1,  1,  0 },  /*  2 -1 peer         */
  {  0,  0,    2,      128,         4,  0,  1,  3 },  /*  3 +3 peer, s4G=2  */
  {  0,  0,    1,        0,         8,  0,  2,  4 },  /*  4 sub-team str=2  */
  {  5,  0,    1,        0,         1,  0,  1,  0 },  /*  5 lsaRank shift   */
  {  0,  0,    1, 0xDEADBEEFull,    1,  0,  1,  0 },  /*  6 large offset    */
  {  0,  0, 0x10,        0,         2,  0,  1,  1 },  /*  7 large stride4G  */
  {  3,  0,    2,      512,         4,  1,  1,  2 },  /*  8 mixed           */
  {  0,  0,    7,        0,         1,  0,  1,  0 },  /*  9 identity, s4G=7 */
  {  0,  0,    1, 0xCAFEBABEull,    2,  0,  1,  1 },  /* 10 large off + +1  */
  {  2,  0,    1,        0,         3,  1,  1,  2 },  /* 11 lsa+sub         */
  {  0,  0,    4,    0x1000,        2,  1,  1,  0 },  /* 12 -1 peer, s4G=4  */
  {  1,  0,    1,        8,         4,  2,  1,  3 },  /* 13 mixed +1        */
  {  0,  0,   15, 0x40000000ull,    1,  0,  1,  0 },  /* 14 identity, big   */
  {  4,  0,    3,       16,         8,  3,  2,  6 },  /* 15 sub + lsa       */
};

int main() {
  int nDev = 0;
  HIP_CHECK(hipGetDeviceCount(&nDev));
  if (nDev <= 0) { std::fprintf(stderr, "No HIP devices.\n"); return 2; }
  std::printf("[peer-inline] devices=%d\n", nDev);

  constexpr int N = sizeof(kCases) / sizeof(kCases[0]);
  /* A nicely-set bit pattern in the upper half of base so add4G's high
   * arithmetic is observable. The pointer is never dereferenced. */
  const uintptr_t base = (uintptr_t)0x100000000ull;

  int failures = 0;

  for (int d = 0; d < nDev; ++d) {
    HIP_CHECK(hipSetDevice(d));
    char devName[128] = {};
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, d));
    std::snprintf(devName, sizeof devName, "gpu%d:%s", d, prop.name);

    TestCase* d_cases = nullptr;
    void**    d_out   = nullptr;
    HIP_CHECK(hipMalloc(&d_cases, sizeof(kCases)));
    HIP_CHECK(hipMalloc(&d_out,   sizeof(void*) * N));
    HIP_CHECK(hipMemcpy(d_cases, kCases, sizeof(kCases),
                        hipMemcpyHostToDevice));

    k_peer_inline<<<1, N>>>((char*)base, d_cases, N, d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<void*> got(N, nullptr);
    HIP_CHECK(hipMemcpy(got.data(), d_out, sizeof(void*) * N,
                        hipMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N; ++i) {
      uintptr_t exp = host_expected(base, kCases[i]);
      uintptr_t obs = (uintptr_t)got[i];
      if (exp != obs) {
        if (bad == 0)
          std::printf("  %s case %d: got=0x%016lx expect=0x%016lx [FAIL]\n",
                      devName, i, (unsigned long)obs, (unsigned long)exp);
        bad++;
      }
    }
    std::printf("[%s] N=%d bad=%d %s\n", devName, N, bad,
                bad ? "[FAIL]" : "[OK]");
    if (bad) failures++;

    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_cases));
  }

  /* Dump the per-test-case expected addresses to a file so the bitcode
   * test can independently validate that it produces the same bytes. */
  if (const char* p = std::getenv("PEER_DUMP")) {
    FILE* f = std::fopen(p, "wb");
    if (f) {
      for (int i = 0; i < N; ++i) {
        uintptr_t v = host_expected(base, kCases[i]);
        std::fwrite(&v, sizeof v, 1, f);
      }
      std::fclose(f);
      std::printf("[peer-inline] expected bytes -> %s (%zu bytes)\n",
                  p, sizeof(uintptr_t) * N);
    }
  }
  return failures ? 1 : 0;
}
