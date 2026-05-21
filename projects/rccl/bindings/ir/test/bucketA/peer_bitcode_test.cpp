/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * peer_bitcode_test.cpp
 *
 * Sister of peer_inline_test.cpp using the BITCODE path.
 *
 *  - Includes <nccl_device_wrapper.h>, which only forward-declares the
 *    extern "C" __device__ thunks (no inline body).
 *  - Calls ncclGetPeerPointerTeam (the C-ABI name for the team overload
 *    of ncclGetPeerPointer) — its definition lives in the bitcode
 *    artifact librccl_device.bc and is supplied at link time.
 *
 * Test cases are intentionally identical to peer_inline_test.cpp so the
 * two binaries' outputs can be byte-compared and their device ISA can
 * be diffed.
 ************************************************************************/
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <nccl.h>                  /* typedef ncclWindow_vidmem* ncclWindow_t */
#include <nccl_device_wrapper.h>   /* extern "C" thunk decls only             */

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

static uintptr_t host_expected(uintptr_t base, const TestCase& c) {
  int      i       = c.lsaRank + (c.peer - c.tm_rank) * c.tm_stride;
  uint32_t delta4G = (uint32_t)((int32_t)i * (int32_t)c.stride4G);
  uint32_t lo      = (uint32_t)(base & 0xFFFFFFFFu);
  uint32_t hi      = (uint32_t)(base >> 32) + delta4G;
  uintptr_t shift  = ((uintptr_t)hi << 32) | lo;
  return shift + c.offset;
}

/* The kernel under test. Identical control-flow to peer_inline_test.cpp
 * except for the function it calls: ncclGetPeerPointerTeam instead of
 * ncclGetPeerPointer. The thunk's body must come from librccl_device.bc
 * at link time. */
__global__ void k_peer_bitcode(char* base, const TestCase* cases, int N,
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
  out[t] = ncclGetPeerPointerTeam(&w, c.offset, tm, c.peer);
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

/* Exact copy of the table in peer_inline_test.cpp. Keep in sync. */
static const TestCase kCases[] = {
  {  0,  0,    0,        0,         1,  0,  1,  0 },
  {  0,  0,    1,       64,         2,  0,  1,  1 },
  {  0,  0,    1,       64,         2,  1,  1,  0 },
  {  0,  0,    2,      128,         4,  0,  1,  3 },
  {  0,  0,    1,        0,         8,  0,  2,  4 },
  {  5,  0,    1,        0,         1,  0,  1,  0 },
  {  0,  0,    1, 0xDEADBEEFull,    1,  0,  1,  0 },
  {  0,  0, 0x10,        0,         2,  0,  1,  1 },
  {  3,  0,    2,      512,         4,  1,  1,  2 },
  {  0,  0,    7,        0,         1,  0,  1,  0 },
  {  0,  0,    1, 0xCAFEBABEull,    2,  0,  1,  1 },
  {  2,  0,    1,        0,         3,  1,  1,  2 },
  {  0,  0,    4,    0x1000,        2,  1,  1,  0 },
  {  1,  0,    1,        8,         4,  2,  1,  3 },
  {  0,  0,   15, 0x40000000ull,    1,  0,  1,  0 },
  {  4,  0,    3,       16,         8,  3,  2,  6 },
};

int main() {
  int nDev = 0;
  HIP_CHECK(hipGetDeviceCount(&nDev));
  if (nDev <= 0) { std::fprintf(stderr, "No HIP devices.\n"); return 2; }
  std::printf("[peer-bitcode] devices=%d\n", nDev);

  constexpr int N = sizeof(kCases) / sizeof(kCases[0]);
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

    k_peer_bitcode<<<1, N>>>((char*)base, d_cases, N, d_out);
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

  if (const char* p = std::getenv("PEER_DUMP")) {
    FILE* f = std::fopen(p, "wb");
    if (f) {
      for (int i = 0; i < N; ++i) {
        uintptr_t v = host_expected(base, kCases[i]);
        std::fwrite(&v, sizeof v, 1, f);
      }
      std::fclose(f);
      std::printf("[peer-bitcode] expected bytes -> %s (%zu bytes)\n",
                  p, sizeof(uintptr_t) * N);
    }
  }
  return failures ? 1 : 0;
}
