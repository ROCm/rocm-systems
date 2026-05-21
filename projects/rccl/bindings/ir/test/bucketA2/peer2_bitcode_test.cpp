/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * peer2_bitcode_test.cpp
 *
 * Sister of peer2_inline_test.cpp using the BITCODE path. Same 2-GPU
 * write/read flow, but every call to the peer-pointer API routes
 * through ncclGetPeerPointerTeam — declared `extern "C" __device__` in
 * <nccl_device_wrapper.h>, defined in librccl_device.bc, supplied at
 * link time via -Xoffload-linker.
 ************************************************************************/
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <nccl.h>
#include <nccl_device_wrapper.h>

static constexpr uint64_t MAGIC       = 0xDEADBEEFCAFEBABEull;
static constexpr uint64_t SLOT_STRIDE = (uint64_t)1 << 32;          /* 4 GiB */
static constexpr size_t   ALLOC_SIZE  = 2 * SLOT_STRIDE + 4096;     /* 8 GiB + pad */

#define HIP_CHECK(stmt) do {                                          \
    hipError_t _e = (stmt);                                           \
    if (_e != hipSuccess) {                                           \
      std::fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n",        \
                   (int)_e, hipGetErrorName(_e),                      \
                   __FILE__, __LINE__, hipGetErrorString(_e));        \
      std::exit(2);                                                   \
    }                                                                 \
  } while (0)

__global__ void k_write(char* base, uint32_t stride4G,
                        uint64_t* wrote_out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ncclWindow_vidmem w{};
    w.winHost = nullptr;  w.lsaFlatBase = base;
    w.lsaRank = 0;        w.worldRank   = 0;
    w.stride4G = stride4G; w.mcOffset4K = 0;

    ncclTeam tm{ 2, 0, 1 };
    uint64_t* peer = (uint64_t*)ncclGetPeerPointerTeam(&w, 0, tm, 1);
    *peer = MAGIC;
    *wrote_out = MAGIC;
  }
}

__global__ void k_read(char* base, uint32_t stride4G,
                       uint64_t* read_out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ncclWindow_vidmem w{};
    w.winHost = nullptr;  w.lsaFlatBase = base;
    w.lsaRank = 1;        w.worldRank   = 1;
    w.stride4G = stride4G; w.mcOffset4K = 0;

    ncclTeam tm{ 2, 1, 1 };
    uint64_t* mine = (uint64_t*)ncclGetPeerPointerTeam(&w, 0, tm, 1);
    *read_out = *mine;
  }
}

int main() {
  int nDev = 0;
  HIP_CHECK(hipGetDeviceCount(&nDev));
  if (nDev < 2) {
    std::fprintf(stderr,
        "[peer2-bitcode] need 2 visible GPUs (got %d). "
        "Try: HIP_VISIBLE_DEVICES=0,1\n", nDev);
    return 2;
  }
  std::printf("[peer2-bitcode] devices=%d\n", nDev);

  char* raw = nullptr;
  HIP_CHECK(hipMallocManaged((void**)&raw, ALLOC_SIZE, hipMemAttachGlobal));
  uintptr_t aligned = ((uintptr_t)raw + SLOT_STRIDE - 1) & ~(SLOT_STRIDE - 1);
  char* base = (char*)aligned;
  std::printf("[peer2-bitcode] managed alloc=%p aligned base=%p "
              "(slot0=%p, slot1=%p)\n",
              raw, base, base, base + SLOT_STRIDE);

  HIP_CHECK(hipMemset(base,                0, sizeof(uint64_t)));
  HIP_CHECK(hipMemset(base + SLOT_STRIDE,  0, sizeof(uint64_t)));

  uint64_t* d_wrote = nullptr;
  uint64_t* d_read  = nullptr;
  HIP_CHECK(hipMallocManaged((void**)&d_wrote, sizeof(uint64_t)));
  HIP_CHECK(hipMallocManaged((void**)&d_read,  sizeof(uint64_t)));
  *d_wrote = 0; *d_read = 0;

  HIP_CHECK(hipSetDevice(0));
  k_write<<<1,1>>>(base, 1, d_wrote);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipSetDevice(1));
  k_read<<<1,1>>>(base, 1, d_read);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::printf("GPU0 wrote: 0x%016lx\n", (unsigned long)*d_wrote);
  std::printf("GPU1 read:  0x%016lx\n", (unsigned long)*d_read);
  const bool ok = (*d_wrote == MAGIC) && (*d_read == MAGIC);
  std::printf("[peer2-bitcode] %s\n", ok ? "[OK]" : "[FAIL]");

  HIP_CHECK(hipFree(d_wrote));
  HIP_CHECK(hipFree(d_read));
  HIP_CHECK(hipFree(raw));
  return ok ? 0 : 1;
}
