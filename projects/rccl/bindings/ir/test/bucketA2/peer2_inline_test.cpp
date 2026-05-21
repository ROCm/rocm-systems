/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * peer2_inline_test.cpp
 *
 * 2-GPU smoke test for bucket A (ncclGetPeerPointer) using the INLINE
 * path: ncclGetPeerPointer is supplied by <nccl_device.h> as inline
 * device code.
 *
 * Flow
 * ----
 *   GPU 0  ->  uses ncclGetPeerPointer(peer=1) to obtain the address
 *              of rank 1's slot, writes MAGIC there, also echoes MAGIC
 *              into host-visible scratch so we can print "what GPU 0
 *              wrote".
 *   GPU 1  ->  uses ncclGetPeerPointer(peer=1) from its own perspective
 *              (lsaRank=1, tm.rank=1, peer=1 -> same slot) to obtain
 *              the same address, reads from it, echoes the value into
 *              host-visible scratch.
 *   Host   ->  prints both values, requires they match.
 *
 * The trick is the symmetric layout. ncclGetPeerPointer assumes rank N's
 * slot lives at  base + N * stride4G * 4GiB. To make that real on two
 * GPUs without RCCL's symmetric-heap infrastructure, we allocate one
 * hipMallocManaged region big enough to span two 4-GiB-spaced slots and
 * use that as a shared address space. Only the two touched cachelines
 * actually back to physical memory.
 ************************************************************************/
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <nccl.h>
#include <nccl_device.h>

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

/* GPU 0 writes MAGIC to peer-1's slot via ncclGetPeerPointer. */
__global__ void k_write(char* base, uint32_t stride4G,
                        uint64_t* wrote_out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ncclWindow_vidmem w{};
    w.winHost = nullptr;  w.lsaFlatBase = base;
    w.lsaRank = 0;        w.worldRank   = 0;
    w.stride4G = stride4G; w.mcOffset4K = 0;

    ncclTeam tm{ /*nRanks=*/2, /*rank=*/0, /*stride=*/1 };
    uint64_t* peer = (uint64_t*)ncclGetPeerPointer(&w, /*offset=*/0, tm,
                                                    /*peer=*/1);
    *peer = MAGIC;
    *wrote_out = MAGIC;
  }
}

/* GPU 1 reads from its own slot via ncclGetPeerPointer (peer=self). */
__global__ void k_read(char* base, uint32_t stride4G,
                       uint64_t* read_out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ncclWindow_vidmem w{};
    w.winHost = nullptr;  w.lsaFlatBase = base;
    w.lsaRank = 1;        w.worldRank   = 1;
    w.stride4G = stride4G; w.mcOffset4K = 0;

    ncclTeam tm{ /*nRanks=*/2, /*rank=*/1, /*stride=*/1 };
    uint64_t* mine = (uint64_t*)ncclGetPeerPointer(&w, /*offset=*/0, tm,
                                                    /*peer=*/1);
    *read_out = *mine;
  }
}

int main() {
  int nDev = 0;
  HIP_CHECK(hipGetDeviceCount(&nDev));
  if (nDev < 2) {
    std::fprintf(stderr,
        "[peer2-inline] need 2 visible GPUs (got %d). "
        "Try: HIP_VISIBLE_DEVICES=0,1\n", nDev);
    return 2;
  }
  std::printf("[peer2-inline] devices=%d\n", nDev);

  /* One managed allocation big enough for two 4-GiB-spaced slots. */
  char* raw = nullptr;
  HIP_CHECK(hipMallocManaged((void**)&raw, ALLOC_SIZE, hipMemAttachGlobal));
  uintptr_t aligned = ((uintptr_t)raw + SLOT_STRIDE - 1) & ~(SLOT_STRIDE - 1);
  char* base = (char*)aligned;
  std::printf("[peer2-inline] managed alloc=%p aligned base=%p "
              "(slot0=%p, slot1=%p)\n",
              raw, base, base, base + SLOT_STRIDE);

  /* Zero the two slot bytes so a stale value can't masquerade as success. */
  HIP_CHECK(hipMemset(base,                0, sizeof(uint64_t)));
  HIP_CHECK(hipMemset(base + SLOT_STRIDE,  0, sizeof(uint64_t)));

  uint64_t* d_wrote = nullptr;
  uint64_t* d_read  = nullptr;
  HIP_CHECK(hipMallocManaged((void**)&d_wrote, sizeof(uint64_t)));
  HIP_CHECK(hipMallocManaged((void**)&d_read,  sizeof(uint64_t)));
  *d_wrote = 0; *d_read = 0;

  /* GPU 0: write peer-1's slot. */
  HIP_CHECK(hipSetDevice(0));
  k_write<<<1,1>>>(base, /*stride4G=*/1, d_wrote);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  /* GPU 1: read its own slot. */
  HIP_CHECK(hipSetDevice(1));
  k_read<<<1,1>>>(base, /*stride4G=*/1, d_read);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::printf("GPU0 wrote: 0x%016lx\n", (unsigned long)*d_wrote);
  std::printf("GPU1 read:  0x%016lx\n", (unsigned long)*d_read);
  const bool ok = (*d_wrote == MAGIC) && (*d_read == MAGIC);
  std::printf("[peer2-inline] %s\n", ok ? "[OK]" : "[FAIL]");

  HIP_CHECK(hipFree(d_wrote));
  HIP_CHECK(hipFree(d_read));
  HIP_CHECK(hipFree(raw));
  return ok ? 0 : 1;
}
