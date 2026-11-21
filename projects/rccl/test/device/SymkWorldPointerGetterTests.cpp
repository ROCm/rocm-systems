/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Contract for ncclSymkWorldPointerGetter (symmetric/symk.cuh, hipified here to symmetric/symk.h):
// a world peer's pointer is the window's LSA pointer for LSA rank window.lsaRank + (peer - rank),
// lsaFlatBase + lsaPeer * stride4G (in 4 GiB units) + offset, for every peer and after advance(),
// whether the window fields are read with vector loads or, for StoreTarget, scalar loads.

#include "DeviceTestBase.hpp"

// Gives this TU its own ncclShmem definition, as each shard of librccl's device-linker build has.
#ifndef RCCL_DEVICE_LINKER
#define RCCL_DEVICE_LINKER
#endif
#include "symmetric/symk.h"

namespace RcclUnitTesting
{
namespace
{

using Pack = BytePack<16>;
constexpr int kNPeers = 4;
constexpr int kAdvance = 7;

struct GetterOut
{
  uintptr_t peer[kNPeers];
  uintptr_t peerAfterAdvance[kNPeers];
};

template <bool StoreTarget>
__global__ void kernelGetterContract(ncclSymPtr<Pack> ptr, ncclDevComm comm, GetterOut* out)
{
  ncclSymkWorldPointerGetter<Pack, StoreTarget> getter(ptr, comm);
  for (int p = 0; p < kNPeers; p++) {
    out->peer[p] = reinterpret_cast<uintptr_t>(getter(p));
  }
  getter.advance(kAdvance);
  for (int p = 0; p < kNPeers; p++) {
    out->peerAfterAdvance[p] = reinterpret_cast<uintptr_t>(getter(p));
  }
}

// add4G() adds lsaPeer * stride4G to the high 32 bits of lsaFlatBase.
uintptr_t ExpectedPeer(void* lsaFlatBase, uint32_t stride4G, int lsaPeer, size_t offset)
{
  uint64_t high = static_cast<uint32_t>(lsaPeer) * stride4G;
  return reinterpret_cast<uintptr_t>(lsaFlatBase) + (high << 32) + offset;
}

class SymkWorldPointerGetterTest : public DeviceTestBase
{
protected:
  void CheckGetter(uint32_t stride4G, int rank, int lsaRank, size_t offset)
  {
    DeviceBuffer<char> backing(1 << 20);
    ncclWindow_vidmem hostWin{};
    hostWin.lsaFlatBase = backing.ptr;
    hostWin.stride4G    = stride4G;
    hostWin.lsaRank     = lsaRank;
    hostWin.worldRank   = rank;

    DeviceBuffer<ncclWindow_vidmem> win(1);
    win.copyFrom(&hostWin, 1);

    ncclDevComm comm{};
    comm.rank    = rank;
    comm.lsaRank = lsaRank;

    DeviceBuffer<GetterOut> vectorOut(1);
    DeviceBuffer<GetterOut> scalarOut(1);
    kernelGetterContract<false><<<1, 1>>>(ncclSymPtr<Pack>(win.ptr, offset), comm, vectorOut.ptr);
    kernelGetterContract<true><<<1, 1>>>(ncclSymPtr<Pack>(win.ptr, offset), comm, scalarOut.ptr);
    syncAndCheck();

    GetterOut got[2] = {vectorOut.download(), scalarOut.download()};
    for (int storeTarget = 0; storeTarget < 2; storeTarget++) {
      for (int p = 0; p < kNPeers; p++) {
        int lsaPeer = lsaRank + (p - rank);
        EXPECT_EQ(got[storeTarget].peer[p], ExpectedPeer(backing.ptr, stride4G, lsaPeer, offset))
          << "StoreTarget " << storeTarget << " peer " << p;
        EXPECT_EQ(got[storeTarget].peerAfterAdvance[p],
                  ExpectedPeer(backing.ptr, stride4G, lsaPeer, offset + kAdvance * sizeof(Pack)))
          << "StoreTarget " << storeTarget << " peer " << p << " after advance";
      }
    }
  }
};

// lsaRank == rank: world and LSA ranks coincide, the common case for the single-LSA-team kernels
// this getter serves.
TEST_F(SymkWorldPointerGetterTest, WorldPeerIsLsaPeerNoShift)
{
  CheckGetter(/*stride4G=*/0x1000, /*rank=*/2, /*lsaRank=*/2, /*offset=*/256);
}

// lsaRank != rank: exercises the (peer - rank) shift onto the window's lsaRank.
TEST_F(SymkWorldPointerGetterTest, WorldPeerIsLsaPeerWithShift)
{
  CheckGetter(/*stride4G=*/0x2000, /*rank=*/5, /*lsaRank=*/1, /*offset=*/4096);
}

} // namespace
} // namespace RcclUnitTesting
