/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Contract for ncclSymkWorldPointerGetter (symmetric/symk.cuh, hipified here to symmetric/symk.h):
// the ReloadPerCall=true instantiation (used by the TDM deep loop) must compute exactly the
// same world-peer pointers as the cached ReloadPerCall=false instantiation, for every peer and
// after advance(), since both derive the same pointer from the same window and offset.

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

struct GetterOut
{
  uintptr_t cached[kNPeers];
  uintptr_t reload[kNPeers];
  uintptr_t cachedAfterAdvance[kNPeers];
  uintptr_t reloadAfterAdvance[kNPeers];
};

__global__ void kernelGetterContract(ncclSymPtr<Pack> ptr, ncclDevComm comm, GetterOut* out)
{
  ncclSymkWorldPointerGetter<Pack, false> cachedGetter(ptr, comm);
  ncclSymkWorldPointerGetter<Pack, true> reloadGetter(ptr, comm);
  for (int p = 0; p < kNPeers; p++) {
    out->cached[p] = reinterpret_cast<uintptr_t>(cachedGetter(p));
    out->reload[p] = reinterpret_cast<uintptr_t>(reloadGetter(p));
  }
  cachedGetter.advance(7);
  reloadGetter.advance(7);
  for (int p = 0; p < kNPeers; p++) {
    out->cachedAfterAdvance[p] = reinterpret_cast<uintptr_t>(cachedGetter(p));
    out->reloadAfterAdvance[p] = reinterpret_cast<uintptr_t>(reloadGetter(p));
  }
}

class SymkWorldPointerGetterTest : public DeviceTestBase {};

// lsaRank == rank: lsaShift is zero, the common case for the single-LSA-team kernels
// this getter serves.
TEST_F(SymkWorldPointerGetterTest, ReloadMatchesCachedNoShift)
{
  DeviceBuffer<char> backing(1 << 20);
  ncclWindow_vidmem hostWin{};
  hostWin.lsaFlatBase = backing.ptr;
  hostWin.stride4G    = 0x1000;

  DeviceBuffer<ncclWindow_vidmem> win(1);
  win.copyFrom(&hostWin, 1);

  ncclDevComm comm{};
  comm.rank    = 2;
  comm.lsaRank = 2;

  ncclSymPtr<Pack> ptr(win.ptr, /*offset=*/256);

  DeviceBuffer<GetterOut> out(1);
  kernelGetterContract<<<1, 1>>>(ptr, comm, out.ptr);
  syncAndCheck();

  GetterOut got = out.download();
  for (int p = 0; p < kNPeers; p++) {
    EXPECT_EQ(got.cached[p], got.reload[p]) << "peer " << p;
    EXPECT_EQ(got.cachedAfterAdvance[p], got.reloadAfterAdvance[p]) << "peer " << p << " after advance";
  }
}

// lsaRank != rank: exercises the (worldPeer + lsaShift) term both instantiations share.
TEST_F(SymkWorldPointerGetterTest, ReloadMatchesCachedWithShift)
{
  DeviceBuffer<char> backing(1 << 20);
  ncclWindow_vidmem hostWin{};
  hostWin.lsaFlatBase = backing.ptr;
  hostWin.stride4G    = 0x2000;

  DeviceBuffer<ncclWindow_vidmem> win(1);
  win.copyFrom(&hostWin, 1);

  ncclDevComm comm{};
  comm.rank    = 5;
  comm.lsaRank = 1;

  ncclSymPtr<Pack> ptr(win.ptr, /*offset=*/4096);

  DeviceBuffer<GetterOut> out(1);
  kernelGetterContract<<<1, 1>>>(ptr, comm, out.ptr);
  syncAndCheck();

  GetterOut got = out.download();
  for (int p = 0; p < kNPeers; p++) {
    EXPECT_EQ(got.cached[p], got.reload[p]) << "peer " << p;
    EXPECT_EQ(got.cachedAfterAdvance[p], got.reloadAfterAdvance[p]) << "peer " << p << " after advance";
  }
}

} // namespace
} // namespace RcclUnitTesting
