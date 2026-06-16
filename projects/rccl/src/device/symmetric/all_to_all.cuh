// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "sym_kernels.h"
#include "symmetric/kernel.h"
#include "symmetric/primitives.h"

template<int BytePerPack, int UnrollPacks, int UnrollPeers>
static __device__ void scatterDeep(
    ncclSymkArgsHandler const& handler, int tn, int t,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    ncclSymPtr<char> input, ncclSymPtr<char> output, bool inPlace,
    int nIters, size_t nElts
  ) {
  using Pack = BytePack<BytePerPack>;
  int wn = tn/WARP_SIZE;
  int w = t/WARP_SIZE;
  int lane = t%WARP_SIZE;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  Pack* inpPacks = (Pack*)input.localPtr() + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  //Pack tmp[nRanks][UnrollPacks];

  nIters -= w;
  //if (0 < nIters) {
  //  #pragma unroll
  //  for (int r=0; r < nRanks; r++) {
  //    #pragma unroll
  //    for (int u=0; u < UnrollPacks; u++) {
  //      tmp[r][u] = inpPacks[r*nElts/BytePerPack + u*WARP_SIZE];
  //    }
  //  }
  //}

  if (waitNeeded) bar.wait(ncclCoopCta(), NCCL_MEM_ORDER_RELAXED);

  if (0 < nIters) {
    while (true) {
      int dr = inPlace ? 1 : 0;
      int r = rank + dr;
      if (r == nRanks) r = 0;
      #pragma unroll 2
      for (int partial=0; partial <= 1; partial++) {
        #pragma unroll 1
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && dr == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              const size_t rank_offset = r*nElts/BytePerPack;
              outPacks.lsaPtr(r)[u*WARP_SIZE] = inpPacks[rank_offset + u*WARP_SIZE];
              //outPacks.lsaPtr(r)[u*WARP_SIZE] = tmp[r][u];
            }
            if (++r == nRanks) r = 0;
          }
        }
      }
      inpPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      outPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

      //Load data for next iteration.
      //#pragma unroll
      //for (int u=0; u < UnrollPacks; u++) {
      //  tmp[r][u] = inpPacks[r*nElts/BytePerPack + u*WARP_SIZE];
      //}
    }
  }
}

template<int UnrollPeers, typename T>
static __device__ void scatterEnds(
    ncclSymkArgsHandler const& handler, int tn, int t,
    ncclSymPtr<T> input, ncclSymPtr<T> output, bool inPlace, size_t nElts, uint32_t nPreElts, size_t nSufElts
  ) {
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;
  BytePack<sizeof(T)>* inpPacks = (BytePack<sizeof(T)>*)input.localPtr();
  ncclSymPtr<BytePack<sizeof(T)>> outPacks = (ncclSymPtr<BytePack<sizeof(T)>>)output;
  #pragma unroll 1
  for (size_t i = t; i < nPreElts+nSufElts; i += tn) {
    size_t elt = i < nPreElts ? i : nElts-nPreElts-nSufElts+i;
    //BytePack<sizeof(T)> tmp = inpPacks[elt];
    BytePack<sizeof(T)> tmp = inpPacks[rank*nElts/sizeof(T) + elt];
    int dr = inPlace ? 1 : 0;
    int r = rank + dr;
    if (r == nRanks) r = 0;
    #pragma unroll 1
    for (; dr + UnrollPeers <= nRanks; dr += UnrollPeers) {
      #pragma unroll UnrollPeers
      for (int u=0; u < UnrollPeers; u++) {
        //outPacks.lsaPtr(r)[elt] = tmp;
        const size_t rank_offset = r*nElts/sizeof(T);
        outPacks.lsaPtr(r)[elt] = inpPacks[rank_offset + elt];
        if (++r == nRanks) r = 0;
      }
    }
    #pragma unroll UnrollPeers
    for (int u=0; u < UnrollPeers; u++) {
      if (dr+u == nRanks) break;
      //outPacks.lsaPtr(r)[elt] = tmp;
      const size_t rank_offset = r*nElts/sizeof(T);
      outPacks.lsaPtr(r)[elt] = inpPacks[rank_offset + elt];
      if (++r == nRanks) r = 0;
    }
  }
}

template<typename T>
static __device__ void scatter(
    ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {
  bool inPlace = (input == output);
  size_t nBytes = nElts*sizeof(T);
  uint32_t nBlocks_rcp32 = nccl::utility::idivRcp32_upto64(nBlocks);

  uint32_t nPreBytes = (16 - input.offset)%16;
  nPreBytes = min((size_t)nPreBytes, nBytes);
  uintptr_t cursor = nPreBytes;

  constexpr int MinWarpPerBlock = 4;

  if ((input.offset - output.offset)%16 == 0) {
    constexpr int BytePerPack = 16, UnrollPacks = 4, UnrollPeers = 2;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      scatterDeep<BytePerPack, UnrollPacks, UnrollPeers>(
        handler, tn, t, waitNeeded, bar,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        inPlace, chunks*MinWarpPerBlock,
        nElts
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (sizeof(T) == 4 || (sizeof(T) < 4 && (input.offset - output.offset)%4 == 0)) {
    constexpr int BytePerPack = 4, UnrollPacks = 4, UnrollPeers = 4;
    constexpr int BytePerChunk = MinWarpPerBlock*UnrollPacks*WARP_SIZE*BytePerPack;
    uint32_t chunks = (nBytes-cursor)/BytePerChunk;
    chunks -= imodFast32(chunks, nBlocks, nBlocks_rcp32);
    if (chunks != 0) {
      uintptr_t cursorAfter = cursor + uintptr_t(chunks)*BytePerChunk;
      scatterDeep<(sizeof(T) <= BytePerPack ? BytePerPack : 0), UnrollPacks, UnrollPeers>(
        handler, tn, t, waitNeeded, bar,
        (ncclSymPtr<char>)input + cursor,
        (ncclSymPtr<char>)output + cursor,
        inPlace, chunks*MinWarpPerBlock,
        nElts
      );
      cursor = cursorAfter;
      waitNeeded = false;
    }
  }

  if (waitNeeded)
    bar.wait(ncclCoopCta(), NCCL_MEM_ORDER_RELAXED);

  constexpr int UnrollPeers = 8;
  size_t nSufElts = (nBytes-cursor)/sizeof(T);
  scatterEnds<UnrollPeers>(handler, tn, t, input, output, inPlace, nElts, nPreBytes/sizeof(T), nSufElts);
}

__device__ __forceinline__ void ncclSymkRun_AlltoAll_ST(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x
  };
  int const& rank = handler.comm.rank;

  bar.arrive(ncclCoopCta(), NCCL_MEM_ORDER_RELAXED);

  bool waitNeeded = true;
  handler.forEachWork<char>(
      [&]__device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                    ncclSymPtr<char> input, ncclSymPtr<char> output) {
        // Threads numbered over rank.
        int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                           block, nBlocks,
                           threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int tn = nBlocks*blockDim.x;

        scatter(handler, tn, t, nBlocks, waitNeeded, bar, input, output + rank*nAllElts, nElts);

        waitNeeded = false;
      }
    );

  bar.sync(ncclCoopCta(), NCCL_MEM_ORDER_RELEASE);
}