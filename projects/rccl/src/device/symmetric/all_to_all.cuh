#include "sym_kernels.h"
#include "symmetric/kernel.h"
#include "symmetric/primitives.h"

template<typename T>
static __device__ void scatter(
    ncclSymkArgsHandler const& handler, int tn, int t, int nBlocks,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    ncclSymPtr<T> input, ncclSymPtr<T> output, size_t nElts
  ) {

  bool inPlace = (input == output);
  static_assert(sizeof(T) == 1);

  int const& rank   = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  // is waitNeeded is needed or should we always do bar.wait()
  if (waitNeeded) {
    bar.wait(ncclCoopCta(), NCCL_MEM_ORDER_RELAXED);
    waitNeeded = false;
  }

  // Stagger the destination order across ranks to spread store traffic.
  for (int i = 0; i < nRanks; i++) {
    int peer = (rank + i) % nRanks;
    if (peer == rank && inPlace) continue;

    ncclSymPtr<char> srcSlice = input  + (size_t)peer * nElts;
    ncclSymPtr<char> dstSlice = output;

    // peer-side destination = r's window at offset (rank*nElts + 0..nElts)
    char*       dst = dstSlice.lsaPtr(peer);
    char const* src = srcSlice.localPtr();

    for (int i = t; i < nElts; i += tn) {
      dst[i] = src[i];
    }
  }
}

__device__ __forceinline__ void ncclSymkRun_AlltoAll_ST(ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclLsaBarrierSession<ncclCoopCta> bar{
    ncclCoopCta(), handler.comm, ncclTeamTagLsa(), blockIdx.x
  };
  int const& rank   = handler.comm.rank;

  bar.arrive(ncclCoopCta(), NCCL_MEM_ORDER_RELAXED);

  bool waitNeeded = true;
  handler.forEachWork<char>(
    [&] __device__ (int block, int nBlocks, size_t nElts, size_t nAllElts,
      ncclSymPtr<char> input, ncclSymPtr<char> output) {
        int t = flattenIx(threadIdx.x%WARP_SIZE, WARP_SIZE,
                          block, nBlocks,
                          threadIdx.x/WARP_SIZE, blockDim.x/WARP_SIZE);
        int tn = nBlocks*blockDim.x;
        scatter(handler, tn, t, nBlocks, waitNeeded, bar, input, output + rank*nElts, nElts);
      }
    );
  bar.sync(ncclCoopCta(), NCCL_MEM_ORDER_RELEASE);
}
