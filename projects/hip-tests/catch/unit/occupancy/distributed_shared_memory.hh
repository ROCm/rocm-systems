#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
namespace cg = cooperative_groups;

static constexpr int ThreadsPerBlock = 256;
static constexpr int ElementsPerThread = 99;

__global__ void distributed_shmem_kernel(int* output) {
  extern __shared__ int smem[];
  int pos = threadIdx.x * ElementsPerThread;
  cg::cluster_group cluster = cg::this_cluster();
  int sum = 0;

  for (int i = 0; i < ElementsPerThread; i++) smem[pos + i] = blockIdx.x;

  cluster.sync();

  for (int rank = 0; rank < cluster.dim_blocks().x; rank++) {
    int* ptr = cluster.map_shared_rank(smem, rank);
    for (int i = 0; i < ThreadsPerBlock * ElementsPerThread; i++) {
      sum += ptr[i];
    }
  }

  output[pos] = sum;

  // make sure block stays alive long enough
  cluster.sync();
}
