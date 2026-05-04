/***************************************************************************
 Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#include "Jacobi.hpp"

// #define OPTIMIZED

#ifdef OPTIMIZED
#define block_size 1024
#else
#define block_size 128
#endif

__global__ void NormKernel1(const int N,
                            const dfloat dx, const dfloat dy,
                            const dfloat*__restrict__ U,
                                  dfloat*__restrict__ tmp) {
  __shared__ dfloat s_dot[block_size];

  const int t = threadIdx.x;
  const int i = blockIdx.x;

  int id = i * block_size + t;

  s_dot[t] = 0.0;
  for ( ; id < N ; id += gridDim.x * block_size ) {
    s_dot[t] += U[id] * U[id] * dx * dy;
  }
  __syncthreads();

  for (int k = block_size / 2; k > 0; k /= 2 ) {
    if ( t < k ) {
      s_dot[t] += s_dot[t + k];
    }
     __syncthreads();
  }

  if (t==0)
    tmp[i] = s_dot[0];
}

__global__ void NormKernel2(const int N,
                            const dfloat*__restrict__ tmp,
                                  dfloat*__restrict__ dprod) {

  __shared__ dfloat s_dot[block_size];

  const int t = threadIdx.x;
  const int i = blockIdx.x;
  int id = i * block_size + t;

  s_dot[t] = 0.0;
  for ( ; id < N ; id += block_size ) {
    s_dot[t] += tmp[id];
  }
  __syncthreads();

  for (int k = block_size / 2; k > 0; k /= 2 ) {
    if ( t < k ) {
      s_dot[t] += s_dot[t + k];
    }
     __syncthreads();
  }

  if (t==0)
    *dprod = s_dot[0];
}

dfloat Norm(grid_t& grid, mesh_t& mesh, hipStream_t stream, dfloat *U) {

  static dfloat* d_tmp = NULL;
  static dfloat* d_norm = NULL;
  static dfloat* h_norm = NULL;

  //first call
  if (h_norm==NULL) {
    //allocate temportary reduction space
    SafeHipCall(hipMalloc(&d_tmp, block_size*sizeof(dfloat)));
    SafeHipCall(hipMalloc(&d_norm, 1*sizeof(dfloat)));
    SafeHipCall(hipHostMalloc(&h_norm, 1*sizeof(dfloat)));
  }

  size_t grid_size = (mesh.N+block_size-1)/block_size;
  grid_size = (grid_size < block_size) ? grid_size : block_size;

  hipLaunchKernelGGL((NormKernel1),
                     dim3(grid_size),
                     dim3(block_size),
                     0, stream,
                     mesh.N, mesh.dx, mesh.dy, U, d_tmp);

  hipLaunchKernelGGL((NormKernel2),
                     dim3(1),
                     dim3(block_size),
                     0, stream,
                     grid_size, d_tmp, d_norm);

  SafeHipCall(hipMemcpy(h_norm, d_norm, 1*sizeof(dfloat), hipMemcpyDeviceToHost));

  dfloat norm;
  MPI_Allreduce(h_norm, &norm, 1, MPI_DFLOAT, MPI_SUM, grid.comm);

  return sqrt(norm)*mesh.invNtotal;
}
