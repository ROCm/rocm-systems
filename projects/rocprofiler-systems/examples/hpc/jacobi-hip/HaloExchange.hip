/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#include "Jacobi.hpp"
#include "markers.h"

void HaloExchange(grid_t& grid, mesh_t& mesh,
                  hipStream_t stream, dfloat* d_U) {

  //copy each side to the haloBuffer
  BEGIN_RANGE("Halo D2H", "Halo Exchange");
  if (grid.Neighbor[SIDE_DOWN]>-1)
    SafeHipCall(hipMemcpy2DAsync(mesh.sendBuffer + mesh.sideOffset[SIDE_DOWN],
                                 mesh.Nx*sizeof(dfloat),
                                 d_U,
                                 mesh.Nx*sizeof(dfloat),
                                 mesh.Nx*sizeof(dfloat), 1,
                                 hipMemcpyDeviceToHost, stream));

  if (grid.Neighbor[SIDE_UP]>-1)
    SafeHipCall(hipMemcpy2DAsync(mesh.sendBuffer + mesh.sideOffset[SIDE_UP],
                                 mesh.Nx*sizeof(dfloat),
                                 d_U+(mesh.Ny-1)*mesh.Nx,
                                 mesh.Nx*sizeof(dfloat),
                                 mesh.Nx*sizeof(dfloat) , 1,
                                 hipMemcpyDeviceToHost, stream));

  if (grid.Neighbor[SIDE_LEFT]>-1)
    SafeHipCall(hipMemcpy2DAsync(mesh.sendBuffer + mesh.sideOffset[SIDE_LEFT],
                                 1*sizeof(dfloat),
                                 d_U,
                                 mesh.Nx*sizeof(dfloat),
                                 1*sizeof(dfloat), mesh.Ny,
                                 hipMemcpyDeviceToHost, stream));

  if (grid.Neighbor[SIDE_RIGHT]>-1)
    SafeHipCall(hipMemcpy2DAsync(mesh.sendBuffer + mesh.sideOffset[SIDE_RIGHT],
                                 1*sizeof(dfloat),
                                 d_U+mesh.Nx-1,
                                 mesh.Nx*sizeof(dfloat),
                                 1*sizeof(dfloat), mesh.Ny,
                                 hipMemcpyDeviceToHost, stream));

  //wait for the data to arrive on host
  hipStreamSynchronize(stream);
  END_RANGE();

  //post recvs & sends
  BEGIN_RANGE("MPI Exchange", "Halo Exchange");
  for (int s=0;s<NSIDES;s++) {
    if (grid.Neighbor[s]>-1) {
      MPI_Irecv(mesh.recvBuffer + mesh.sideOffset[s], mesh.sideLength[s], MPI_DFLOAT,
                grid.Neighbor[s], 0, grid.comm, mesh.requests+2*s);
      MPI_Isend(mesh.sendBuffer + mesh.sideOffset[s], mesh.sideLength[s], MPI_DFLOAT,
                grid.Neighbor[s], 0, grid.comm, mesh.requests+2*s+1);
    }
  }

  // Wait for all sent messages to have left and received messages to have arrived
  MPI_Waitall(2*NSIDES, mesh.requests, mesh.status);
  END_RANGE();

  //copy recvbuffer to haloBuffer on device
  BEGIN_RANGE("Halo H2D", "Halo Exchange");
  for (int s=0;s<NSIDES;s++) {
    if (grid.Neighbor[s]>-1) {
      SafeHipCall(hipMemcpyAsync(mesh.d_haloBuffer + mesh.sideOffset[s],
                                 mesh.recvBuffer + mesh.sideOffset[s],
                                 mesh.sideLength[s]*sizeof(dfloat),
                                 hipMemcpyHostToDevice, stream));
    }
  }

  //wait for the data to arrive on device
  hipStreamSynchronize(stream);
  END_RANGE();
}
