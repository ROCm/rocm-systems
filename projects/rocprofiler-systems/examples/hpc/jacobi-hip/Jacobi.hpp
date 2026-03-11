/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#ifndef __JACOBI_HPP__
#define __JACOBI_HPP__

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "defines.hpp"
#include "hip/hip_runtime.h"

struct grid_t
{
    MPI_Comm comm;
    int      rank, size;

    int device_id;

    int Nrow, Ncol;
    int myrow, mycol;

    int Neighbor[NSIDES];
};

struct mesh_t
{
    int N;
    int Nx, Ny;

    int Nhalo;

    dfloat Lx, Ly;
    dfloat dx, dy;

    dfloat invNtotal;

    dfloat* x;
    dfloat* y;

    dfloat *sendBuffer, *recvBuffer;
    dfloat* d_haloBuffer;

    int sideLength[NSIDES], sideOffset[NSIDES];

    MPI_Request requests[2 * NSIDES];
    MPI_Status  status[2 * NSIDES];
};

class Jacobi_t
{
private:
    grid_t& grid;
    mesh_t& mesh;

    // host buffers
    dfloat *h_U, *h_RHS, *h_AU, *h_RES;

    // device buffers
    dfloat *d_U, *d_RHS, *d_AU, *d_RES;

    double timerStart, timerStop, elasped, avgTransferTime;
    double totalCommTime, totalLocalComputeTime;
    int    iterations;

    hipStream_t computeStream;
    hipStream_t dataStream;

    hipEvent_t JacobiLocalStart, JacobiLocalEnd;

    void ApplyTopology();

    void CreateMesh();

    void InitializeData();

    void PrintResults();

public:
    Jacobi_t(grid_t& grid_, mesh_t& mesh_);

    ~Jacobi_t();

    void Run();
};

dfloat
ForcingFunction(dfloat x, dfloat y);
dfloat
BoundaryFunction(dfloat x, dfloat y);

void
ParseCommandLineArguments(int argc, char** argv, MPI_Comm comm, grid_t& grid,
                          mesh_t& mesh);

void
HaloExchange(grid_t& grid, mesh_t& mesh, hipStream_t stream, dfloat* d_U);

dfloat
Norm(grid_t& grid, mesh_t& mesh, hipStream_t stream, dfloat* U);

void
LocalLaplacian(grid_t& grid, mesh_t& mesh, hipStream_t stream, dfloat* d_U, dfloat* d_AU);

void
HaloLaplacian(grid_t& grid, mesh_t& mesh, hipStream_t stream, dfloat* d_U, dfloat* d_AU);

void
JacobiIteration(grid_t& grid, mesh_t& mesh, hipStream_t stream, dfloat* d_RHS,
                dfloat* d_AU, dfloat* d_RES, dfloat* d_U);

#endif  // __JACOBI_HPP__
