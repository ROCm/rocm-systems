/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#include "Jacobi.hpp"

void Jacobi_t::Run() {

  OnePrintf((!grid.rank), "Starting Jacobi run.\n");

  iterations = 0;

  //compute initial residual (assumes zero initial guess)
  dfloat residual = Norm(grid, mesh, computeStream, d_RES);

  OnePrintf((!grid.rank),
      "Iteration:   %d - Residual: %.6f\n", iterations, residual);

  hipDeviceSynchronize();
  MPI_Barrier(grid.comm);
  timerStart = MPI_Wtime();

  totalCommTime = 0.0;
  totalLocalComputeTime = 0.0;

  while ((iterations < JACOBI_MAX_LOOPS) && (residual > JACOBI_TOLERANCE)) {

    //record when local compute starts on compute stream
    hipEventRecord(JacobiLocalStart, computeStream);
    double commStart = MPI_Wtime();

    //queue local part of Laplacian to compute stream
    LocalLaplacian(grid, mesh, computeStream, d_U, d_AU);

    //record when local compute ends on compute stream
    hipEventRecord(JacobiLocalEnd, computeStream);

    //Extract data off GPU exchange Halo with MPI
    HaloExchange(grid, mesh, dataStream, d_U);

    //comms are complete at this point, record the elapsed time
    double commElapsed = MPI_Wtime() - commStart;

    //use halo data to complete Laplacian computation
    HaloLaplacian(grid, mesh, computeStream, d_U, d_AU);

    //Jacobi iterative method
    // U = U + D^{-1}*(RHS - AU)
    JacobiIteration(grid, mesh, computeStream, d_RHS, d_AU, d_RES, d_U);

    //residual = ||U||
    residual = Norm(grid, mesh, computeStream, d_RES);

    //finish everything on device
    hipDeviceSynchronize();

    //query the completed events to find the time the local laplacian took
    float localLaplacianTime = 0.0;
    hipEventElapsedTime(&localLaplacianTime,
                        JacobiLocalStart, JacobiLocalEnd);

    //keep running totals
    totalCommTime += commElapsed;
    totalLocalComputeTime += localLaplacianTime/1000.0;

    ++iterations;
    OnePrintf((!grid.rank) && ((iterations) % 100 == 0),
      "Iteration: %d - Residual: %.6f\n", iterations, residual);
  }

  MPI_Barrier(grid.comm);
  timerStop = MPI_Wtime();
  elasped = timerStop - timerStart;

  OnePrintf((!grid.rank), "Stopped after %d iterations with residue %.6f\n", iterations, residual);

  PrintResults();
}
