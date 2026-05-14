/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
***************************************************************************/

#include "Jacobi.hpp"

/**
 * @file JacobiMain.cpp
 * @brief This contains the application entry point
 */

/**
 * @brief The application entry point
 *
 * @param[in] argc  The number of command-line arguments
 * @param[in] argv  The list of command-line arguments
 */
int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  grid_t grid;
  mesh_t mesh;

  // Extract topology and domain dimensions from the command-line arguments
  ParseCommandLineArguments(argc, argv,
                            comm,
                            grid,
                            mesh);

  Jacobi_t Jacobi(grid, mesh);

  Jacobi.Run();

  // Finalize the MPI process
  MPI_Finalize();
  return STATUS_OK;
}
