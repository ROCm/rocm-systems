// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// This sample code solves the 2D Laplace equation using the Jacobi iterative
// method on a rectangle with periodic boundary conditions. 

// The domain is initialized with a smooth Dirac delta function at the center.

// The grid is partitioned among multiple MPI processes in the vertical direction.
// Each process exchanges halo regions with its neighboring processes after each
// iteration. Communication uses explicit GPU->Host->MPI->Host->GPU transfers
// (not GPU-aware MPI) for halo exchange.

// The code allows users to specify the number of iterations and the
// order of kernel launches via command-line arguments.


// Reference:  Finite Difference Methods for Ordinary and Partial Differential
// Equations by Randall J. LeVeque, Page 69-71


#include <algorithm>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

// Helper macro for HIP error checking
#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

void initialize(float *U, int N) {
#ifdef USE_MPI
  // Get the number of processes
  int num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int N_total = N * num_procs;
#endif // USE_MPI

  // The grid is partitioned among MPI processes in the vertical direction
  for (int i = 0; i < N; ++i) {
#ifdef USE_MPI
    // The halo regions are not initialized here
    // They are obtained from the neighboring rank
    for (int j = 1; j <= N; ++j) {
      int j_global = rank * N + j;
      float y = 2.0f * (float)j_global / (float)N_total - 1.0f;
#else
    for (int j = 0; j < N; ++j) {
      float y = 2.0f * (float)j / (float)N - 1.0f;
#endif // USE_MPI
      float x = 2.0f * (float)i / (float)N - 1.0f;
      U[i * N + j] = 1.0f / (1.0f + x * x + y * y); // Initial guess
    }
  }
}

__global__ void jacobi_iteration(float *__restrict__ U_new,
                                 const float *__restrict__ U_old, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = j + i * N; // Flattened index for 2D grid

  extern __shared__ float U[];
  int s_i = threadIdx.x + 1;
  int s_j = threadIdx.y + 1;

  int tile_height = blockDim.y + 2;
  int s_idx = s_j + s_i * tile_height;

  // Load data into shared memory with halo regions
  if (i < N && j < N) {
    U[s_idx] = U_old[idx];

    // Load halo regions
    if (threadIdx.x == 0) {
      U[s_j] = (i > 0) ? U_old[(i - 1) * N + j] : U_old[(N - 1) * N + j];
    }

    if (threadIdx.x == blockDim.x - 1) {
      U[s_j + (blockDim.x + 1) * tile_height] =
          (i < N - 1) ? U_old[(i + 1) * N + j] : U_old[j];
    }

    if (threadIdx.y == 0) {
      U[s_i * tile_height] =
          (j > 0) ? U_old[i * N + (j - 1)] : U_old[i * N + (N - 1)];
    }

    if (threadIdx.y == blockDim.y - 1) {
      U[s_i * tile_height + blockDim.y + 1] =
          (j < N - 1) ? U_old[i * N + (j + 1)] : U_old[i * N];
    }
  }

  __syncthreads();

  if (i < N && j < N) {
    int east = s_idx - tile_height;
    int west = s_idx + tile_height;
    int north = s_idx + 1;
    int south = s_idx - 1;
    U_new[idx] = 0.25f * (U[east] + U[west] + U[north] + U[south]);
  }
}

int main(int argc, char *argv[]) {
#ifdef USE_MPI
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get the number of processes
  int num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "Rank " << rank << " of " << num_procs << std::endl;
#else
  int num_procs = 1;
  int rank = 0;
  std::cout << "Running without MPI (single process)" << std::endl;
#endif // USE_MPI

  int iter = 1000;
  std::vector<int> launch_order;

#ifdef USE_MPI
  if (rank == 0) {
#endif // USE_MPI
    // Parse command-line arguments
    for (int i = 0; i < argc; i++) {
      if (std::string(argv[i]) == "--mpi-info") {
#ifdef USE_MPI
        std::cout << "MPI support: enabled" << std::endl;
        MPI_Finalize();
#else
        std::cout << "MPI support: disabled" << std::endl;
#endif // USE_MPI
        return 0;
      }
      if (std::string(argv[i]) == "-i" && i + 1 < argc) {
        iter = std::atoi(argv[i + 1]);
      } else if (std::string(argv[i]) == "-o" && i + 1 < argc) {
        int n_order = std::atoi(argv[i + 1]);
        if ((i + n_order + 1) >= argc) {
          std::cerr << "Insufficient arguments for -o option" << std::endl;
          return EXIT_FAILURE;
        } else {
          for (int j = 0; j < n_order; j++) {
            int kernel_id = std::atoi(argv[i + 2 + j]);
            if (kernel_id < 0 || kernel_id > 2) {
              std::cerr << "Invalid kernel ID: " << kernel_id << std::endl;
              return EXIT_FAILURE;
            }
            launch_order.push_back(kernel_id);
          }
          i += n_order;
        }
      }
    }

    if (launch_order.empty()) {
      launch_order = {0, 1, 2};
    }
#ifdef USE_MPI
  }

  // Broadcast number of iterations to all processes
  MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Broadcast launch_order size and data to all processes
  int order_size = launch_order.size();
  MPI_Bcast(&order_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0)
    launch_order.resize(order_size);

  MPI_Bcast(launch_order.data(), order_size, MPI_INT, 0, MPI_COMM_WORLD);
#endif // USE_MPI

  // Problem size
  const int n_small = 1024;
  const int n_mid = 2048;
  const int n_large = 4096;

  // Define block and grid sizes
  dim3 blockSize_small(16, 16);
  dim3 gridSize_small((n_small + blockSize_small.x - 1) / blockSize_small.x,
                      (n_small + blockSize_small.y - 1) / blockSize_small.y);
  dim3 blockSize_mid(24, 24);
  dim3 gridSize_mid((n_mid + blockSize_mid.x - 1) / blockSize_mid.x,
                    (n_mid + blockSize_mid.y - 1) / blockSize_mid.y);
  dim3 blockSize_large(32, 32);
  dim3 gridSize_large((n_large + blockSize_large.x - 1) / blockSize_large.x,
                      (n_large + blockSize_large.y - 1) / blockSize_large.y);

  const int nx_small = n_small;
  const int nx_mid = n_mid;
  const int nx_large = n_large;

#ifdef USE_MPI
  // Set device based on rank
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  HIP_CHECK(hipSetDevice(rank % num_devices));

  const int ny_small = n_small + 2;
  const int ny_mid = n_mid + 2;
  const int ny_large = n_large + 2;
#else
  const int ny_small = n_small;
  const int ny_mid = n_mid;
  const int ny_large = n_large;
#endif // USE_MPI

  // Host memory pointers
  float *h_U_small = new float[nx_small * ny_small];
  float *h_U_mid = new float[nx_mid * ny_mid];
  float *h_U_large = new float[nx_large * ny_large];

  // Initialize host arrays
  initialize(h_U_small, n_small);
  initialize(h_U_mid, n_mid);
  initialize(h_U_large, n_large);

  // Device memory pointers
  float *d_U_small_new, *d_U_small_old;
  float *d_U_mid_new, *d_U_mid_old;
  float *d_U_large_new, *d_U_large_old;

  // Allocate device memory
  HIP_CHECK(hipMalloc(&d_U_small_new, nx_small * ny_small * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_U_small_old, nx_small * ny_small * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_U_mid_new, nx_mid * ny_mid * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_U_mid_old, nx_mid * ny_mid * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_U_large_new, nx_large * ny_large * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_U_large_old, nx_large * ny_large * sizeof(float)));

  // Copy host arrays to device
  HIP_CHECK(hipMemcpy(d_U_small_old, h_U_small,
                      nx_small * ny_small * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_U_mid_old, h_U_mid, nx_mid * ny_mid * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_U_large_old, h_U_large,
                      nx_large * ny_large * sizeof(float),
                      hipMemcpyHostToDevice));

  std::vector<dim3> grid_sizes = {gridSize_small, gridSize_mid, gridSize_large};
  std::vector<dim3> block_sizes = {blockSize_small, blockSize_mid,
                                   blockSize_large};
  std::vector<float *> d_U_news = {d_U_small_new, d_U_mid_new, d_U_large_new};
  std::vector<float *> d_U_olds = {d_U_small_old, d_U_mid_old, d_U_large_old};
  std::vector<int> n_sizes = {n_small, n_mid, n_large};

#ifdef USE_MPI
  int left = rank == 0 ? num_procs - 1 : rank - 1;
  int right = (rank + 1) % num_procs;

  // Allocate host buffers for halo exchange (GPU -> Host -> MPI -> Host -> GPU)
  // Each problem size needs 4 buffers: send_left, send_right, recv_left, recv_right
  std::vector<float *> h_send_left(n_sizes.size());
  std::vector<float *> h_send_right(n_sizes.size());
  std::vector<float *> h_recv_left(n_sizes.size());
  std::vector<float *> h_recv_right(n_sizes.size());

  for (size_t i = 0; i < n_sizes.size(); ++i) {
    h_send_left[i] = new float[n_sizes[i]];
    h_send_right[i] = new float[n_sizes[i]];
    h_recv_left[i] = new float[n_sizes[i]];
    h_recv_right[i] = new float[n_sizes[i]];
  }

  int num_requests = 4 * n_sizes.size();
  std::vector<MPI_Request> requests(num_requests);

  // Initial halo exchange: GPU -> Host -> MPI -> Host -> GPU
  for (size_t i = 0; i < n_sizes.size(); ++i) {
    // Copy halo data from GPU to host send buffers
    HIP_CHECK(hipMemcpy(h_send_left[i], &d_U_olds[i][n_sizes[i]],
                        n_sizes[i] * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_send_right[i], &d_U_olds[i][n_sizes[i] * (n_sizes[i] + 1)],
                        n_sizes[i] * sizeof(float), hipMemcpyDeviceToHost));
  }

  for (size_t i = 0; i < n_sizes.size(); ++i) {
    // Send data to left node, receive from right node
    MPI_Isend(h_send_left[i], n_sizes[i], MPI_FLOAT, left, 2 * i,
              MPI_COMM_WORLD, &requests[4 * i]);
    MPI_Irecv(h_recv_right[i], n_sizes[i], MPI_FLOAT, right, 2 * i,
              MPI_COMM_WORLD, &requests[4 * i + 1]);

    // Send data to right node, receive from left node
    MPI_Isend(h_send_right[i], n_sizes[i], MPI_FLOAT, right, 2 * i + 1,
              MPI_COMM_WORLD, &requests[4 * i + 2]);
    MPI_Irecv(h_recv_left[i], n_sizes[i], MPI_FLOAT, left, 2 * i + 1,
              MPI_COMM_WORLD, &requests[4 * i + 3]);
  }

  MPI_Waitall(num_requests, requests.data(), MPI_STATUSES_IGNORE);

  // Copy received halo data from host to GPU
  for (size_t i = 0; i < n_sizes.size(); ++i) {
    HIP_CHECK(hipMemcpy(&d_U_olds[i][n_sizes[i] * n_sizes[i]], h_recv_right[i],
                        n_sizes[i] * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(&d_U_olds[i][0], h_recv_left[i],
                        n_sizes[i] * sizeof(float), hipMemcpyHostToDevice));
  }
#endif // USE_MPI

  // Perform Jacobi iterations
  for (int it = 0; it < iter; ++it) {
    int i = launch_order[it % launch_order.size()];
    jacobi_iteration<<<grid_sizes[i], block_sizes[i],
                       (block_sizes[i].x + 2) * (block_sizes[i].y + 2) *
                           sizeof(float)>>>(d_U_news[i], d_U_olds[i],
                                            n_sizes[i]);

    HIP_CHECK(hipDeviceSynchronize());
    // Swap pointers
    std::swap(d_U_news[i], d_U_olds[i]);

#ifdef USE_MPI
    std::vector<MPI_Request> requests(4);

    // Copy halo data from GPU to host send buffers
    HIP_CHECK(hipMemcpy(h_send_left[i], &d_U_olds[i][n_sizes[i]],
                        n_sizes[i] * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_send_right[i], &d_U_olds[i][n_sizes[i] * (n_sizes[i] + 1)],
                        n_sizes[i] * sizeof(float), hipMemcpyDeviceToHost));

    // Send data to left node, receive from right node
    MPI_Isend(h_send_left[i], n_sizes[i], MPI_FLOAT, left, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(h_recv_right[i], n_sizes[i], MPI_FLOAT, right, 0,
              MPI_COMM_WORLD, &requests[1]);

    // Send data to right node, receive from left node
    MPI_Isend(h_send_right[i], n_sizes[i], MPI_FLOAT, right, 1,
              MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(h_recv_left[i], n_sizes[i], MPI_FLOAT, left, 1,
              MPI_COMM_WORLD, &requests[3]);

    MPI_Waitall(4, requests.data(), MPI_STATUSES_IGNORE);

    // Copy received halo data from host to GPU
    HIP_CHECK(hipMemcpy(&d_U_olds[i][n_sizes[i] * n_sizes[i]], h_recv_right[i],
                        n_sizes[i] * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(&d_U_olds[i][0], h_recv_left[i],
                        n_sizes[i] * sizeof(float), hipMemcpyHostToDevice));
#endif // USE_MPI
  }

  // Pedagogical
  // Copy device data to host
  HIP_CHECK(hipMemcpy(h_U_small, d_U_small_old,
                      nx_small * ny_small * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_U_mid, d_U_mid_old, nx_mid * ny_mid * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_U_large, d_U_large_old,
                      nx_large * ny_large * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Free device memory
  HIP_CHECK(hipFree(d_U_small_new));
  HIP_CHECK(hipFree(d_U_small_old));
  HIP_CHECK(hipFree(d_U_mid_new));
  HIP_CHECK(hipFree(d_U_mid_old));
  HIP_CHECK(hipFree(d_U_large_new));
  HIP_CHECK(hipFree(d_U_large_old));

  delete[] h_U_small;
  delete[] h_U_mid;
  delete[] h_U_large;

#ifdef USE_MPI
  // Free host buffers for halo exchange
  for (size_t i = 0; i < n_sizes.size(); ++i) {
    delete[] h_send_left[i];
    delete[] h_send_right[i];
    delete[] h_recv_left[i];
    delete[] h_recv_right[i];
  }

  // Finalize MPI
  MPI_Finalize();
#endif // USE_MPI

  return 0;
}