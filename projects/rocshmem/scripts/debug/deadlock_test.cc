/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

/*
 * deadlock_test — rocSHMEM deadlock analysis script tester.
 *
 * Intentional deadlock scenario:
 *   - PE 0 skips the rocshmem_barrier_all_wg() call and proceeds to finalize.
 *   - All other PEs call rocshmem_barrier_all_wg() inside a GPU kernel, and
 *     get stuck waiting for PE 0 to participate.
 *
 * This exercises the barrier/wait_until deadlock path that the analysis
 * script should detect and annotate.
 *
 * Build (adjust ROCSHMEM, MPI_HOME, GPU_TARGET for your installation):
 *
 *   hipcc -O2 -g -std=c++17 -fgpu-rdc \
 *         --offload-arch=${GPU_TARGET} \
 *         -I${ROCSHMEM}/include -I${MPI_HOME}/include \
 *         -x hip deadlock_test.cc -x none \
 *         ${ROCSHMEM}/lib/librocshmem.a ${MPI_HOME}/lib/libmpi.so \
 *         -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 \
 *         -Wl,-rpath,${MPI_HOME}/lib \
 *         -o deadlock_test
 *
 * Run:
 *   mpirun -np 2 ./deadlock_test
 */

#include <rocshmem/rocshmem.hpp>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

using namespace rocshmem;

/*
 * Kernel: the workgroup performs a collective barrier.
 * When PE 0 never calls this, all other PEs stall here indefinitely.
 */
__global__ void barrier_kernel() {
    rocshmem_barrier_all_wg();
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    /* Select GPU by local MPI rank */
    const char *local_rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local_rank = local_rank_str ? atoi(local_rank_str) : 0;
    hipError_t dev_err = hipSetDevice(local_rank);
    if (dev_err != hipSuccess) {
        fprintf(stderr, "[rank %d] hipSetDevice(%d) failed: %s\n",
                rank, local_rank, hipGetErrorString(dev_err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    rocshmem_init();
    int my_pe = rocshmem_my_pe();
    printf("[PE %d/%d] initialized\n", my_pe, nranks);
    fflush(stdout);

    if (my_pe == 1) {
        /*
         * PE 0 intentionally skips the kernel: finishes immediately,
         * leaving all other PEs stuck in the barrier forever.
         */
        printf("[PE 0] skipping barrier — other PEs will deadlock\n");
        fflush(stdout);
        /* Give other PEs time to enter the kernel before we exit. */
        sleep(3);
    } else {
        /*
         * All other PEs launch the barrier kernel and spin indefinitely
         * waiting for PE 0 to join the collective.
         */
        printf("[PE %d] launching barrier kernel (will deadlock)\n", my_pe);
        fflush(stdout);
        barrier_kernel<<<dim3(1), dim3(64), 0, 0>>>();
        hipError_t launch_err = hipGetLastError();
        if (launch_err != hipSuccess) {
            fprintf(stderr, "[PE %d] barrier_kernel launch failed: %s\n",
                    my_pe, hipGetErrorString(launch_err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* This hipDeviceSynchronize blocks until the kernel finishes — never. */
        hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
            fprintf(stderr, "[PE %d] hipDeviceSynchronize failed: %s\n",
                    my_pe, hipGetErrorString(sync_err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    printf("[PE %d] done\n", my_pe);
    fflush(stdout);

    rocshmem_finalize();
    MPI_Finalize();
    return 0;
}
