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
 * First find your offload target, and if xnack is enabled/disabled using

   rocminfo | grep amdgcn

 * It should output a string like so:

   "Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"

 * This lists the offload taret (gfx942) and that xnack is disabled (xnack-).
 * Therefore, we need to specify --offload-arch=gfx942:xnack- to our link and compile commands.
 * Please modify the compile and link commands to suit your system

 * To compile:
   hipcc -c -fgpu-rdc -x hip rocshmem_broadcast_test.cc \
         --offload-arch=<target>:<xnack>                \
         -I/opt/rocm/include                            \
         -I$ROCSHMEM_INSTALL_DIR/include                \
         -I$OPENMPI_UCX_INSTALL_DIR/include/

 * To link:
   hipcc -fgpu-rdc --hip-link rocshmem_broadcast_test.o -o rocshmem_broadcast_test \
         --offload-arch=<target>:<xnack>                                           \
         $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a                                   \
         $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so                                    \
         -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

 * To run:
   mpirun -np 8 -x ROCSHMEM_MAX_NUM_CONTEXTS=2 ./rocshmem_broadcast_test

 */

#include <rocshmem/rocshmem.hpp>

#include "util.h"

using namespace rocshmem;

__global__ void broadcast_test(int *source, int *dest, size_t nelem,
        int root, rocshmem_team_t team) {
    __shared__ rocshmem_ctx_t ctx;
    int64_t ctx_type = 0;

    rocshmem_wg_ctx_create(ctx_type, &ctx);
    int num_pes = rocshmem_ctx_n_pes(ctx);

    rocshmem_ctx_int_broadcast_wg(ctx, team, dest, source, nelem, root);

    rocshmem_ctx_quiet(ctx);
    __syncthreads();

    rocshmem_wg_ctx_destroy(&ctx);
}

static void init_sendbuf(int *source, int nelem, int my_pe)
{
    for (int i = 0; i < nelem; i++) {
        source[i] = i;
    }
}

static bool check_recvbuf(int *dest, int nelem, int my_pe, int npes)
{
    bool res=true;

    for (int i = 0; i < npes; i++) {
        if (dest[i] != i) {
            res = false;
#ifdef VERBOSE
            printf("PE: %d, dest[%d] = %d, expected %d \n", my_pe, i, dest[i], i);
#endif
        }
    }

    return res;
}

#define MAX_ELEM 256

int main(int argc, char **argv)
{
    int nelem = MAX_ELEM;

    if (argc > 1) {
        nelem = atoi(argv[1]);
    }

    CHECK_HIP(hipSetDevice(get_launcher_local_rank()));

    rocshmem_init();

    int my_pe = rocshmem_my_pe();
    int npes =  rocshmem_n_pes();

    int *source = (int *)rocshmem_malloc(nelem * sizeof(int));
    int *dest = (int *)rocshmem_malloc(nelem * sizeof(int));
    if (NULL == source || NULL == dest) {
        std::cout << "Error allocating memory from symmetric heap" << std::endl;
        std::cout << "source: " << source << ", dest: " << dest << ", size: "
          << sizeof(int) * nelem << std::endl;
        rocshmem_global_exit(1);
    }

    init_sendbuf(source, nelem, my_pe);
    for (int i=0; i<nelem; i++) {
        dest[i] = -1;
    }

    int root = 0;
    rocshmem_team_t team_reduce_world_dup;
    team_reduce_world_dup = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, npes, nullptr, 0,
                               &team_reduce_world_dup);

    CHECK_HIP(hipDeviceSynchronize());

    int threadsPerBlock=256;
    broadcast_test<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(source, dest,
                        nelem, root, team_reduce_world_dup);
    CHECK_HIP(hipDeviceSynchronize());

    if(my_pe != root) {
        bool pass = check_recvbuf(dest, nelem, my_pe, npes);
        printf("Test %s \t nelem %d %s\n", argv[0], nelem, pass ? "[PASS]" : "[FAIL]");
    }

    rocshmem_free(source);
    rocshmem_free(dest);

    rocshmem_finalize();
    return 0;
}
