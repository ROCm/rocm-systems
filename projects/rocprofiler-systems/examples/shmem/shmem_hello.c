// MIT License
//
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <shmem.h>
#include <stdio.h>

int
main(void)
{
    // 1. Initialize the SHMEM environment
    shmem_init();

    // 2. Get basic information about my place in the world
    int me   = shmem_my_pe();  // my Processing Element (like rank)
    int npes = shmem_n_pes();  // total number of PEs (like size)

    // Simple output from every process
    printf("Hello from PE %d of %d\n", me, npes);

    // -------------------------------------------------------------------------
    // A bit more interesting: simple point-to-point communication
    // -------------------------------------------------------------------------

    // Allocate one integer in the symmetric heap (visible to all PEs)
    int* value = (int*) shmem_malloc(sizeof(int));

    // Everyone initializes their own slot to their PE number
    *value = me;

    // Barrier so everyone has written their value
    shmem_barrier_all();

    // Each PE reads the value from the next PE (with wrap-around)
    int next_pe = (me + 1) % npes;
    int received;

    // Blocking get from next PE
    shmem_int_get(&received, value, 1, next_pe);

    printf("PE %d received value %d from PE %d\n", me, received, next_pe);

    // Optional: make sure remote memory operations are completed
    shmem_quiet();

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    shmem_free(value);

    // Finalize SHMEM (required in modern versions)
    shmem_finalize();

    return 0;
}