// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
