// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Exercises ROCPROFSYS_SELECTED_REGIONS against MPI tracing. MPI_Allreduce is only
// ever called inside "TracedRegion" and MPI_Bcast only ever outside it, so a filtered
// trace must contain the former and none of the latter.

#include <mpi.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <cstdio>

namespace
{
constexpr int CALLS_PER_PHASE = 8;

void
run_untraced_phase()
{
    int value = 0;
    for(int i = 0; i < CALLS_PER_PHASE; ++i)
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void
run_traced_phase()
{
    roctxRangePushA("TracedRegion");
    for(int i = 0; i < CALLS_PER_PHASE; ++i)
    {
        int local = 1;
        int total = 0;
        MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    roctxRangePop();
}
}  // namespace

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    run_untraced_phase();
    run_traced_phase();
    run_untraced_phase();

    if(rank == 0) printf("mpi-roctx-regions completed on %d rank(s)\n", size);

    MPI_Finalize();
    return 0;
}
