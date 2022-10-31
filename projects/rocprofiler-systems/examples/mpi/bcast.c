// Author: Wes Kendall
// Copyright 2011 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Comparison of MPI_Bcast with the my_bcast function
//
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void
my_bcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator)
{
    int world_rank;
    MPI_Comm_rank(communicator, &world_rank);
    int world_size;
    MPI_Comm_size(communicator, &world_size);

    if(world_rank == root)
    {
        // If we are the root process, send our data to everyone
        int i;
        for(i = 0; i < world_size; i++)
        {
            if(i != world_rank)
            {
                MPI_Send(data, count, datatype, i, 0, communicator);
            }
        }
    }
    else
    {
        // If we are a receiver process, receive the data from the root
        MPI_Recv(data, count, datatype, root, 0, communicator, MPI_STATUS_IGNORE);
    }
}

void
my_ibcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator)
{
    int world_rank;
    MPI_Comm_rank(communicator, &world_rank);
    int world_size;
    MPI_Comm_size(communicator, &world_size);
    MPI_Request request = MPI_REQUEST_NULL;

    if(world_rank == root)
    {
        // If we are the root process, send our data to everyone
        int i;
        for(i = 0; i < world_size; i++)
        {
            if(i != world_rank)
            {
                MPI_Isend(data, count, datatype, i, 0, communicator, &request);
            }
        }
    }
    else
    {
        // If we are a receiver process, receive the data from the root
        MPI_Irecv(data, count, datatype, root, 0, communicator, &request);
    }

    MPI_Status status;
    // bloks and waits for destination process to receive data
    MPI_Wait(&request, &status);
}

int
main(int argc, char** argv)
{
    int num_elements = 30;
    int num_trials   = 50;

    if(argc != 3) fprintf(stderr, "Usage: compare_bcast [num_elements] [num_trials]\n");

    if(argc > 1) num_elements = atoi(argv[1]);
    if(argc > 2) num_trials = atoi(argv[2]);

    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double total_my_bcast_time  = 0.0;
    double total_my_ibcast_time = 0.0;
    double total_mpi_bcast_time = 0.0;
    int    i;
    int*   data = (int*) malloc(sizeof(int) * num_elements);
    assert(data != NULL);

    for(i = 0; i < num_trials; i++)
    {
        // Time my_bcast
        // Synchronize before starting timing
        MPI_Barrier(MPI_COMM_WORLD);
        total_my_bcast_time -= MPI_Wtime();
        my_bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
        // Synchronize again before obtaining final time
        MPI_Barrier(MPI_COMM_WORLD);
        total_my_bcast_time += MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);
        total_my_ibcast_time -= MPI_Wtime();
        my_ibcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
        // Synchronize again before obtaining final time
        MPI_Barrier(MPI_COMM_WORLD);
        total_my_ibcast_time += MPI_Wtime();

        // Time MPI_Bcast
        MPI_Barrier(MPI_COMM_WORLD);
        total_mpi_bcast_time -= MPI_Wtime();
        MPI_Bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        total_mpi_bcast_time += MPI_Wtime();
    }

    // Print off timing information
    if(world_rank == 0)
    {
        printf("Data size = %d, Trials = %d\n", num_elements * (int) sizeof(int),
               num_trials);
        printf("Avg my_bcast  time = %lf\n", total_my_bcast_time / num_trials);
        printf("Avg my_ibcast time = %lf\n", total_my_ibcast_time / num_trials);
        printf("Avg MPI_Bcast time = %lf\n", total_mpi_bcast_time / num_trials);
    }

    free(data);
    MPI_Finalize();
}
