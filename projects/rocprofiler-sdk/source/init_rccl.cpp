#include <iostream>
#include <vector>
#include <thread> // For std::this_thread
#include <chrono> // For std::chrono
#include <cmath>  // For std::abs

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

// Simple error checking macro for HIP calls
#define HIP_CHECK(call) do { \
  hipError_t err = call; \
  if (err != hipSuccess) { \
    fprintf(stderr, "HIP Error at %s:%d, %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

// Simple error checking macro for RCCL calls
#define RCCL_CHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    fprintf(stderr, "RCCL Error at %s:%d, %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

// Helper function to verify the results
void verify_result(int rank, const std::vector<float>& buffer, float expected_value) {
    bool success = true;
    for (size_t i = 0; i < buffer.size(); ++i) {
        if (std::abs(buffer[i] - expected_value) > 1e-5) {
            success = false;
            break;
        }
    }
    if (success) {
        printf("Rank %d: SUCCESS! Result matches expected value of %.1f.\n", rank, expected_value);
    } else {
        printf("Rank %d: FAILED! Result does not match expected value.\n", rank);
    }
}


int main(int argc, char* argv[]) {
    // --- MPI Initialization ---
    int my_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    fprintf(stderr, "MPI_COMM_WORLD_RANK: %d\n", my_rank);
    fprintf(stderr, "MPI_COMM_WORLD_SIZE: %d\n", world_size);

    if (world_size != 2) {
        fprintf(stderr, "This example is designed to run with exactly 2 processes (nodes).\n");
        if (my_rank == 0) {
            fprintf(stderr, "This example is designed to run with exactly 2 processes (nodes).\n");
        }
        // MPI_Finalize();
        // return 1;
    }

    // --- GPU and Device Setup ---
    int num_devices;
    HIP_CHECK(hipGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        fprintf(stderr, "Rank %d: No HIP devices found.\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device_id = my_rank % num_devices;
    HIP_CHECK(hipSetDevice(device_id));
    printf("Rank %d is using device %d\n", my_rank, device_id);

    // --- RCCL Communicator and Stream Setup ---
    ncclComm_t comm;
    ncclUniqueId id;
    hipStream_t stream;

    HIP_CHECK(hipStreamCreate(&stream));
    fprintf(stderr, "Rank %d: Created stream\n", my_rank);
    if (my_rank == 0) {
        RCCL_CHECK(ncclGetUniqueId(&id));
    }
    fprintf(stderr, "Rank %d: Got unique ID\n", my_rank);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    fprintf(stderr, "Rank %d: Broadcasted unique ID\n", my_rank);
    RCCL_CHECK(ncclCommInitRank(&comm, world_size, id, my_rank));
    fprintf(stderr, "Rank %d: RCCL communicator initialized\n", my_rank);
    // auto sockaddr = comm->ncclPeerInfo;
    // --- Data Preparation ---
    const int data_size = 10;
    std::vector<float> host_send_buffer(data_size);
    std::vector<float> host_recv_buffer(data_size, 0.0f);
    
    // **CRITICAL FIX**: Allocate buffers in GPU device memory, not on the host stack.
    float *device_send_buffer;
    float *device_recv_buffer;
    HIP_CHECK(hipMalloc(&device_send_buffer, data_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_recv_buffer, data_size * sizeof(float)));

    // Initialize data for the first AllReduce
    // Rank 0: [0.0, 0.0, ...]
    // Rank 1: [1.0, 1.0, ...]
    // Expected result: Sum = [1.0, 1.0, ...]
    for (int i = 0; i < data_size; ++i) {
        host_send_buffer[i] = (float)my_rank;
    }

    // Copy data from host to device
    HIP_CHECK(hipMemcpy(device_send_buffer, host_send_buffer.data(), data_size * sizeof(float), hipMemcpyHostToDevice));
    
    // --- First AllReduce Operation ---
    printf("\nRank %d: Performing first AllReduce...\n", my_rank);
    RCCL_CHECK(ncclAllReduce(device_send_buffer, device_recv_buffer, data_size, ncclFloat, ncclSum, comm, stream));
    
    // Synchronize stream to ensure the operation is complete
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Copy the result back from device to host for verification
    HIP_CHECK(hipMemcpy(host_recv_buffer.data(), device_recv_buffer, data_size * sizeof(float), hipMemcpyDeviceToHost));
    
    verify_result(my_rank, host_recv_buffer, 1.0f);

    // --- Pause / Sleep ---
    if (my_rank == 0) {
        printf("\nPausing for 5 seconds before the next operation...\n\n");
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // --- Second AllReduce Operation ---
    // Re-initialize data to show a different operation
    // Rank 0: [10.0, 10.0, ...]
    // Rank 1: [20.0, 20.0, ...]
    // Expected result: Sum = [30.0, 30.0, ...]
    for (int i = 0; i < data_size; ++i) {
        host_send_buffer[i] = 10.0f * (my_rank + 1);
    }
    HIP_CHECK(hipMemcpy(device_send_buffer, host_send_buffer.data(), data_size * sizeof(float), hipMemcpyHostToDevice));

    printf("Rank %d: Performing second AllReduce...\n", my_rank);
    RCCL_CHECK(ncclAllReduce(device_send_buffer, device_recv_buffer, data_size, ncclFloat, ncclSum, comm, stream));
    
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(host_recv_buffer.data(), device_recv_buffer, data_size * sizeof(float), hipMemcpyDeviceToHost));

    verify_result(my_rank, host_recv_buffer, 30.0f);


    // --- Cleanup ---
    HIP_CHECK(hipFree(device_send_buffer));
    HIP_CHECK(hipFree(device_recv_buffer));
    HIP_CHECK(hipStreamDestroy(stream));
    RCCL_CHECK(ncclCommDestroy(comm));
    MPI_Finalize();

    return 0;
}

