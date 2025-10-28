#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <unistd.h>  // gethostname
#include <ctime>     // clock_gettime

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include "lib/common/utility.hpp"  // ROCprofiler headers for timestamp_ns

// Error macros (HIP/RCCL)
#define HIP_CHECK(call) do { \
  hipError_t err = call; \
  if (err != hipSuccess) { \
    fprintf(stderr, "HIP Error at %s:%d, %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

#define RCCL_CHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    fprintf(stderr, "RCCL Error at %s:%d, %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

// // Timestamp function (from rocprofiler common or standalone)
// inline uint64_t timestamp_ns(clockid_t clk_id) {
//     struct timespec ts{};
//     if (clock_gettime(clk_id, &ts) != 0) {
//         fprintf(stderr, "clock_gettime failed\n");
//         return 0;
//     }
//     constexpr uint64_t ns_per_sec = 1000000000ULL;
//     return (static_cast<uint64_t>(ts.tv_sec) * ns_per_sec) + static_cast<uint64_t>(ts.tv_nsec);
// }


int main(int argc, char* argv[]) {
    // --- Args ---
    if (argc < 3) {
        fprintf(stderr, "Usage: mpirun -np 2 ./test_rccl_clocks <clock_type> <num_allreduces> [sleep_between_ms]\n");
        fprintf(stderr, "  clock_type: tai | realtime\n");
        fprintf(stderr, "  num_allreduces: e.g., 10, 50, 100\n");
        fprintf(stderr, "  sleep_between_ms: optional, e.g., 1000 (1s)\n");
        return 1;
    }
    std::string clock_type = argv[1];
    int num_ars = std::atoi(argv[2]);
    int sleep_ms = (argc > 3) ? std::atoi(argv[3]) : 0;  // Optional sleep

    clockid_t clk_id = (clock_type == "tai") ? CLOCK_TAI : CLOCK_REALTIME;
    if (clock_type != "tai" && clock_type != "realtime") {
        fprintf(stderr, "Invalid clock_type. Use 'tai' or 'realtime'.\n");
        return 1;
    }
    std::function<uint64_t()> timestamp_func;
    if (clock_type == "tai") {
        timestamp_func = []() { return rocprofiler::common::timestamp_ns<CLOCK_TAI>(); };
    } else {
        timestamp_func = []() { return rocprofiler::common::timestamp_ns<CLOCK_REALTIME>(); };
    }

    // Initial timestamp (t0)
    uint64_t t0 = timestamp_func();

    // --- MPI Init ---
    int my_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (my_rank == 0) fprintf(stderr, "Designed for 2 ranks, got %d\n", world_size);
        // MPI_Finalize();
        // return 1;
    }

    // --- Node Info ---
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    std::string log_file = "rccl_clocks_rank" + std::to_string(my_rank) + "_" + clock_type + ".csv";
    std::ofstream log(log_file);
    log << "Rank,Hostname,ClockType,Iteration,PreAR_ns,PostAR_ns,Duration_ns\n";  // CSV header

    printf("Rank %d on %s, Clock: %s, NumARs: %d, SleepMs: %d\n", my_rank, hostname, clock_type.c_str(), num_ars, sleep_ms);

    // --- GPU Setup ---
    int num_devices;
    HIP_CHECK(hipGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        fprintf(stderr, "Rank %d: No GPUs\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device_id = my_rank % num_devices;
    HIP_CHECK(hipSetDevice(device_id));
    printf("Rank %d using device %d\n", my_rank, device_id);

    // --- RCCL Init ---
    ncclComm_t comm;
    ncclUniqueId id;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    if (my_rank == 0) {
        RCCL_CHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    RCCL_CHECK(ncclCommInitRank(&comm, world_size, id, my_rank));
    printf("Rank %d: RCCL ready\n", my_rank);

    // --- Data Buffers ---
    const int data_size = 1024;  // Floats per AR
    float *d_send, *d_recv;
    HIP_CHECK(hipMalloc(&d_send, data_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_recv, data_size * sizeof(float)));

    std::vector<float> h_send(data_size, 1.0f * (my_rank + 1));  // Rank-specific init
    HIP_CHECK(hipMemcpy(d_send, h_send.data(), data_size * sizeof(float), hipMemcpyHostToDevice));

    // --- Experiment Loop ---
    uint64_t exp_start = timestamp_func();
    printf("Rank %d: Starting experiment at t=%lu ns\n", my_rank, exp_start);

    for (int i = 0; i < num_ars; ++i) {
        // Timestamp before AR
        uint64_t pre_ar = timestamp_func();

        // All-Reduce
        RCCL_CHECK(ncclAllReduce(d_send, d_recv, data_size, ncclFloat, ncclSum, comm, stream));
        HIP_CHECK(hipStreamSynchronize(stream));  // Critical for accurate post-AR timestamp

        // Timestamp after AR
        uint64_t post_ar = timestamp_func();
        uint64_t duration = post_ar - pre_ar;

        // Log
        log << my_rank << "," << hostname << "," << clock_type << "," << i << "," 
            << pre_ar << "," << post_ar << "," << duration << "\n";

        // Optional: Verify result (first AR only to save time)
        if (i == 0) {
            std::vector<float> h_recv(data_size);
            HIP_CHECK(hipMemcpy(h_recv.data(), d_recv, data_size * sizeof(float), hipMemcpyDeviceToHost));
            float expected = 1.0f + 2.0f;  // Rank0(1) + Rank1(2) = 3
            bool ok = (std::abs(h_recv[0] - expected) < 1e-5);
            printf("Rank %d AR %d: Verify %s (got %.1f, exp %.1f)\n", my_rank, i, ok ? "OK" : "FAIL", h_recv[0], expected);
        }

        // Optional sleep (simulate training step)
        if (sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }

    uint64_t exp_end = timestamp_func();
    printf("Rank %d: Experiment done at t=%lu ns, total=%lu ns\n", my_rank, exp_end, exp_end - exp_start);

    // --- Cleanup ---
    HIP_CHECK(hipFree(d_send));
    HIP_CHECK(hipFree(d_recv));
    HIP_CHECK(hipStreamDestroy(stream));
    RCCL_CHECK(ncclCommDestroy(comm));
    log.close();
    MPI_Finalize();

    if (my_rank == 0) {
        printf("Logs: rccl_clocks_rank*.csv\n");
    }
    return 0;
}