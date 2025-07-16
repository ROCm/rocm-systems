#include <unistd.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <hip/hip_runtime.h>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

const char* sem_name = "/ipc_sem";
const char* shm_name = "/ipc_shm";

#define HIP_CHECK_KERNEL(call)                                                                     \
  {                                                                                                \
    do {                                                                                           \
      call;                                                                                        \
      hipError_t err = hipGetLastError();                                                          \
      if (err != hipSuccess) {                                                                     \
        std::cerr << std::endl                                                                     \
                  << "error: '" << hipGetErrorString(err) << "'(" << err << ") at " << __FILE__    \
                  << ":" << __LINE__ << std::endl                                                  \
                  << std::endl;                                                                    \
        exit(1);                                                                                   \
      }                                                                                            \
    } while (0);                                                                                   \
  }

__global__ void kernel1(double* a, std::size_t N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (std::size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
    a[i] = a[i] * a[i];
  }
}

void die(const char* name, int rank) {
  perror(name);
  if (rank == 0) {
    exit(EXIT_FAILURE);  // parent
  } else {
    _exit(EXIT_FAILURE);  // child
  }
}

void barrier(int size) {
  // Semaphore to count the number of processes that arrived
  const char* sem_count_name = "/ipc_count";
  sem_t* count = sem_open(sem_count_name, O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                          /* initial value */ 0);  // 0644
  if (count == SEM_FAILED) die("sem_open", 0);

  // Need another semaphore for the actual barrier
  const char* sem_barrier_name = "/ipc_barrier";
  sem_t* barrier = sem_open(sem_barrier_name, O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                            /* initial value */ 0);  // 0644
  if (barrier == SEM_FAILED) die("sem_open", 0);

  // Each process increments count
  sem_post(count);

  int count_value;
  sem_getvalue(count, &count_value);

  if (count_value == size) {
    // All processes completed the increment.  The last one that gets here unlocks the barrier
    sem_post(barrier);
  }

  // Processes will wait until the last one sees `count` is `size` and when that happens, one
  // process is unblocked, and that unblocked process will unblock the next one, and so on...
  sem_wait(barrier);
  sem_post(barrier);

  // Cleanup
  sem_unlink(sem_count_name);
  sem_close(count);
  sem_unlink(sem_barrier_name);
  sem_close(barrier);
}

int execute(int rank) {
  int size = 2;  // There are only two processes in this test
  int device_count;
  HIPCHECK(hipGetDeviceCount(&device_count));

  if (device_count < 2) {
    if (rank == 0) {
      printf("Expected 2 GPUs, but only %d visible\n", device_count);
    }
    return EXIT_FAILURE;
  }

  HIPCHECK(hipSetDevice(rank));

  if (rank == 0) printf("Allocating memory and launching warmup workload\n");

  std::size_t N = 1000 * 1000 * 1000;
  printf("rank %d allocating %zu bytes on the host\n", rank, N * sizeof(double));
  double* a = (double*)malloc(sizeof(double) * N);  // 8 GB
  double* da;

  printf("rank %d allocating %zu bytes on the device\n", rank, N * sizeof(double));
  HIPCHECK(hipMalloc(&da, sizeof(double) * N));
  printf("rank done %d allocating %zu bytes on the device\n", rank, N * sizeof(double));

  // Fill a
  for (std::size_t i = 0; i < N; i++) {
    a[i] = 0.5 * i;
  }

  // Copy to device
  HIPCHECK(hipMemcpy(da, a, sizeof(double) * N, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  // Warm up
  int num_kernels = 10;
  for (int i = 0; i < num_kernels; i++) {
    // HIP_CHECK_KERNEL(hipLaunchKernelGGL(kernel1, 1024, 1024, 0, stream, da, N));
    hipLaunchKernelGGL(kernel1, 1024, 1024, 0, stream, da, N);
  }

  hipEvent_t end_warmup;
  HIPCHECK(hipEventCreate(&end_warmup));

  // Time the workload
  if (rank == 0) {
    num_kernels = 1000;  // Make device zero very busy
  }

  for (int i = 0; i < num_kernels; i++) {
    // HIP_CHECK_KERNEL(hipLaunchKernelGGL(kernel1, 1024, 1024, 0, stream, da, N));
    hipLaunchKernelGGL(kernel1, 1024, 1024, 0, stream, da, N);
  }
  HIPCHECK(hipEventRecord(end_warmup, stream));

  // Wait for the warmup to finish
  HIPCHECK(hipEventSynchronize(end_warmup));

  // Let's flush everything
  HIPCHECK(hipDeviceSynchronize());

  hipEvent_t local_event;
  hipEvent_t remote_event;
  hipIpcEventHandle_t local_event_handle;
  hipIpcEventHandle_t remote_event_handle;
  HIPCHECK(hipEventCreateWithFlags(&local_event, hipEventDisableTiming | hipEventInterprocess));
  HIPCHECK(hipIpcGetEventHandle(&local_event_handle, local_event));

  // Open a semaphore -- we don't care which process gets here first
  sem_t* sem = sem_open(sem_name, O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH,
                        /* initial value */ 0);  // 0644
  if (sem == SEM_FAILED) die("sem_open", rank);

  // Now open the shm region for the ipc handle
  int fd = shm_open(shm_name, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);  // 0644
  if (fd < 0) die("shm_open", rank);

  if (ftruncate(fd, sizeof(hipIpcEventHandle_t)) !=
      0) {  // What if this thing is an opaque pointer type?
    die("ftruncate", rank);
  }

  void* shm = mmap(NULL, sizeof(hipIpcEventHandle_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (shm == MAP_FAILED) die("mmap", rank);

  // Share ipc event handle with rank 1
  // Only rank 0 writes
  if (rank == 0) {
    memcpy(shm, &local_event_handle, sizeof(hipIpcEventHandle_t));
    if (sem_post(sem)) die("sem_post", rank);
  } else {
    if (sem_wait(sem)) die("sem_wait", rank);
    memcpy(&remote_event_handle, shm, sizeof(hipIpcEventHandle_t));

    // open handle
    HIPCHECK(hipIpcOpenEventHandle(&remote_event, remote_event_handle));
  }

  // Now reproduce the problem

  // kernel in stream
  if (rank == 0) printf("Launching workload\n");
  for (int i = 0; i < num_kernels; i++) {
    // HIP_CHECK_KERNEL(hipLaunchKernelGGL(kernel1, 1024, 1024, 0, stream, da, N));
    hipLaunchKernelGGL(kernel1, 1024, 1024, 0, stream, da, N);
  }

  if (rank == 0) {
    printf("Testing ipc event sync\n");
    HIPCHECK(hipEventRecord(local_event, stream));
  }

  barrier(size);

  // At this point, rank 1 knows that the remote ipc event has been recorded

  // wait on remote event
  double rank_0_time, rank_1_time;
  if (rank == 1) {
    auto ipc_start = std::chrono::steady_clock::now();
    HIPCHECK(hipEventSynchronize(remote_event));
    auto ipc_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> ipc_elapsed = ipc_end - ipc_start;

    double elapsed_time = ipc_elapsed.count();

    // Copy time to shm so that rank 0 can later read it
    memcpy(shm, &elapsed_time, sizeof(double));
  }
  if (rank == 0) {
    auto local_start = std::chrono::steady_clock::now();
    HIPCHECK(hipDeviceSynchronize());
    auto local_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> local_elapsed = local_end - local_start;
    rank_0_time = local_elapsed.count();
  }

  barrier(size);

  // Compute the difference in recorded times
  // Re-use the shm region we created to communicate the child's computed time and compare
  int ret_code = EXIT_FAILURE;
  if (rank == 0) {
    memcpy(&rank_1_time, shm, sizeof(double));
    printf("hipDeviceSynchronize time: %lf s\n", rank_0_time);
    printf("IPC hipEventSynchronize time: %lf s\n", rank_1_time);

    double diff = rank_0_time - rank_1_time;
    double err = diff / rank_0_time;

    // Pass if the error isn't "too large"
    if (err < 0.3) {
      printf("Test: PASSED\n");
      ret_code = EXIT_SUCCESS;
    } else {
      printf("Test: FAILED\n");
      ret_code = EXIT_FAILURE;
    }
  } else {
    // Child process doesn't fail
    ret_code = EXIT_SUCCESS;
  }

  // Clean up
  munmap(shm, sizeof(hipIpcEventHandle_t));
  shm_unlink(shm_name);
  sem_unlink(sem_name);
  sem_close(sem);

  // Clean up
  // We don't need to destroy the remote event because we destroy the exported event.
  HIPCHECK(hipEventDestroy(local_event));

  HIPCHECK(hipEventDestroy(end_warmup));
  HIPCHECK(hipStreamDestroy(stream));
  HIPCHECK(hipFree(da));
  free(a);

  return ret_code;
}

TEST_CASE("Unit_hipIpcEventSynchronize_Test") {
  int ret;
  pid_t pid = fork();
  switch (pid) {
    case -1:
      perror("fork");
      exit(EXIT_FAILURE);
    case 0:
      // We're the child process
      ret = execute(1);
      _exit(ret);
    default:
      // We're the parent process
      ret = execute(0);
      exit(ret);
  }
}
