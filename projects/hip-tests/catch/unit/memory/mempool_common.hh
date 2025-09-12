/*
   Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */
#pragma once

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <resource_guards.hh>
#include <utils.hh>

#ifdef __linux__
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <sys/socket.h>
#include <memory.h>
#include <sys/un.h>
#endif

namespace {
constexpr auto wait_ms = 500;
}  // anonymous namespace

#define checkMempoolSupported(device) {\
  int deviceSupportsMemoryPools = 0;\
  HIP_CHECK(hipDeviceGetAttribute(&deviceSupportsMemoryPools,\
        hipDeviceAttributeMemoryPoolsSupported, device));\
  if (0 == deviceSupportsMemoryPools) {\
    HipTest::HIP_SKIP_TEST("Memory Pool not supported. Skipping Test..");\
    return;\
  }\
}

#define checkIfMultiDev(numOfDev) {\
  if (numOfDev < 2) {\
    HipTest::HIP_SKIP_TEST("Multiple GPUs not available. Skipping Test..");\
    return;\
  }\
}

template <typename T> __global__ void kernel_500ms(T* host_res, int clk_rate) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  host_res[tid] = tid + 1;
  __threadfence_system();
  // expecting that the data is getting flushed to host here!
  uint64_t start = clock64() / clk_rate, cur;
  if (clk_rate > 1) {
    do {
      cur = clock64() / clk_rate - start;
    } while (cur < wait_ms);
  } else {
    do {
      cur = clock64() / start;
    } while (cur < wait_ms);
  }
}

template <typename T> __global__ void kernel_500ms_gfx11(T* host_res, int clk_rate) {
#if HT_AMD
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  host_res[tid] = tid + 1;
  __threadfence_system();
  // expecting that the data is getting flushed to host here!
  uint64_t start = clock_function() / clk_rate, cur;
  if (clk_rate > 1) {
    do {
      cur = clock_function() / clk_rate - start;
    } while (cur < wait_ms);
  } else {
    do {
      cur = clock_function() / start;
    } while (cur < wait_ms);
  }
#endif
}

template <typename T> __global__ void notifiedKernel(T* host_res, volatile unsigned int* notified) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  host_res[tid] = tid + 1;
  __threadfence_system();
  while (*notified == 0) { }
}

template <typename F> void MallocMemPoolAsync_OneAlloc(F malloc_func, const MemPools mempool_type) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int *notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);
  MemPoolGuard mempool(mempool_type, device_id);

  int* alloc_mem;
  StreamGuard stream(Streams::created);

  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem), allocation_size, mempool.mempool(),
                        stream.stream()));

  int blocks = 16;
  hipMemPoolAttr attr;
  notifiedKernel<<<32, blocks, 0, stream.stream()>>>(alloc_mem, notified);

  const auto element_count = allocation_size / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 17;
  VectorSet<<<block_count, thread_count, 0, stream.stream()>>>(alloc_mem, expected_value,
                                                               element_count);

  HIP_CHECK(hipMemcpyAsync(host_alloc.host_ptr(), alloc_mem, allocation_size, hipMemcpyDeviceToHost,
                           stream.stream()));

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem), stream.stream()));

  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_before_sync));
  *notified = 1;
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_sync));
  // Sync must release memory to OS
  REQUIRE(res_after_sync <= res_before_sync);

  std::uint64_t used_mem = 10;
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  REQUIRE(0 == used_mem);

  ArrayFindIfNot(host_alloc.host_ptr(), expected_value, element_count);
  HIP_CHECK(hipHostFree(notified));
}

template <typename F>
void MallocMemPoolAsync_TwoAllocs(F malloc_func, const MemPools mempool_type) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int *notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);
  MemPoolGuard mempool(mempool_type, device_id);

  int* alloc_mem1;
  int* alloc_mem2;
  StreamGuard stream(Streams::created);

  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem1), allocation_size, mempool.mempool(),
                        stream.stream()));
  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem2), allocation_size, mempool.mempool(),
                        stream.stream()));

  int blocks = 16;
  hipMemPoolAttr attr;
  notifiedKernel<<<32, blocks, 0, stream.stream()>>>(alloc_mem1, notified);

  const auto element_count = allocation_size / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 17;
  VectorSet<<<block_count, thread_count, 0, stream.stream()>>>(alloc_mem1, expected_value,
                                                               element_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpyAsync(alloc_mem2, alloc_mem1, allocation_size, hipMemcpyDeviceToDevice,
                           stream.stream()));

  HIP_CHECK(hipMemcpyAsync(host_alloc.host_ptr(), alloc_mem2, allocation_size,
                           hipMemcpyDeviceToHost, stream.stream()));

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream.stream()));

  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_before_sync));
  *notified = 1;
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_sync));
  // Sync must release memory to OS
  REQUIRE(res_after_sync <= res_before_sync);

  std::uint64_t used_mem = 0;
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  // Make sure the current usage query works - just second buffer is left
  REQUIRE(allocation_size == used_mem);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  // Make sure the high watermark usage works - both buffers must be reported
  REQUIRE((2 * allocation_size) == used_mem);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  // Make sure the current usage query works - none of the buffers are used
  REQUIRE(0 == used_mem);

  ArrayFindIfNot(host_alloc.host_ptr(), expected_value, element_count);
  HIP_CHECK(hipHostFree(notified));
}

template <typename F> void MallocMemPoolAsync_Reuse(F malloc_func, const MemPools mempool_type) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int *notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  MemPoolGuard mempool(mempool_type, device_id);

  int *alloc_mem1, *alloc_mem2, *alloc_mem3;
  StreamGuard stream(Streams::created);

  size_t allocation_size1 = kPageSize * kPageSize * 2;
  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem1), allocation_size1, mempool.mempool(),
                        stream.stream()));

  size_t allocation_size2 = kPageSize;
  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem3), allocation_size2, mempool.mempool(),
                        stream.stream()));

  int blocks = 2;

  notifiedKernel<<<32, blocks, 0, stream.stream()>>>(alloc_mem1, notified);

  hipMemPoolAttr attr;
  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream.stream()));

  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem2), allocation_size1, mempool.mempool(),
                        stream.stream()));
  // Runtime must reuse the pointer
  REQUIRE(alloc_mem1 == alloc_mem2);

  // Make a sync before the second kernel launch to make sure memory B isn't gone
  *notified = 1;
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  *notified = 0;
  // Second kernel launch with new memory
  notifiedKernel<<<32, blocks, 0, stream.stream()>>>(alloc_mem2, notified);
  *notified = 1;
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  attr = hipMemPoolAttrUsedMemCurrent;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the current usage reports the both buffers
  REQUIRE((allocation_size1 + allocation_size2) == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE((allocation_size1 + allocation_size2) == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream.stream()));
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the current usage reports just one buffer, because the above free doesn't hold memory
  REQUIRE(allocation_size2 == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem3), stream.stream()));
  HIP_CHECK(hipHostFree(notified));
}

// definitions
#define THREADS_PER_BLOCK 512
#define LAUNCH_ITERATIONS 5
#define NUMBER_OF_THREADS 5
#define NUM_OF_STREAM 3

enum eTestValue {
  testdefault,
  testMaximum,
  testDisabled,
  testEnabled
};

class streamMemAllocTest {
  int *A_h, *B_h, *C_h;
  int *A_d, *B_d, *C_d;
  int size;
  size_t byte_size;
  hipMemPool_t mem_pool;

 public:
  explicit streamMemAllocTest(int N) : size(N) {
    byte_size = N*sizeof(int);
  }
  // Create host buffers and initialize them with input data
  void createHostBufferWithData() {
    A_h = reinterpret_cast<int*>(malloc(byte_size));
    REQUIRE(A_h != nullptr);
    B_h = reinterpret_cast<int*>(malloc(byte_size));
    REQUIRE(B_h != nullptr);
    C_h = reinterpret_cast<int*>(malloc(byte_size));
    REQUIRE(C_h != nullptr);
    // set data to host
    for (int i = 0; i < size; i++) {
      A_h[i] = 2*i + 1;  // Odd
      B_h[i] = 2*i;      // Even
      C_h[i] = 0;
    }
  }
  // Instead of creating a mempool in class use the global mempool.
  void useCommonMempool(hipMemPool_t mempool) {
    mem_pool = mempool;
  }
  // Create the mempool
  void createMempool(hipMemPoolAttr attr, enum eTestValue testtype,
                    int dev) {
    // Create mempool in current device
    hipMemPoolProps pool_props{};
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.location.id = dev;
    pool_props.location.type = hipMemLocationTypeDevice;
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
    if (attr == hipMemPoolAttrReleaseThreshold) {
      uint64_t setThreshold = 0;
      if (testtype == testMaximum) {
        setThreshold = UINT64_MAX;
      }
      HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &setThreshold));
    } else if ((attr == hipMemPoolReuseFollowEventDependencies) ||
              (attr == hipMemPoolReuseAllowOpportunistic) ||
              (attr == hipMemPoolReuseAllowInternalDependencies)) {
      int value = 0;
      if (testtype == testEnabled) {
        value = 1;
      }
      HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));
    }
  }
  // allocate device memory from mempool.
  void allocFromMempool(hipStream_t stream) {
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d),
              byte_size, mem_pool, stream));
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B_d),
              byte_size, mem_pool, stream));
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C_d),
              byte_size, mem_pool, stream));
  }
  // Transfer data from host to device asynchronously.
  void transferToMempool(hipStream_t stream) {
    HIP_CHECK(hipMemcpyAsync(A_d, A_h, byte_size, hipMemcpyHostToDevice,
              stream));
    HIP_CHECK(hipMemcpyAsync(B_d, B_h, byte_size, hipMemcpyHostToDevice,
              stream));
  }
  // allocate from default mempool.
  void allocFromDefMempool(hipStream_t stream) {
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&A_d),
              byte_size, stream));
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&B_d),
              byte_size, stream));
    HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&C_d),
              byte_size, stream));
  }
  // Execute Kernel to process input data and wait for it.
  void runKernel(hipStream_t stream) {
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(size / THREADS_PER_BLOCK),
                        dim3(THREADS_PER_BLOCK), 0, stream,
                        static_cast<const int*>(A_d),
                        static_cast<const int*>(B_d), C_d, size);
    HIP_CHECK(hipGetLastError());
  }
  // Transfer data from device to host asynchronously.
  void transferFromMempool(hipStream_t stream) {
    HIP_CHECK(hipMemcpyAsync(C_h, C_d, byte_size, hipMemcpyDeviceToHost,
                        stream));
  }
  // Validate the data returned from device.
  bool validateResult() {
    for (int i = 0; i < size; i++) {
      if (C_h[i] != (A_h[i] + B_h[i])) {
        return false;
      }
    }
    return true;
  }
  // Free device memory
  void freeDevBuf(hipStream_t stream) {
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B_d), stream));
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C_d), stream));
  }
  // Free mempool if not using global mempool
  void freeMempool() {
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
  // Free all host buffers
  void freeHostBuf() {
    free(A_h);
    free(B_h);
    free(C_h);
  }
};

#ifdef __linux__

#define checkSysCallErrors(result)                                                                 \
  if (result == -1) {                                                                              \
    fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); exit(EXIT_FAILURE);                 \
  }

#ifdef HT_AMD
typedef int64_t hipShareableHdl;
#else
typedef int hipShareableHdl;
#endif

typedef pid_t Process;

struct ipcHdl {
    int socket;
    char *name;
};

class ipcSocketCom {
  ipcHdl *handle;
  // method to create socket from server
  int createSocket() {
    int server_fd;
    struct sockaddr_un servaddr;

    char name[16];
    // Create a unique socket name based on current pid
    sprintf(name, "%u", getpid());

    // Create the socket handle
    handle = new ipcHdl;
    if (nullptr == handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }

    memset(handle, 0, sizeof(*handle));
    handle->socket = -1;
    handle->name = NULL;

    // Creating socket
    if ((server_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == 0) {
      perror("Socket failure: Socket creation failed");
      return -1;
    }

    unlink(name);
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sun_family = AF_UNIX;

    size_t len = strlen(name);
    if (len > (sizeof(servaddr.sun_path) - 1)) {
      perror("Socket failure: Cannot bind provided name to socket. Name too large");
      return -1;
    }

    strncpy(servaddr.sun_path, name, len);

    if (bind(server_fd, (struct sockaddr *)&servaddr, SUN_LEN(&servaddr)) < 0) {
      perror("Socket failure: Binding socket failed");
      return -1;
    }

    handle->name = new char[strlen(name) + 1];
    strcpy(handle->name, name);
    handle->socket = server_fd;
    return 0;
  }
  // method to create socket from client
  int openSocket() {
    int sock = 0;
    struct sockaddr_un cliaddr;

    handle = new ipcHdl;
    if (nullptr == handle) {
      perror("Socket failure: Handle memory allocation failed");
      return -1;
    }
    memset(handle, 0, sizeof(*handle));

    if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
      perror("IPC failure:Socket creation error");
      return -1;
    }

    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char name[16];

    // Create a unique socket name based on current process id.
    sprintf(name, "%u", getpid());

    strcpy(cliaddr.sun_path, name);
    if (bind(sock, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
      perror("Socket failure: Binding socket failed");
      return -1;
    }

    handle->socket = sock;
    handle->name = new char[strlen(name) + 1];
    strcpy(handle->name, name);

    return 0;
  }
  // method to close socket
  int closeSocket() {
    if (!handle) {
      return -1;
    }

    if (handle->name) {
      unlink(handle->name);
      delete[] handle->name;
    }
    close(handle->socket);
    delete handle;
    return 0;
  }
public:
  ipcSocketCom() = default;
  ipcSocketCom(bool isServer) {
    if (isServer) {
      checkSysCallErrors(createSocket());
    } else {
      checkSysCallErrors(openSocket());
    }
  }
  ~ipcSocketCom() {
  }
  int closeThisSock() {
    return closeSocket();
  }
  // method to receive shareable handle via socket
  int recvShareableHdl(hipShareableHdl *shHandle) {
    struct msghdr msg;
    struct iovec iov[1];

    // Union to guarantee alignment requirements for control array
    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    ssize_t n;
    int receivedfd;
    int dummy_data;

    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);
    iov[0].iov_base = &dummy_data;
    iov[0].iov_len = sizeof(dummy_data);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    if ((n = recvmsg(handle->socket, &msg, 0)) <= 0) {
      perror("Socket failure: Receiving data over socket failed");
      return -1;
    }

    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
       (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        return -1;
      }

      memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
      *(int *)shHandle = receivedfd;
    } else {
      return -1;
    }

    return 0;
  }
  // method to send shareable handle via sockets
  int sendShareableHdl(hipShareableHdl shareableHdl, Process process) {
    struct msghdr msg;
    struct iovec iov[1];
    int dummy_data = 0;

    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr *cmptr;
    struct sockaddr_un cliaddr;

    // Construct client address to send this SHareable handle to
    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    strcpy(cliaddr.sun_path, std::to_string(process).c_str());

    // Send corresponding shareable handle to the client
    int sendfd = (int)shareableHdl;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

    msg.msg_name = (void *)&cliaddr;
    msg.msg_namelen = sizeof(struct sockaddr_un);
    iov[0].iov_base = &dummy_data;
    iov[0].iov_len = sizeof(dummy_data);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t sendResult = sendmsg(handle->socket, &msg, 0);
    if (sendResult <= 0) {
      perror("Socket failure: Sending data over socket failed");
      return -1;
    }
    return 0;
  }
};
#endif
