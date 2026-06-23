/**
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemFabricHandle hipMemFabricHandle
 * @{
 * @ingroup VirtualMemoryManagementTest
 * Tests for fabric handle export/import via VMM + MPI.
 * Exercises the full multi-rank path: VMM allocate -> export fabric handle ->
 * MPI_Send to partner rank -> MPI_Recv -> import fabric handle ->
 * map imported handle -> kernel read/write -> verify data coherence.
 *
 * When run with a single MPI rank (or without mpirun), falls back to a
 * local self-test exercising VMM without fabric handles.
*/

#include <hip_test_common.hh>
#include "hip_vmm_common.hh"
#include <mpi.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdarg>

/* ========================================================================
 * Test Configuration
 * ======================================================================== */

enum AccessOrder {
  MEMCPY_BEFORE_KERNEL = 0,
  KERNEL_BEFORE_MEMCPY = 1,
};

enum MemcpyMode {
  MEMCPY_SYNC = 0,
  MEMCPY_ASYNC = 1,
};

struct TestConfig {
  size_t bufferSize;
  AccessOrder accessOrder;
  MemcpyMode memcpyMode;

  std::string to_string() const {
    char buf[256];
    snprintf(buf, sizeof(buf), "bufSize=%zu order=%s memcpy=%s", bufferSize,
             accessOrder == MEMCPY_BEFORE_KERNEL ? "memcpy_first" : "kernel_first",
             memcpyMode == MEMCPY_SYNC ? "sync" : "async");
    return buf;
  }
};

/* ========================================================================
 * Dynamic Test Data Layout
 *   Memory layout: [src | dst | write_dst | src_copy], each bufferSize ints.
 * ======================================================================== */

struct TestDataLayout {
  size_t bufferSize;
  size_t total_size;

  TestDataLayout(size_t bs) : bufferSize(bs), total_size(4 * bs * sizeof(int)) {}

  int* src(void* base) const { return reinterpret_cast<int*>(base); }
  int* dst(void* base) const {
    return reinterpret_cast<int*>(static_cast<char*>(base) + bufferSize * sizeof(int));
  }
  int* write_dst(void* base) const {
    return reinterpret_cast<int*>(static_cast<char*>(base) + 2 * bufferSize * sizeof(int));
  }
  int* src_copy(void* base) const {
    return reinterpret_cast<int*>(static_cast<char*>(base) + 3 * bufferSize * sizeof(int));
  }
};

/* ========================================================================
 * Globals (rank-local state, populated by ensureMpiInit)
 * ======================================================================== */

static char gProcessorName[MPI_MAX_PROCESSOR_NAME];
static int gRank = -1;
static int gPartnerRank = -1;
static int gNumGpus = 0;
static int gWorldSize = 0;
static bool gMultiHost = false;

/* ========================================================================
 * Logging
 * ======================================================================== */

enum LogLevel { LOG_ERROR = 0, LOG_INFO = 1, LOG_DEBUG = 2 };
static int gVerboseLevel = LOG_DEBUG;

static void log_printf(const char* file, int line, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  char message[4096];
  vsnprintf(message, sizeof(message), format, ap);
  va_end(ap);
  printf("[%s|%d] (%s:%d) %s", gProcessorName, gRank, file, line, message);
  fflush(stdout);
}

#define LogPrint(verbose, format, ...)                                                              \
  do {                                                                                             \
    if (verbose <= gVerboseLevel) {                                                                \
      log_printf(__FILE__, __LINE__, format, ##__VA_ARGS__);                                       \
    }                                                                                              \
  } while (false)

/* ========================================================================
 * HIP_CHECK variant for MPI helper functions.
 * Returns -1 on failure rather than asserting, so that MPI exchanges
 * can complete and partner ranks don't deadlock.
 * ======================================================================== */

#define HIP_CHECK_MPI(cmd)                                                                         \
  do {                                                                                             \
    hipError_t error = (cmd);                                                                      \
    if (error != hipSuccess) {                                                                     \
      LogPrint(LOG_ERROR, "HIP error: '%s'(%d) from '%s'\n", hipGetErrorString(error), error,      \
               #cmd);                                                                              \
      if (gWorldSize > 1) {                                                                        \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));                                        \
      }                                                                                            \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)

/* ========================================================================
 * Helpers
 * ======================================================================== */

template <typename T> static T AlignUp(T value, size_t alignment) {
  return (T)(((size_t)value + alignment - 1) / alignment * alignment);
}

static char fabric_str_buf[256];
static const char* fabric_handle_str(const hipMemFabricHandle_t& handle) {
  char* p = fabric_str_buf;
  for (size_t i = 0; i < sizeof(handle.data) && i < 64; i++)
    p += sprintf(p, "%02x ", handle.data[i]);
  return fabric_str_buf;
}

/* ========================================================================
 * GPU Kernel: reads src into dst, writes index pattern into write_dst
 * ======================================================================== */

__global__ void gpuReadWrite(const int* src, int* write_dst, int* dst, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx];
    write_dst[idx] = (int)idx;
  }
}

/* ========================================================================
 * Memcpy wrapper: supports both sync and async modes
 * ======================================================================== */

static int DeviceMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind,
                        MemcpyMode mode) {
  if (mode == MEMCPY_ASYNC) {
    hipStream_t stream;
    HIP_CHECK_MPI(hipStreamCreate(&stream));
    HIP_CHECK_MPI(hipMemcpyAsync(dst, src, size, kind, stream));
    HIP_CHECK_MPI(hipStreamSynchronize(stream));
    HIP_CHECK_MPI(hipStreamDestroy(stream));
  } else {
    HIP_CHECK_MPI(hipMemcpy(dst, src, size, kind));
  }
  return 0;
}

/* ========================================================================
 * Data Verification
 * ======================================================================== */

static int VerifyData(const int* dst, const int* write_dst, const int* src_copy, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (dst[i] != src_copy[i]) {
      LogPrint(LOG_ERROR, "dst verification failed index:%zu dst:%d expected:%d\n", i, dst[i],
               src_copy[i]);
      return -1;
    }
  }
  LogPrint(LOG_DEBUG, "GPU copied data to dst successfully\n");

  for (size_t i = 0; i < size; i++) {
    if (write_dst[i] != (int)i) {
      LogPrint(LOG_ERROR, "write_dst verification failed index:%zu got:%d expected:%d\n", i,
               write_dst[i], (int)i);
      return -1;
    }
  }
  LogPrint(LOG_DEBUG, "GPU wrote data to write_dst successfully\n");
  return 0;
}

/* ========================================================================
 * VMM Allocation Management
 * ======================================================================== */

struct VmmAllocation {
  void* dptr;
  size_t size;
  hipMemGenericAllocationHandle_t memHandle;
  hipMemAllocationProp prop;
  size_t granularity;
};

static int VmmAllocAndMap(VmmAllocation* alloc, int deviceId,
                          hipMemAllocationHandleType handleType, size_t requestedSize) {
  memset(&alloc->prop, 0, sizeof(alloc->prop));
  alloc->prop.type = hipMemAllocationTypePinned;
  alloc->prop.location.type = hipMemLocationTypeDevice;
  alloc->prop.location.id = deviceId;
  alloc->prop.requestedHandleTypes = handleType;

  alloc->granularity = 0;
  HIP_CHECK_MPI(
      hipMemGetAllocationGranularity(&alloc->granularity, &alloc->prop,
                                     hipMemAllocationGranularityMinimum));

  alloc->size = AlignUp(requestedSize, alloc->granularity);
  LogPrint(LOG_DEBUG, "VMM alloc: requested=%zu aligned=%zu granularity=%zu handleType=0x%x\n",
           requestedSize, alloc->size, alloc->granularity, handleType);

  alloc->dptr = nullptr;
  HIP_CHECK_MPI(hipMemAddressReserve(&alloc->dptr, alloc->size, alloc->granularity, nullptr, 0));
  HIP_CHECK_MPI(hipMemCreate(&alloc->memHandle, alloc->size, &alloc->prop, 0));
  HIP_CHECK_MPI(hipMemMap(alloc->dptr, alloc->size, 0, alloc->memHandle, 0));

  hipMemAccessDesc accessDesc = {};
  accessDesc.location = alloc->prop.location;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK_MPI(hipMemSetAccess(alloc->dptr, alloc->size, &accessDesc, 1));

  return 0;
}

static int VmmReserveAndMapImported(VmmAllocation* alloc, int deviceId,
                                    hipMemGenericAllocationHandle_t importedHandle,
                                    size_t requestedSize) {
  memset(&alloc->prop, 0, sizeof(alloc->prop));
  alloc->prop.type = hipMemAllocationTypePinned;
  alloc->prop.location.type = hipMemLocationTypeDevice;
  alloc->prop.location.id = deviceId;
  alloc->prop.requestedHandleTypes = hipMemHandleTypeFabric;

  alloc->granularity = 0;
  HIP_CHECK_MPI(
      hipMemGetAllocationGranularity(&alloc->granularity, &alloc->prop,
                                     hipMemAllocationGranularityMinimum));

  // requestedSize is the exporter-aligned allocation size. Map exactly that
  // region so we never map beyond what the exported handle backs. The importer
  // granularity is used only as the VA-reservation alignment, not to re-size.
  alloc->size = requestedSize;
  alloc->memHandle = importedHandle;

  alloc->dptr = nullptr;
  HIP_CHECK_MPI(hipMemAddressReserve(&alloc->dptr, alloc->size, alloc->granularity, nullptr, 0));
  HIP_CHECK_MPI(hipMemMap(alloc->dptr, alloc->size, 0, alloc->memHandle, 0));

  hipMemAccessDesc accessDesc = {};
  accessDesc.location = alloc->prop.location;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK_MPI(hipMemSetAccess(alloc->dptr, alloc->size, &accessDesc, 1));

  LogPrint(LOG_DEBUG, "VMM import mapped: size=%zu dptr=%p\n", alloc->size, alloc->dptr);
  return 0;
}

static int VmmFree(VmmAllocation* alloc) {
  if (alloc->dptr) {
    HIP_CHECK_MPI(hipMemUnmap(alloc->dptr, alloc->size));
    HIP_CHECK_MPI(hipMemRelease(alloc->memHandle));
    HIP_CHECK_MPI(hipMemAddressFree(alloc->dptr, alloc->size));
    alloc->dptr = nullptr;
  }
  return 0;
}

/* ========================================================================
 * Initialize test data on host and copy to device
 * ======================================================================== */

static int InitTestData(void* dptr, const TestDataLayout& layout,
                        std::vector<int>& host_src_copy) {
  size_t bs = layout.bufferSize;
  host_src_copy.resize(bs);

  std::vector<char> host_buf(layout.total_size, 0);
  void* hbase = host_buf.data();

  unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)gRank;
  for (size_t i = 0; i < bs; i++) {
    int val = 1 + HipTest::RAND_R(&seed) % 100;
    layout.src(hbase)[i] = val;
    layout.src_copy(hbase)[i] = val;
    host_src_copy[i] = val;
  }

  HIP_CHECK_MPI(hipMemcpy(dptr, hbase, layout.total_size, hipMemcpyHostToDevice));
  return 0;
}

/* ========================================================================
 * RunKernelAndVerify
 *   When host_src_copy is nullptr (importer path), fetches src_copy from
 *   device memory. The fetch order relative to the kernel dispatch is
 *   controlled by config.accessOrder, and sync/async by config.memcpyMode.
 *   When host_src_copy is provided (local/exporter path), skips the fetch.
 * ======================================================================== */

static int RunKernelAndVerify(int deviceId, void* dptr, const TestDataLayout& layout,
                              const TestConfig& config, const int* host_src_copy = nullptr) {
  HIP_CHECK_MPI(hipSetDevice(deviceId));

  bool need_fetch = (host_src_copy == nullptr);
  std::vector<int> fetched_src_copy;
  if (need_fetch) fetched_src_copy.resize(layout.bufferSize);

  if (need_fetch && config.accessOrder == MEMCPY_BEFORE_KERNEL) {
    LogPrint(LOG_DEBUG, "Fetching src_copy from device (before kernel, %s)\n",
             config.memcpyMode == MEMCPY_ASYNC ? "async" : "sync");
    if (DeviceMemcpy(fetched_src_copy.data(), layout.src_copy(dptr),
                     layout.bufferSize * sizeof(int), hipMemcpyDeviceToHost, config.memcpyMode))
      return -1;
  }

  dim3 block(256);
  dim3 grid((layout.bufferSize + block.x - 1) / block.x);
  LogPrint(LOG_DEBUG, "Launching GPU kernel (device:%d bufferSize:%zu)\n", deviceId,
           layout.bufferSize);
  gpuReadWrite<<<grid, block>>>(layout.src(dptr), layout.write_dst(dptr), layout.dst(dptr),
                                layout.bufferSize);
  HIP_CHECK_MPI(hipGetLastError());
  HIP_CHECK_MPI(hipDeviceSynchronize());

  if (need_fetch && config.accessOrder == KERNEL_BEFORE_MEMCPY) {
    LogPrint(LOG_DEBUG, "Fetching src_copy from device (after kernel, %s)\n",
             config.memcpyMode == MEMCPY_ASYNC ? "async" : "sync");
    if (DeviceMemcpy(fetched_src_copy.data(), layout.src_copy(dptr),
                     layout.bufferSize * sizeof(int), hipMemcpyDeviceToHost, config.memcpyMode))
      return -1;
  }

  const int* verify_src = need_fetch ? fetched_src_copy.data() : host_src_copy;

  std::vector<int> h_dst(layout.bufferSize);
  std::vector<int> h_write_dst(layout.bufferSize);
  HIP_CHECK_MPI(hipMemcpy(h_dst.data(), layout.dst(dptr), layout.bufferSize * sizeof(int),
                           hipMemcpyDeviceToHost));
  HIP_CHECK_MPI(hipMemcpy(h_write_dst.data(), layout.write_dst(dptr),
                           layout.bufferSize * sizeof(int), hipMemcpyDeviceToHost));

  return VerifyData(h_dst.data(), h_write_dst.data(), verify_src, layout.bufferSize);
}

/* ========================================================================
 * VerifyDeviceResults
 *   Used by exporter to check shared memory after importer's kernel ran.
 * ======================================================================== */

static int VerifyDeviceResults(void* dptr, const TestDataLayout& layout,
                               const int* host_src_copy) {
  std::vector<int> h_dst(layout.bufferSize);
  std::vector<int> h_write_dst(layout.bufferSize);
  HIP_CHECK_MPI(hipMemcpy(h_dst.data(), layout.dst(dptr), layout.bufferSize * sizeof(int),
                           hipMemcpyDeviceToHost));
  HIP_CHECK_MPI(hipMemcpy(h_write_dst.data(), layout.write_dst(dptr),
                           layout.bufferSize * sizeof(int), hipMemcpyDeviceToHost));
  return VerifyData(h_dst.data(), h_write_dst.data(), host_src_copy, layout.bufferSize);
}

/* ========================================================================
 * Mode 1: Local self-test (world_size == 1)
 * ======================================================================== */

static int runLocal(int deviceId, const TestConfig& config) {
  TestDataLayout layout(config.bufferSize);
  LogPrint(LOG_INFO, "=== Local self-test (device:%d rank:%d %s) ===\n", deviceId, gRank,
           config.to_string().c_str());
  HIP_CHECK_MPI(hipSetDevice(deviceId));

  VmmAllocation alloc = {};
  if (VmmAllocAndMap(&alloc, deviceId, hipMemHandleTypeNone, layout.total_size)) return -1;

  std::vector<int> host_src_copy;
  if (InitTestData(alloc.dptr, layout, host_src_copy)) return -1;

  int ret = RunKernelAndVerify(deviceId, alloc.dptr, layout, config, host_src_copy.data());
  LogPrint(LOG_INFO, "Local test: %s\n", ret ? "FAILED" : "PASS");

  VmmFree(&alloc);
  return ret;
}

/* ========================================================================
 * Mode 2 & 3: Exporter
 *   Allocates VMM with fabric handle type, initialises data, exports the
 *   fabric handle and sends it to the partner (importer) via MPI. Waits
 *   for the importer's result, then verifies the shared memory from its
 *   own VA.
 * ======================================================================== */

static int runExporter(int deviceId, const TestConfig& config) {
  TestDataLayout layout(config.bufferSize);
  LogPrint(LOG_INFO, "=== Exporter (device:%d rank:%d partner:%d %s) ===\n", deviceId, gRank,
           gPartnerRank, config.to_string().c_str());
  HIP_CHECK_MPI(hipSetDevice(deviceId));

  VmmAllocation alloc = {};
  if (VmmAllocAndMap(&alloc, deviceId, hipMemHandleTypeFabric, layout.total_size)) return -1;

  std::vector<int> host_src_copy;
  if (InitTestData(alloc.dptr, layout, host_src_copy)) return -1;

  hipMemFabricHandle_t fabricHandle = {};
  LogPrint(LOG_DEBUG, "Exporting fabric handle\n");
  HIP_CHECK_MPI(hipMemExportToShareableHandle(&fabricHandle, alloc.memHandle,
                                              hipMemHandleTypeFabric, 0));

  LogPrint(LOG_DEBUG, "Sending fabric handle to importer [%s]\n",
           fabric_handle_str(fabricHandle));
  MPI_Send(&fabricHandle, sizeof(fabricHandle), MPI_UNSIGNED_CHAR, gPartnerRank, 0,
           MPI_COMM_WORLD);

  // Send the exporter-aligned allocation size so the importer maps exactly the
  // backed region. The importer's own granularity may differ; re-aligning
  // locally could exceed the exported allocation and make hipMemMap fail.
  uint64_t exported_size = alloc.size;
  MPI_Send(&exported_size, 1, MPI_UINT64_T, gPartnerRank, 0, MPI_COMM_WORLD);

  LogPrint(LOG_DEBUG, "Waiting for importer result\n");
  int remote_status = -1;
  MPI_Recv(&remote_status, 1, MPI_INT, gPartnerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  int ret = 0;
  if (remote_status == 0) {
    LogPrint(LOG_DEBUG, "Verifying local data updated by remote kernel\n");
    ret = VerifyDeviceResults(alloc.dptr, layout, host_src_copy.data());
    LogPrint(LOG_INFO, "Exporter verification: %s\n", ret ? "FAILED" : "PASS");
  } else {
    LogPrint(LOG_INFO, "Importer reported failure, skipping local verification\n");
    ret = -1;
  }

  VmmFree(&alloc);
  LogPrint(LOG_INFO, "Exporter exiting\n");
  return ret;
}

/* ========================================================================
 * Mode 2 & 3: Importer
 *   Receives the fabric handle from the partner (exporter) via MPI,
 *   imports it, maps to a new VA, runs the kernel, verifies, and sends
 *   the result back. config.accessOrder controls whether the src_copy
 *   fetch happens before or after the kernel, and config.memcpyMode
 *   selects sync vs async.
 * ======================================================================== */

static int runImporter(int deviceId, const TestConfig& config) {
  TestDataLayout layout(config.bufferSize);
  LogPrint(LOG_INFO, "=== Importer (device:%d rank:%d partner:%d %s) ===\n", deviceId, gRank,
           gPartnerRank, config.to_string().c_str());
  HIP_CHECK_MPI(hipSetDevice(deviceId));

  hipMemFabricHandle_t fabricHandle = {};
  LogPrint(LOG_DEBUG, "Waiting for fabric handle from exporter\n");
  MPI_Recv(&fabricHandle, sizeof(fabricHandle), MPI_UNSIGNED_CHAR, gPartnerRank, 0,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  LogPrint(LOG_DEBUG, "Received fabric handle [%s]\n", fabric_handle_str(fabricHandle));

  // Receive the exporter-aligned allocation size and map exactly that region,
  // rather than re-aligning total_size with the importer's own granularity.
  uint64_t exported_size = 0;
  MPI_Recv(&exported_size, 1, MPI_UINT64_T, gPartnerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  LogPrint(LOG_DEBUG, "Received exported allocation size: %zu\n",
           static_cast<size_t>(exported_size));

  hipMemGenericAllocationHandle_t memHandle;
  HIP_CHECK_MPI(
      hipMemImportFromShareableHandle(&memHandle, &fabricHandle, hipMemHandleTypeFabric));

  VmmAllocation alloc = {};
  if (VmmReserveAndMapImported(&alloc, deviceId, memHandle,
                               static_cast<size_t>(exported_size)))
    return -1;

  int test_result = RunKernelAndVerify(deviceId, alloc.dptr, layout, config);
  LogPrint(LOG_INFO, "Importer verification: %s\n", test_result ? "FAILED" : "PASS");

  LogPrint(LOG_DEBUG, "Sending result to exporter\n");
  MPI_Send(&test_result, 1, MPI_INT, gPartnerRank, 0, MPI_COMM_WORLD);

  VmmFree(&alloc);
  LogPrint(LOG_INFO, "Importer exiting\n");
  return test_result;
}

/* ========================================================================
 * Multi-host Detection
 * ======================================================================== */

static bool detect_multihost(int world_rank, int world_size) {
  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL,
                      &node_comm);
  int size_of_node;
  MPI_Comm_size(node_comm, &size_of_node);
  MPI_Comm_free(&node_comm);
  return (world_size > size_of_node);
}

/* ========================================================================
 * MPI Environment Init
 *
 *   1. Catch2 owns main(), so we pass nullptr instead of argc/argv
 *      (valid per MPI-2).
 *   2. MPI_Finalize is registered via std::atexit since there is no
 *      explicit finalize point in a Catch2 binary.
 *
 * A static guard ensures this runs at most once across test cases.
 * ======================================================================== */

static int mpi_init_env() {
  static bool done = false;
  if (done) return 0;
  done = true;

  MPI_Init(nullptr, nullptr);
  std::atexit([]() { MPI_Finalize(); });

  MPI_Comm_size(MPI_COMM_WORLD, &gWorldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &gRank);

  int name_len;
  MPI_Get_processor_name(gProcessorName, &name_len);

  if (hipGetDeviceCount(&gNumGpus) != hipSuccess || gNumGpus == 0) {
    LogPrint(LOG_ERROR, "No HIP GPUs found\n");
    return -1;
  }

  gMultiHost = detect_multihost(gRank, gWorldSize);

  LogPrint(LOG_INFO, "Processor:%s rank:%d world_size:%d GPUs:%d multi_host:%d\n",
           gProcessorName, gRank, gWorldSize, gNumGpus, gMultiHost);
  return 0;
}

/* ========================================================================
 * Test Runner
 *   Selects mode based on world_size and topology, then dispatches to
 *   runLocal / runExporter / runImporter with the given config.
 * ======================================================================== */

static int test_runner(const TestConfig& config) {
  int ret = 0;

  if (gWorldSize == 1) {
    gPartnerRank = -1;
    ret = runLocal(0, config);

  } else if (!gMultiHost) {
    int gpuIndex = gRank % gNumGpus;
    if (!(gRank & 0x1)) {
      gPartnerRank = gRank + 1;
      ret = runExporter(gpuIndex, config);
    } else {
      gPartnerRank = gRank - 1;
      ret = runImporter(gpuIndex, config);
    }
  } else {
    if (!(gRank & 0x1)) {
      gPartnerRank = gRank + 1;
      int gpuIndex = (gRank / 2) % gNumGpus;
      ret = runExporter(gpuIndex, config);
    } else {
      gPartnerRank = gRank - 1;
      int gpuIndex = ((gRank - 1) / 2) % gNumGpus;
      ret = runImporter(gpuIndex, config);
    }
  }

  return ret;
}

/* ========================================================================
 * Test 1: Basic fabric handle test
 * ========================================================================
 * Test Description
 * ------------------------
 *    - Basic fabric handle round-trip with default sync memcpy and
 *      memcpy-before-kernel ordering. With 1 rank: local VMM self-test.
 *      With 2+ ranks: exporter sends fabric handle to importer via MPI,
 *      importer maps it and runs a GPU kernel, both sides verify.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemFabricHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - Device supports VMM and fabric handles (multi-rank)
 *    - MPI runtime available
 *    - HIP_VERSION >= 7.13
 */
HIP_TEST_CASE(Unit_hipExportToFabricHandle_Basic) {
  REQUIRE(mpi_init_env() == 0);
  if (gWorldSize > 1 && (gWorldSize % 2) != 0) {
    HIP_SKIP_TEST("This test requires an even MPI world size (rank pairs).");
  }
  checkVMMSupported(0);
  if (gWorldSize > 1) {
    checkFabricHandleSupported(0);
  }
  REQUIRE(test_runner({1024, MEMCPY_BEFORE_KERNEL, MEMCPY_SYNC}) == 0);
}

/* ========================================================================
 * Test 2: Stress test -- buffer-size sweep x access-order x memcpy-mode
 * ========================================================================
 * Test Description
 * ------------------------
 *    - Sweeps over multiple buffer sizes.  For each size, four dynamic
 *      sections cover every combination of {sync, async} memcpy mode and
 *      {memcpy-before-kernel, kernel-before-memcpy} access ordering.
 *      The full MPI export/import path is exercised in each section.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemFabricHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - Device supports VMM and fabric handles (multi-rank)
 *    - MPI runtime available
 *    - HIP_VERSION >= 7.13
 */
HIP_TEST_CASE(Unit_hipExportToFabricHandle_Stress) {
  REQUIRE(mpi_init_env() == 0);

  if (gWorldSize > 1 && (gWorldSize % 2) != 0) {
    HIP_SKIP_TEST("This test requires an even MPI world size (rank pairs).");
  }
  checkVMMSupported(0);
  if (gWorldSize > 1) {
    checkFabricHandleSupported(0);
  }

  size_t bufferSize = GENERATE(1024, 4096, 8192, 32768, 65536);

  SECTION("SyncMemcpy_MemcpyBeforeKernel") {
    REQUIRE(test_runner({bufferSize, MEMCPY_BEFORE_KERNEL, MEMCPY_SYNC}) == 0);
  }

  SECTION("SyncMemcpy_KernelBeforeMemcpy") {
    REQUIRE(test_runner({bufferSize, KERNEL_BEFORE_MEMCPY, MEMCPY_SYNC}) == 0);
  }

  SECTION("AsyncMemcpy_MemcpyBeforeKernel") {
    REQUIRE(test_runner({bufferSize, MEMCPY_BEFORE_KERNEL, MEMCPY_ASYNC}) == 0);
  }

  SECTION("AsyncMemcpy_KernelBeforeMemcpy") {
    REQUIRE(test_runner({bufferSize, KERNEL_BEFORE_MEMCPY, MEMCPY_ASYNC}) == 0);
  }
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */