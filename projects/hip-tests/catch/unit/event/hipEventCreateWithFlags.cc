/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include <stdlib.h>

constexpr size_t buffer_size = (1024 * 1024);
static int test_iteration_hstvismem() { return isQuickLevel() ? 3 : 5; }
static int test_iteration_noncohmem() { return isQuickLevel() ? 5 : 10; }
constexpr int block_size = 512;

// Atomic store required as events are created with special flag hipEventDisableSystemFence [Ref : SWDEV-523177]
template <typename T> __global__ void vector_square_system_scope_atomic(const T* A_d, T* C_d, size_t N_ELMTS) {
  size_t i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N_ELMTS) {
#if HT_AMD
    T result = A_d[i] * A_d[i];
    __hip_atomic_store(&C_d[i], result, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    C_d[i] = A_d[i] * A_d[i];
#endif
  }
}

/**
 * @addtogroup hipEventCreateWithFlags hipEventCreateWithFlags
 * @{
 * @ingroup EventTest
 * `hipEventCreateWithFlags (hipEvent_t *event, unsigned flags)` -
 * begins graph capture on a stream
 */

/**
 * Test Description
 * ------------------------
 *    - Test simple event creation with hipEventCreateWithFlags api for each flag
 * Test source
 * ------------------------
 *    - catch\unit\event\hipEventCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipEventCreateWithFlags_Positive) {
#if HT_AMD
  const unsigned int flagUnderTest =
      GENERATE(hipEventDefault, hipEventBlockingSync, hipEventDisableTiming,
               hipEventInterprocess | hipEventDisableTiming, hipEventReleaseToDevice,
               hipEventReleaseToSystem);
#else
  // On Non-AMD platforms hipEventReleaseToDevice / hipEventReleaseToSystem
  // are not defined.
  const unsigned int flagUnderTest =
      GENERATE(hipEventDefault, hipEventBlockingSync, hipEventDisableTiming,
               hipEventInterprocess | hipEventDisableTiming);
#endif

  hipEvent_t event;
  HIP_CHECK(hipEventCreateWithFlags(&event, flagUnderTest));
  REQUIRE(event != nullptr);
  HIP_CHECK(hipEventDestroy(event));
}

/**
Since flags hipEventReleaseToSystem, hipEventDisableSystemFence and hipEventReleaseToDevice
are AMD specific flags, hence the following tests enabled only for AMD.
*/
#if HT_AMD
enum class eSyncToTest {
  eStreamSynchronize,
  eDeviceSynchronize,
  eStreamWaitEvent,
  eEventSynchronize
};

enum class eMemoryToTest { eHostVisibleMemory, eNonCoherentHostMemory, eCoherentHostMemory };

static void init_input(int* a, size_t size) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < size; i++) {
    a[i] = (HipTest::RAND_R(&seed) & 0xFF);
  }
}

static void check_output(int* inp, int* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    REQUIRE(out[i] == (inp[i] * inp[i]));
  }
}
// local function
static void testMemCoherency(eSyncToTest test, eMemoryToTest mem, uint32_t flags) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  // If the GPU is not large bar then exit the test
  if (prop.isLargeBar != 1) {
    HIP_SKIP_TEST("large BAR (resizable BAR) is not supported on this device.");
  }
  constexpr auto blocksPerCU = 6;
  unsigned grid_size = HipTest::setNumBlocks(blocksPerCU, block_size, buffer_size);
  hipEvent_t event;
  HIP_CHECK(hipEventCreateWithFlags(&event, flags));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, 0x0));
  int *ibuf_h, *buf_d;
  ibuf_h = new int[buffer_size];
  REQUIRE(ibuf_h != nullptr);
  int total_iter = 0;
  if (mem == eMemoryToTest::eHostVisibleMemory) {
    HIP_CHECK(hipMalloc(&buf_d, buffer_size * sizeof(int)));
    total_iter = test_iteration_hstvismem();
  } else if (mem == eMemoryToTest::eNonCoherentHostMemory) {
    HIP_CHECK(hipHostMalloc(&buf_d, buffer_size * sizeof(int), hipHostMallocNonCoherent));
    total_iter = test_iteration_noncohmem();
  } else if (mem == eMemoryToTest::eCoherentHostMemory) {
    HIP_CHECK(hipHostMalloc(&buf_d, buffer_size * sizeof(int), hipHostMallocCoherent));
    total_iter = test_iteration_noncohmem();
  }
  for (int iter = 0; iter < total_iter; iter++) {
    // Inititalize the buffer with random data
    init_input(ibuf_h, buffer_size);
    HIP_CHECK(hipMemcpy(buf_d, ibuf_h, sizeof(int) * buffer_size, hipMemcpyDefault));
    if (flags & hipEventDisableSystemFence) {
      vector_square_system_scope_atomic<int><<<grid_size, block_size, 0, stream>>>(buf_d, buf_d, buffer_size);
    } else {
      HipTest::vector_square<int><<<grid_size, block_size, 0, stream>>>(buf_d, buf_d, buffer_size);
    }
    HIP_CHECK(hipEventRecord(event, stream));
    // test different synchronization APIs
    if (test == eSyncToTest::eStreamSynchronize) {
      HIP_CHECK(hipStreamSynchronize(stream));
    } else if (test == eSyncToTest::eDeviceSynchronize) {
      HIP_CHECK(hipDeviceSynchronize());
    } else if (test == eSyncToTest::eEventSynchronize) {
      HIP_CHECK(hipEventSynchronize(event));
    } else if (test == eSyncToTest::eStreamWaitEvent) {
      HIP_CHECK(hipStreamWaitEvent(stream, event, 0));
    }
    check_output(ibuf_h, buf_d, buffer_size);
  }
  delete[] ibuf_h;
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(event));
  if (mem == eMemoryToTest::eHostVisibleMemory) {
    HIP_CHECK(hipFree(buf_d));
  } else if ((mem == eMemoryToTest::eNonCoherentHostMemory) ||
             (mem == eMemoryToTest::eCoherentHostMemory)) {
    HIP_CHECK(hipHostFree(buf_d));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Check Synchronization effect on Host Visible Memory.
 * Disable System fence when creating an event. Create a chunk of Host Visisble
 * Memory using hipMalloc and initialize the memory with user data. Launch a
 * kernel that writes to this memory location. Perform different synchronizations
 * and validate that updated values are seen from host.
 * Test source
 * ------------------------
 *    - catch\unit\event\hipEventCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipEventCreateWithFlags_DisableSystemFence_HstVisMem) {
  SECTION("Check with hipStreamSynchronize") {
    INFO("Check with hipStreamSynchronize");
    testMemCoherency(eSyncToTest::eStreamSynchronize, eMemoryToTest::eHostVisibleMemory,
                     hipEventDisableSystemFence);
  }
  SECTION("Check with hipDeviceSynchronize") {
    INFO("Check with hipDeviceSynchronize");
    testMemCoherency(eSyncToTest::eDeviceSynchronize, eMemoryToTest::eHostVisibleMemory,
                     hipEventDisableSystemFence);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Check Synchronization effect on Host Visible Memory.
 * Use Default Flag when creating an event. Create a chunk of Host Visisble
 * Memory using hipMalloc and initialize the memory with user data. Launch
 * a kernel that writes to this memory location. Perform different synchronizations
 * and validate that updated values are seen from host.
 * ------------------------
 *    - catch\unit\event\hipEventCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipEventCreateWithFlags_DefaultFlg_HstVisMem) {
  SECTION("Check with hipStreamSynchronize") {
    INFO("Check with hipStreamSynchronize");
    testMemCoherency(eSyncToTest::eStreamSynchronize, eMemoryToTest::eHostVisibleMemory,
                     hipEventDefault);
  }
  SECTION("Check with hipDeviceSynchronize") {
    INFO("Check with hipDeviceSynchronize");
    testMemCoherency(eSyncToTest::eDeviceSynchronize, eMemoryToTest::eHostVisibleMemory,
                     hipEventDefault);
  }
  SECTION("Check with hipEventSynchronize") {
    INFO("Check with hipEventSynchronize");
    testMemCoherency(eSyncToTest::eEventSynchronize, eMemoryToTest::eHostVisibleMemory,
                     hipEventDefault);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Check Synchronization effect on Non Coherent Host Memory.
 * Disable System fence when creating an event. Create a chunk of Host Visisble
 * Memory using hipMalloc and initialize the memory with user data. Launch a
 * kernel that writes to this memory location. Perform different synchronizations
 * and validate that updated values are seen from host.
 * ------------------------
 *    - catch\unit\event\hipEventCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipEventCreateWithFlags_DisableSystemFence_NonCohHstMem) {
  SECTION("Check with hipStreamSynchronize") {
    INFO("Check with hipStreamSynchronize");
    testMemCoherency(eSyncToTest::eStreamSynchronize, eMemoryToTest::eNonCoherentHostMemory,
                     hipEventDisableSystemFence);
  }
  SECTION("Check with hipDeviceSynchronize") {
    INFO("Check with hipDeviceSynchronize");
    testMemCoherency(eSyncToTest::eDeviceSynchronize, eMemoryToTest::eNonCoherentHostMemory,
                     hipEventDisableSystemFence);
  }
}


/**
 * Test Description
 * ------------------------
 *    - Check Synchronization effect on Non Coherent Host Memory.
 * Use Default Flag when creating an event. Create a chunk of Host Visisble
 * Memory using hipMalloc and initialize the memory with user data. Launch
 * a kernel that writes to this memory location. Perform different
 * synchronizations and validate that updated values are seen from host.
 * ------------------------
 *    - catch\unit\event\hipEventCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipEventCreateWithFlags_DefaultFlg_NonCohHstMem) {
  SECTION("Check with hipStreamSynchronize") {
    INFO("Check with hipStreamSynchronize");
    testMemCoherency(eSyncToTest::eStreamSynchronize, eMemoryToTest::eNonCoherentHostMemory,
                     hipEventDefault);
  }
  SECTION("Check with hipDeviceSynchronize") {
    INFO("Check with hipDeviceSynchronize");
    testMemCoherency(eSyncToTest::eDeviceSynchronize, eMemoryToTest::eNonCoherentHostMemory,
                     hipEventDefault);
  }
  SECTION("Check with hipEventSynchronize") {
    INFO("Check with hipEventSynchronize");
    testMemCoherency(eSyncToTest::eEventSynchronize, eMemoryToTest::eNonCoherentHostMemory,
                     hipEventDefault);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Check Synchronization effect on Coherent Host Memory.
 * Disable System fence when creating an event. Create a chunk of Host Visisble
 * Memory using hipMalloc and initialize the memory with user data. Launch a
 * kernel that writes to this memory location. Perform different synchronizations
 * and validate that updated values are seen from host.
 * ------------------------
 *    - catch\unit\event\hipEventCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipEventCreateWithFlags_DisableSystemFence_CohHstMem) {
  SECTION("Check with hipStreamSynchronize") {
    INFO("Check with hipStreamSynchronize");
    testMemCoherency(eSyncToTest::eStreamSynchronize, eMemoryToTest::eCoherentHostMemory,
                     hipEventDisableSystemFence);
  }
  SECTION("Check with hipDeviceSynchronize") {
    INFO("Check with hipDeviceSynchronize");
    testMemCoherency(eSyncToTest::eDeviceSynchronize, eMemoryToTest::eCoherentHostMemory,
                     hipEventDisableSystemFence);
  }
  SECTION("Check with hipEventSynchronize") {
    INFO("Check with hipEventSynchronize");
    testMemCoherency(eSyncToTest::eEventSynchronize, eMemoryToTest::eCoherentHostMemory,
                     hipEventDisableSystemFence);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Check Synchronization effect on Coherent Host Memory.
 * Use Default Flag when creating an event. Create a chunk of Host Visisble
 * Memory using hipMalloc and initialize the memory with user data. Launch a
 * kernel that writes to this memory location. Perform different synchronizations
 * and validate that updated values are seen from host.
 * ------------------------
 *    - catch\unit\event\hipEventCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipEventCreateWithFlags_DefaultFlg_CohHstMem) {
  SECTION("Check with hipStreamSynchronize") {
    INFO("Check with hipStreamSynchronize");
    testMemCoherency(eSyncToTest::eStreamSynchronize, eMemoryToTest::eCoherentHostMemory,
                     hipEventDefault);
  }
  SECTION("Check with hipDeviceSynchronize") {
    INFO("Check with hipDeviceSynchronize");
    testMemCoherency(eSyncToTest::eDeviceSynchronize, eMemoryToTest::eCoherentHostMemory,
                     hipEventDefault);
  }
  SECTION("Check with hipEventSynchronize") {
    INFO("Check with hipEventSynchronize");
    testMemCoherency(eSyncToTest::eEventSynchronize, eMemoryToTest::eCoherentHostMemory,
                     hipEventDefault);
  }
}
#endif

/**
 * Test Description
 * ------------------------
 *    - Test event creation with hipEventCreateWithFlags while stream is capturing
 * Test source
 * ------------------------
 *    - catch\unit\event\hipEventCreateWithFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipEventCreateWithFlags_Verify_Capture) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipStreamCaptureMode mode = GENERATE(hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal,
                                       hipStreamCaptureModeRelaxed);
  HIP_CHECK(hipStreamBeginCapture(stream, mode));

  const unsigned int flags = GENERATE(hipEventDefault, hipEventBlockingSync, hipEventDisableTiming,
                                      hipEventInterprocess | hipEventDisableTiming);
  hipEvent_t event;
  HIP_CHECK(hipEventCreateWithFlags(&event, flags));
  REQUIRE(event != nullptr);
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group hipEventCreateWithFlags.
 * @}
 */
