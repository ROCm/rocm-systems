/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */


#include <hip_test_common.hh>
#include "hipArrayCommon.hh"
#include "utils.hh"
#include <array>


/*
 * hipFreeMipmappedArray API test scenarios
 * 1. Check that hipFreeMipmappedArray implicitly synchronises the device.
 * 2. Perform multiple allocations and then call hipFreeMipmappedArray on each pointer concurrently
 * (from unique threads) for different memory types and different allocation sizes.
 * 3. Pass nullptr as argument and check that correct error code is returned.
 * 4. Call hipFreeMipmappedArray twice on the same pointer and check that the implementation handles
 * the second call correctly.
 */
TEMPLATE_TEST_CASE(Unit_hipFreeMipmappedArrayImplicitSyncArray, char, float) {
  hipMipmappedArray_t arrayPtr{};
  hipExtent extent{};
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();
  hipDeviceProp_t props;
  std::array<unsigned int, 3> levels = {1, 5, 7};

#if HT_AMD
  const unsigned int flags = hipArrayDefault;
#else
  const unsigned int flags = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore);
#endif

  extent.width = GENERATE(64, 256, 1024);
  extent.height = GENERATE(64, 256, 1024);
  extent.depth = GENERATE(0, 64, 256, 1024);

  HIP_CHECK(hipGetDeviceProperties(&props, 0))

  for (auto numLevels : levels) {
    INFO(" extent: (" << extent.width << ", " << extent.height << ", " << extent.depth << ") and "
                      << numLevels << " levels. Total VRAM: " << props.totalGlobalMem);
    if (extent.width * extent.height * extent.depth * numLevels * sizeof(TestType) >
        props.totalGlobalMem / 2) {
      // some devices will not have enough memory allocate the 6GB required for the biggest extent
      // We would skip the test if the extent would require more than half of the global memory.
      // Note that totalGlobalMem is not an exact measurement of the available memory for
      // compute and we cannot use it as an exact value, so we use half
      // (we use SUCCEED as no warning is needed)
      SUCCEED(
          "Device might not have enough global memory to allocate a mipmapped array using this "
          "extent; "
          "test will not be run. Total global memory: "
          << props.totalGlobalMem);
      continue;
    }

    HIP_CHECK_IGNORED_RETURN(hipMallocMipmappedArray(&arrayPtr, &desc, extent, numLevels, flags),
                             hipErrorNotSupported);

    LaunchDelayKernel(std::chrono::milliseconds{50}, nullptr);
    // make sure device is busy
    HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
    HIP_CHECK(hipFreeMipmappedArray(arrayPtr));
    HIP_CHECK(hipStreamQuery(nullptr));
  }
}

TEST_CASE(Unit_hipFreeMipmappedArray_Negative_Nullptr) {
#if HT_AMD
  HIP_CHECK_ERROR(hipFreeMipmappedArray(nullptr), hipErrorInvalidValue);
#else
  HIP_CHECK(hipFreeMipmappedArray(nullptr));
#endif
}

TEMPLATE_TEST_CASE(Unit_hipFreeMipmappedArrayMultiTArray, char, int) {
  constexpr size_t numAllocs = 10;
  std::vector<std::thread> threads;
  std::vector<hipMipmappedArray_t> ptrs(numAllocs);
  hipExtent extent{};
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  const unsigned int numLevels = GENERATE(1, 5, 7);

#if HT_AMD
  const unsigned int flags = hipArrayDefault;
#else
  const unsigned int flags = GENERATE(hipArrayDefault, hipArraySurfaceLoadStore);
#endif

  extent.width = GENERATE(64, 256, 1024);
  extent.height = GENERATE(64, 256, 1024);
  extent.depth = GENERATE(0, 64, 256, 1024);

  int i = 0;
  for (; i < ptrs.size(); i++) {
    if (hipErrorOutOfMemory == hipMallocMipmappedArray(&ptrs[i], &desc, extent, numLevels, flags)) {
      break;
    }
  }

  for (int j = 0; j < i; j++) {
    threads.emplace_back([ptrs, j] {
      if (hipSuccess != hipFreeMipmappedArray(ptrs[j])) {
        return;
      }
      if (hipSuccess != hipStreamQuery(nullptr)) {
        return;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  HIP_CHECK_THREAD_FINALIZE();
}
