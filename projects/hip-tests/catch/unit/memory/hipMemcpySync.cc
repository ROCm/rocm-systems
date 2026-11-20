/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "MemUtils.hh"
#include <memcpy1d_tests_common.hh>
/*
 * These testcases verify that synchronization behaviour for memcpy functions with respect to
 * the host.
 */

using namespace mem_utils;

// value used for memset operations
constexpr int testValue = 0x11;


/*
 * Set of helper functions handling the different cases for memcpy
 */

static inline hipMemcpyKind getMemcpyType(allocType type, bool fromHost) {
  // This pointer is a device alias of registered host storage. Let every
  // dimensional copy API infer its memory type from the allocation.
  if (type == allocType::devRegistered) {
    return hipMemcpyDefault;
  }
  if (fromHost) {
    switch (type) {
      case allocType::deviceMalloc:
        return hipMemcpyHostToDevice;
        break;
      default:  // host
        return hipMemcpyHostToHost;
        break;
    }
  } else {
    switch (type) {
      case allocType::deviceMalloc:
        return hipMemcpyDeviceToDevice;
        break;
      default:  // host
        return hipMemcpyDeviceToHost;
        break;
    }
  }
}

static inline hipError_t memcpyCheck(allocType type, memType memType, char* aPtr, MultiDData& data,
                                     char* fillerData, bool async, hipStream_t stream,
                                     bool fromHost) {
  auto cpyType = getMemcpyType(type, fromHost);
  auto sizeInBytes = data.pitch * data.getH() * data.getD() * sizeof(char);
  switch (memType) {
    case memType::hipMem:
      return async ? hipMemcpyAsync(aPtr + data.offset, fillerData, sizeInBytes, cpyType, stream)
                   : hipMemcpy(aPtr + data.offset, fillerData, sizeInBytes, cpyType);
    case memType::hipMem2D:
      return async ? hipMemcpy2DAsync(aPtr + data.offset, data.pitch, fillerData, sizeInBytes,
                                      data.width, data.getH(), cpyType, stream)
                   : hipMemcpy2D(aPtr + data.offset, data.pitch, fillerData, sizeInBytes,
                                 data.width, data.getH(), cpyType);
    case memType::hipMem3D: {
      hipMemcpy3DParms params{};
      params.kind = cpyType;
      params.srcPos = make_hipPos(0, 0, 0);
      params.dstPos = make_hipPos(data.offset, data.offset, data.offset);
      params.srcPtr = make_hipPitchedPtr(fillerData, data.width, data.width, data.getH());
      params.dstPtr = make_hipPitchedPtr(aPtr, data.pitch, data.width, data.getH());
      params.extent = make_hipExtent(data.width * sizeof(char), data.getH(), data.getD());
      return async ? hipMemcpy3DAsync(&params, stream) : hipMemcpy3D(&params);
    }
    default:
      return hipErrorInvalidValue;
  }
}

static inline char* createFillerData(size_t count, size_t value, bool fromHost) {
  if (fromHost) {
    char* fillerData = new char[count];
    std::fill(fillerData, fillerData + count, value);
    return fillerData;
  } else {
    char* fillerData;
    HIP_CHECK(hipMalloc(&fillerData, count * sizeof(char)));
    HIP_CHECK(hipMemset(fillerData, value, count * sizeof(char)));
    return fillerData;
  }
}

// Helper function to run tests for hipMemset allocation types
static void runMemcpyTests(hipStream_t stream, bool async, allocType type, memType memType,
                           MultiDData data) {
  bool fromHost = GENERATE(true, false);

  std::pair<char*, char*> aPtr = initMemory<char>(type, memType, data);
  size_t sizeInBytes = data.getCount();

  // filler data for device memory created beforehand as it uses memset
  // which might interfere with synchronization testing
  auto fillerData = createFillerData(sizeInBytes, testValue, fromHost);
  CAPTURE(type, memType, data.width, data.height, data.depth, stream, async, fromHost, sizeInBytes);

  // The source is pageable when fromHost is true, so HIP synchronizes even
  // for the Async APIs. Device-to-pinned-host copies synchronize only for the
  // synchronous APIs; device-to-device copies do not require a host wait.
  const bool shouldSync = fromHost || (type != allocType::deviceMalloc && !async);
  MemcpySyncBehaviorCheck(
      [&] {
        return memcpyCheck(type, memType, aPtr.first, data, fillerData, async, stream, fromHost);
      },
      shouldSync, stream);
  // verify
  HIP_CHECK(hipStreamSynchronize(stream));
  verifyData(aPtr.first, testValue, data, type, memType);
  if (type == allocType::devRegistered) {
    freeStuff(aPtr.second, type);
  } else {
    freeStuff(aPtr.first, type);
  }
  if (fromHost) {
    delete[] fillerData;
  } else {
    HIP_CHECK(hipFree(fillerData));
  }
}

#if HT_AMD  // These cases check HIP synchronization rules for pageable host memory.

HIP_TEST_CASE(Unit_hipMemcpySync) {
  allocType type = GENERATE(allocType::deviceMalloc, allocType::hostMalloc, allocType::hostRegisted,
                            allocType::devRegistered);
  memType memcpy_type = memType::hipMem;
  MultiDData data;
  data.width = 1;

  doMemTest<char>(runMemcpyTests, type, memcpy_type, data);
}

HIP_TEST_CASE(Unit_hipMemcpy2DSync) {
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);

  memType memcpy_type = memType::hipMem2D;
  MultiDData data;
  data.width = 1;
  data.height = 1;

  doMemTest<char>(runMemcpyTests, mallocType, memcpy_type, data);
}

HIP_TEST_CASE(Unit_hipMemcpy3DSync) {
  allocType mallocType = GENERATE(allocType::deviceMalloc, allocType::hostMalloc,
                                  allocType::hostRegisted, allocType::devRegistered);

  memType memcpy_type = memType::hipMem3D;
  MultiDData data;
  data.width = 1;
  data.height = 1;
  data.depth = 1;

  doMemTest<char>(runMemcpyTests, mallocType, memcpy_type, data);
}

#endif
