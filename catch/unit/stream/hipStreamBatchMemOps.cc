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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include <hip_test_common.hh>
#include <random>

#if CUDA_VERSION >= CUDA_12000

#define ARRAY_SIZE 8

static hipStreamBatchMemOpParams paramArray[ARRAY_SIZE + ARRAY_SIZE / 2];

/**
 * @brief This test verifies the functionality of the hipStreamBatchMemOp API by performing
 * batched memory operations in a HIP stream. It tests 32-bit and 64-bit wait conditions
 * (hipStreamWaitValueEq, hipStreamWaitValueGte, hipStreamWaitValueAnd, hipStreamWaitValueNor)
 * and write operations. The test releases wait conditions asynchronously using a separate stream,
 * then verifies that the operations were correctly executed and synchronized by reading back the
 * memory values and checking against the expected conditions.
 *
 **/

TEST_CASE("Unit_hipStreamBatchMemOp_functional") {

  hipStream_t stream, releaseStream;

  // Create a HIP stream for batch operations
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamDefault));
  INFO("Main stream created.");

  // Create a separate stream for releasing waits
  HIP_CHECK(hipStreamCreateWithFlags(&releaseStream, hipStreamDefault));
  INFO("Release stream created.");

  // Allocate memory for the addresses and initialize the array
  std::vector<hipDeviceptr_t> waitAddrs(ARRAY_SIZE);
  std::vector<hipDeviceptr_t> writeAddrs(ARRAY_SIZE / 2);

  // Allocate memory for wait operations
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    if (i % 2 == 0) {
        HIP_CHECK(hipMalloc((void**)&waitAddrs[i], sizeof(uint32_t)));
        INFO("Allocated 32-bit waitAddr[" << i << "] at address " << (void*)waitAddrs[i]);
    } else {
        HIP_CHECK(hipMalloc((void**)&waitAddrs[i], sizeof(uint64_t)));
        INFO("Allocated 64-bit waitAddr[" << i << "] at address " << (void*)waitAddrs[i]);
    }
}

  // Allocate memory for write operations
  for (int i = 0; i < ARRAY_SIZE / 2; ++i) {
    if (i % 2 == 0) {
      HIP_CHECK(hipMalloc((void**)&writeAddrs[i], sizeof(uint32_t)));
      INFO("Allocated 32-bit writeAddr[" << i << "] at address " << (void*)writeAddrs[i]);
    } else {
      HIP_CHECK(hipMalloc((void**)&writeAddrs[i], sizeof(uint64_t)));
      INFO("Allocated 64-bit writeAddr[" << i << "] at address " << (void*)writeAddrs[i]);
    }
  }

  INFO("Memory allocated for wait and write operations.");

  // Random number generator for test values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dist32(1, UINT32_MAX); // Avoid zero
  std::uniform_int_distribution<uint64_t> dist64(1, UINT64_MAX); // Avoid zero

  // Initialize the array with wait operations covering all four flags
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    int flagIndex = i % 4;
    if (i % 2 == 0) {
      // Even indices: 32-bit wait operations
      paramArray[i].operation = hipStreamMemOpWaitValue32;
      paramArray[i].waitValue.address = waitAddrs[i];
      paramArray[i].waitValue.value = dist32(gen);
      switch (flagIndex) {
        case 0:
          paramArray[i].waitValue.flags = hipStreamWaitValueEq;
          break;
        case 1:
          paramArray[i].waitValue.flags = hipStreamWaitValueGte;
          break;
        case 2:
          paramArray[i].waitValue.flags = hipStreamWaitValueAnd;
          break;
        case 3:
          paramArray[i].waitValue.flags = hipStreamWaitValueNor;
          break;
        }
        paramArray[i].waitValue.alias = 0;
        INFO("Initialized wait operation [" << i << "] (32-bit): address=" << (void*)waitAddrs[i]
                 << ", value=" << paramArray[i].waitValue.value
                 << ", flag=" << paramArray[i].waitValue.flags);
        } else {
          // Odd indices: 64-bit wait operations
          paramArray[i].operation = hipStreamMemOpWaitValue64;
          paramArray[i].waitValue.address = waitAddrs[i];
          paramArray[i].waitValue.value64 = dist64(gen);
          switch (flagIndex) {
            case 0:
              paramArray[i].waitValue.flags = hipStreamWaitValueEq;
              break;
              case 1:
                paramArray[i].waitValue.flags = hipStreamWaitValueGte;
                break;
              case 2:
                paramArray[i].waitValue.flags = hipStreamWaitValueAnd;
                break;
              case 3:
                paramArray[i].waitValue.flags = hipStreamWaitValueNor;
                break;
          }
          paramArray[i].waitValue.alias = 0;
          INFO("Initialized wait operation [" << i << "] (64-bit): address=" << (void*)waitAddrs[i]
                 << ", value=" << paramArray[i].waitValue.value64
                 << ", flag=" << paramArray[i].waitValue.flags);
        }
  }

  // Initialize the array with write operations
  int totalOps = ARRAY_SIZE + ARRAY_SIZE / 2;  // Total operations including write operations
  for (int i = 0; i < ARRAY_SIZE / 2; ++i) {
    int idx = ARRAY_SIZE + i;  // Index in paramArray for write operations
    if (i % 2 == 0) {
      // Even indices: 32-bit write operations
      paramArray[idx].operation = hipStreamMemOpWriteValue32;
      paramArray[idx].writeValue.address = writeAddrs[i];
      paramArray[idx].writeValue.value = dist32(gen);
      paramArray[idx].writeValue.flags = 0x0;
      paramArray[idx].writeValue.alias = 0;
      INFO("Initialized write operation [" << idx << "] (32-bit): address=" << (void*)writeAddrs[i]
                 << ", value=" << paramArray[idx].writeValue.value);
      } else {
        // Odd indices: 64-bit write operations
        paramArray[idx].operation = hipStreamMemOpWriteValue64;
        paramArray[idx].writeValue.address = writeAddrs[i];
        paramArray[idx].writeValue.value64 = dist64(gen);
        paramArray[idx].writeValue.flags = 0x0;
        paramArray[idx].writeValue.alias = 0;
        INFO("Initialized write operation [" << idx << "] (64-bit): address="
              << (void*)writeAddrs[i]
              << ", value=" << paramArray[idx].writeValue.value64);
      }
  }

  // Write initial values to the wait addresses to ensure the wait works
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    if (i % 2 == 0) {
      uint32_t initialValue32 = 0x0;
      HIP_CHECK(hipMemcpyHtoD(waitAddrs[i], &initialValue32, sizeof(uint32_t)));
      INFO("Initialized waitAddr[" << i << "] (32-bit) with initial value " << initialValue32);
    } else {
      uint64_t initialValue64 = 0x0;
      HIP_CHECK(hipMemcpyHtoD(waitAddrs[i], &initialValue64, sizeof(uint64_t)));
      INFO("Initialized waitAddr[" << i << "] (64-bit) with initial value " << initialValue64);
    }
  }

  // Execute batched memory operations
  INFO("Launching hipStreamBatchMemOp with totalOps = " << totalOps);
  HIP_CHECK(hipStreamBatchMemOp(stream, totalOps, paramArray, 0));
  INFO("hipStreamBatchMemOp launched successfully.");

  // Release the wait by writing the expected values to the addresses in a different stream
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    if (i % 2 == 0) {
      uint32_t releaseValue32;
      switch (paramArray[i].waitValue.flags) {
        case hipStreamWaitValueEq:
        case hipStreamWaitValueGte:
          releaseValue32 = paramArray[i].waitValue.value;
        break;
        case hipStreamWaitValueAnd:
          releaseValue32 = paramArray[i].waitValue.value | 0xFFFFFFFF;  // Ensure condition is met
          break;
        case hipStreamWaitValueNor:
          releaseValue32 = ~(paramArray[i].waitValue.value);
          break;
        default:
          releaseValue32 = paramArray[i].waitValue.value;
        }
        HIP_CHECK(hipMemcpyHtoDAsync(waitAddrs[i], &releaseValue32, sizeof(uint32_t),
                                     releaseStream));
        INFO("Released wait operation [" << i << "] (32-bit): wrote value " <<
              releaseValue32 << " to address " << (void*)waitAddrs[i]);
    } else {
      uint64_t releaseValue64;
      switch (paramArray[i].waitValue.flags) {
        case hipStreamWaitValueEq:
        case hipStreamWaitValueGte:
          releaseValue64 = paramArray[i].waitValue.value64;
          break;
        case hipStreamWaitValueAnd:
          releaseValue64 = paramArray[i].waitValue.value64 | 0xFFFFFFFFFFFFFFFF;
          break;
        case hipStreamWaitValueNor:
          releaseValue64 = ~(paramArray[i].waitValue.value64);
        break;
          default:
          releaseValue64 = paramArray[i].waitValue.value64;
        }
        HIP_CHECK(hipMemcpyHtoDAsync(waitAddrs[i], &releaseValue64, sizeof(uint64_t),
                  releaseStream));
        INFO("Released wait operation [" << i << "] (64-bit): wrote value " <<
              releaseValue64 << " to address " << (void*)waitAddrs[i]);
    }
  }

  // Synchronize the release stream
  INFO("Synchronizing release stream...");
  HIP_CHECK(hipStreamSynchronize(releaseStream));
  INFO("Release stream synchronized.");

  // Wait for the original stream to complete
  INFO("Waiting for original stream to complete...");
  HIP_CHECK(hipStreamSynchronize(stream));
  INFO("Original stream synchronized.");

  // Verify wait operations
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    if (i % 2 == 0) {
      uint32_t value32;
      HIP_CHECK(hipMemcpyDtoH(&value32, waitAddrs[i], sizeof(uint32_t)));
      INFO("Wait operation [" << i << "] (32-bit): read value " << value32 << " from address " <<
            (void*)waitAddrs[i]);
      bool conditionMet = false;
      switch (paramArray[i].waitValue.flags) {
        case hipStreamWaitValueEq:
          conditionMet = (value32 == paramArray[i].waitValue.value);
          break;
        case hipStreamWaitValueGte:
          conditionMet = (value32 >= paramArray[i].waitValue.value);
          break;
        case hipStreamWaitValueAnd:
          conditionMet = ((value32 & paramArray[i].waitValue.value) != 0);
          break;
        case hipStreamWaitValueNor:
          conditionMet = ((~(value32 | paramArray[i].waitValue.value)) == 0);
          break;
        }
        INFO("Verification for wait operation [" << i << "] (32-bit): conditionMet = " <<
             conditionMet);
        REQUIRE(conditionMet);
    } else {
      uint64_t value64;
      HIP_CHECK(hipMemcpyDtoH(&value64, waitAddrs[i], sizeof(uint64_t)));
      INFO("Wait operation [" << i << "] (64-bit): read value " << value64 << " from address " <<
           (void*)waitAddrs[i]);
      bool conditionMet = false;
      switch (paramArray[i].waitValue.flags) {
      case hipStreamWaitValueEq:
        conditionMet = (value64 == paramArray[i].waitValue.value64);
        break;
      case hipStreamWaitValueGte:
        conditionMet = (value64 >= paramArray[i].waitValue.value64);
        break;
      case hipStreamWaitValueAnd:
        conditionMet = ((value64 & paramArray[i].waitValue.value64) != 0);
        break;
      case hipStreamWaitValueNor:
        conditionMet = ((~(value64 | paramArray[i].waitValue.value64)) == 0);
        break;
      }
      INFO("Verification for wait operation [" << i << "] (64-bit): conditionMet = " <<
           conditionMet);
      REQUIRE(conditionMet);
    }
  }

  // Verify write operations
  for (int i = 0; i < ARRAY_SIZE / 2; ++i) {
    int idx = ARRAY_SIZE + i;
    if (i % 2 == 0) {
      uint32_t value32;
      HIP_CHECK(hipMemcpyDtoH(&value32, writeAddrs[i], sizeof(uint32_t)));
      INFO("Write operation [" << idx << "] (32-bit): expected value = " <<
           paramArray[idx].writeValue.value
            << ", actual value = " << value32 << " at address " << (void*)writeAddrs[i]);
      REQUIRE(paramArray[idx].writeValue.value == value32);
      } else {
        uint64_t value64;
        HIP_CHECK(hipMemcpyDtoH(&value64, writeAddrs[i], sizeof(uint64_t)));
        INFO("Write operation [" << idx << "] (64-bit): expected value = " <<
             paramArray[idx].writeValue.value64 << ", actual value = " << value64 <<
             " at address " << (void*)writeAddrs[i]);
        REQUIRE(paramArray[idx].writeValue.value64 == value64);
      }
    }

  // Cleanup
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    HIP_CHECK(hipFree((void*)waitAddrs[i]));
    INFO("Freed waitAddr[" << i << "] at address " << (void*)waitAddrs[i]);
  }
  for (int i = 0; i < ARRAY_SIZE / 2; ++i) {
    HIP_CHECK(hipFree((void*)writeAddrs[i]));
    INFO("Freed writeAddr[" << i << "] at address " << (void*)writeAddrs[i]);
  }
  HIP_CHECK(hipStreamDestroy(releaseStream));
  INFO("Release stream destroyed.");
  HIP_CHECK(hipStreamDestroy(stream));
  INFO("Main stream destroyed.");
  INFO("hipStreamBatchMemOp PASSED.\n");
}

#endif // CUDA_VERSION > CUDA_12000
