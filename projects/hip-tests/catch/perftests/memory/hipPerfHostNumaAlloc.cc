/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @addtogroup hipMemcpyKernel hipMemcpyKernel
 * @{
 * @ingroup perfMemoryTest
 * `hipMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind)` -
 * Copies data between host and device.
 */
#include <unistd.h>
#include <numaif.h>
#include <numa.h>
#include <hip_test_common.hh>
// #define ENABLE_DEBUG 1
// To run it correctly, we must not export HIP_VISIBLE_DEVICES.
// And we must explicitly link libnuma because of numa api move_pages().
#define NUM_PAGES 100
char* h = nullptr;
char* d_h = nullptr;
char* m = nullptr;
char* d_m = nullptr;
int page_size = 0;

const int mode[] = {MPOL_DEFAULT, MPOL_BIND, MPOL_PREFERRED, MPOL_INTERLEAVE};
const char* modeStr[] = {"MPOL_DEFAULT", "MPOL_BIND", "MPOL_PREFERRED", "MPOL_INTERLEAVE"};

bool test(int cpuId, int gpuId, int numaMode, unsigned int hostMallocflags) {
  void* pages[NUM_PAGES];
  int status[NUM_PAGES];
  int ret_code;

  CONSOLE_PRINT("set cpu %d, gpu %d, numaMode %d, hostMallocflags %u\n", cpuId, gpuId, numaMode,
                hostMallocflags);
  if (gpuId >= 0) {
    HIP_CHECK(hipSetDevice(gpuId));
  }

  if (cpuId >= 0) {
    unsigned long nodeMask = 1 << cpuId;           // NOLINT
    unsigned long maxNode = sizeof(nodeMask) * 8;  // NOLINT
    // Will override existing numa policy in memory
    if (set_mempolicy(numaMode, numaMode == MPOL_DEFAULT ? NULL : &nodeMask,
                      numaMode == MPOL_DEFAULT ? 0 : maxNode) == -1) {
      WARN("set_mempolicy() failed with err " << errno << "\n");
      return false;
    }
  }

  posix_memalign(reinterpret_cast<void**>(&m), page_size, page_size * NUM_PAGES);
  HIP_CHECK(hipHostRegister(m, page_size * NUM_PAGES, hipHostRegisterMapped));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&d_m), m, 0));

  status[0] = -1;
  pages[0] = m;
  for (int i = 1; i < NUM_PAGES; i++) {
    pages[i] = reinterpret_cast<char*>(pages[0]) + page_size;
  }

  ret_code = move_pages(0, NUM_PAGES, pages, NULL, status, 0);
  CONSOLE_PRINT("Memory (malloc) ret %d at %p (dev %p) is at node: ", ret_code, m, d_m);
  for (int i = 0; i < NUM_PAGES; i++) {
    CONSOLE_PRINT("%d ", status[i]);  // Don't verify as it's out of our control
  }
  CONSOLE_PRINT("\n");

  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), page_size * NUM_PAGES, hostMallocflags));
  pages[0] = h;
  for (int i = 1; i < NUM_PAGES; i++) {
    pages[i] = reinterpret_cast<char*>(pages[0]) + page_size;
  }
  ret_code = move_pages(0, NUM_PAGES, pages, NULL, status, 0);
  d_h = nullptr;
  if (hostMallocflags & hipHostMallocMapped) {
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&d_h), h, 0));
    CONSOLE_PRINT("Memory (hipHostMalloc) ret %d at %p (dev %p) is at node: ", ret_code, h, d_h);
  } else {
    CONSOLE_PRINT("Memory (hipHostMalloc) ret %d at %p is at node: ", ret_code, h);
  }
  for (int i = 0; i < NUM_PAGES; i++) {
    CONSOLE_PRINT("%d ", status[i]);  // Always print it even if it's wrong. Verify later
  }
  CONSOLE_PRINT("\n");

  HIP_CHECK(hipHostFree(reinterpret_cast<void*>(h)));
  HIP_CHECK(hipHostUnregister(m));
  free(m);

  if (cpuId >= 0 && (numaMode == MPOL_BIND || numaMode == MPOL_PREFERRED)) {
    for (int i = 0; i < NUM_PAGES; i++) {
      if (status[i] != cpuId) {  // Now verify
        WARN("Failed at " << i << " status[i] = " << status[i] << " cpuId " << cpuId << "\n");
        return false;
      }
    }
  }
  return true;
}

bool runTest(const int& cpuCount, const int& gpuCount, unsigned int hostMallocflags,
             const std::string& str) {
  CONSOLE_PRINT("Test- %s\n", str.c_str());

  for (int m = 0; m < sizeof(mode) / sizeof(mode[0]); m++) {
    CONSOLE_PRINT("Testing %s\n", modeStr[m]);

    for (int i = 0; i < cpuCount; i++) {
      for (int j = 0; j < gpuCount; j++) {
        if (!test(i, j, mode[m], hostMallocflags)) {
          return false;
        }
      }
    }
  }
  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfHostNumaAlloc status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfHostNumaAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfHostNumaAlloc_test") {
  int gpuCount = 0;
  HIP_CHECK(hipGetDeviceCount(&gpuCount));
  int cpuCount = numa_max_node() + 1; // number of numa nodes
  page_size = getpagesize();
  CONSOLE_PRINT("Cpu count %d, Gpu count %d, page_size %d\n", cpuCount, gpuCount, page_size);

  if (cpuCount < 0 || gpuCount < 0) {
    SUCCEED(
        "Skipped testcase hipPerfHostNumaAlloc as "
        "there is no device to test.\n");
    return;
  }

  REQUIRE(true == runTest(cpuCount, gpuCount, hipHostMallocDefault | hipHostMallocNumaUser,
                          "Testing hipHostMallocDefault | hipHostMallocNumaUser......"));

  REQUIRE(true == runTest(cpuCount, gpuCount, hipHostMallocMapped | hipHostMallocNumaUser,
                          "Testing hipHostMallocMapped | hipHostMallocNumaUser......."));
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
