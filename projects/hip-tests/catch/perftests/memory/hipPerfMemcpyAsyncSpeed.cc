/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#define NUM_SIZES 6
// 256 Bytes, 512 Bytes, 1024 Bytes, 2048 Bytes, 3072 Bytes, 4096 Bytes
constexpr uint32_t Mi = 1024 * 1024;
static const unsigned int Sizes[NUM_SIZES] = {256, 512, 1024, 2048, 3072, 4096};
static const unsigned int Iterations[2] = {1000, 1000};

#define BUF_TYPES 1
//  16 ways to combine 4 different buffer types
#define NUM_SUBTESTS (BUF_TYPES * BUF_TYPES)

void checkData(void* ptr, unsigned int size, char value) {
  char* ptr2 = (char*)ptr;
  for (unsigned int i = 0; i < size; i++) {
    if (ptr2[i] != value) {
      CONSOLE_PRINT("Data validation failed at %d!  Got 0x%08x\n", i, ptr2[i]);
      CONSOLE_PRINT("Expected 0x%08x\n", value);
      CONSOLE_PRINT("Data validation failed!");
      break;
    }
  }
}

bool extraWarmup = true;
TEST_CASE("Perf_hipPerfMemcpyAsyncSpeed_test") {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  CONSOLE_PRINT("Set device to %d : %s\n", 0, props.name);
  HIP_CHECK(hipSetDevice(0));

  unsigned int bufSize_;
  bool hostMalloc[2] = {false};
  bool hostRegister[2] = {false};
  bool unpinnedMalloc[2] = {false};
  unsigned int numIter;
  void* srcBuffer = nullptr;
  void* dstBuffer = nullptr;

  for (int t = 0; t < 3; ++t) {
    int numTests = NUM_SIZES * NUM_SUBTESTS - 1;
    int test = 0;
    uint32_t kMaxSize = (t == 0) ? 128 * 1024 * 1024 : 1024 * 1024 * 1024;
    if (t < 2) {
      CONSOLE_PRINT("\n----- Global buffer (MiB): %d", kMaxSize / (1024 * 1024));
    } else {
      CONSOLE_PRINT("\n----- Same buffer copy repeat");
    }
    for (; test <= numTests; test++) {
      bufSize_ = Sizes[test % NUM_SIZES];
      hostMalloc[0] = hostMalloc[1] = false;
      hostRegister[0] = hostRegister[1] = false;
      unpinnedMalloc[0] = unpinnedMalloc[1] = false;
      srcBuffer = dstBuffer = 0;

      numIter = Iterations[test / (NUM_SIZES * NUM_SUBTESTS)];
      uint32_t totalSize = bufSize_ * numIter;
      if (t == 2) {
        totalSize = bufSize_;
        kMaxSize = bufSize_;
      }
      totalSize = std::max(totalSize, kMaxSize);
      HIP_CHECK(hipMalloc(&srcBuffer, totalSize));
      HIP_CHECK(hipMemset(srcBuffer, 0xd0, totalSize));
      HIP_CHECK(hipMalloc(&dstBuffer, totalSize));

      // warm up
      uint32_t warm_up = (extraWarmup) ? numIter : 1;
      for (unsigned int i = 0; i < warm_up; i++) {
        size_t bufSize_warm = (t == 2) ? bufSize_ : 16 * Mi;
        char* src = reinterpret_cast<char*>(srcBuffer) + bufSize_warm * i;
        if ((t == 2) || (src >= reinterpret_cast<char*>(srcBuffer) + kMaxSize)) {
          src = reinterpret_cast<char*>(srcBuffer);
        }
        char* dst = reinterpret_cast<char*>(dstBuffer) + bufSize_warm * i;
        if ((t == 2) || (dst >= reinterpret_cast<char*>(dstBuffer) + kMaxSize)) {
          dst = reinterpret_cast<char*>(dstBuffer);
        }

        HIP_CHECK(hipMemcpyAsync(dst, src, bufSize_warm, hipMemcpyDefault, nullptr));
      }

      HIP_CHECK(hipStreamSynchronize(nullptr));
      auto start = std::chrono::steady_clock::now();
      for (unsigned int i = 0; i < numIter; i++) {
        char* src = reinterpret_cast<char*>(srcBuffer) + bufSize_ * i;
        if ((t == 2) || (src >= reinterpret_cast<char*>(srcBuffer) + kMaxSize)) {
          src = reinterpret_cast<char*>(srcBuffer);
        }
        char* dst = reinterpret_cast<char*>(dstBuffer) + bufSize_ * i;
        if ((t == 2) || (dst >= reinterpret_cast<char*>(dstBuffer) + kMaxSize)) {
          dst = reinterpret_cast<char*>(dstBuffer);
        }

        HIP_CHECK(hipMemcpyAsync(dst, src, bufSize_, hipMemcpyDefault, nullptr));
      }
      auto timer_cpu = std::chrono::steady_clock::now();
      HIP_CHECK(hipStreamSynchronize(nullptr));
      auto timer = std::chrono::steady_clock::now();
      std::chrono::duration<double> sec = timer - start;
      std::chrono::duration<double> sec_cpu = timer_cpu - start;
      // Buffer copy bandwidth in GB/S
      double perf = ((double)bufSize_ * numIter * (double)(1e-09)) / sec.count();

      const char* strSrc = "dM";
      const char* strDst = "dM";
      // Double results when src and dst are both on device
      perf *= 2.0;
      CONSOLE_PRINT(
          "hipMemcpyAsync[%d]\t(%8d bytes)\ts:%s d:%s\ti:%4d\t(GB/s) "
          "perf\t%.2f, time per iter(us):\t%.1f, time per iter CPU (us):\t%.1f",
          test, bufSize_, strSrc, strDst, numIter, (float)perf, sec.count() / numIter * 1000 * 1000,
          sec_cpu.count() / numIter * 1000 * 1000);

      // Verification
      void* temp = malloc(bufSize_ + 4096);
      void* chkBuf = (void*)(((size_t)temp + 4095) & ~4095);
      HIP_CHECK(hipMemcpy(chkBuf, dstBuffer, bufSize_, hipMemcpyDefault));
      checkData(chkBuf, bufSize_, 0xd0);
      free(temp);

      // Free src and dst
      HIP_CHECK(hipFree(srcBuffer));
      HIP_CHECK(hipFree(dstBuffer));
    }
  }
}
