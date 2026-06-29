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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

constexpr size_t wave32 = 32;
constexpr size_t wave64 = 64;
constexpr size_t scratchBits12X = 18;
constexpr size_t scratchBits9X = 15;
constexpr size_t compilerRequired = 64;
constexpr size_t maxStackSize12X = (((1 << scratchBits12X) - 1) * 256 / wave32) - compilerRequired;
constexpr size_t maxStackSize11X = (((1 << scratchBits9X) - 1) * 256 / wave32) - compilerRequired;
constexpr size_t maxStackSize9X = (((1 << scratchBits9X) - 1) * 256 / wave64) - compilerRequired;

__global__ void sumFromPrivateMem(int* global, size_t size, unsigned long long* result) {
  constexpr size_t elems =
  #if (__GFX12__)
    maxStackSize12X / sizeof(int);
  #elif (__GFX11__ || __GFX10__)
    maxStackSize11X / sizeof(int);
  #elif (__GFX9__ || __GFX8__)
    maxStackSize9X / sizeof(int);
  #else
    1024;
  #endif
  // printf("\nfrom Kernel : %zu,  %zu\n", elems, size);
  int privateArray[elems];
  unsigned long long sum = 0;
  for (int i = 0; i < size; ++i) {
    int index = i % elems; // Use modulo to cycle through privateArray
    if (index == 0) {
      privateArray[index] = 1 + global[i];
    } else {
      privateArray[index] = privateArray[index - 1] + global[i];
    }
    sum += privateArray[index];
  }
  *result = sum;
}

size_t GetMaxStackSize() {
  int device = -1;
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string arch = std::string(props.gcnArchName);
  if (arch.find("gfx12") != std::string::npos) {
    return maxStackSize12X;
  } else if (arch.find("gfx11") != std::string::npos ||
              arch.find("gfx10") != std::string::npos) {
    return maxStackSize11X;
  } else if (arch.find("gfx9") != std::string::npos ||
              arch.find("gfx8") != std::string::npos) {
    return maxStackSize9X;
  }
  return 1024;
}

HIP_TEST_CASE(Unit_KernelStackSize) {
  // maxArrayBytes is chosen as maxStackSize12X being largest stack size amongst
  // which make sure to cover whole range of stack size for any ASIC.
  size_t maxArrayBytes = maxStackSize12X;
  size_t maxArraySize = maxArrayBytes / sizeof(int);

  int* h_a = new int[maxArraySize];
  for (int i = 0; i < maxArraySize; i++) {
    h_a[i] = rand() % 1000;
  }

  int *d_a;
  HIP_CHECK(hipMalloc(&d_a, maxArrayBytes));
  HIP_CHECK(hipMemcpy(d_a, h_a, maxArrayBytes, hipMemcpyHostToDevice));

  unsigned long long *d_sum;
  HIP_CHECK(hipMalloc(&d_sum, sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(d_sum, 0, sizeof(unsigned long long)));

  sumFromPrivateMem<<<1,1,0,0>>>(d_a, maxArraySize, d_sum);
  unsigned long long h_sum = 0;
  HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(unsigned long long), hipMemcpyDeviceToHost));

  size_t stackSize = GetMaxStackSize() / sizeof(int);
  int* privateArray = new int[stackSize];
  unsigned long long sum = 0;
  for (int i = 0; i < maxArraySize; ++i) {
    int index = i % stackSize; // Use modulo to cycle through privateArray
    if (index == 0) {
      privateArray[index] = 1 + h_a[i];
    } else {
      privateArray[index] = privateArray[index - 1] + h_a[i];
    }
    sum += privateArray[index];
  }
  // printf("\nfrom host : %zu, %llu , %llu\n", stackSize, h_sum, sum);
  REQUIRE(sum == h_sum);
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_sum));
  delete [] privateArray;
  delete [] h_a;
}