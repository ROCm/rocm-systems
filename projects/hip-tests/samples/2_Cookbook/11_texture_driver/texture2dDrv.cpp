/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "hip_helper.h"

#define fileName "tex2dKernel.code"

bool testResult = true;

template<typename T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type *t = nullptr>
static inline hipArray_Format getArrayFormat() {
  if (std::is_same<char, T>::value) {
    return HIP_AD_FORMAT_SIGNED_INT8;
  } else if (std::is_same<short, T>::value) {
    return HIP_AD_FORMAT_SIGNED_INT16;
  } else if (std::is_same<int, T>::value) {
    return HIP_AD_FORMAT_SIGNED_INT32;
  } else if (std::is_same<float, T>::value) {
    return HIP_AD_FORMAT_FLOAT;
  }
  return HIP_AD_FORMAT_HALF;
}

template<typename T,
    typename std::enable_if<!std::is_arithmetic<T>::value>::type *t = nullptr>
static inline hipArray_Format getArrayFormat() {
  return getArrayFormat<decltype(T::x)>();
}

template<typename T>
static inline constexpr int rank() {
  return sizeof(T) / sizeof(decltype(T::x));
}

#ifdef __HIP_PLATFORM_NVIDIA__
template <typename T,
    typename std::enable_if<std::is_same<T, int4>::value ||
                            std::is_same<T, short4>::value ||
                            std::is_same<T, char4>::value ||
                            std::is_same<T, float4>::value>::type *t = nullptr>
static inline bool operator!=(const T& a, const T& b)
{
    return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w);
}
#endif


template<typename T>
static inline T getRandom() {
  double r = 0;
  if (std::is_signed < T > ::value) {
    r = (std::rand() - RAND_MAX / 2.0) / (RAND_MAX / 2.0 + 1.);
  } else {
    r = std::rand() / (RAND_MAX + 1.);
  }
  return static_cast<T>(std::numeric_limits < T > ::max() * r);
}

template<typename T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
static inline constexpr int getChannels() {
  return 1;
}

template<typename T,
    typename std::enable_if<!std::is_arithmetic<T>::value>::type *t = nullptr,
    typename std::enable_if<rank<T>() != 0>::type *r = nullptr>
static inline constexpr int getChannels() {
  return rank<T>();
}

template<typename T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
static inline void printDiff(const int &i, const int &j, const T &expected,
                             const T &output) {
  std::cout << "Difference [" << i << " " << j << "]: " << expected << " - "
      << output << "\n";
}

template<typename T,
    typename std::enable_if<!std::is_arithmetic<T>::value>::type* = nullptr,
    typename std::enable_if<rank<T>() == 4>::type* = nullptr>
static inline void printDiff(const int &i, const int &j, const T &expected,
                             const T &output) {
  std::cout << "Difference [" << i << " " << j << "]: " << expected.x << ","
      << expected.y << "," << expected.z << "," << expected.w << " - "
      << output.x << "," << output.y << "," << output.z << "," << output.w
      << "\n";
}

template<typename T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
static inline void initVal(T &val) {
  val = getRandom<T>();
}

template<typename T,
    typename std::enable_if<!std::is_arithmetic<T>::value>::type* = nullptr,
    typename std::enable_if<rank<T>() == 4>::type* = nullptr>
static inline void initVal(T &val) {
  val.x = getRandom<decltype(T::x)>();
  val.y = getRandom<decltype(T::x)>();
  val.z = getRandom<decltype(T::x)>();
  val.w = getRandom<decltype(T::x)>();
}

template<typename T>
bool runTest(hipModule_t &module, const char *refName, const char *funcName) {
  hipArray_Format format = getArrayFormat<T>();
  int channels = getChannels<T>();
  unsigned int width = 256;
  unsigned int height = 256;
  unsigned int size = width * height * sizeof(T);
  T *hData = (T*) malloc(size);
  memset(hData, 0, size);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      initVal(hData[i * width + j]);
    }
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();
  hipArray_t array;
  checkHipErrors(hipMallocArray(&array, &channelDesc, width, height));

  const size_t spitch = width * sizeof(T);

  checkHipErrors(hipMemcpy2DToArray(array, 0, 0, hData, spitch, width * sizeof(T),
                        height, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = array;

  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = hipAddressModeClamp;
  texDesc.addressMode[1] = hipAddressModeClamp;
  texDesc.filterMode = hipFilterModePoint;
  texDesc.readMode = hipReadModeElementType;
  texDesc.normalizedCoords = 0;

  hipTextureObject_t texObj;
  checkHipErrors(hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

  T *dData = NULL;
  checkHipErrors(hipMalloc((void** )&dData, size));

  struct {
    void *_Ad;
    hipTextureObject_t _texObj;
    unsigned int _Bd;
    unsigned int _Cd;
  } args;
  args._Ad = (void*) dData;
  args._texObj = texObj;
  args._Bd = width;
  args._Cd = height;

  size_t sizeTemp = sizeof(args);

  void *config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &sizeTemp, HIP_LAUNCH_PARAM_END };

  hipFunction_t Function;
  checkHipErrors(hipModuleGetFunction(&Function, module, funcName));

  int temp1 = width / 16;
  int temp2 = height / 16;
  checkHipErrors(
      hipModuleLaunchKernel(Function, 16, 16, 1, temp1, temp2, 1, 0, 0, NULL,
                            (void** )&config));
  checkHipErrors(hipDeviceSynchronize());

  T *hOutputData = (T*) malloc(size);
  memset(hOutputData, 0, size);
  checkHipErrors(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (hData[i * width + j] != hOutputData[i * width + j]) {
        printDiff<T>(i, j, hData[i * width + j], hOutputData[i * width + j]);
        testResult = false;
        break;
      }
    }
  }
  checkHipErrors(hipDestroyTextureObject(texObj));
  checkHipErrors(hipFree(dData));
  checkHipErrors(hipFreeArray(array));
  free(hOutputData);
  free(hData);
  printf("%s test  %s ...\n", funcName, testResult ? "PASSED" : "FAILED");
  return testResult;
}

inline bool isImageSupported() {
    int imageSupport = 1;
#ifdef __HIP_PLATFORM_AMD__
    checkHipErrors(hipDeviceGetAttribute(&imageSupport, hipDeviceAttributeImageSupport,
                              0));
#endif
  return imageSupport != 0;
}

int main(int argc, char** argv) {
  if (!isImageSupported()) {
    printf("Texture is not support on the device. Skipped.\n");
    printf("texture2dDrv %s ...\n", "PASSED");
    return 0;
  }
  checkHipErrors(hipInit(0));
  checkHipErrors(hipSetDevice(0));
  hipModule_t module;
  checkHipErrors(hipModuleLoad(&module, fileName));
  testResult = testResult && runTest<char>(module, "texChar", "tex2dKernelChar");
  testResult = testResult && runTest<short>(module, "texShort", "tex2dKernelShort");
  testResult = testResult && runTest<int>(module, "texInt", "tex2dKernelInt");
  testResult = testResult && runTest<float>(module, "texFloat", "tex2dKernelFloat");
  testResult = testResult && runTest<char4>(module, "texChar4", "tex2dKernelChar4");
  testResult = testResult && runTest<short4>(module, "texShort4", "tex2dKernelShort4");
  testResult = testResult && runTest<int4>(module, "texInt4", "tex2dKernelInt4");
  testResult = testResult && runTest<float4>(module, "texFloat4", "tex2dKernelFloat4");

  checkHipErrors(hipModuleUnload(module));
  printf("texture2dDrv %s ...\n", testResult ? "PASSED" : "FAILED");
  return testResult ? EXIT_SUCCESS : EXIT_FAILURE;
}
