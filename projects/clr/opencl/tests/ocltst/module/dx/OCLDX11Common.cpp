/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "OCLDX11Common.h"

#define D3D_FEATURE_LEVEL_11_1 0xb100

#define INITPFN(x)                                                                                 \
  x = (x##_fn)clGetExtensionFunctionAddressForPlatform(platform_, #x);                             \
  if ((x) == NULL) {                                                                               \
    char* buf = (char*)malloc(4096);                                                               \
    _errorFlag = true;                                                                             \
    int rc = snprintf(buf, 4096, "Failed to get function pointer for %s", #x);                     \
    assert(rc >= 0 && rc < (int)4096);                                                             \
    printf("%s:%d - %s\n", __FILE__, __LINE__, buf);                                               \
    _errorMsg = std::string(buf);                                                                  \
    _crcword += 1;                                                                                 \
    free(buf);                                                                                     \
    return;                                                                                        \
  }

OCLDX11Common::OCLDX11Common() : OCLTestImp() {
  clGetDeviceIDsFromD3D11KHR = NULL;
  clCreateFromD3D11BufferKHR = NULL;
  clCreateFromD3D11Texture2DKHR = NULL;
  clCreateFromD3D11Texture3DKHR = NULL;
  clEnqueueAcquireD3D11ObjectsKHR = NULL;
  clEnqueueReleaseD3D11ObjectsKHR = NULL;
  clGetPlaneFromImageAMD = NULL;
}

OCLDX11Common::~OCLDX11Common() {}

void OCLDX11Common::ExtensionCheck() {
  cl_int result = CL_SUCCESS;
  char extensions[1024];

  result = _wrapper->clGetPlatformInfo(platform_, CL_PLATFORM_EXTENSIONS, sizeof(extensions),
                                       extensions, NULL);
  CHECK_RESULT(result != CL_SUCCESS, "Failed to list platform extensions.");

  extensionsAvailable = strstr(extensions, "cl_khr_d3d11_sharing") ? true : false;
  if (!extensionsAvailable) {
    printf("cl_khr_d3d11_sharing extension is required for this test!\n");
  }

  OSVERSIONINFOEX versionInfo = {0};
  versionInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
  versionInfo.dwMajorVersion = 6;

  DWORDLONG conditionMask = 0;
  VER_SET_CONDITION(conditionMask, VER_MAJORVERSION, VER_GREATER_EQUAL);
  if (VerifyVersionInfo(&versionInfo, VER_MAJORVERSION, conditionMask)) {
    CHECK_RESULT(!extensionsAvailable, "Extension should be exported on Windows >= 6");
  } else {
    CHECK_RESULT(extensionsAvailable, "Extension should not be exported on Windows < 6");
  }

  result = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_EXTENSIONS, sizeof(extensions),
                                     extensions, NULL);
  CHECK_RESULT(result != CL_SUCCESS, "Failed to list device extensions.");

  extensionsAvailable = strstr(extensions, "cl_amd_planar_yuv") ? true : false;
  if (!extensionsAvailable) {
    printf("cl_amd_planar_yuv extension is required for this test!\n");
  }
}

void OCLDX11Common::open(unsigned int test, char* units, double& conversion,
                         unsigned int deviceId) {
  // OpenCL Initialization
  // OCLTestImp::open(test, units, conversion, deviceId);
  BaseTestImp::open();
  devices_ = 0;
  deviceCount_ = 0;
  context_ = 0;
  program_ = 0;
  kernel_ = 0;
  _queue = 0;
  _deviceId = deviceId;

  dxD3D11Context = NULL;
  dxD3D11Device = NULL;

  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test (%d)", error_);

  cl_uint numPlatforms = 0;
  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetPlatformIDs failed");
  CHECK_RESULT((numPlatforms == 0), "No platform found");

  cl_platform_id* platforms = new cl_platform_id[numPlatforms];
  error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");

  platform_ = platforms[_platformIndex];
  CHECK_RESULT((platform_ == 0), "AMD Platform not found");

  delete[] platforms;

  error_ = _wrapper->clGetDeviceIDs(platform_, type_, 0, NULL, &deviceCount_);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs() failed");

  devices_ = new cl_device_id[deviceCount_];
  error_ = _wrapper->clGetDeviceIDs(platform_, type_, deviceCount_, devices_, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs() failed");

  ExtensionCheck();
  if (!extensionsAvailable) {
    return;
  }

  // extract function pointers for exported functions
  INITPFN(clGetDeviceIDsFromD3D11KHR);
  INITPFN(clCreateFromD3D11BufferKHR);
  INITPFN(clCreateFromD3D11Texture2DKHR);
  INITPFN(clCreateFromD3D11Texture3DKHR);
  INITPFN(clEnqueueAcquireD3D11ObjectsKHR);
  INITPFN(clEnqueueReleaseD3D11ObjectsKHR);
  INITPFN(clGetPlaneFromImageAMD);

  char name[1024] = {0};
  size_t size = 0;

  if (deviceId >= deviceCount_) {
    _errorFlag = true;
    return;
  }

  HRESULT hr = S_OK;

  UINT createDeviceFlags = 0;

  D3D_FEATURE_LEVEL featureLevels[] = {
      (D3D_FEATURE_LEVEL)D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0

  };
  D3D_FEATURE_LEVEL featureLevel;
  // Create only the device, not the swapchain. We can't create the swapchain
  // anyways without a handle to a window we explicitly own
  hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags, featureLevels,
                         _countof(featureLevels), D3D11_SDK_VERSION, &dxD3D11Device, &featureLevel,
                         &dxD3D11Context);

  if (FAILED(hr)) {
    hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags,
                           featureLevels + 1, _countof(featureLevels) - 1, D3D11_SDK_VERSION,
                           &dxD3D11Device, &featureLevel, &dxD3D11Context);
  }
  if (FAILED(hr)) {
    hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_SOFTWARE, NULL, createDeviceFlags, featureLevels,
                           _countof(featureLevels), D3D11_SDK_VERSION, &dxD3D11Device,
                           &featureLevel, &dxD3D11Context);
  }

  if (FAILED(hr)) {
    hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_SOFTWARE, NULL, createDeviceFlags,
                           featureLevels + 1, _countof(featureLevels) - 1, D3D11_SDK_VERSION,
                           &dxD3D11Device, &featureLevel, &dxD3D11Context);
  }

  cl_int status = 0;
  cl_context_properties cps[7] = {CL_CONTEXT_D3D11_DEVICE_KHR,
                                  (cl_context_properties)(ID3D11Device*)dxD3D11Device,
                                  CL_CONTEXT_INTEROP_USER_SYNC,
                                  CL_FALSE,
                                  CL_CONTEXT_PLATFORM,
                                  (cl_context_properties)platform_,
                                  0};
  cl_context_properties* cprops = (NULL == platform_) ? NULL : cps;

  cl_uint deviceListSize = 0;
  clGetDeviceIDsFromD3D11KHR(platform_, CL_D3D11_DEVICE_KHR, dxD3D11Device,
                             CL_PREFERRED_DEVICES_FOR_D3D11_KHR, 0, NULL, &deviceListSize);

  std::vector<cl_device_id> devices;
  devices.resize(deviceListSize);
  clGetDeviceIDsFromD3D11KHR(platform_, CL_D3D11_DEVICE_KHR, dxD3D11Device,
                             CL_PREFERRED_DEVICES_FOR_D3D11_KHR, deviceListSize, &devices[0], NULL);

  bool ret = false;
  // Check that current device can be associated with OpenGL context
  for (unsigned int i = 0; i < deviceListSize; i++) {
    if (devices[i] == devices_[_deviceId]) {
      ret = true;
      break;
    }
  }
  if (ret) {
    char buf[2000];
    _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_EXTENSIONS, sizeof(buf), buf, NULL);

    context_ = clCreateContext(cprops, 1, &devices_[_deviceId], NULL, NULL, &status);
    _queue = clCreateCommandQueue(context_, devices_[_deviceId], 0, &status);
  }
  CHECK_RESULT((ret != true), "Can't find D3D device!");
}

unsigned int OCLDX11Common::close(void) {
  clReleaseCommandQueue(_queue);
  unsigned int retVal = OCLTestImp::close();
  // deleteDXDevice(hDX_);
  if (dxD3D11Context) dxD3D11Context->Release();
  if (dxD3D11Device) dxD3D11Device->Release();
  return retVal;
}
