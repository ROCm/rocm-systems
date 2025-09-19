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

#include "OCLOfflineCompilation.h"

#include <Timer.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "CL/cl_ext.h"
#include "cl_kernel_info_amd.h"

typedef CL_API_ENTRY cl_int(CL_API_CALL* clGetKernelInfoAMD_fn)(
    cl_kernel kernel, cl_device_id device, cl_kernel_info_amd param_name, size_t param_value_size,
    void* param_value, size_t* param_value_size_ret);

clGetKernelInfoAMD_fn clGetKernelInfoAMDp;

#define BLIT_KERNEL(...) #__VA_ARGS__

const char* strKernel12 = BLIT_KERNEL(
\n const constant uint test = 1; __kernel void factorial(__global uint* out) {
  uint id = get_global_id(0);
  uint factorial = 1;
  out[id] = factorial + test;
}
\n);

const char* strKernel20 = BLIT_KERNEL(
\n const constant uint test = 1; global uint test2 = 0;
    __kernel void factorial(__global uint* out) {
      uint id = get_global_id(0);
      uint factorial = 1;
      out[id] = factorial + test;
      if (id == 0) {
        out[id] += test2++;
      }
    }
\n);

OCLOfflineCompilation::OCLOfflineCompilation() { _numSubTests = 1; }

OCLOfflineCompilation::~OCLOfflineCompilation() {}

void OCLOfflineCompilation::open(unsigned int test, char* units, double& conversion,
                                 unsigned int deviceId) {
  size_t nDevices = 0;
  cl_device_id* devices = NULL;

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  _wrapper->clReleaseContext(context_);

  cl_context_properties cprops[5];
  clGetKernelInfoAMDp = (clGetKernelInfoAMD_fn)clGetExtensionFunctionAddressForPlatform(
      platform_, "clGetKernelInfoAMD");
  if (clGetKernelInfoAMDp == NULL) {
    testDescString = "clGetKernelInfoAMD not found!\n";
    return;
  }

  // Utilize the CL_CONTEXT_OFFLINE_DEVICES_AMD platform option to allow for
  // the generation of binary kernel without target device installed in build
  // system.
  cprops[0] = CL_CONTEXT_PLATFORM;
  cprops[1] = (cl_context_properties)platform_;
  cprops[2] = CL_CONTEXT_OFFLINE_DEVICES_AMD;
  cprops[3] = (cl_context_properties)1;
  cprops[4] = (cl_context_properties)0;  // end of options list marker

  // Create a context with all of the available devices.
  context_ = _wrapper->clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, NULL, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateContextFromType()  failed");

  size_t deviceListSize = 0;
  error_ = _wrapper->clGetContextInfo(context_, CL_CONTEXT_NUM_DEVICES, sizeof(size_t),
                                      &deviceListSize, NULL);
  CHECK_RESULT(((error_ != CL_SUCCESS) || (deviceListSize == 0)), "clGetContextInfo()  failed");

  devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceListSize);
  CHECK_RESULT((devices == NULL), "clGetContextInfo()  failed");

  memset(devices, 0, deviceListSize);

  error_ = _wrapper->clGetContextInfo(context_, CL_CONTEXT_DEVICES,
                                      sizeof(cl_device_id) * deviceListSize, devices, &nDevices);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetContextInfo()  failed");

  for (unsigned version = 1; version <= 2; ++version) {
    std::string options;
    const char* strKernel;

    switch (version) {
      case 1:
        options = "";
        strKernel = strKernel12;
        break;
      case 2:
        options = "-cl-std=CL2.0";
        strKernel = strKernel20;
        break;
      default:
        assert(false);
        return;
    }

    program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

    for (unsigned int i = 0; i < deviceListSize; ++i) {
      char name[128];
      char strVersion[128];
      _wrapper->clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
      if (strstr(name, "-generic") != NULL) {
        continue;  // Skip generic target because it needs code object version 6
      }
      error_ = _wrapper->clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(strVersion),
                                         strVersion, 0);
      CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

      if (version == 2 && strVersion[7] < '2') {
        continue;
      }

      // skipping the test on gfx9+ for now till we add compiler support for al
      // the gfx10+ subdevices
      cl_uint gfxip_major = 0;
      cl_uint gfxip_minor = 0;
      clGetDeviceInfo(devices[i], CL_DEVICE_GFXIP_MAJOR_AMD, sizeof(gfxip_major), &gfxip_major,
                      NULL);
      clGetDeviceInfo(devices[i], CL_DEVICE_GFXIP_MINOR_AMD, sizeof(gfxip_minor), &gfxip_minor,
                      NULL);

      printf("Building on %s, OpenCL version %s, (options '%s')\n", name,
             (version == 2 ? "2.0" : "1.2"), options.c_str());
      error_ = _wrapper->clBuildProgram(program_, 1, &devices[i], options.c_str(), NULL, NULL);
      if (error_ != CL_SUCCESS) {
        char programLog[1024];
        _wrapper->clGetProgramBuildInfo(program_, devices[i], CL_PROGRAM_BUILD_LOG, 1024,
                                        programLog, 0);
        printf("\n%s\n", programLog);
        fflush(stdout);
        break;
      }
      kernel_ = _wrapper->clCreateKernel(program_, "factorial", &error_);
      CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

      size_t usedVGPRs = 0;
      error_ = clGetKernelInfoAMDp(kernel_, devices[i], CL_KERNELINFO_USED_VGPRS, sizeof(usedVGPRs),
                                   &usedVGPRs, NULL);
      CHECK_RESULT(((error_ != CL_SUCCESS) || (usedVGPRs == 0)), "clGetKernelInfoAMD() failed");

      _wrapper->clReleaseKernel(kernel_);
      kernel_ = nullptr;

      size_t binSize;
      error_ = _wrapper->clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES, sizeof(size_t),
                                          &binSize, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clGetProgramInfo() failed");
      char* binary = new char[binSize];
      error_ =
          _wrapper->clGetProgramInfo(program_, CL_PROGRAM_BINARIES, sizeof(char*), &binary, NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clGetProgramInfo() failed");
      delete[] binary;
    }
    if (version == 1) {
      error_ = _wrapper->clReleaseProgram(program_);
      CHECK_RESULT((error_ != CL_SUCCESS), "clReleaseProgram() failed");
    }
  }
  free(devices);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLOfflineCompilation::run(void) {}

unsigned int OCLOfflineCompilation::close(void) { return OCLTestImp::close(); }
