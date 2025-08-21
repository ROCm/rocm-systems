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

#include "OCLImage2DFromBuffer.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define GROUP_SIZE 256
const unsigned int OCLImage2DFromBuffer::imageWidth = 1920;
const unsigned int OCLImage2DFromBuffer::imageHeight = 1080;

const static char strKernel[] =
    "__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | "
    "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; \n"
    "__kernel void image2imageCopy(                                            "
    "                             \n"
    "    __read_only image2d_t input,                                          "
    "                              \n"
    "    __write_only image2d_t output)                                        "
    "                              \n"
    "{                                                                         "
    "                             \n"
    "    int2 coord = (int2)(get_global_id(0), get_global_id(1));              "
    "                              \n"
    "    uint4 temp = read_imageui(input, imageSampler, coord);                "
    "                              \n"
    "    write_imageui(output, coord, temp);                                   "
    "                              \n"
    "}                                                                         "
    "                             \n";

typedef CL_API_ENTRY cl_mem(CL_API_CALL* clConvertImageAMD_fn)(cl_context context, cl_mem image,
                                                               const cl_image_format* image_format,
                                                               cl_int* errcode_ret);

clConvertImageAMD_fn clConvertImageAMD;

OCLImage2DFromBuffer::OCLImage2DFromBuffer() : OCLTestImp() {
  _numSubTests = 6;
  blockSizeX = GROUP_SIZE;
  blockSizeY = 1;
}

OCLImage2DFromBuffer::~OCLImage2DFromBuffer() {}

void OCLImage2DFromBuffer::open(unsigned int test, char* units, double& conversion,
                                unsigned int deviceId) {
  buffer = clImage2DOriginal = clImage2D = clImage2DOut = NULL;
  done_ = false;
  pitchAlignment = 0;

  _openTest = test;
  // Initialize random number seed
  srand((unsigned int)time(NULL));

  OCLTestImp::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    testDescString = "GPU device is required for this test!\n";
    done_ = true;
    return;
  }

  cl_bool imageSupport;
  size_t size;
  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_IMAGE_SUPPORT, sizeof(imageSupport),
                            &imageSupport, &size);
  if (!imageSupport) {
    testDescString = "Image not supported, skipping this test! ";
    done_ = true;
    return;
  }

  if (_openTest >= 4) {
    clConvertImageAMD = (clConvertImageAMD_fn)clGetExtensionFunctionAddressForPlatform(
        platform_, "clConvertImageAMD");
    if (clConvertImageAMD == NULL) {
      testDescString = "clConvertImageAMD not found!\n";
      done_ = true;
      return;
    }
  }

  CompileKernel();
  AllocateOpenCLImage();
}

void OCLImage2DFromBuffer::run(void) {
  if (_errorFlag || done_) {
    return;
  }

  if ((_openTest % 2) == 0) {
    testReadImage(clImage2D);
  } else {
    testKernel();
  }
}

void OCLImage2DFromBuffer::AllocateOpenCLImage() {
  const bool pitchTest = (_openTest == 2 || _openTest == 3);
  cl_int status = 0;

  size_t size = 0;
  pitchAlignment = 0;
  status = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_IMAGE_PITCH_ALIGNMENT,
                                     sizeof(cl_uint), &pitchAlignment, &size);

  if (pitchAlignment != 0) {
    pitchAlignment--;
  }

  const unsigned int requiredPitch = ((imageWidth + pitchAlignment) & ~pitchAlignment);
  const unsigned int pitch = (!pitchTest) ? requiredPitch : imageWidth;
  const size_t bufferSize = pitch * imageHeight;
  CHECK_RESULT(bufferSize == 0, "ERROR: calculated image size is zero");

  unsigned char* sourceData = new unsigned char[bufferSize];

  // init data
  for (unsigned int y = 0; y < imageHeight; y++) {
    for (unsigned int x = 0; x < imageWidth / 4; x++) {
      for (unsigned int p = 0; p < 4; p++) {
        *(sourceData + y * pitch + x * 4 + p) = p;
      }
    }
  }
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, bufferSize,
                                    sourceData, &status);

  {
    // testing clConvertImageAMD
    if (_openTest == 4 || _openTest == 5) {
      const cl_image_format format = {CL_R, CL_UNSIGNED_INT8};
#if defined(CL_VERSION_2_0)
      const cl_image_desc desc = {
          CL_MEM_OBJECT_IMAGE2D, imageWidth, imageHeight, 0, 0, pitch, 0, 0, 0, {buffer}};
#else
      const cl_image_desc desc = {
          CL_MEM_OBJECT_IMAGE2D, imageWidth, imageHeight, 0, 0, pitch, 0, 0, 0, buffer};
#endif
      clImage2DOriginal =
          _wrapper->clCreateImage(context_, CL_MEM_READ_WRITE, &format, &desc, NULL, &status);
      CHECK_RESULT(status != CL_SUCCESS, "clCreateImage() failed");

      const cl_image_format formatRGBA = {CL_RGBA, CL_UNSIGNED_INT8};

      clImage2D = clConvertImageAMD(context_, clImage2DOriginal, &formatRGBA, &status);
      CHECK_RESULT(status != CL_SUCCESS, "clConvertImageAMD() failed");

      cl_mem fishyBuffer = 0;
      status = clGetImageInfo(clImage2D, CL_IMAGE_BUFFER, sizeof(fishyBuffer), &fishyBuffer, 0);
      CHECK_RESULT(status != CL_SUCCESS, "clGetImageInfo(CL_IMAGE_BUFFER) failed");
      CHECK_RESULT(fishyBuffer != buffer, "clGetImageInfo() failed, buffer != fishyBuffer");
    } else {
      const cl_image_format format = {CL_RGBA, CL_UNSIGNED_INT8};
#if defined(CL_VERSION_2_0)
      const cl_image_desc desc = {
          CL_MEM_OBJECT_IMAGE2D, imageWidth / 4, imageHeight, 0, 0, pitch, 0, 0, 0, {buffer}};
#else
      const cl_image_desc desc = {
          CL_MEM_OBJECT_IMAGE2D, imageWidth / 4, imageHeight, 0, 0, pitch, 0, 0, 0, buffer};
#endif

      clImage2D =
          _wrapper->clCreateImage(context_, CL_MEM_READ_WRITE, &format, &desc, NULL, &status);
    }

    // testing pitch alignment correct check in the runtime
    if (pitchTest) {
      CHECK_RESULT(requiredPitch != pitch &&
                       (clImage2D != NULL || status != CL_INVALID_IMAGE_FORMAT_DESCRIPTOR),
                   "AllocateOpenCLImage() failed: (clImage2D!=NULL || "
                   "status!=CL_INVALID_IMAGE_FORMAT_DESCRIPTOR) <=> (%p, %x)",
                   clImage2D, status);
      if (requiredPitch != pitch) {
        done_ = true;
        return;
      }
    }
  }

  delete[] sourceData;

  {
    const cl_image_format format = {CL_RGBA, CL_UNSIGNED_INT8};
#if defined(CL_VERSION_2_0)
    const cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE2D, imageWidth / 4, imageHeight, 0, 0, 0, 0, 0, 0, {NULL}};
#else
    const cl_image_desc desc = {
        CL_MEM_OBJECT_IMAGE2D, imageWidth / 4, imageHeight, 0, 0, 0, 0, 0, 0, NULL};
#endif
    clImage2DOut =
        _wrapper->clCreateImage(context_, CL_MEM_READ_WRITE, &format, &desc, NULL, &status);
  }
  CHECK_RESULT(clImage2D == NULL, "AllocateOpenCLImage() failed");
}

void OCLImage2DFromBuffer::testReadImage(cl_mem image) {
  cl_int status = 0;
  size_t bufferSize = imageWidth * imageHeight;
  unsigned char* dstData = new unsigned char[bufferSize];

  size_t origin[] = {0, 0, 0};
  size_t region[] = {imageWidth / 4, imageHeight, 1};

  status =
      clEnqueueReadImage(cmdQueues_[_deviceId], image, 1, origin, region, 0, 0, dstData, 0, 0, 0);

  ::clFinish(cmdQueues_[_deviceId]);

  for (unsigned int y = 0; y < imageHeight; y++) {
    for (unsigned int x = 0; x < imageWidth / 4; x++) {
      for (unsigned int p = 0; p < 4; p++) {
        if (*(dstData + y * imageWidth + x * 4 + p) != p) {
          CHECK_RESULT(true, "CheckCLImage: *(dstData+y*imageWidth+x*4+p)!=p => %i != %i",
                       *(dstData + y * imageWidth + x * 4 + p), p);
          goto cleanup;
        }
      }
    }
  }
cleanup:

  delete[] dstData;
}

void OCLImage2DFromBuffer::testKernel() {
  CopyOpenCLImage(clImage2D);

  testReadImage(clImage2DOut);
}

unsigned int OCLImage2DFromBuffer::close(void) {
  if (clImage2DOriginal != NULL) clReleaseMemObject(clImage2DOriginal);
  if (clImage2D != NULL) clReleaseMemObject(clImage2D);
  if (clImage2DOut != NULL) clReleaseMemObject(clImage2DOut);
  if (buffer != NULL) clReleaseMemObject(buffer);
  return OCLTestImp::close();
}

void OCLImage2DFromBuffer::CopyOpenCLImage(cl_mem clImageSrc) {
  cl_int status = 0;

  // Set appropriate arguments to the kernel2D

  // input buffer image
  status = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &clImageSrc);
  CHECK_RESULT((status != CL_SUCCESS),
               "CopyOpenCLImage() failed at "
               "clSetKernelArg(kernel_,0,sizeof(cl_mem),&clImageSrc)");
  status = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &clImage2DOut);
  CHECK_RESULT((status != CL_SUCCESS),
               "CopyOpenCLImage() failed at "
               "clSetKernelArg(kernel_,1,sizeof(cl_mem),&clImage2DOut)");

  // Enqueue a kernel run call.
  size_t global_work_offset[] = {0, 0};
  size_t globalThreads[] = {imageWidth / 4, imageHeight};
  size_t localThreads[] = {blockSizeX, blockSizeY};

  status = clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, globalThreads, NULL, 0,
                                  NULL, 0);
  CHECK_RESULT((status != CL_SUCCESS), "CopyOpenCLImage() failed at clEnqueueNDRangeKernel");

  status = clFinish(cmdQueues_[_deviceId]);
  CHECK_RESULT((status != CL_SUCCESS), "CopyOpenCLImage() failed at clFinish");
}

void OCLImage2DFromBuffer::CompileKernel() {
  cl_int status = 0;

  size_t kernelSize = sizeof(strKernel);
  const char* strs = (const char*)&strKernel[0];

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strs, &kernelSize, &status);

  status = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], NULL, NULL, NULL);
  if (status != CL_SUCCESS) {
    if (status == CL_BUILD_PROGRAM_FAILURE) {
      cl_int logStatus;
      size_t buildLogSize = 0;
      logStatus = clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG,
                                        buildLogSize, NULL, &buildLogSize);
      std::string buildLog;
      buildLog.resize(buildLogSize);

      logStatus = clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG,
                                        buildLogSize, &buildLog[0], NULL);
      printf("%s", buildLog.c_str());
    }
    return;
  }
  // get a kernel object handle for a kernel with the given name
  kernel_ = _wrapper->clCreateKernel(program_, "image2imageCopy", &status);

  size_t kernel2DWorkGroupSize = 0;
  status = clGetKernelWorkGroupInfo(kernel_, devices_[_deviceId], CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(size_t), &kernel2DWorkGroupSize, 0);

  if ((blockSizeX * blockSizeY) > kernel2DWorkGroupSize) {
    if (blockSizeX > kernel2DWorkGroupSize) {
      blockSizeX = kernel2DWorkGroupSize;
      blockSizeY = 1;
    }
  }
}
