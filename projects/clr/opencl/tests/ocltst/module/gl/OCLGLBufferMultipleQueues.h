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

#ifndef _OCL_GL_BUFFER_MULTIPLE_QUEUES_H_
#define _OCL_GL_BUFFER_MULTIPLE_QUEUES_H_

#include "OCLGLCommon.h"

class OCLGLBufferMultipleQueues : public OCLGLCommon {
 public:
  OCLGLBufferMultipleQueues();
  virtual ~OCLGLBufferMultipleQueues();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  static const int BUFFER_ELEMENTS_COUNT = 1024;
  static const int QUEUES_PER_DEVICE_COUNT = 2;
  std::vector<cl_command_queue> deviceCmdQueues_;  // Multiple queues per device (single device)
  std::vector<cl_mem> inputGLBufferPerQueue_;      // Input GL buffer per queue
  std::vector<cl_mem> outputGLBufferPerQueue_;     // Output GL buffer per queue
  std::vector<cl_mem> outputCLBufferPerQueue_;     // Input CL buffer per queue
  std::vector<GLuint> inGLBufferIDs_;              // Input GL buffers IDs
  std::vector<GLuint> outGLBufferIDs_;             // Output GL buffers IDs
};

#endif  // _OCL_GL_BUFFER_MULTIPLE_QUEUES_H_
