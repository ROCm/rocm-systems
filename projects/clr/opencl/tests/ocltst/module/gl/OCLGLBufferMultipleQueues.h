/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
