/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_TEXTUREMEMLATENCY_H_
#define _OCL_TEXTUREMEMLATENCY_H_

#include "OCLTestImp.h"

class OCLPerfTextureMemLatency : public OCLTestImp {
 public:
  OCLPerfTextureMemLatency();
  virtual ~OCLPerfTextureMemLatency();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void genShader(void);
  void setData(cl_mem buffer, unsigned int data);
  void checkData(cl_mem buffer);

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  cl_kernel kernel2_;
  cl_mem inBuffer_;
  cl_mem outBuffer_;
  cl_int error_;

  unsigned int width_;
  unsigned int height_;
  size_t image_row_pitch;
  size_t image_slice_pitch;
  unsigned int bufSizeDW_;
  unsigned int repeats_;
  unsigned int maxSize_;
};

#endif  // _OCL_TEXTUREMEMLATENCY_H_
