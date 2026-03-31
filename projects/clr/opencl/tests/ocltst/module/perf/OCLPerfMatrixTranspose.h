/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_MATRIX_TRANSPOSE_H_
#define _OCL_MATRIX_TRANSPOSE_H_

#include "OCLTestImp.h"

class OCLPerfMatrixTranspose : public OCLTestImp {
 public:
  OCLPerfMatrixTranspose();
  virtual ~OCLPerfMatrixTranspose();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void setData(cl_mem buffer);
  void fillData(cl_mem buffer, unsigned int data);
  void checkData(cl_mem buffer);

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  cl_mem inBuffer_;
  cl_mem outBuffer_;
  cl_int error_;

  unsigned int width_;
  unsigned int height_;
  unsigned int bufSize_;
  unsigned int blockSize_;
  static const unsigned int MAX_ITERATIONS = 50;
};

#endif  // _OCL_MATRIX_TRANSPOSE_H_
