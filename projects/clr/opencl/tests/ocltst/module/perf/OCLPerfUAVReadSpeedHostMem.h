/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_UAVReadSpeedHostMem_H_
#define _OCL_UAVReadSpeedHostMem_H_

#include "OCLTestImp.h"

class OCLPerfUAVReadSpeedHostMem : public OCLTestImp {
 public:
  OCLPerfUAVReadSpeedHostMem();
  virtual ~OCLPerfUAVReadSpeedHostMem();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void genShader(unsigned int type, unsigned int vecWidth, unsigned int numReads);
  void setData(cl_mem buffer, float data);
  void checkData(cl_mem buffer);

  static const unsigned int NUM_ITER = 100;

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  cl_mem inBuffer_;
  cl_mem outBuffer_;
  cl_mem constBuffer_;
  cl_int error_;

  unsigned int width_;
  unsigned int bufSize_;
  unsigned int vecSizeIdx_;
  unsigned int numReads_;
  unsigned int typeIdx_;
  bool isAMD;
  bool cached_;
};

#endif  // _OCL_UAVReadSpeedHostMem_H_
