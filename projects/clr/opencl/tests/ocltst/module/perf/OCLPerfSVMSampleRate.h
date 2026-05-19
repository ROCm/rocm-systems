/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_SVMSAMPLERATE_H_
#define _OCL_SVMSAMPLERATE_H_

#include "OCLTestImp.h"

class OCLPerfSVMSampleRate : public OCLTestImp {
 public:
  OCLPerfSVMSampleRate();
  virtual ~OCLPerfSVMSampleRate();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void setData(void* buffer, unsigned int data);
  void checkData(void* buffer);
  void setKernel(void);

  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  void** inBuffer_;
  void* outBuffer_;

  unsigned int width_;
  unsigned int bufSize_;
  unsigned int outBufSize_;
  static const unsigned int MAX_ITERATIONS = 25;
  unsigned int numBufs_;
  unsigned int typeIdx_;
  unsigned int svmMode_;

  bool skip_;
  bool coarseGrainBuffer_;
  bool fineGrainBuffer_;
  bool fineGrainSystem_;
  std::string testdesc;
};

#endif  // _OCL_SVMSAMPLERATE_H_
