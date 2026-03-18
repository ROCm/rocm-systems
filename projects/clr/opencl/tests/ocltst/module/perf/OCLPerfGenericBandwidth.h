/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_GenericBandwidth_H_
#define _OCL_GenericBandwidth_H_

#include "OCLTestImp.h"

class OCLPerfGenericBandwidth : public OCLTestImp {
 public:
  OCLPerfGenericBandwidth();
  virtual ~OCLPerfGenericBandwidth();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void genShader(unsigned int idx);
  void setData(cl_mem buffer, float data);
  void checkData(cl_mem buffer);

  static const unsigned int NUM_ITER = 100;

  cl_mem inBuffer_;
  cl_mem outBuffer_;

  unsigned int width_;
  unsigned int bufSize_;
  unsigned int vecSizeIdx_;
  unsigned int numReads_;
  unsigned int shaderIdx_;
  unsigned int dataSizeBytes_;
  cl_char useLDS_;
  bool failed;
};

#endif  // _OCL_GenericBandwidth_H_
