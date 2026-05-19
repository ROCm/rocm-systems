/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCLTestImp_H_
#define _OCLTestImp_H_

#include <string>
#include <vector>

#include "BaseTestImp.h"
#include "CL/cl.h"
#include "OCL/Thread.h"
#include "OCLTest.h"
#include "OCLWrapper.h"

class OCLTestImp : public BaseTestImp {
 public:
  OCLTestImp();
  virtual ~OCLTestImp();

 public:
  //! Abstract functions being defined here
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId,
                    unsigned int platformIndex);
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void) = 0;
  virtual unsigned int close(void);
  //! Functions to set class members

 public:
  void useCPU();
  int genIntRand(int a, int b);
  int genBitRand(int n);
  void accumulateCRC(const void* buffer, int len);
  void setOCLWrapper(OCLWrapper* wrapper);
  OCLTestImp* toOCLTestImp() { return this; }

  static OCLutil::Lock openDeviceLock;
  static OCLutil::Lock compileLock;

 protected:
  const std::vector<cl_mem>& buffers() const { return buffers_; }

  OCLWrapper* _wrapper;

  int _seed;

  // Common data of any CL program
  cl_int error_;
  cl_uint type_;
  cl_uint deviceCount_;
  cl_device_id* devices_;
  cl_platform_id platform_;
  std::vector<cl_command_queue> cmdQueues_;
  cl_context context_;

  cl_program program_;
  cl_kernel kernel_;
  std::vector<cl_mem> buffers_;
};

// useful for initialization of an array of data types for a test
#define DTYPE(x, y) DataType(x, #x, (unsigned int)y)

#endif
