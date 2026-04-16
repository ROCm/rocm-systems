/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_AtomicSpeed20_H_
#define _OCL_AtomicSpeed20_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "OCLTestImp.h"

#define DEFAULT_WG_SIZE 256
#define NBINS 256
#define BITS_PER_PIX 8
#define NBANKS 16

#include "OCLPerfAtomicSpeed.h"

typedef struct {
  AtomicType atomicType;
  int inputScale;
} testOCLPerfAtomicSpeed20Struct;

// Define the OCLPerfAtomicSpeed20 class.
class OCLPerfAtomicSpeed20 : public OCLTestImp {
 public:
  OCLPerfAtomicSpeed20();
  virtual ~OCLPerfAtomicSpeed20();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  cl_command_queue cmd_queue_;
  std::vector<cl_program> _programs;
  std::vector<cl_kernel> _kernels;

  bool _atomicsSupported;
  bool _dataSizeTooBig;
  cl_uint _numLoops;

  // Histogram related stuff...
 private:
  cl_ulong _maxMemoryAllocationSize;
  cl_uint _inputNBytes;
  cl_uint _outputNBytes;

  cl_uint _nCurrentInputScale;
  cl_uint _workgroupSize;
  //    cl_uint nLoops;
  cl_uint _nThreads;
  cl_uint _nThreadsPerGroup;
  cl_uint _nGroups;
  cl_uint _n4Vectors;
  cl_uint _n4VectorsPerThread;
  cl_uint _nBins;
  cl_uint _nBytesLDSPerGrp;

  cl_uint* _input;
  cl_uint* _output;
  cl_mem _inputBuffer;
  cl_mem _outputBuffer;
  bool skip_;

  cl_uint _cpuhist[NBINS];
  cl_uint _cpuReductionSum;

  void calculateHostBin();
  void setupHistogram();
  bool VerifyResults(const AtomicType atomicType);
  void ResetGlobalOutput();

  // Methods that does the actual NDRange.
  void RunGlobalHistogram(const AtomicType atomicType);

  void CreateKernels(const AtomicType atomicType);
  void SetKernelArguments(const AtomicType atomicType);
  void PrintResults(const AtomicType atomicType, double totalTime);
};

#endif  // _OCL_AtomicSpeed20_H_
