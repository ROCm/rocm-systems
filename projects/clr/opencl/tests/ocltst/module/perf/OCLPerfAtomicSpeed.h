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

#ifndef _OCL_AtomicSpeed_H_
#define _OCL_AtomicSpeed_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "OCLTestImp.h"

#define DEFAULT_WG_SIZE 256
#define NBINS 256
#define BITS_PER_PIX 8
#define NBANKS 16

// Define the atomic type to test.
enum AtomicType {
  LocalHistogram = 0,
  GlobalHistogram,
  Global4Histogram,
  LocalReductionNoAtomics,
  Local4ReductionNoAtomics,
  LocalReductionAtomics,
  Local4ReductionAtomics,
  GlobalWGReduction,
  Global4WGReduction,
  GlobalAllToZeroReduction,
  Global4AllToZeroReduction,
};

typedef struct {
  AtomicType atomicType;
  int inputScale;
} testOCLPerfAtomicSpeedStruct;

// Define the OCLPerfAtomicSpeed class.
class OCLPerfAtomicSpeed : public OCLTestImp {
 public:
  OCLPerfAtomicSpeed();
  virtual ~OCLPerfAtomicSpeed();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  cl_context context_;
  cl_command_queue cmd_queue_;
  std::vector<cl_program> _programs;
  std::vector<cl_kernel> _kernels;
  cl_device_id device;

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

  cl_uint _cpuhist[NBINS];
  cl_uint _cpuReductionSum;

  void calculateHostBin();
  void setupHistogram();
  bool VerifyResults(const AtomicType atomicType);
  void ResetGlobalOutput();

  // Methods that does the actual NDRange.
  void RunLocalHistogram();
  void RunLocalReduction(const AtomicType atomicType);
  void RunGlobalHistogram(const AtomicType atomicType);

  void CreateKernels(const AtomicType atomicType);
  bool IsReduction(const AtomicType atomicType);
  void SetKernelArguments(const AtomicType atomicType);
  void PrintResults(const AtomicType atomicType, double totalTime);
};

#endif  // _OCL_AtomicSpeed_H_
