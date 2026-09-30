/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _COMPILER_HIPCLASSKERNEL_H_
#define _COMPILER_HIPCLASSKERNEL_H_

#include <hip_test_common.hh>

static const int BLOCKS = 512;
static const int THREADS_PER_BLOCK = 1;
size_t NBOOL = BLOCKS * sizeof(bool);

class testFuncOvld {
 public:
  int __host__ __device__ func1(int a) { return a + 10; }

  int __host__ __device__ func1(int a, int b) { return a + b + 10; }
};

class testOvrB {
 public:
  int __host__ __device__ ovrdFunc1() { return 10; }
};

class testOvrD : public testOvrB {
 public:
  int __host__ __device__ ovrdFunc1() {
    int x = testOvrB::ovrdFunc1();
    return x + 20;
  }
};

class testFrndA {
 private:
  int fa = 10;

 public:
  friend class testFrndB;
};

class testFrndB {
 public:
  __host__ __device__ int showA() {
    testFrndA x;
    return x.fa;
  }
};

class testClassEmpty {};

class testPassByValue {
 public:
  int exI;
  char exC;
};

class testSizeA {
 public:
  float xa;
  int ia;
  double da;
  static char ca;
};

class testSizeDerived : testSizeA {
 public:
  float fd;
};

#pragma pack(push, 4)
class testSizeDerived2 : testSizeA {
 public:
  float fd;
};
#pragma pack(pop)

class testSizeB {
 public:
  char ab;
  int ib;
  char cb;
};

class testSizeVirtDer : public virtual testSizeB {
 public:
  int ivd;
};

class testSizeVirtDer1 : public virtual testSizeB {
 public:
  int ivd1;
};

class testSizeDerMulti : public testSizeVirtDer, public testSizeVirtDer1 {
 public:
  int ivd2;
};

#pragma pack(push, 4)
class testSizeVirtDerPack : public virtual testSizeB {
 public:
  int ivd;
};
#pragma pack(pop)

class testSizeC {
 public:
  char ac;
  int ic;
  int bc[2];
};

class testSizeDV {
 public:
  virtual void __host__ __device__ func1();

 private:
  int iDV;
};

class testSizeDerivedDV : testSizeDV {
 public:
  virtual void __host__ __device__ funcD1();

 private:
  int iDDV;
};

struct refStructSizes {
  size_t sizeOftestSizeDV;
  size_t sizeOftestSizeDerivedDV;
  size_t sizeOftestSizeVirtDer;
  size_t sizeOftestSizeVirtDerPack;
  size_t sizeOftestSizeDerMulti;
};

#pragma pack(push, 1)
class testSizeP1 {
 public:
  char ap;
  int ip;
  char cp;
};
#pragma pack(pop)

#pragma pack(push, 1)
class testSizeP2 {
 public:
  char ap1;
  int ip1;
  int bp1[2];
};
#pragma pack(pop)

#pragma pack(push, 2)
class testSizeP3 {
 public:
  char ap2;
  int ip2;
  char cp2;
};
#pragma pack(pop)

class testDeviceClass {
 public:
  int iVar;
  __host__ __device__ testDeviceClass();
  __host__ __device__ testDeviceClass(int a);
  __host__ __device__ ~testDeviceClass();
};

__host__ __device__ testDeviceClass::testDeviceClass() { iVar = 5; }

__host__ __device__ testDeviceClass::testDeviceClass(int a) { iVar = a; }

bool* AllocateHostMemory(void) {
  bool* result_ech;
  HIPCHECK(hipHostMalloc(&result_ech, NBOOL, hipHostMallocDefault));
  return result_ech;
}

bool* AllocateDeviceMemory(void) {
  bool* result_ecd;
  HIPCHECK(hipMalloc(&result_ecd, NBOOL));
  HIPCHECK(hipMemset(result_ecd, false, NBOOL));
  return result_ecd;
}

void VerifyResult(bool* result_ech, bool* result_ecd) {
  HIPCHECK(hipMemcpy(result_ech, result_ecd, BLOCKS * sizeof(bool), hipMemcpyDeviceToHost));
  // validation on host side
  for (int i = 0; i < BLOCKS; i++) {
    HIPASSERT(result_ech[i] == true);
  }
}

void FreeMem(bool* result_ech, bool* result_ecd) {
  HIPCHECK(hipHostFree(result_ech));
  HIPCHECK(hipFree(result_ecd));
}

#endif  // _HIPCLASSKERNEL_H_
