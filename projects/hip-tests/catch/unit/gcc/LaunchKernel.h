
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef __cplusplus
extern "C" {
#endif

struct things {
  char c;
  short s;
  int i;
};

typedef enum func { mykernel, mykernel1, mykernel2, mykernel3, mykernel4 } func;

extern const void* getKernelFunc(enum func f);

int launchKernel();
int hipMallocfunc();

#ifdef __cplusplus
}
#endif
