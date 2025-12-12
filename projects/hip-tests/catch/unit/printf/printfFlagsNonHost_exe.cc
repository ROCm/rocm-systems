/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

__global__ void test_kernel() {
  printf("%08d\n", 42);
  printf("%08i\n", -42);
  printf("%08u\n", 42);
  printf("%08g\n", 123.456);
  printf("%0+8d\n", 42);
  printf("%+d\n", -42);
  printf("%+08d\n", 42);
  printf("%-8s\n", "xyzzy");
  printf("% i\n", -42);
  printf("%-16.8d\n", 42);
  printf("%16.8d\n", 42);
}

int main() {
  test_kernel<<<1, 1>>>();
  static_cast<void>(hipDeviceSynchronize());
}
