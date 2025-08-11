/*
##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el



A instruction mix exerciser example, written by Gina Sitaraman and Nicholas Curtis [AMD].
Although inline assembly is inherently unportable, this is expected to work on all CDNA accelerators.
*/


#include "common.h"

__global__ void kernelasm() {
  // int32
  int i, j;
  asm volatile("v_add_u32_e32 %0, %1, %0\n" : "=v"(j) : "v"(i));

  // int 64
  long int l1, l2;
  asm volatile("v_cmp_eq_i64 %0, %1\n" : "=v"(l2) : "v"(l1), "v"(i));

  // fp32: add, mul, transcendental and fma
  float f1, f2;
  asm volatile(
      "v_add_f32_e32 %0, %1, %0\n"
      "v_mul_f32_e32 %0, %1, %0\n"
      "v_sqrt_f32 %0, %1\n"
      "v_fma_f32 %0, %1, %0, %1\n"
      : "=v"(f1)
      : "v"(f2));

  // fp64: add, mul, transcendental and fma
  double d1, d2, d3, d4;
  asm volatile(
      "v_add_f64 %0, %1, %0\n"
      "v_mul_f64 %0, %1, %0\n"
      "v_fma_f64 %0, %1, %0, %1\n"
      "v_sqrt_f64 %0, %1\n"
      "v_min_f64 %0, %1, %0\n"
      : "+v"(d1)
      : "v"(d2));

  // fp16: add, mul, transcendental and fma
  _Float16 h1, h2;
  asm volatile(
      "v_add_f16_e32 %0, %1, %0\n"
      "v_mul_f16_e32 %0, %1, %0\n"
      "v_sqrt_f16 %0, %1\n"
      "v_cvt_f16_f32 %0 %2\n"
      "v_fma_f16 %0, %1, %0, %0\n"
      : "=v"(h2)
      : "v"(h1), "v"(f1));

  // MFMA  ops
  double2 dd;
  unsigned short us;
  long2 ll;
#if defined(__gfx90a__)
  asm volatile("v_mfma_f64_4x4x4f64 %0 %1 %2 %3\n"
               : "=v"(d4)
               : "v"(d1), "v"(d2), "v"(d3));
  asm volatile("v_mfma_f32_16x16x4f32 %0 %1 %2 1\n"
               : "=v"(dd)
               : "v"(f1), "v"(f2));
  asm volatile("v_mfma_f32_16x16x16f16 %0 %1 %2 1\n"
               : "=v"(dd)
               : "v"(d1), "v"(d2));
  asm volatile("v_mfma_f32_16x16x8bf16 %0 %1 %2 1\n"
               : "=v"(dd)
               : "v"(f1), "v"(f2));
  asm volatile("v_mfma_i32_16x16x16i8 %0 %1 %2 1\n"
               : "=v"(ll)
               : "v"(i), "v"(j));
#endif

  // Scalar op
  asm volatile("s_add_i32 %0 %1 %0\n" : "=s"(j) : "s"(i));

  // LDS
  asm volatile("ds_read_b32 %0 %0\n" : "=v"(i) : "v"(j));

  // Branch
  asm volatile(
      "s_branch .LDUMMY\n"
      ".LDUMMY:\n"
      "s_endpgm\n");
}
int main() {
  kernelasm<<<1, 64>>>();
  hipCheck(hipDeviceSynchronize());
}
