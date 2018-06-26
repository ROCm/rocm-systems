/******************************************************************************
MIT License

Copyright (c) 2018 ROCm Core Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

#include <hsa.h>
#include <string.h>
#include <iostream>

#include "ctrl/run_kernel.h"
#include "ctrl/test_aql.h"
#include "simple_convolution/simple_convolution.h"

int main(int argc, char** argv) {
  const char* kiter_s = getenv("ROCP_KITER");
  const char* diter_s = getenv("ROCP_DITER");
  const int kiter = (kiter_s != NULL) ? atol(kiter_s) : 1;
  const int diter = (diter_s != NULL) ? atol(diter_s) : 1;
  TestHsa::HsaInstantiate();
  for (int i = 0; i < kiter; ++i) RunKernel<SimpleConvolution, TestAql>(argc, argv, diter);
  TestHsa::HsaShutdown();
  return 0;
}
