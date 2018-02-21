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
