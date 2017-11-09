#include <hsa.h>
#include <string.h>
#include <iostream>

#include "ctrl/run_kernel.h"
#include "ctrl/test_aql.h"
#include "simple_convolution/simple_convolution.h"

int main(int argc, char** argv) {
  TestHsa::HsaInstantiate();
  for (int i = 0; i < 3; ++i) RunKernel<SimpleConvolution, TestAql>(argc, argv);
  TestHsa::HsaShutdown();
  return 0;
}
