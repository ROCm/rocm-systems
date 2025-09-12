#include "memoryCommon.hh"
#include "memoryGlobal.hh"

void set_value(int const value) {
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalVar), &value, sizeof(value)));
}

int get_value() {
  int value;
  HIP_CHECK(hipMemcpyFromSymbol(&value, HIP_SYMBOL(globalVar), sizeof(value)));
  return value;
}
