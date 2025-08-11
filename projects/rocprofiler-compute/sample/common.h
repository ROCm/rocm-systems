#pragma once

#include <hip/hip_runtime.h>
#include <iostream>

#define hipCheck(stmt)                                                         \
  do {                                                                         \
    hipError_t err = stmt;                                                     \
    if (err != hipSuccess) {                                                   \
      char msg[256];                                                           \
      sprintf(msg, "%s in file %s, function %s, line %d\n", #stmt, __FILE__,   \
              __FUNCTION__, __LINE__);                                         \
      std::string errstring = hipGetErrorString(err);                          \
      std::cerr << msg << "\t" << errstring << std::endl;                      \
      throw std::runtime_error(msg);                                           \
    }                                                                          \
  } while (0)
