#ifndef COMMON_COMMON_HPP
#define COMMON_COMMON_HPP

#include <cstdlib>
#include <iostream>

#include "hsa.h"
#include "hsa_ext_finalize.h"
#include "hsa_ext_amd.h"

#if defined(_MSC_VER)
  #define ALIGNED_(x) __declspec(align(x))
#else
  #if defined(__GNUC__)
    #define ALIGNED_(x) __attribute__ ((aligned(x)))
  #endif // __GNUC__
#endif // _MSC_VER

#define MULTILINE(...) # __VA_ARGS__

void ErrorCheck(hsa_status_t hsa_error_code);

hsa_status_t FindGpuDevice(hsa_agent_t agent, void *data);

hsa_status_t FindHostRegion(hsa_region_t region, void *data);

#endif // COMMON_COMMON_HPP
