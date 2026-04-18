// opencl/amdocl/cl_mem.hpp
// AMD OpenCL memory extension flags — opencl-layer only.
#pragma once
#include "CL/opencl.h"
// AMD OpenCL extension mem flags (not part of standard CL spec)
// These values match MemFlags::VaRangeAmd and MemFlags::FollowUserNumaPolicy
// in amd_types.hpp and are only needed at the OpenCL API boundary.
#define CL_MEM_FOLLOW_USER_NUMA_POLICY ((cl_mem_flags)(uint64_t(1) << 31))
#define CL_MEM_VA_RANGE_AMD            ((cl_mem_flags)(uint64_t(1) << 28))
