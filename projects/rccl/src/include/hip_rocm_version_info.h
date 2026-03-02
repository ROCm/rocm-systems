/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RCCL_HIP_ROCM_VERSION_INFO_H_
#define RCCL_HIP_ROCM_VERSION_INFO_H_

#define STR2(v) #v
#define STR(v) STR2(v)

// HIP version info retrieval
#if ROCM_VERSION >= 50000
   #define HIP_BUILD_INFO STR(HIP_VERSION_MAJOR) "." STR(HIP_VERSION_MINOR) "." STR(HIP_VERSION_PATCH) "-" HIP_VERSION_GITHASH
// HIP Githash info not available in older ROCm versions < 5.0
#elif ROCM_VERSION >= 40000
   #define HIP_BUILD_INFO STR(HIP_VERSION_MAJOR) "." STR(HIP_VERSION_MINOR) "." STR(HIP_VERSION_PATCH)
#else
   #define HIP_BUILD_INFO "Unknown"
#endif

// ROCm version info retrieval  
#if ROCM_VERSION >= 60000
   // rocm_version.h moved to rocm/include/rocm-core from ROCm 6.0
   #include <rocm-core/rocm_version.h>
#else
   // rocm-core/rocm_version.h not present in some ROCm versions < 6.0. 
   // So, including it from rocm/include/rocm_version.h
   #if ROCM_VERSION >= 50000
      #include <rocm_version.h>
      //ROCM_BUILD_INFO not defined in ROCm Versions < 5.50
      #ifndef ROCM_BUILD_INFO
         #define ROCM_BUILD_INFO STR(ROCM_VERSION_MAJOR) "." STR(ROCM_VERSION_MINOR) "." STR(ROCM_VERSION_PATCH)
      #endif
   //ROCm version info not available for ROCm versions < 5.0
   #else
      #define ROCM_BUILD_INFO "Unknown"
   #endif
#endif

#endif