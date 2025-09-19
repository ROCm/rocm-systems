/*
 * Copyright (c) 2016-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * OpenCL is a trademark of Apple Inc. used under license by Khronos.
 */

#ifndef _ICD_H_
#define _ICD_H_

#include "icd_platform.h"

#ifndef CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#include <CL/cl.h>
#include <CL/cl_ext.h>

/*
 * type definitions
 */

typedef CL_API_ENTRY cl_int(CL_API_CALL* pfn_clIcdGetPlatformIDs)(
    cl_uint num_entries, cl_platform_id* platforms,
    cl_uint* num_platforms) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int(CL_API_CALL* pfn_clGetPlatformInfo)(
    cl_platform_id platform, cl_platform_info param_name, size_t param_value_size,
    void* param_value, size_t* param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY void*(CL_API_CALL* pfn_clGetExtensionFunctionAddress)(
    const char* function_name)CL_API_SUFFIX__VERSION_1_0;

typedef struct KHRicdVendorRec KHRicdVendor;

/*
 * KHRicdVendor
 *
 * Data for a single ICD vendor platform.
 */
struct KHRicdVendorRec {
  // the loaded library object (true type varies on Linux versus Windows)
  void* library;

  // the file name of the library
  char* libName;

  // the extension suffix for this platform
  char* suffix;

  // function pointer to the ICD platform IDs extracted from the library
  pfn_clGetExtensionFunctionAddress clGetExtensionFunctionAddress;

  // the platform retrieved from clGetIcdPlatformIDsKHR
  cl_platform_id platform;

  // next vendor in the list vendors
  KHRicdVendor* next;
};

// the global state
extern KHRicdVendor* khrIcdVendors;

/*
 * khrIcd interface
 */

// read vendors from system configuration and store the data
// loaded into khrIcdState.  this will call the OS-specific
// function khrIcdEnumerateVendors.  this is called at every
// dispatch function which may be a valid first call into the
// API (e.g, getPlatformIDs, etc).
void khrIcdInitialize(void);

// go through the list of vendors (in /etc/OpenCL.conf or through
// the registry) and call khrIcdVendorAdd for each vendor encountered
// n.b, this call is OS-specific
void khrIcdOsVendorsEnumerateOnce(void);

// read vendors from environment variables
void khrIcdVendorsEnumerateEnv(void);

// add a vendor's implementation to the list of libraries
void khrIcdVendorAdd(const char* libraryName);

// dynamically load a library.  returns NULL on failure
// n.b, this call is OS-specific
void* khrIcdOsLibraryLoad(const char* libraryName);

// get a function pointer from a loaded library.  returns NULL on failure.
// n.b, this call is OS-specific
void* khrIcdOsLibraryGetFunctionAddress(void* library, const char* functionName);

// unload a library.
// n.b, this call is OS-specific
void khrIcdOsLibraryUnload(void* library);

// parse properties and determine the platform to use from them
void khrIcdContextPropertiesGetPlatform(const cl_context_properties* properties,
                                        cl_platform_id* outPlatform);

// internal tracing macros
#if 0
#include <stdio.h>
#define KHR_ICD_TRACE(...)                                                                         \
  do {                                                                                             \
    fprintf(stderr, "KHR ICD trace at %s:%d: ", __FILE__, __LINE__);                               \
    fprintf(stderr, __VA_ARGS__);                                                                  \
  } while (0)
#ifdef _WIN32
#define KHR_ICD_WIDE_TRACE(...)                                                                    \
  do {                                                                                             \
    fwprintf(stderr, L"KHR ICD trace at %hs:%d: ", __FILE__, __LINE__);                            \
    fwprintf(stderr, __VA_ARGS__);                                                                 \
  } while (0)
#else
#define KHR_ICD_WIDE_TRACE(...)
#endif
#define KHR_ICD_ASSERT(x)                                                                          \
  do {                                                                                             \
    if (!(x)) {                                                                                    \
      fprintf(stderr, "KHR ICD assert at %s:%d: %s failed", __FILE__, __LINE__, #x);               \
    }                                                                                              \
  } while (0)
#else
#define KHR_ICD_TRACE(...)
#define KHR_ICD_WIDE_TRACE(...)
#define KHR_ICD_ASSERT(x)
#endif

// if handle is NULL then return invalid_handle_error_code
#define KHR_ICD_VALIDATE_HANDLE_RETURN_ERROR(handle, invalid_handle_error_code)                    \
  do {                                                                                             \
    if (!handle) {                                                                                 \
      return invalid_handle_error_code;                                                            \
    }                                                                                              \
  } while (0)

// if handle is NULL then set errcode_ret to invalid_handle_error and return NULL
// (NULL being an invalid handle)
#define KHR_ICD_VALIDATE_HANDLE_RETURN_HANDLE(handle, invalid_handle_error)                        \
  do {                                                                                             \
    if (!handle) {                                                                                 \
      if (errcode_ret) {                                                                           \
        *errcode_ret = invalid_handle_error;                                                       \
      }                                                                                            \
      return NULL;                                                                                 \
    }                                                                                              \
  } while (0)


#endif
