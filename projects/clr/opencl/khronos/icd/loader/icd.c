/* Modifications Copyright(C) 2022 Advanced Micro Devices, Inc.
 * All rights reserved.
 */

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

#include "icd.h"
#include "icd_dispatch.h"
#include "icd_envvars.h"
#include <stdlib.h>
#include <string.h>

KHRicdVendor* khrIcdVendors = NULL;

// entrypoint to initialize the ICD and add all vendors
void khrIcdInitialize(void) {
  // enumerate vendors present on the system
  khrIcdOsVendorsEnumerateOnce();
}

void khrIcdVendorAdd(const char* libraryName) {
  void* library = NULL;
  cl_int result = CL_SUCCESS;
  pfn_clGetExtensionFunctionAddress p_clGetExtensionFunctionAddress = NULL;
  pfn_clIcdGetPlatformIDs p_clIcdGetPlatformIDs = NULL;
  cl_uint i = 0;
  cl_uint platformCount = 0;
  cl_platform_id* platforms = NULL;
  KHRicdVendor* vendorIterator = NULL;

  // require that the library name be valid
  if (!libraryName) {
    goto Done;
  }
  KHR_ICD_TRACE("attempting to add vendor %s...\n", libraryName);

  // load its library and query its function pointers
  library = khrIcdOsLibraryLoad(libraryName);
  if (!library) {
    KHR_ICD_TRACE("failed to load library %s\n", libraryName);
    goto Done;
  }

  // get the library's file name
  const char* libName = libraryName;
  const char* c;
  for (c = libraryName; *c; ++c) {
    if ((*c == '\\') || (*c == '/')) {
      libName = c + 1;
    }
  }

  // ensure that we haven't already loaded this vendor
  for (vendorIterator = khrIcdVendors; vendorIterator; vendorIterator = vendorIterator->next) {
    if (vendorIterator->library == library) {
      KHR_ICD_TRACE("already loaded vendor %s, nothing to do here\n", libraryName);
      goto Done;
    }
    if (!strcmp(vendorIterator->libName, libName)) {
      KHR_ICD_TRACE("already loaded library %s, nothing to do here\n", libName);
      goto Done;
    }
  }

  // get the library's clGetExtensionFunctionAddress pointer
  p_clGetExtensionFunctionAddress =
      (pfn_clGetExtensionFunctionAddress)(size_t)khrIcdOsLibraryGetFunctionAddress(
          library, "clGetExtensionFunctionAddress");
  if (!p_clGetExtensionFunctionAddress) {
    KHR_ICD_TRACE("failed to get function address clGetExtensionFunctionAddress\n");
    goto Done;
  }

  // use that function to get the clIcdGetPlatformIDsKHR function pointer
  p_clIcdGetPlatformIDs =
      (pfn_clIcdGetPlatformIDs)(size_t)p_clGetExtensionFunctionAddress("clIcdGetPlatformIDsKHR");
  if (!p_clIcdGetPlatformIDs) {
    KHR_ICD_TRACE("failed to get extension function address clIcdGetPlatformIDsKHR\n");
    goto Done;
  }

  // query the number of platforms available and allocate space to store them
  result = p_clIcdGetPlatformIDs(0, NULL, &platformCount);
  if (CL_SUCCESS != result) {
    KHR_ICD_TRACE("failed clIcdGetPlatformIDs\n");
    goto Done;
  }
  platforms = (cl_platform_id*)malloc(platformCount * sizeof(cl_platform_id));
  if (!platforms) {
    KHR_ICD_TRACE("failed to allocate memory\n");
    goto Done;
  }
  memset(platforms, 0, platformCount * sizeof(cl_platform_id));
  result = p_clIcdGetPlatformIDs(platformCount, platforms, NULL);
  if (CL_SUCCESS != result) {
    KHR_ICD_TRACE("failed clIcdGetPlatformIDs\n");
    goto Done;
  }

  // for each platform, add it
  for (i = 0; i < platformCount; ++i) {
    KHRicdVendor* vendor = NULL;
    char* suffix;
    size_t suffixSize;

    // call clGetPlatformInfo on the returned platform to get the suffix
    if (!platforms[i]) {
      continue;
    }
    result = platforms[i]->dispatch->clGetPlatformInfo(platforms[i], CL_PLATFORM_ICD_SUFFIX_KHR, 0,
                                                       NULL, &suffixSize);
    if (CL_SUCCESS != result) {
      continue;
    }
    suffix = (char*)malloc(suffixSize);
    if (!suffix) {
      continue;
    }
    result = platforms[i]->dispatch->clGetPlatformInfo(platforms[i], CL_PLATFORM_ICD_SUFFIX_KHR,
                                                       suffixSize, suffix, NULL);
    if (CL_SUCCESS != result) {
      free(suffix);
      continue;
    }

    // allocate a structure for the vendor
    vendor = (KHRicdVendor*)malloc(sizeof(*vendor));
    if (!vendor) {
      free(suffix);
      KHR_ICD_TRACE("failed to allocate memory\n");
      continue;
    }
    memset(vendor, 0, sizeof(*vendor));

    // populate vendor data
    vendor->library = khrIcdOsLibraryLoad(libraryName);
    if (!vendor->library) {
      free(suffix);
      free(vendor);
      KHR_ICD_TRACE("failed get platform handle to library\n");
      continue;
    }
    vendor->libName = (char*)malloc(strlen(libName) + 1);
    strcpy(vendor->libName, libName);
    vendor->clGetExtensionFunctionAddress = p_clGetExtensionFunctionAddress;
    vendor->platform = platforms[i];
    vendor->suffix = suffix;

    // add this vendor to the list of vendors at the tail
    {
      KHRicdVendor** prevNextPointer = NULL;
      for (prevNextPointer = &khrIcdVendors; *prevNextPointer;
           prevNextPointer = &((*prevNextPointer)->next));
      *prevNextPointer = vendor;
    }

    KHR_ICD_TRACE("successfully added vendor %s with suffix %s\n", libraryName, suffix);
  }

Done:

  if (library) {
    khrIcdOsLibraryUnload(library);
  }
  if (platforms) {
    free(platforms);
  }
}

// Get next file or dirname given a string list or registry key path.
// Note: the input string may be modified!
static char* loader_get_next_path(char* path) {
  size_t len;
  char* next;

  if (path == NULL) return NULL;
  next = strchr(path, PATH_SEPARATOR);
  if (next == NULL) {
    len = strlen(path);
    next = path + len;
  } else {
    *next = '\0';
    next++;
  }

  return next;
}

void khrIcdVendorsEnumerateEnv(void) {
  char* icdFilenames = khrIcd_secure_getenv("OCL_ICD_FILENAMES");
  char* cur_file = NULL;
  char* next_file = NULL;
  if (icdFilenames) {
    KHR_ICD_TRACE("Found OCL_ICD_FILENAMES environment variable.\n");

    next_file = icdFilenames;
    while (NULL != next_file && *next_file != '\0') {
      cur_file = next_file;
      next_file = loader_get_next_path(cur_file);

      khrIcdVendorAdd(cur_file);
    }

    khrIcd_free_getenv(icdFilenames);
  }
}

void khrIcdContextPropertiesGetPlatform(const cl_context_properties* properties,
                                        cl_platform_id* outPlatform) {
  if (properties == NULL && khrIcdVendors != NULL) {
    *outPlatform = khrIcdVendors[0].platform;
  } else {
    const cl_context_properties* property = (cl_context_properties*)NULL;
    *outPlatform = NULL;
    for (property = properties; property && property[0]; property += 2) {
      if ((cl_context_properties)CL_CONTEXT_PLATFORM == property[0]) {
        *outPlatform = (cl_platform_id)property[1];
      }
    }
  }
}
