/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
*******************************************************************************/

#include <hsa.h>
#include <hsa_api_trace.h>
#include <hsa_ven_amd_loader.h>
#include <stdio.h>
#include <stdlib.h>

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define HSA_RT(call) \
  do { \
    const hsa_status_t status = call; \
    if (status != HSA_STATUS_SUCCESS) { \
      printf("error \"%s\"\n", #call); fflush(stdout); \
      abort(); \
    } \
  } while(0)

// HSA API intercepting primitives
decltype(hsa_executable_freeze)* hsa_executable_freeze_fn;
hsa_ven_amd_loader_1_01_pfn_t loader_api_table{};

hsa_status_t code_object_callback(
  hsa_executable_t executable,
  hsa_loaded_code_object_t loaded_code_object,
  void* arg)
{
  printf("code_object_callback\n"); fflush(stdout);

  uint64_t load_base = 0;
  uint64_t load_size = 0;
  uint64_t load_delta = 0;
  uint32_t uri_len = 0;
  char* uri_str = NULL;

  HSA_RT(loader_api_table.hsa_ven_amd_loader_loaded_code_object_get_info(
    loaded_code_object,
    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
    &load_base));
  HSA_RT(loader_api_table.hsa_ven_amd_loader_loaded_code_object_get_info(
    loaded_code_object,
    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
    &load_size));
  HSA_RT(loader_api_table.hsa_ven_amd_loader_loaded_code_object_get_info(
    loaded_code_object,
    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
    &load_delta));
  HSA_RT(loader_api_table.hsa_ven_amd_loader_loaded_code_object_get_info(
    loaded_code_object,
    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
    &uri_len));

  uri_str = (char*)calloc(uri_len + 1, sizeof(char));
  if (!uri_str) {
    perror("calloc");
    abort();
  }

  HSA_RT(loader_api_table.hsa_ven_amd_loader_loaded_code_object_get_info(
    loaded_code_object,
    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI,
    uri_str));

  printf("load_base(0x%lx)\n", load_base); fflush(stdout);
  printf("load_size(0x%lx)\n", load_size); fflush(stdout);
  printf("load_delta(0x%lx)\n", load_delta); fflush(stdout);
  printf("uri_len(%u)\n", uri_len); fflush(stdout);
  printf("uri_str(\"%s\")\n", uri_str); fflush(stdout);

  free(uri_str);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_executable_freeze_interceptor(
  hsa_executable_t executable,
  const char *options)
{
  HSA_RT(loader_api_table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
    executable,
    code_object_callback,
    NULL));
  HSA_RT(hsa_executable_freeze_fn(
    executable,
    options));
  return HSA_STATUS_SUCCESS;
}

// HSA-runtime tool on-load method
extern "C" PUBLIC_API bool OnLoad(HsaApiTable* table,
                                  uint64_t runtime_version,
                                  uint64_t failed_tool_count,
                                  const char* const* failed_tool_names)
{
  printf("OnLoad: begin\n"); fflush(stdout);
  // intercepting hsa_executable_freeze API
  hsa_executable_freeze_fn = table->core_->hsa_executable_freeze_fn;
  table->core_->hsa_executable_freeze_fn = hsa_executable_freeze_interceptor;
  // Fetching AMD Loader HSA extension API
  HSA_RT(hsa_system_get_major_extension_table(
    HSA_EXTENSION_AMD_LOADER,
    1,
    sizeof(hsa_ven_amd_loader_1_01_pfn_t),
    &loader_api_table));
  printf("OnLoad: end\n"); fflush(stdout);
  return true;
}

extern "C" PUBLIC_API void OnUnload() {
  printf("OnUnload\n"); fflush(stdout);
}

extern "C" CONSTRUCTOR_API void constructor() {
  printf("constructor\n"); fflush(stdout);
}

extern "C" DESTRUCTOR_API void destructor() {
  printf("destructor\n"); fflush(stdout);
}
