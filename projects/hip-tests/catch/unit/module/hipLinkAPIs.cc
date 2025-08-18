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

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <vector>

#define SPIRV_FILE "addKernel.spv"
#define SPIRV_BUNDLED_FILE "addKernel-bundle.spv"

#if HT_AMD
static inline bool load_co_from_file(const char *filename,
                                     std::vector<char> *co_source) {
  std::ifstream file_stream{filename,
                            std::ios_base::in | std::ios_base::binary};
  if (!file_stream.good()) {
    return false;
  }

  file_stream.seekg(0, std::ios::end);
  std::streampos file_size = file_stream.tellg();
  file_stream.seekg(0, std::ios::beg);

  // Read the file contents
  co_source->resize(file_size);
  file_stream.read(co_source->data(), file_size);

  file_stream.close();

  return true;
}

/**
 * Test Description
 * ------------------------
 * - This test case tests the following scenario:-
 * - 1) Create hip link and extract module from Bundled/Un-bundled
 * -    file using hip link APIs,
 * -      Section 1 : Bundled with Data
 * -      Section 2 : Un-bundled with Data
 * -      Section 3 : Bundled with File
 * -      Section 4 : Un-bundled with File
 * - 4) Get the Global device data from module and validate it.
 * Test source
 * ------------------------
 * - unit/module/hipLinkAPIs.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipLinkAPIs_GlobalDeviceData") {
  const char *isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm",
                           "-inlinehint-threshold=1"};
  std::vector<hipJitOption> jit_options = {hipJitOptionIRtoISAOptExt,
                                           hipJitOptionIRtoISAOptCountExt};
  size_t isaoptssize = 4;
  void *lopts[] = {reinterpret_cast<void *>(isaopts),
                   reinterpret_cast<void *>(isaoptssize)};

  hipLinkState_t linkState;

  HIP_CHECK(hipLinkCreate(jit_options.size(), jit_options.data(),
                          lopts, &linkState));

  const char *filename = SPIRV_FILE;
  bool from_file;

  SECTION("Link Add Data with Bundled Spirv") {
    from_file = false;
    filename = SPIRV_BUNDLED_FILE;
  }
  SECTION("Link Add Data with UnBundled Spirv") {
    from_file = false;
    filename = SPIRV_FILE;
  }
  SECTION("Link Add File with Bundled Spirv") {
    from_file = true;
    filename = SPIRV_BUNDLED_FILE;
  }
  SECTION("Link Add File with UnBundled Spirv") {
    from_file = true;
    filename = SPIRV_FILE;
  }

  if (!from_file) {
    std::vector<char> co_source;
    REQUIRE(load_co_from_file(filename, &co_source) == true);
    HIP_CHECK(hipLinkAddData(linkState, hipJitInputSpirv,
                             reinterpret_cast<void *>(co_source.data()),
                             co_source.size(), "LinkSPIRV1", 0,
                             nullptr, nullptr));
  } else {
    HIP_CHECK(hipLinkAddFile(linkState, hipJitInputSpirv, filename, 0, nullptr,
                             nullptr));
  }

  void *linkOut;
  size_t linkSize = 0;
  HIP_CHECK(hipLinkComplete(linkState, &linkOut, &linkSize));

  hipModule_t module;
  HIP_CHECK(hipModuleLoadData(&module, linkOut));

  hipDeviceptr_t dptr = nullptr;
  size_t bytes = 0;
  HIP_CHECK(hipModuleGetGlobal(&dptr, &bytes, module, "globalDevData"));
  REQUIRE(dptr != nullptr);
  REQUIRE(bytes == 4);

  int hostData = 0;
  HIP_CHECK(hipMemcpy(&hostData, dptr, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(hostData == 10);

  HIP_CHECK(hipLinkDestroy(linkState));
  HIP_CHECK(hipModuleUnload(module));
}

/**
 * Test Description
 * ------------------------
 * - This test case tests the following scenario:-
 * - 1) Create hip link and extract module from Bundled/Un-bundled
 * -    file using hip link APIs,
 * -      Section 1 : Bundled with Data
 * -      Section 2 : Un-bundled with Data
 * -      Section 3 : Bundled with File
 * -      Section 4 : Un-bundled with File
 * - 4) Get the texture reference from module and validate it.
 * Test source
 * ------------------------
 * - unit/module/hipLinkAPIs.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipLinkAPIs_Texture") {
  CHECK_IMAGE_SUPPORT

  const char *isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm",
                           "-inlinehint-threshold=1"};
  std::vector<hipJitOption> jit_options = {hipJitOptionIRtoISAOptExt,
                                           hipJitOptionIRtoISAOptCountExt};
  size_t isaoptssize = 4;
  void *lopts[] = {reinterpret_cast<void *>(isaopts),
                   reinterpret_cast<void *>(isaoptssize)};

  hipLinkState_t linkState;

  HIP_CHECK(hipLinkCreate(jit_options.size(), jit_options.data(),
                          lopts, &linkState));

  const char *filename = SPIRV_FILE;
  bool from_file;

  SECTION("Link Add Data with Bundled Spirv") {
    from_file = false;
    filename = SPIRV_BUNDLED_FILE;
  }
  SECTION("Link Add Data with UnBundled Spirv") {
    from_file = false;
    filename = SPIRV_FILE;
  }
  SECTION("Link Add File with Bundled Spirv") {
    from_file = true;
    filename = SPIRV_BUNDLED_FILE;
  }
  SECTION("Link Add File with UnBundled Spirv") {
    from_file = true;
    filename = SPIRV_FILE;
  }

  if (!from_file) {
    std::vector<char> co_source;
    REQUIRE(load_co_from_file(filename, &co_source) == true);
    HIP_CHECK(hipLinkAddData(linkState, hipJitInputSpirv,
                             reinterpret_cast<void *>(co_source.data()),
                             co_source.size(), "LinkSPIRV1", 0,
                             nullptr, nullptr));
  } else {
    HIP_CHECK(hipLinkAddFile(linkState, hipJitInputSpirv, filename, 0, nullptr,
                             nullptr));
  }

  void *linkOut;
  size_t linkSize = 0;
  HIP_CHECK(hipLinkComplete(linkState, &linkOut, &linkSize));

  hipModule_t module;
  HIP_CHECK(hipModuleLoadData(&module, linkOut));

  hipTexRef texRef = nullptr;
  HIP_CHECK(hipModuleGetTexRef(&texRef, module, "tex"));
  REQUIRE(texRef != nullptr);
  REQUIRE(texRef->format == HIP_AD_FORMAT_FLOAT);

  HIP_CHECK(hipLinkDestroy(linkState));
  HIP_CHECK(hipModuleUnload(module));
}

/**
 * Test Description
 * ------------------------
 * - This test case tests the following Negetive scenarios
 * - for the hipLinkAddData API:-
 * -  1) With Invalid link state
 * -  2) With Invalid data
 * -  3) With Invalid data size
 * Test source
 * ------------------------
 * - unit/module/hipLinkAPIs.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipLinkAPIs_hipLinkAddData_Negative") {
  const char *isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm",
                           "-inlinehint-threshold=1"};
  std::vector<hipJitOption> jit_options = {hipJitOptionIRtoISAOptExt,
                                           hipJitOptionIRtoISAOptCountExt};
  size_t isaoptssize = 4;
  void *lopts[] = {reinterpret_cast<void *>(isaopts),
                   reinterpret_cast<void *>(isaoptssize)};

  hipLinkState_t linkState;

  HIP_CHECK(hipLinkCreate(jit_options.size(), jit_options.data(),
                          lopts, &linkState));

  const char *filename = SPIRV_FILE;
  std::vector<char> co_source;
  REQUIRE(load_co_from_file(filename, &co_source) == true);

  SECTION("Invalid link state") {
    HIP_CHECK_ERROR(hipLinkAddData(nullptr, hipJitInputSpirv,
                                   reinterpret_cast<void *>(co_source.data()),
                                   co_source.size(), "LinkSPIRV1", 0, nullptr,
                                   nullptr),
                    hipErrorInvalidHandle);
  }

  SECTION("Invalid data") {
    HIP_CHECK_ERROR(hipLinkAddData(nullptr, hipJitInputSpirv, nullptr,
                                   co_source.size(), "LinkSPIRV1", 0, nullptr,
                                   nullptr),
                    hipErrorInvalidImage);
  }

  SECTION("Invalid data size") {
    HIP_CHECK_ERROR(hipLinkAddData(nullptr, hipJitInputSpirv,
                                   reinterpret_cast<void *>(co_source.data()),
                                   0, "LinkSPIRV1", 0, nullptr, nullptr),
                    hipErrorInvalidImage);
  }

  void *linkOut;
  size_t linkSize = 0;
  HIP_CHECK(hipLinkComplete(linkState, &linkOut, &linkSize));
  HIP_CHECK(hipLinkDestroy(linkState));
}

/**
 * Test Description
 * ------------------------
 * - This test case tests the following Negetive scenarios
 * - for the hipLinkComplete API:-
 * -  1) With Invalid link state
 * -  2) With Invalid link out
 * -  3) With Invalid link size
 * Test source
 * ------------------------
 * - unit/module/hipLinkAPIs.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipLinkAPIs_hipLinkComplete_Negative") {
  const char *isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm",
                           "-inlinehint-threshold=1"};
  std::vector<hipJitOption> jit_options = {hipJitOptionIRtoISAOptExt,
                                           hipJitOptionIRtoISAOptCountExt};
  size_t isaoptssize = 4;
  void *lopts[] = { reinterpret_cast<void *>(isaopts),
                    reinterpret_cast<void *>(isaoptssize) };

  hipLinkState_t linkState;
  HIP_CHECK(hipLinkCreate(jit_options.size(), jit_options.data(),
                          lopts, &linkState));
  HIP_CHECK(hipLinkAddFile(linkState, hipJitInputSpirv, SPIRV_FILE, 0, nullptr,
                           nullptr));

  void *linkOut;
  size_t linkSize = 0;

  SECTION("Invalid link state") {
    HIP_CHECK_ERROR(hipLinkComplete(nullptr, &linkOut, &linkSize),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid link out") {
    HIP_CHECK_ERROR(hipLinkComplete(linkState, nullptr, &linkSize),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid link size") {
    HIP_CHECK_ERROR(hipLinkComplete(linkState, &linkOut, nullptr),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipLinkDestroy(linkState));
}

/**
 * Test Description
 * ------------------------
 * - This test case tests the Negetive scenario
 * - for the hipLinkComplete With Invalid link state
 * Test source
 * ------------------------
 * - unit/module/hipLinkAPIs.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipLinkAPIs_hipLinkDestroy_Negative") {
  HIP_CHECK_ERROR(hipLinkDestroy(nullptr), hipErrorInvalidValue);
}

#endif
