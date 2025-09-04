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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <fstream>
#ifdef __linux__
#include <unistd.h>
#endif

constexpr int LEN = 64;
constexpr auto SIZE = (LEN << 2);
constexpr auto CODE_OBJ_SINGLEARCH = "vcpy_kernel.code";
constexpr auto kernel_name = "hello_world";
#ifdef __linux__
constexpr int COMMAND_LEN = 256;
constexpr auto CODE_OBJ_MULTIARCH = "vcpy_kernel_multarch.code";
#endif

/**
 * @addtogroup hipModuleLoad
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleLoad(hipModule_t* module, const char* fname)` -
 * Loads code object from file into a module
 */

/**
 * Test Description
 * ------------------------
 * - Test case to load and execute a code object file for the current GPU architecture.
 * - Test case to load and execute a code object file for the multiple GPU architectures including
 the current

 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleLoad.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
*/

bool testCodeObjFile(const char* codeObjFile) {
  float *A, *B, *Ad, *Bd;
  A = new float[LEN];
  B = new float[LEN];

  for (uint32_t i = 0; i < LEN; i++) {
    A[i] = i * 1.0f;
    B[i] = 0.0f;
  }

  HIP_CHECK(hipMalloc(&Ad, SIZE));
  HIP_CHECK(hipMalloc(&Bd, SIZE));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

  hipModule_t Module;
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, codeObjFile));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void* _Ad;
    void* _Bd;
  } args;
  args._Ad = reinterpret_cast<void*>(Ad);
  args._Bd = reinterpret_cast<void*>(Bd);
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, NULL,
                                  reinterpret_cast<void**>(&config)));

  HIP_CHECK(hipStreamDestroy(stream));

  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));

  bool btestPassed = true;
  for (uint32_t i = 0; i < LEN; i++) {
    if (A[i] != B[i]) {
      btestPassed = false;
      break;
    }
  }
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Ad));
  delete[] B;
  delete[] A;
  HIP_CHECK(hipModuleUnload(Module));
  return btestPassed;
}

#ifdef __linux__
// Check if environment variable $ROCM_PATH is defined
bool isRocmPathSet() {
  FILE* fpipe;
  char const* command = "echo $ROCM_PATH";
  fpipe = popen(command, "r");

  if (fpipe == nullptr) {
    INFO("Unable to create command\n");
    return false;
  }
  char command_op[COMMAND_LEN];
  if (fgets(command_op, COMMAND_LEN, fpipe)) {
    size_t len = strlen(command_op);
    if (len > 1) {  // This is because fgets always adds newline character
      pclose(fpipe);
      return true;
    }
  }
  pclose(fpipe);
  return false;
}
#endif

bool testMultiTargArchCodeObj() {
  bool btestPassed = true;
#ifdef __linux__
  char command[COMMAND_LEN];
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  // Hardcoding the codeobject lines in multiple string to avoid cpplint warning
  std::string CodeObjL1 = "#include \"hip/hip_runtime.h\"\n";
  std::string CodeObjL2 = "extern \"C\" __global__ void hello_world(float* a, float* b) {\n";
  std::string CodeObjL3 = "  int tx = threadIdx.x;\n";
  std::string CodeObjL4 = "  b[tx] = a[tx];\n";
  std::string CodeObjL5 = "}";
  // Creating the full code object string
  static std::string CodeObj = CodeObjL1 + CodeObjL2 + CodeObjL3 + CodeObjL4 + CodeObjL5;
  std::ofstream ofs("/tmp/vcpy_kernel.cpp", std::ofstream::out);
  ofs << CodeObj;
  ofs.close();
  // Copy the file into current working location if not available
  if (access("/tmp/vcpy_kernel.cpp", F_OK) == -1) {
    INFO("Code Object File: /tmp/vcpy_kernel.cpp not found \n");
    return true;
  }
  // Generate the command to generate multi architecture code object file
  const char* hipcc_path = nullptr;
  if (isRocmPathSet()) {
    hipcc_path = "$ROCM_PATH/bin/hipcc";
  } else {
    hipcc_path = "/opt/rocm/bin/hipcc";
  }
  /* Putting these command parameters into a variable to shorten the string
    literal length in order to avoid multiline string literal cpplint warning
  */
  const char* genco_option = "--offload-arch";
  const char* input_codeobj = "/tmp/vcpy_kernel.cpp";
  const char* rocm_enumerator = "${ROCM_PATH}/bin/rocm_agent_enumerator";
  snprintf(command, COMMAND_LEN, rocm_enumerator, hipcc_path, genco_option, props.gcnArchName,
           input_codeobj, CODE_OBJ_MULTIARCH);

  system((const char*)command);
  // Check if the code object file is created
  snprintf(command, COMMAND_LEN, "./%s", CODE_OBJ_MULTIARCH);

  if (access(command, F_OK) == -1) {
    INFO("Code Object File not found \n");
    return true;
  }
  btestPassed = testCodeObjFile(CODE_OBJ_MULTIARCH);
#else
  INFO("This test is skipped due to non linux environment.\n");
#endif
  return btestPassed;
}

TEST_CASE("Unit_hipModule_Functional") {
  bool TestPassed = true;
  SECTION("Code object file test on current GPU") {
    TestPassed &= testCodeObjFile(CODE_OBJ_SINGLEARCH);
    REQUIRE(TestPassed == true);
  }
  SECTION("Code object file test on multiple GPUs") {
    TestPassed &= testMultiTargArchCodeObj();
    REQUIRE(TestPassed == true);
  }
}
