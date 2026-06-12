/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#endif
#ifdef _WIN64
#include <windows.h>
#endif

#ifdef _WIN64
#define setenv(x, y, z) _putenv_s(x, y)
#define unsetenv(x) _putenv_s(x, "")
#endif

constexpr int LEN = 64;
constexpr auto SIZE_BYTES = (LEN << 2);
constexpr auto codeObjFile = "copyKernel.code";
constexpr auto kernel_name = "copy_ker";

// Helper to run test logic in a forked child process
// This ensures each test gets a fresh HIP runtime initialization
#ifdef __linux__
void runTestInFork(std::function<void()> testFunc) {
  pid_t pid = fork();

  if (pid < 0) {
    FAIL("Failed to fork process");
  }

  if (pid == 0) {
    // Child process: run the test
    testFunc();
    exit(0);  // Test passed in child
  } else {
    // Parent process: wait for child
    int status;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status)) {
      int exitCode = WEXITSTATUS(status);
      if (exitCode != 0) {
        FAIL("Test failed in child process with exit code " << exitCode);
      }
    } else {
      FAIL("Test terminated abnormally in child process");
    }
  }
}
#else
// Windows: just run the test directly (fork not available)
// Tests may interfere with each other on Windows
void runTestInFork(std::function<void()> testFunc) {
  WARN("fork() not available on Windows, tests may interfere");
  testFunc();
}
#endif

/**
 * @addtogroup hipModuleGetLoadingMode hipModuleGetLoadingMode
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleGetLoadingMode(hipModuleLoadingMode_t* mode)` -
 * Function gets the current module load mode
 */

void kernelExecutionFunction(hipModule_t module) {
  float *Ad, *Bd;
  std::vector<float> A(LEN, 1.0f), B(LEN, 0.0f);
  HIP_CHECK(hipMalloc(&Ad, SIZE_BYTES));
  HIP_CHECK(hipMalloc(&Bd, SIZE_BYTES));
  HIP_CHECK(hipMemcpy(Ad, A.data(), SIZE_BYTES, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B.data(), SIZE_BYTES, hipMemcpyHostToDevice));
  hipFunction_t Function = nullptr;
  HIP_CHECK(hipModuleGetFunction(&Function, module, kernel_name));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void *_Ad;
    void *_Bd;
    size_t _size;
  } args;
  args._Ad = reinterpret_cast<void *>(Ad);
  args._Bd = reinterpret_cast<void *>(Bd);
  args._size = LEN;
  size_t size = sizeof(args);

  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, NULL,
                                  reinterpret_cast<void **>(&config)));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(B.data(), Bd, SIZE_BYTES, hipMemcpyDeviceToHost));
  for (uint32_t i = 0; i < LEN; i++) {
    REQUIRE(A[i] == B[i]);
  }
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Ad));
}

/**
 * Test Description
 * ------------------------
 * - Test case verifies the default mode when no environment variable is set.
 * - Verifies the default mode is LAZY.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_DefaultModeCheck") {
  runTestInFork([]() {
    INFO("Testing default mode (no env var set)");

    // Ensure HIP_MODULE_LOADING is not set
    unsetenv("HIP_MODULE_LOADING");
    const char* val = getenv("HIP_MODULE_LOADING");
    INFO("Environment variable HIP_MODULE_LOADING = " << (val ? val : "(not set)"));

    hipModuleLoadingMode_t mode;
    HIP_CHECK(hipModuleGetLoadingMode(&mode));

    INFO("Default mode value: " << mode);
    INFO("HIP_MODULE_LAZY_LOADING = " << HIP_MODULE_LAZY_LOADING);
    INFO("HIP_MODULE_EAGER_LOADING = " << HIP_MODULE_EAGER_LOADING);
    INFO("Default mode (no env var): " << mode);

    REQUIRE(mode == HIP_MODULE_LAZY_LOADING);
  });
}

/**
 * Test Description
 * ------------------------
 * - Test case verifies LAZY mode with explicit env var set.
 * - Sets HIP_MODULE_LOADING=LAZY and verifies the mode.
 * - Tests kernel execution in LAZY mode.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_LazyModeCheck") {
  runTestInFork([]() {
    INFO("Testing LAZY mode with explicit env var set");

    // Set env var to LAZY
    REQUIRE(setenv("HIP_MODULE_LOADING", "LAZY", 1) == 0);

    // Verify the env var is set
    const char* val = getenv("HIP_MODULE_LOADING");
    INFO("Environment variable HIP_MODULE_LOADING = " << (val ? val : "(not set)"));

    hipModuleLoadingMode_t mode;
    HIP_CHECK(hipModuleGetLoadingMode(&mode));

    INFO("Mode value: " << mode);
    INFO("HIP_MODULE_LAZY_LOADING = " << HIP_MODULE_LAZY_LOADING);
    INFO("HIP_MODULE_EAGER_LOADING = " << HIP_MODULE_EAGER_LOADING);
    INFO("Mode with HIP_MODULE_LOADING=LAZY: " << mode);

    REQUIRE(mode == HIP_MODULE_LAZY_LOADING);

    // Execute kernel to verify lazy loading works with actual kernel execution
    INFO("Testing kernel execution with LAZY mode...");
    hipModule_t module;
    auto start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipModuleLoad(&module, codeObjFile));
    auto stop = std::chrono::high_resolution_clock::now();
    auto result = std::chrono::duration<double, std::milli>(stop - start);
    INFO("Module load time: " << result.count() << " milliseconds");

    kernelExecutionFunction(module);
    HIP_CHECK(hipModuleUnload(module));

    unsetenv("HIP_MODULE_LOADING");
  });
}

/**
 * Test Description
 * ------------------------
 * - Test case verifies EAGER mode after setting the env var HIP_MODULE_LOADING.
 * - Sets HIP_MODULE_LOADING=EAGER and verifies the mode.
 * - Tests kernel execution in EAGER mode.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_EagerModeCheck") {
  runTestInFork([]() {
    // Set env var to EAGER
    REQUIRE(setenv("HIP_MODULE_LOADING", "EAGER", 1) == 0);

    // Verify the env var is set
    const char* val = getenv("HIP_MODULE_LOADING");
    INFO("Environment variable HIP_MODULE_LOADING = " << (val ? val : "(not set)"));

    hipModuleLoadingMode_t mode;
    HIP_CHECK(hipModuleGetLoadingMode(&mode));

    INFO("Mode with HIP_MODULE_LOADING=EAGER: " << mode);
    REQUIRE(mode == HIP_MODULE_EAGER_LOADING);

    // Execute kernel to verify eager loading works with actual kernel execution
    INFO("Testing kernel execution with EAGER mode...");
    hipModule_t module;
    auto start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipModuleLoad(&module, codeObjFile));
    auto stop = std::chrono::high_resolution_clock::now();
    auto result = std::chrono::duration<double, std::milli>(stop - start);
    INFO("Module load time: " << result.count() << " milliseconds");

    kernelExecutionFunction(module);
    HIP_CHECK(hipModuleUnload(module));

    unsetenv("HIP_MODULE_LOADING");
  });
}

/**
 * Test Description
 * ------------------------
 * - Test case verifies case-insensitive environment variable handling.
 * - Tests "eager" (lowercase) works for EAGER mode.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_CaseInsensitive") {
  runTestInFork([]() {
    hipModuleLoadingMode_t mode;

    // Test lowercase "eager"
    REQUIRE(setenv("HIP_MODULE_LOADING", "eager", 1) == 0);
    HIP_CHECK(hipModuleGetLoadingMode(&mode));
    INFO("Mode with 'eager': " << mode);
    REQUIRE(mode == HIP_MODULE_EAGER_LOADING);

    unsetenv("HIP_MODULE_LOADING");
  });
}

void setMode() { setenv("HIP_MODULE_LOADING", "EAGER", 1); }

void ChkMode() {
  hipModuleLoadingMode_t mode;
  HIP_CHECK(hipModuleGetLoadingMode(&mode));
  INFO("Mode in thread: " << mode);
  REQUIRE(mode == HIP_MODULE_EAGER_LOADING);
  unsetenv("HIP_MODULE_LOADING");
}

/**
 * Test Description
 * ------------------------
 * - Test case verifies the multithread scenario.
 * - Set mode env in one thread. Verify mode in another thread.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_MultiThread") {
  runTestInFork([]() {
    // Set env var to EAGER first
    setenv("HIP_MODULE_LOADING", "EAGER", 1);

    // Create Thread one.
    std::thread t1(setMode);
    t1.join();
    // Create Thread two
    std::thread t2(ChkMode);
    t2.join();
  });
}

/**
 * Test Description
 * ------------------------
 * - Test case verifies that module loading mode is cached at initialization.
 * - Sets HIP_MODULE_LOADING=EAGER and verifies the mode.
 * - Changes env var to LAZY and verifies mode remains EAGER (cached).
 * - This validates that the mode doesn't change dynamically after initialization.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleGetLoadingMode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipModuleGetLoadingMode_Change") {
  runTestInFork([]() {
    hipModuleLoadingMode_t mode;

    // Set env var to EAGER BEFORE HIP initialization
    REQUIRE(setenv("HIP_MODULE_LOADING", "EAGER", 1) == 0);

    // Verify the env var is set
    const char* val = getenv("HIP_MODULE_LOADING");
    INFO("Environment variable HIP_MODULE_LOADING = " << (val ? val : "(not set)"));

    // Get the mode - this initializes HIP runtime and caches the mode
    HIP_CHECK(hipModuleGetLoadingMode(&mode));
    INFO("Mode with HIP_MODULE_LOADING=EAGER: " << mode);
    REQUIRE(mode == HIP_MODULE_EAGER_LOADING);

    // Change env var to LAZY - mode should NOT change (cached at init)
    REQUIRE(setenv("HIP_MODULE_LOADING", "LAZY", 1) == 0);
    const char* val1 = getenv("HIP_MODULE_LOADING");
    INFO("Environment variable HIP_MODULE_LOADING = " << (val1 ? val1 : "(not set)"));

    // Get the mode again - should still be EAGER (cached)
    HIP_CHECK(hipModuleGetLoadingMode(&mode));
    INFO("Mode after env var changed to LAZY: " << mode);
    INFO("Verifying mode is still EAGER (not affected by env var change)");
    REQUIRE(mode == HIP_MODULE_EAGER_LOADING);

    unsetenv("HIP_MODULE_LOADING");
  });
}

/**
 * End doxygen group ModuleTest.
 * @}
 */
