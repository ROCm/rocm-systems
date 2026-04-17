#pragma once

#include <gtest/gtest.h>

#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

// Global test configuration, set by main() before RUN_ALL_TESTS().
struct TestConfig {
  bool testAll = false;
  std::vector<std::string> corpusDirs;
  std::vector<std::string> raiseDirs;
  int corpusLimit = 200;
};

extern TestConfig g_config;

// HIP assertion macro for use in void-returning TEST_F() bodies and helpers.
// Unlike the old HIP_CHECK that called exit(1), this produces a proper
// GoogleTest fatal failure that stops the current test without killing the
// process.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

#define HIP_ASSERT(call)                                                       \
  do {                                                                         \
    hipError_t _err = (call);                                                  \
    ASSERT_EQ(_err, hipSuccess)                                                \
        << "HIP error " << static_cast<int>(_err) << " ("                      \
        << hipGetErrorString(_err) << ") at " << __FILE__ << ":" << __LINE__;  \
  } while (0)

// Fixture for GPU tests.  TearDown resets the device so that a failure in one
// test (e.g. hipError 700 / illegal address) does not cascade into subsequent
// tests sharing the same process.
class GpuTest : public ::testing::Test {
protected:
  void TearDown() override { (void)hipDeviceReset(); }
};
#endif

static inline bool fileExists(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

// Build the MFMA code object if it doesn't already exist.
// Returns true if the .co is available after the call.
// Uses execvp instead of system() to avoid shell injection.
static inline bool ensureMfmaCo(const std::string &transpilerSrcDir) {
  std::string co = transpilerSrcDir + "/build/mfma_gemm_gfx942_unbundled.co";
  if (fileExists(co))
    return true;

  std::string hip = transpilerSrcDir + "/tests/mfma_gemm.hip";
  if (!fileExists(hip))
    return false;

  std::string buildDir = transpilerSrcDir + "/build";
  mkdir(buildDir.c_str(), 0755);

  std::string bundled = buildDir + "/mfma_gemm_gfx942.co";

  // Compile with hipcc via fork/exec (no shell).
  {
    pid_t pid = fork();
    if (pid == 0) {
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
      execlp("hipcc", "hipcc", "--genco", "--offload-arch=gfx942",
             "-o", bundled.c_str(), hip.c_str(), nullptr);
      _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
      return false;
  }

  const char *bundlers[] = {
      "/opt/rocm/lib/llvm/bin/clang-offload-bundler",
      "/opt/rocm/llvm/bin/clang-offload-bundler",
  };
  for (auto *b : bundlers) {
    if (!fileExists(b))
      continue;
    pid_t pid = fork();
    if (pid == 0) {
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
      execl(b, b, "--type=o",
            ("--input=" + bundled).c_str(),
            ("--output=" + co).c_str(),
            "--targets=hipv4-amdgcn-amd-amdhsa--gfx942",
            "--unbundle", nullptr);
      _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    if (WIFEXITED(st) && WEXITSTATUS(st) == 0 && fileExists(co))
      return true;
  }
  return false;
}
