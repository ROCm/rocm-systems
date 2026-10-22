// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

namespace {
volatile sig_atomic_t file_size_signal = 0;
void on_file_size_signal(int) { file_size_signal = 1; }

__global__ void write_cookie(unsigned char *data) { data[0] = 0x37; }

#define CHECK_HIP(call) ASSERT_EQ((call), hipSuccess)

TEST(HipFileSizeLimitTest, PreservesApplicationLimit) {
  const char *phase = std::getenv("RJ_TEST_PHASE");
  if (phase && std::strcmp(phase, "late") == 0) {
    void *warmup = nullptr;
    CHECK_HIP(hipMalloc(&warmup, 4096));
    CHECK_HIP(hipMemset(warmup, 0, 4096));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(warmup));
  }
  char path[] = "/tmp/rocjitsu-fsize-XXXXXX";
  const int file = mkstemp(path);
  ASSERT_GE(file, 0);
  ASSERT_EQ(unlink(path), 0);
  struct sigaction action {};
  action.sa_handler = on_file_size_signal;
  sigemptyset(&action.sa_mask);
  ASSERT_EQ(sigaction(SIGXFSZ, &action, nullptr), 0);

  rlimit limit{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &limit), 0);
  const char *kind = std::getenv("RJ_TEST_LIMIT");
  limit.rlim_cur = kind && std::strcmp(kind, "zero") == 0 ? 0 : 2 * 1024 * 1024;
  if (!kind || std::strcmp(kind, "soft") != 0)
    limit.rlim_max = limit.rlim_cur;
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);

  constexpr size_t bytes = 4 * 1024 * 1024;
  unsigned char *device = nullptr;
  CHECK_HIP(hipMalloc(&device, bytes));
  // COMGR's first-use compilation of HIP copy builtins writes temporary files,
  // independently of rocjitsu. Exercise startup/allocation under early zero;
  // late zero covers copies and dispatch after the runtime's compiler warmup.
  if (limit.rlim_cur != 0 || (phase && std::strcmp(phase, "late") == 0)) {
    std::vector<unsigned char> source(bytes, 0xA5), result(bytes);
    CHECK_HIP(hipMemcpy(device, source.data(), bytes, hipMemcpyHostToDevice));
    write_cookie<<<1, 1>>>(device);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(result.data(), device, bytes, hipMemcpyDeviceToHost));
    source[0] = 0x37;
    EXPECT_EQ(result, source);
  }
  CHECK_HIP(hipFree(device));
  CHECK_HIP(hipDeviceReset());
  EXPECT_EQ(file_size_signal, 0);

  rlimit observed{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &observed), 0);
  EXPECT_EQ(observed.rlim_cur, limit.rlim_cur);
  EXPECT_EQ(observed.rlim_max, limit.rlim_max);
  struct sigaction observed_action {};
  ASSERT_EQ(sigaction(SIGXFSZ, nullptr, &observed_action), 0);
  EXPECT_EQ(observed_action.sa_handler, on_file_size_signal);
  EXPECT_EQ(pwrite(file, "x", 1, limit.rlim_cur), -1);
  EXPECT_EQ(errno, EFBIG);
  EXPECT_EQ(file_size_signal, 1);
  file_size_signal = 0;
  EXPECT_EQ(ftruncate(file, limit.rlim_cur + 1), -1);
  EXPECT_EQ(errno, EFBIG);
  EXPECT_EQ(file_size_signal, 1);
  close(file);
}
} // namespace

int main(int argc, char **argv) {
  const char *phase = std::getenv("RJ_TEST_PHASE");
  if (phase && std::strcmp(phase, "inherited") == 0 && !std::getenv("RJ_TEST_LIMIT_INHERITED")) {
    // Re-exec so both the interposer constructor and its keeper inherit the
    // finite hard limit, as when the shell starts rocjitsu under ulimit -f.
    const rlimit limit{2 * 1024 * 1024, 2 * 1024 * 1024};
    if (setrlimit(RLIMIT_FSIZE, &limit) != 0 || setenv("RJ_TEST_LIMIT_INHERITED", "1", 1) != 0)
      return 2;
    execv("/proc/self/exe", argv);
    return 3;
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
