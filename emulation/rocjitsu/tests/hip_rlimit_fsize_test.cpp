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
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
volatile sig_atomic_t file_size_signal = 0;
void on_file_size_signal(int) { file_size_signal = 1; }

__global__ void write_cookie(unsigned char *data) { data[0] = 0x37; }

#define CHECK_HIP(call) ASSERT_EQ((call), hipSuccess)

void check_vmm(size_t bytes, bool allow_rejection) {
  hipMemAllocationProp property{};
  property.type = hipMemAllocationTypePinned;
  property.location.type = hipMemLocationTypeDevice;
  property.location.id = 0;
  property.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  size_t granularity = 0;
  CHECK_HIP(
      hipMemGetAllocationGranularity(&granularity, &property, hipMemAllocationGranularityMinimum));
  const size_t allocation_size = (bytes + granularity - 1) / granularity * granularity;
  hipMemGenericAllocationHandle_t allocation{}, imported{};
  const auto created = hipMemCreate(&allocation, allocation_size, &property, 0);
  if (created != hipSuccess) {
    EXPECT_TRUE(allow_rejection) << hipGetErrorString(created);
    return;
  }
  int shared_fd = -1;
  const auto exported =
      hipMemExportToShareableHandle(&shared_fd, allocation, hipMemHandleTypePosixFileDescriptor, 0);
  if (exported != hipSuccess) {
    EXPECT_TRUE(allow_rejection) << hipGetErrorString(exported);
    CHECK_HIP(hipMemRelease(allocation));
    return;
  }
  EXPECT_EQ(lseek(shared_fd, 0, SEEK_END), allocation_size);
  EXPECT_EQ(lseek64(shared_fd, 0, SEEK_END), allocation_size);
  EXPECT_EQ(lseek(shared_fd, 0, SEEK_SET), 0);
  CHECK_HIP(hipMemImportFromShareableHandle(
      &imported, reinterpret_cast<void *>(static_cast<intptr_t>(shared_fd)),
      hipMemHandleTypePosixFileDescriptor));
  CHECK_HIP(hipMemRelease(imported));
  CHECK_HIP(hipMemRelease(allocation));
  close(shared_fd);
}

void check_application_limit() {
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
  // The exported seek aliases must preserve native file and pipe behavior.
  EXPECT_EQ(lseek(file, 17, SEEK_SET), 17);
  EXPECT_EQ(lseek64(file, 0, SEEK_CUR), 17);
  int pipe_fds[2];
  ASSERT_EQ(pipe(pipe_fds), 0);
  EXPECT_EQ(lseek(pipe_fds[0], 0, SEEK_END), -1);
  EXPECT_EQ(errno, ESPIPE);
  EXPECT_EQ(lseek64(pipe_fds[0], 0, SEEK_END), -1);
  EXPECT_EQ(errno, ESPIPE);
  close(pipe_fds[0]);
  close(pipe_fds[1]);
  EXPECT_EQ(lseek(-1, 0, SEEK_SET), -1);
  EXPECT_EQ(errno, EBADF);
  EXPECT_EQ(lseek64(-1, 0, SEEK_SET), -1);
  EXPECT_EQ(errno, EBADF);
  struct sigaction action {};
  action.sa_handler = on_file_size_signal;
  sigemptyset(&action.sa_mask);
  ASSERT_EQ(sigaction(SIGXFSZ, &action, nullptr), 0);

  rlimit limit{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &limit), 0);
  const char *kind = std::getenv("RJ_TEST_LIMIT");
  limit.rlim_cur = kind && (std::strcmp(kind, "zero") == 0 || std::strcmp(kind, "soft-zero") == 0)
                       ? 0
                       : 2 * 1024 * 1024;
  if (!kind || (std::strcmp(kind, "soft") != 0 && std::strcmp(kind, "soft-zero") != 0))
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
  const char *vmm = std::getenv("RJ_TEST_VMM");
  check_vmm(bytes, vmm && std::strcmp(vmm, "allow-rejection") == 0);
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
TEST(HipFileSizeLimitTest, PreservesApplicationLimit) {
  const char *phase = std::getenv("RJ_TEST_PHASE");
  if (!phase || std::strcmp(phase, "fork") != 0) {
    check_application_limit();
    return;
  }
  rlimit limit{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &limit), 0);
  limit.rlim_cur = 0;
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    check_application_limit();
    _exit(::testing::Test::HasFailure() ? 1 : 0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
