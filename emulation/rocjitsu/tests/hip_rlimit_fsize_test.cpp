// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#if __has_feature(thread_sanitizer)
#include <sanitizer/common_interface_defs.h>
#endif

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
volatile sig_atomic_t file_size_signal = 0;
void on_file_size_signal(int) { file_size_signal = 1; }

void expect_file_size_signal() {
  // TSan can defer SIGXFSZ until another intercepted blocking call.
  for (unsigned attempt = 0; attempt < 100 && !file_size_signal; ++attempt)
    poll(nullptr, 0, 10);
  EXPECT_EQ(static_cast<int>(file_size_signal), 1);
}

__global__ void write_cookie(unsigned char *data) { data[0] = 0x37; }
__global__ void copy_cookie(const unsigned char *source, unsigned char *result) {
  result[0] = source[0];
}

#define CHECK_HIP(call) ASSERT_EQ((call), hipSuccess)

void check_daemon_host_mapping(size_t bytes, bool can_dispatch) {
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  const size_t page = static_cast<size_t>(page_size);
  ASSERT_LE(3 * page, bytes);
  unsigned char *host = nullptr;
  CHECK_HIP(hipHostMalloc(&host, bytes));
  std::memset(host, 0x5A, 2 * page);
  const std::vector<unsigned char> expected(page, 0x5A);
  // Both requests must retain the daemon's shared backing, including when a
  // private anonymous request replaces only one page of a fallback allocation.
  const int types[] = {MAP_PRIVATE, MAP_SHARED};
  for (size_t index = 0; index < 2; ++index) {
    auto *addr = host + index * page;
    ASSERT_EQ(
        mmap(addr, page, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_ANONYMOUS | types[index], -1, 0),
        addr)
        << std::strerror(errno);
    EXPECT_EQ(std::memcmp(addr, expected.data(), page), 0);
    if (can_dispatch) {
      // Use an untouched page as the result so the kernel must read the remap.
      auto *result = host + 2 * page;
      addr[1] = 0xA5;
      result[0] = 0;
      copy_cookie<<<1, 1>>>(addr + 1, result);
      CHECK_HIP(hipGetLastError());
      CHECK_HIP(hipDeviceSynchronize());
      EXPECT_EQ(result[0], 0xA5);
      write_cookie<<<1, 1>>>(addr);
      CHECK_HIP(hipGetLastError());
      CHECK_HIP(hipDeviceSynchronize());
      EXPECT_EQ(addr[0], 0x37);
    }
  }
  CHECK_HIP(hipHostFree(host));
}

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
  const bool can_dispatch = limit.rlim_cur != 0 || (phase && std::strcmp(phase, "late") == 0);
  if (can_dispatch) {
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
  const char *mode = std::getenv("RJ_TEST_MODE");
  if (mode && std::strcmp(mode, "daemon") == 0)
    check_daemon_host_mapping(bytes, can_dispatch);
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
  expect_file_size_signal();
  file_size_signal = 0;
  EXPECT_EQ(ftruncate(file, limit.rlim_cur + 1), -1);
  EXPECT_EQ(errno, EFBIG);
  expect_file_size_signal();
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
#if __has_feature(thread_sanitizer)
  // The symbolizer inherits LD_PRELOAD. Start it before lowering FSIZE so its
  // TSan runtime can initialize and resolve the existing ROCr suppressions.
  char symbol[1024];
  __sanitizer_symbolize_pc(reinterpret_cast<void *>(&main), "%f", symbol, sizeof(symbol));
#endif
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
