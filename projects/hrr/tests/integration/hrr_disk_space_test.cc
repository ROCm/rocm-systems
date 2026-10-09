/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR capture free-space reserve
 * @{
 * @ingroup HRRTest
 * Tests for how capture stops before the file system holding the archive
 * fills.
 *
 * The writer keeps free the smaller of 15% of that file system and 4 GiB. It
 * refuses to open below that, and stops, leaving an archive marked incomplete,
 * when a write would eat into it. CI runners have far more free space than any
 * workload can fill, so this file fakes the file system instead: the test
 * binary defines statvfs, statvfs64 and fsync itself. The dynamic linker binds
 * the HIP runtime's calls to these definitions before libc's, and they forward
 * to libc for every path outside HIP_HRR_CAPTURE_OUTPUT.
 *
 * Under that directory they report a file system of HRR_TEST_FAKE_FS_TOTAL
 * bytes. Free space starts at HRR_TEST_FAKE_FS_AVAIL (1 TiB when unset), and a
 * workload can then pin it relative to the reserve: either fixed, or shrinking
 * as the archive grows on disk. Two one-shot holds park the next statvfs or
 * fsync, so a test can line up the races the writer has to survive.
 *
 * POSIX only: Windows reads free space through GetDiskFreeSpaceExA, which this
 * file does not fake.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32) && defined(HRR_TEST_EXE)

#include <dlfcn.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>

// One struct serves both statvfs and statvfs64: their layouts only differ on
// 32-bit targets.
static_assert(sizeof(void*) == 8, "the statvfs fake assumes an LP64 target");

// Set by a workload in another file of this binary. Each statvfs and statvfs64
// call runs it first, faked or not. The capture writer reads free space on its
// pid-<pid> directory while it opens the archive, after it has removed a stale
// active marker and before it creates its own.
std::atomic<void (*)(const char*)> g_hrr_statvfs_hook{nullptr};

namespace {

constexpr uint64_t kBlock = 4096;
constexpr uint64_t kDefaultAvail = 1ull << 40;
// Free space one block short of the reserve: every check fails, whatever is
// pending.
constexpr int64_t kBelowReserve = -static_cast<int64_t>(kBlock);

enum FakeMode : int { kOff = 0, kFixed = 1, kShrinking = 2 };

std::atomic<int>      g_mode{kOff};
std::atomic<int64_t>  g_headroom{0};
std::atomic<uint64_t> g_used_base{0};
std::atomic<uint64_t> g_fake_calls{0};

std::atomic<bool> g_statvfs_hold{false};
std::atomic<bool> g_statvfs_entered{false};
std::atomic<bool> g_statvfs_release{false};
std::atomic<bool> g_fsync_hold{false};
std::atomic<bool> g_fsync_entered{false};

std::atomic<pid_t> g_quick_exit_pid{0};

const char* fake_root() {
  const char* root = getenv("HIP_HRR_CAPTURE_OUTPUT");
  return root != nullptr && root[0] != '\0' ? root : nullptr;
}

uint64_t fake_total() {
  const char* total = getenv("HRR_TEST_FAKE_FS_TOTAL");
  return total != nullptr ? strtoull(total, nullptr, 10) : 0;
}

bool is_faked(const char* path) {
  const char* root = fake_root();
  if (root == nullptr || path == nullptr || fake_total() == 0) return false;
  const size_t n = strlen(root);
  return strncmp(path, root, n) == 0 && (path[n] == '\0' || path[n] == '/');
}

uint64_t round_up(uint64_t v) { return (v + kBlock - 1) / kBlock * kBlock; }

// The writer's reserve, computed the same way from the same total.
uint64_t keep_free(uint64_t total) {
  return std::min<uint64_t>(total / 100 * 15, 4ull << 30);
}

uint64_t check_interval(uint64_t total) {
  return std::max<uint64_t>(std::min<uint64_t>(64ull << 20, keep_free(total) / 4), 1);
}

// Bytes the capture directory takes on disk, each file in whole blocks.
uint64_t used_bytes() {
  const char* root = fake_root();
  if (root == nullptr) return 0;
  uint64_t sum = 0;
  std::error_code ec;
  fs::recursive_directory_iterator it(root, ec), end;
  for (; !ec && it != end; it.increment(ec)) {
    std::error_code fe;
    if (!it->is_regular_file(fe) || fe) continue;
    const uint64_t size = it->file_size(fe);
    if (!fe) sum += round_up(size);
  }
  return sum;
}

uint64_t fake_avail() {
  const int mode = g_mode.load();
  if (mode == kOff) {
    const char* avail = getenv("HRR_TEST_FAKE_FS_AVAIL");
    return avail != nullptr ? strtoull(avail, nullptr, 10) : kDefaultAvail;
  }
  int64_t avail = static_cast<int64_t>(round_up(keep_free(fake_total()))) + g_headroom.load();
  if (mode == kShrinking)
    avail -= static_cast<int64_t>(used_bytes()) - static_cast<int64_t>(g_used_base.load());
  return avail > 0 ? static_cast<uint64_t>(avail) : 0;
}

bool wait_for(const std::atomic<bool>& flag, int timeout_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (!flag.load()) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

int fake_statvfs(const char* name, const char* path, struct statvfs* buf) {
  if (auto hook = g_hrr_statvfs_hook.load(std::memory_order_acquire)) hook(path);
  if (!is_faked(path)) {
    using Fn = int (*)(const char*, struct statvfs*);
    static Fn real[2] = {reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "statvfs")),
                         reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "statvfs64"))};
    Fn fn = real[strcmp(name, "statvfs") == 0 ? 0 : 1];
    if (fn == nullptr) {
      errno = ENOSYS;
      return -1;
    }
    return fn(path, buf);
  }
  g_fake_calls.fetch_add(1);
  if (g_statvfs_hold.exchange(false)) {
    g_statvfs_entered.store(true);
    (void)wait_for(g_statvfs_release, 5000);
  }
  memset(buf, 0, sizeof(*buf));
  buf->f_bsize   = kBlock;
  buf->f_frsize  = kBlock;
  buf->f_blocks  = fake_total() / kBlock;
  buf->f_bfree   = fake_avail() / kBlock;
  buf->f_bavail  = buf->f_bfree;
  buf->f_files   = 1u << 20;
  buf->f_ffree   = 1u << 20;
  buf->f_favail  = 1u << 20;
  buf->f_namemax = 255;
  return 0;
}

}  // namespace

// Exported through the dynamic list in CMakeLists.txt, so the HIP runtime binds
// to them. The asm labels keep the C++ names apart from the libc declarations.
extern "C" int hrr_disk_space_statvfs(const char* path, struct statvfs* buf)
    __asm__("statvfs");
extern "C" int hrr_disk_space_statvfs64(const char* path, struct statvfs* buf)
    __asm__("statvfs64");
extern "C" int hrr_disk_space_fsync(int fd) __asm__("fsync");

extern "C" int hrr_disk_space_statvfs(const char* path, struct statvfs* buf) {
  return fake_statvfs("statvfs", path, buf);
}

extern "C" int hrr_disk_space_statvfs64(const char* path, struct statvfs* buf) {
  return fake_statvfs("statvfs64", path, buf);
}

// Defined in hrr_workload_test.cc: the fork cases there hold a thread in fsync.
extern "C" void hrr_workload_fsync_hook(int fd);

extern "C" int hrr_disk_space_fsync(int fd) {
  hrr_workload_fsync_hook(fd);
  using Fn = int (*)(int);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "fsync"));
  if (g_fsync_hold.exchange(false)) {
    g_fsync_entered.store(true);
    std::this_thread::sleep_for(std::chrono::seconds(3));
  }
  if (real == nullptr) {
    errno = ENOSYS;
    return -1;
  }
  return real(fd);
}

namespace {

void arm_fixed(int64_t headroom) {
  g_headroom.store(headroom);
  g_mode.store(kFixed);
}

// Free space starts `headroom` above the reserve and shrinks by whatever the
// capture writes from here on.
void arm_shrinking(int64_t headroom) {
  g_used_base.store(used_bytes());
  g_headroom.store(headroom);
  g_mode.store(kShrinking);
}

void disarm() { g_mode.store(kOff); }

// Leave through _exit right after hip_capture_shutdown when getpid() matches:
// registered before HIP initializes, so it runs after the capture's own atexit
// handler, and skips HIP teardown in a forked child or under a parked thread.
void quick_exit_hook() {
  if (g_quick_exit_pid.load() != getpid()) return;
  fflush(stdout);
  fflush(stderr);
  _exit(0);
}

// Every workload starts here, before its first HIP call.
void begin_workload() {
  // Capture opens at HIP initialization and reads free space at once, so no
  // call yet means HIP has not initialized and the hook goes in first.
  REQUIRE(g_fake_calls.load() == 0);
  REQUIRE(std::atexit(quick_exit_hook) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  // The HIP runtime reached the fake. Without this a missing export would turn
  // every test below into a capture on the real disk.
  REQUIRE(g_fake_calls.load() > 0);
}

// Distinct bytes per seed, so content-addressed blobs never deduplicate.
std::vector<unsigned char> pattern(size_t len, uint32_t seed) {
  std::vector<unsigned char> v(len);
  uint32_t x = seed * 2654435761u + 0x9e3779b9u;
  for (size_t i = 0; i < len; ++i) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    v[i] = static_cast<unsigned char>(x);
  }
  return v;
}

void copy_to_device(void* dev, uint32_t seed, size_t len) {
  const std::vector<unsigned char> host = pattern(len, seed);
  HRR_HIP_CHECK(hipMemcpy(dev, host.data(), len, hipMemcpyHostToDevice));
}

// A forked child opens its archive on its first record, not at fork. This
// records one without touching the device, which a forked child must not use:
// a launch configuration pushed and popped through the compiler dispatch table.
void record_in_child() {
  dim3 grid, block;
  size_t shared = 0;
  hipStream_t stream = nullptr;
  (void)__hipPushCallConfiguration(dim3(1), dim3(1), 0, nullptr);
  (void)__hipPopCallConfiguration(&grid, &block, &shared, &stream);
}

// Wait for a forked child for at most 60 s; -1 when it had to be killed.
int wait_child(pid_t child) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t r = waitpid(child, &status, WNOHANG);
    if (r == child) return status;
    if (r < 0 && errno != EINTR) return -1;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  kill(child, SIGKILL);
  waitpid(child, &status, 0);
  return -1;
}

constexpr uint64_t kTotal64K  = 64ull << 10;
constexpr uint64_t kTotal16M  = 16ull << 20;
constexpr uint64_t kTotal64M  = 64ull << 20;
// At least the check interval for a 64 MiB file system, so one copy is checked.
constexpr size_t   kBigCopy   = 4u << 20;
constexpr size_t   kStepCopy  = 256u << 10;
constexpr int      kStepCount = 16;
constexpr size_t   kTinyCopy  = 64;
constexpr int      kTinyCount = 1500;
constexpr int      kMemsetCount = 200;
// Events still buffered when the capture stops are flushed into events.bin.
constexpr uint64_t kEventSlack = 64u << 10;

struct HrrDiskSpaceBig {
  unsigned char bytes[3000];
};

// One launch record carrying it stays under the check interval of a 64 KiB file
// system, and two go over it.
struct HrrDiskSpaceMid {
  unsigned char bytes[1400];
};

}  // namespace

__global__ void hrr_disk_space_big_arg(HrrDiskSpaceBig big, int* out) {
  if (threadIdx.x == 0) out[0] = big.bytes[0] + big.bytes[sizeof(big.bytes) - 1];
}

__global__ void hrr_disk_space_mid_arg(HrrDiskSpaceMid mid, int* out) {
  if (threadIdx.x == 0) out[0] = mid.bytes[0] + mid.bytes[sizeof(mid.bytes) - 1];
}

// ===========================================================================
// Workloads. Each runs in its own process under the fake file system.
// ===========================================================================

TEST_CASE("Unit_HRR_DiskSpace_RefusedAtStart_Direct", "[.][hrr-direct]") {
  begin_workload();
  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kStepCopy));
  const std::vector<unsigned char> in = pattern(kStepCopy, 1);
  std::vector<unsigned char> out(kStepCopy);
  HRR_HIP_CHECK(hipMemcpy(dev, in.data(), kStepCopy, hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipMemcpy(out.data(), dev, kStepCopy, hipMemcpyDeviceToHost));
  REQUIRE(in == out);
  HRR_HIP_CHECK(hipFree(dev));
}

TEST_CASE("Unit_HRR_DiskSpace_StopsBeforeReserve_Direct", "[.][hrr-direct]") {
  begin_workload();
  const uint64_t interval = check_interval(fake_total());
  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, 1u << 20));
  // Larger than the interval, so it is checked and the count starts from zero.
  copy_to_device(dev, 100, 1u << 20);
  const fs::path active =
      fs::path(fake_root()) / ("pid-" + std::to_string(getpid())) / "active";
  REQUIRE(fs::exists(active));

  const int64_t headroom = 2 << 20;
  arm_shrinking(headroom);
  for (int i = 0; i < kStepCount; ++i) copy_to_device(dev, 200 + i, kStepCopy);
  const uint64_t used = used_bytes() - g_used_base.load();
  disarm();

  // Stopped while the program runs on, so producers must stop writing sidecars.
  CHECK_FALSE(fs::exists(active));

  // The capture went no further into the reserve than one check interval.
  INFO("used " << used << " headroom " << headroom << " interval " << interval);
  CHECK(used <= static_cast<uint64_t>(headroom) + interval + kEventSlack);

  std::vector<unsigned char> out(kStepCopy);
  HRR_HIP_CHECK(hipMemcpy(out.data(), dev, kStepCopy, hipMemcpyDeviceToHost));
  REQUIRE(out == pattern(kStepCopy, 200 + kStepCount - 1));
  HRR_HIP_CHECK(hipFree(dev));
}

TEST_CASE("Unit_HRR_DiskSpace_SmallBlobsCountedInBlocks_Direct", "[.][hrr-direct]") {
  begin_workload();
  const uint64_t interval = check_interval(fake_total());
  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kBigCopy));
  copy_to_device(dev, 300, kBigCopy);

  // Each blob is 64 bytes and one block on disk. Counted in bytes, the whole
  // loop stays under one check interval and nothing is ever checked.
  const int64_t headroom = 1 << 20;
  arm_shrinking(headroom);
  for (int i = 0; i < kTinyCount; ++i) copy_to_device(dev, 1000 + i, kTinyCopy);
  const uint64_t used = used_bytes() - g_used_base.load();
  disarm();

  INFO("used " << used << " headroom " << headroom << " interval " << interval);
  CHECK(used <= static_cast<uint64_t>(headroom) + interval + kEventSlack);
  HRR_HIP_CHECK(hipFree(dev));
}

TEST_CASE("Unit_HRR_DiskSpace_ForkAfterStop_Direct", "[.][hrr-direct]") {
  begin_workload();
  const fs::path root = fake_root();
  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kBigCopy));
  arm_fixed(kBelowReserve);
  copy_to_device(dev, 400, kBigCopy);
  // Plenty free again: a child that ignored the stop would open an archive.
  disarm();

  const pid_t parent = getpid();
  fflush(stdout);
  fflush(stderr);
  const pid_t child = fork();
  if (child == 0) {
    g_quick_exit_pid.store(getpid());
    record_in_child();
    std::exit(0);
  }
  REQUIRE(child > 0);
  const int status = wait_child(child);
  REQUIRE(status != -1);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);

  // The child opened nothing and finalized nothing, its own or the parent's.
  CHECK_FALSE(fs::exists(root / ("pid-" + std::to_string(child))));
  CHECK_FALSE(fs::exists(root / "manifest.json"));
  CHECK_FALSE(fs::exists(root / ("pid-" + std::to_string(parent)) / "manifest.json"));
  HRR_HIP_CHECK(hipFree(dev));
}

TEST_CASE("Unit_HRR_DiskSpace_RefusedForkChild_Direct", "[.][hrr-direct]") {
  begin_workload();
  const fs::path root = fake_root();
  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kStepCopy));
  copy_to_device(dev, 500, kStepCopy);

  arm_fixed(kBelowReserve);
  fflush(stdout);
  fflush(stderr);
  const pid_t child = fork();
  if (child == 0) {
    g_quick_exit_pid.store(getpid());
    // Its open, and so the refusal, comes with its first record.
    record_in_child();
    std::exit(0);
  }
  REQUIRE(child > 0);
  const int status = wait_child(child);
  disarm();
  REQUIRE(status != -1);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);

  CHECK_FALSE(fs::exists(root / ("pid-" + std::to_string(child))));
  CHECK_FALSE(fs::exists(root / "manifest.json"));
  // The parent's capture is untouched.
  copy_to_device(dev, 501, kStepCopy);
  HRR_HIP_CHECK(hipFree(dev));
}

TEST_CASE("Unit_HRR_DiskSpace_CrashAfterStop_Direct", "[.][hrr-direct]") {
  begin_workload();
  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kBigCopy));
  arm_fixed(kBelowReserve);
  copy_to_device(dev, 600, kBigCopy);

  const struct rlimit no_core = {0, 0};
  (void)setrlimit(RLIMIT_CORE, &no_core);
  fflush(stdout);
  fflush(stderr);
  abort();
}

TEST_CASE("Unit_HRR_DiskSpace_StopWhileExiting_Direct", "[.][hrr-direct]") {
  begin_workload();
  void* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, kBigCopy));
  HRR_HIP_CHECK(hipMemset(dev, 0x5a, kBigCopy));
  static std::vector<unsigned char> host(kBigCopy);

  // The stop parks in its fsync, between closing events.bin and marking the
  // archive incomplete, while this thread exits and flush() runs.
  arm_fixed(kBelowReserve);
  g_fsync_hold.store(true);
  std::thread([dev] {
    (void)hipMemcpy(host.data(), dev, kBigCopy, hipMemcpyDeviceToHost);
    for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
  }).detach();

  REQUIRE(wait_for(g_fsync_entered, 10000));
  g_quick_exit_pid.store(getpid());
  std::exit(0);
}

TEST_CASE("Unit_HRR_DiskSpace_DuplicateRacingStop_Direct", "[.][hrr-direct]") {
  begin_workload();
  void* dev_a = nullptr;
  void* dev_b = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev_a, kBigCopy));
  HRR_HIP_CHECK(hipMalloc(&dev_b, kBigCopy));
  const std::vector<unsigned char> a = pattern(kBigCopy, 800);
  const std::vector<unsigned char> b = a;

  // The first copy parks in its free-space check. The second copy of the same
  // bytes must not take the hash as written while that blob may never be.
  arm_fixed(kBelowReserve);
  g_statvfs_hold.store(true);
  hipError_t first = hipErrorUnknown;
  std::thread worker([&] {
    first = hipMemcpy(dev_a, a.data(), kBigCopy, hipMemcpyHostToDevice);
  });
  const bool parked = wait_for(g_statvfs_entered, 10000);
  hipError_t second = hipErrorUnknown;
  if (parked) second = hipMemcpy(dev_b, b.data(), kBigCopy, hipMemcpyHostToDevice);
  g_statvfs_release.store(true);
  worker.join();
  disarm();

  REQUIRE(parked);
  HRR_HIP_CHECK(first);
  HRR_HIP_CHECK(second);
  HRR_HIP_CHECK(hipFree(dev_a));
  HRR_HIP_CHECK(hipFree(dev_b));
}

TEST_CASE("Unit_HRR_DiskSpace_LargeEventCheckedBeforeWrite_Direct", "[.][hrr-direct]") {
  begin_workload();
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, sizeof(int)));
  HrrDiskSpaceBig big{};
  big.bytes[0] = 3;
  big.bytes[sizeof(big.bytes) - 1] = 4;

  // The launch record carries the 3000-byte argument, above the 2456-byte check
  // interval of a 64 KiB file system.
  hipLaunchKernelGGL(hrr_disk_space_big_arg, dim3(1), dim3(64), 0, nullptr, big, out);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());
  void* scratch = nullptr;
  HRR_HIP_CHECK(hipMalloc(&scratch, kBlock));
  copy_to_device(scratch, 900, kBlock);

  arm_fixed(kBelowReserve);
  hipLaunchKernelGGL(hrr_disk_space_big_arg, dim3(1), dim3(64), 0, nullptr, big, out);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());
  disarm();

  int host = 0;
  HRR_HIP_CHECK(hipMemcpy(&host, out, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(host == 7);
  HRR_HIP_CHECK(hipFree(scratch));
  HRR_HIP_CHECK(hipFree(out));
}

TEST_CASE("Unit_HRR_DiskSpace_EventsCheckedAcrossInterval_Direct", "[.][hrr-direct]") {
  begin_workload();
  int* out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&out, sizeof(int)));
  HrrDiskSpaceMid mid{};
  mid.bytes[0] = 3;
  mid.bytes[sizeof(mid.bytes) - 1] = 4;

  // The first launch writes the code object while free space is plenty.
  hipLaunchKernelGGL(hrr_disk_space_mid_arg, dim3(1), dim3(64), 0, nullptr, mid, out);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());
  // A one-block blob reaches the interval, so the count starts again from here.
  void* scratch = nullptr;
  HRR_HIP_CHECK(hipMalloc(&scratch, kBlock));
  copy_to_device(scratch, 970, kBlock);

  // Neither record reaches the interval alone. The second one makes the check
  // due, and that check has to run before it is written.
  arm_fixed(kBelowReserve);
  hipLaunchKernelGGL(hrr_disk_space_mid_arg, dim3(1), dim3(64), 0, nullptr, mid, out);
  hipLaunchKernelGGL(hrr_disk_space_mid_arg, dim3(1), dim3(64), 0, nullptr, mid, out);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipDeviceSynchronize());
  disarm();

  int host = 0;
  HRR_HIP_CHECK(hipMemcpy(&host, out, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(host == 7);
  HRR_HIP_CHECK(hipFree(scratch));
  HRR_HIP_CHECK(hipFree(out));
}

TEST_CASE("Unit_HRR_DiskSpace_BufferedEventsCounted_Direct", "[.][hrr-direct]") {
  begin_workload();
  int* dev = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dev, sizeof(int)));
  void* scratch = nullptr;
  HRR_HIP_CHECK(hipMalloc(&scratch, kBlock));
  copy_to_device(scratch, 950, kBlock);

  // No blobs from here on, only small buffered records.
  arm_fixed(kBelowReserve);
  for (int i = 0; i < kMemsetCount; ++i)
    HRR_HIP_CHECK(hipMemsetAsync(dev, i & 0xff, sizeof(int), nullptr));
  HRR_HIP_CHECK(hipStreamSynchronize(nullptr));
  disarm();

  HRR_HIP_CHECK(hipFree(scratch));
  HRR_HIP_CHECK(hipFree(dev));
}

// ===========================================================================
// Drivers.
// ===========================================================================

namespace {

struct WorkloadRun {
  int rc = -1;
  std::string out;
};

WorkloadRun run_workload(const std::string& name, const fs::path& cap, uint64_t total,
                         uint64_t avail = 0) {
  hrr::test::SpawnProc proc(hrr_self_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
  proc.setEnv("HRR_TEST_FAKE_FS_TOTAL", std::to_string(total));
  if (avail != 0) proc.setEnv("HRR_TEST_FAKE_FS_AVAIL", std::to_string(avail));
  set_proc_search_path(proc);
  WorkloadRun run;
  run.rc = proc.runWithTimeout("\"" + name + "\"", 300);
  run.out = proc.getOutput();
  return run;
}

size_t count_of(const std::string& text, const std::string& needle) {
  size_t n = 0;
  for (size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size()))
    ++n;
  return n;
}

constexpr const char* kStopped  = "[HRR capture] Capture stopped";
constexpr const char* kDisabled = "[HRR capture] Capture disabled";
constexpr const char* kIncomplete = "\"complete\": false";

// The archive a stopped capture leaves: stopped once, not torn, marked
// incomplete in both manifests, and every recorded copy's blob on disk.
void load_stopped_archive(const fs::path& cap, const WorkloadRun& run, hrr::Archive& arc) {
  CHECK(count_of(run.out, kStopped) == 1);
  const fs::path proc_dir = hrr_single_process_archive(cap);
  REQUIRE(hrr::load_archive(proc_dir.string(), arc));
  CHECK_FALSE(arc.complete);
  CHECK_FALSE(arc.truncated);
  CHECK(read_text_file(proc_dir / "manifest.json").find(kIncomplete) != std::string::npos);
  CHECK(read_text_file(cap / "manifest.json").find(kIncomplete) != std::string::npos);

  size_t missing = 0;
  for (const auto& ev : arc.events) {
    if (ev.header().event_type != HRR_API_HIPMEMCPY) continue;
    if (ev.memcpy_ev.hash_lo == 0 && ev.memcpy_ev.hash_hi == 0) continue;
    std::vector<uint8_t> data;
    if (!hrr::read_blob(arc, ev.memcpy_ev.hash_lo, ev.memcpy_ev.hash_hi, data)) ++missing;
  }
  CHECK(missing == 0);
}

size_t count_h2d(const hrr::Archive& arc, uint64_t size) {
  size_t n = 0;
  for (const auto& ev : arc.events)
    if (ev.header().event_type == HRR_API_HIPMEMCPY &&
        ev.memcpy_ev.kind == static_cast<int32_t>(hipMemcpyHostToDevice) &&
        ev.memcpy_ev.size == size)
      ++n;
  return n;
}

size_t count_processes(const fs::path& cap) {
  return count_of(read_text_file(cap / "manifest.json"), "{ \"pid\": ");
}

fs::path cap_dir(const char* name) {
  return fs::temp_directory_path() / (std::string("hrr_disk_space_") + name + ".hrr");
}

}  // namespace

// Below the reserve at start, capture stays off, leaves nothing behind, and the
// program runs as if it were never asked to capture.
HRR_TEST_CASE(Unit_HRR_DiskSpace_RefusedAtStart) {
  ScopedDir cap(cap_dir("refused"));
  const WorkloadRun run = run_workload("Unit_HRR_DiskSpace_RefusedAtStart_Direct", cap.path,
                                       kTotal64M, /*avail=*/1u << 20);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 0);
  CHECK(count_of(run.out, kDisabled) == 1);
  CHECK_FALSE(fs::exists(cap.path / "manifest.json"));
  std::error_code ec;
  for (const auto& ent : fs::directory_iterator(cap.path, ec))
    CHECK(ent.path().filename().string().rfind("pid-", 0) != 0);
}

// Free space shrinks with every blob written. The capture stops within one
// check interval of the reserve and leaves an archive that is incomplete but
// whole.
HRR_TEST_CASE(Unit_HRR_DiskSpace_StopsBeforeReserve) {
  ScopedDir cap(cap_dir("stops"));
  const WorkloadRun run =
      run_workload("Unit_HRR_DiskSpace_StopsBeforeReserve_Direct", cap.path, kTotal16M);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 0);
  hrr::Archive arc;
  load_stopped_archive(cap.path, run, arc);
  const size_t steps = count_h2d(arc, kStepCopy);
  CHECK(steps > 0);
  CHECK(steps < static_cast<size_t>(kStepCount));
}

// Small blobs are counted in file system blocks, so a stream of them reaches a
// check, and stops, long before their byte count would.
HRR_TEST_CASE(Unit_HRR_DiskSpace_SmallBlobsCountedInBlocks) {
  ScopedDir cap(cap_dir("blocks"));
  const WorkloadRun run =
      run_workload("Unit_HRR_DiskSpace_SmallBlobsCountedInBlocks_Direct", cap.path, kTotal64M);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 0);
  hrr::Archive arc;
  load_stopped_archive(cap.path, run, arc);
  CHECK(count_h2d(arc, kTinyCopy) < static_cast<size_t>(kTinyCount));
}

// A child forked after the stop opens no archive of its own and does not
// finalize the parent's when it exits.
HRR_TEST_CASE(Unit_HRR_DiskSpace_ForkAfterStop) {
  ScopedDir cap(cap_dir("fork_stop"));
  const WorkloadRun run =
      run_workload("Unit_HRR_DiskSpace_ForkAfterStop_Direct", cap.path, kTotal64M);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 0);
  hrr::Archive arc;
  load_stopped_archive(cap.path, run, arc);
  CHECK(count_processes(cap.path) == 1);
}

// A child whose own open is refused leaves no trace, and the parent's archive
// stays complete.
HRR_TEST_CASE(Unit_HRR_DiskSpace_RefusedForkChild) {
  ScopedDir cap(cap_dir("fork_refused"));
  const WorkloadRun run =
      run_workload("Unit_HRR_DiskSpace_RefusedForkChild_Direct", cap.path, kTotal64M);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 0);
  CHECK(count_of(run.out, kDisabled) == 1);
  CHECK(count_of(run.out, kStopped) == 0);
  CHECK(count_processes(cap.path) == 1);
  const std::string root = read_text_file(cap.path / "manifest.json");
  CHECK(root.find("\"complete\": true") != std::string::npos);
  CHECK(root.find(kIncomplete) == std::string::npos);
}

// A crash after the stop still leaves a manifest that says the archive is
// incomplete, although events.bin is already closed.
HRR_TEST_CASE(Unit_HRR_DiskSpace_CrashAfterStop) {
  ScopedDir cap(cap_dir("crash"));
  const WorkloadRun run =
      run_workload("Unit_HRR_DiskSpace_CrashAfterStop_Direct", cap.path, kTotal64M);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 128 + SIGABRT);
  CHECK(count_of(run.out, kStopped) == 1);
  const fs::path proc_dir = hrr_single_process_archive(cap.path);
  REQUIRE(fs::exists(proc_dir / "manifest.json"));
  CHECK(read_text_file(proc_dir / "manifest.json").find(kIncomplete) != std::string::npos);
}

// The process exits while another thread is stopping the capture. The final
// manifests still say incomplete.
HRR_TEST_CASE(Unit_HRR_DiskSpace_StopWhileExiting) {
  ScopedDir cap(cap_dir("exiting"));
  const WorkloadRun run =
      run_workload("Unit_HRR_DiskSpace_StopWhileExiting_Direct", cap.path, kTotal64M);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 0);
  hrr::Archive arc;
  load_stopped_archive(cap.path, run, arc);
}

// Two threads copy the same bytes while the capture stops. Neither copy may be
// recorded against a blob that was never written.
HRR_TEST_CASE(Unit_HRR_DiskSpace_DuplicateRacingStop) {
  ScopedDir cap(cap_dir("duplicate"));
  const WorkloadRun run =
      run_workload("Unit_HRR_DiskSpace_DuplicateRacingStop_Direct", cap.path, kTotal64M);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 0);
  hrr::Archive arc;
  load_stopped_archive(cap.path, run, arc);
}

// A record of at least the check interval is checked before it is written, so
// the launch that would cross into the reserve is not recorded.
HRR_TEST_CASE(Unit_HRR_DiskSpace_LargeEventCheckedBeforeWrite) {
  ScopedDir cap(cap_dir("large_event"));
  const WorkloadRun run =
      run_workload("Unit_HRR_DiskSpace_LargeEventCheckedBeforeWrite_Direct", cap.path, kTotal64K);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 0);
  hrr::Archive arc;
  load_stopped_archive(cap.path, run, arc);
  size_t launches = 0;
  for (const auto& ev : arc.events)
    if (ev.kernel_launch != nullptr &&
        ev.kernel_launch->kernel_name.find("hrr_disk_space_big_arg") != std::string::npos)
      ++launches;
  CHECK(launches == 1);
}

// Records under the check interval are checked before the one that makes the
// check due is written, so less than one interval goes in unchecked, however
// the bytes add up.
HRR_TEST_CASE(Unit_HRR_DiskSpace_EventsCheckedAcrossInterval) {
  ScopedDir cap(cap_dir("event_interval"));
  const WorkloadRun run = run_workload("Unit_HRR_DiskSpace_EventsCheckedAcrossInterval_Direct",
                                       cap.path, kTotal64K);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 0);
  hrr::Archive arc;
  load_stopped_archive(cap.path, run, arc);

  // Everything recorded from the scratch copy on, the last check that passed.
  const uint64_t interval = check_interval(kTotal64K);
  size_t from = arc.events.size();
  for (size_t i = 0; i < arc.events.size(); ++i)
    if (arc.events[i].header().event_type == HRR_API_HIPMEMCPY &&
        arc.events[i].memcpy_ev.kind == static_cast<int32_t>(hipMemcpyHostToDevice) &&
        arc.events[i].memcpy_ev.size == kBlock)
      from = i;
  REQUIRE(from < arc.events.size());
  uint64_t unchecked = 0;
  size_t launches = 0;
  for (size_t i = from; i < arc.events.size(); ++i) {
    const auto& ev = arc.events[i];
    unchecked += ev.header().payload_length;
    if (ev.kernel_launch != nullptr &&
        ev.kernel_launch->kernel_name.find("hrr_disk_space_mid_arg") != std::string::npos) {
      // Each launch alone stays under the interval, so it takes the counted path.
      CHECK(ev.header().payload_length < interval);
      ++launches;
    }
  }
  INFO("unchecked " << unchecked << " interval " << interval << " launches " << launches);
  CHECK(unchecked < interval);
  CHECK(launches < 2);
}

// Event records count toward the next check like blobs do, so a run of
// buffered events with no blobs still stops.
HRR_TEST_CASE(Unit_HRR_DiskSpace_BufferedEventsCounted) {
  ScopedDir cap(cap_dir("events"));
  const WorkloadRun run =
      run_workload("Unit_HRR_DiskSpace_BufferedEventsCounted_Direct", cap.path, kTotal64K);
  INFO("Capture output:\n" << run.out);
  REQUIRE(run.rc == 0);
  hrr::Archive arc;
  load_stopped_archive(cap.path, run, arc);
  size_t memsets = 0;
  for (const auto& ev : arc.events)
    if (ev.header().event_type == HRR_API_HIPMEMSETASYNC) ++memsets;
  CHECK(memsets > 0);
  CHECK(memsets < static_cast<size_t>(kMemsetCount));
}

#endif  // !_WIN32 && HRR_TEST_EXE

/**
 * @}
 */
