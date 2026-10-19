// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "scoped_temp.h"

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"
#include "rocjitsu/vm/amdgpu/pci/mmio_registers.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"
#include "rocjitsu/vm/amdgpu/pci/scratch_pci_device.h"
#include "rocjitsu/vmm/vfu/vfio_device_host.h"
#include "rocjitsu/vmm/vfu/vfio_server.h"
#include "util/diagnostic.h"
#include "vfio_user_client.h"

#include <gtest/gtest.h>

// libvfio-user's headers pull in <stdint.h>, <sys/uio.h> and others before
// opening their own linkage block, so those are included first here, outside the
// wrapper, rather than being dragged into C linkage by it. The C++ spelling is
// used where there is one; the rest are POSIX interfaces with no C++ header.
//
// The wrapper is then only around libvfio-user's own declarations. v0.8 guards
// them itself, which makes this nest harmlessly, but releases before it do not,
// and a consumer building against one of those otherwise has to work around the
// name mangling from outside the project.
#include <cstdint>

#include <sys/queue.h>
#include <sys/uio.h>
#include <syslog.h>
#include <unistd.h>

extern "C" {
#include <libvfio-user.h>
#include <pci_caps/px.h>
}

#include <poll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

enum class RunCtxFaultState { Idle, Armed, Fired };
std::atomic<RunCtxFaultState> run_ctx_fault_state = RunCtxFaultState::Idle;

} // namespace

extern "C" int __real_vfu_run_ctx(vfu_ctx_t *ctx);
extern "C" int __wrap_vfu_run_ctx(vfu_ctx_t *ctx) {
  // Answer the request before reporting a transport failure. This leaves the
  // client usable while bypassing libvfio-user's lost-connection reset callback.
  const int result = __real_vfu_run_ctx(ctx);
  RunCtxFaultState expected = RunCtxFaultState::Armed;
  if (result >= 0 &&
      run_ctx_fault_state.compare_exchange_strong(expected, RunCtxFaultState::Fired)) {
    errno = ENOTCONN;
    return -1;
  }
  return result;
}

namespace {

class ChildProcessGuard {
public:
  explicit ChildProcessGuard(pid_t pid) : pid_(pid) {}
  ~ChildProcessGuard() {
    if (pid_ <= 0)
      return;
    (void)::kill(pid_, SIGKILL);
    int status = 0;
    while (::waitpid(pid_, &status, 0) == -1 && errno == EINTR) {
    }
  }

  ChildProcessGuard(const ChildProcessGuard &) = delete;
  ChildProcessGuard &operator=(const ChildProcessGuard &) = delete;

  void release() { pid_ = -1; }

private:
  pid_t pid_;
};

class ScopedFd {
public:
  explicit ScopedFd(int fd = -1) : fd_(fd) {}
  ~ScopedFd() { reset(); }

  ScopedFd(const ScopedFd &) = delete;
  ScopedFd &operator=(const ScopedFd &) = delete;

  [[nodiscard]] int get() const { return fd_; }
  void reset(int fd = -1) {
    if (fd_ >= 0)
      ::close(fd_);
    fd_ = fd;
  }

private:
  int fd_;
};

constexpr simdojo::PciId kTestId = {.vendor = 0x1002,
                                    .device = 0x1250,
                                    .subsys_vendor = 0x1002,
                                    .subsys = 0x4321,
                                    .cls = 0x12,
                                    .subcls = 0x00,
                                    .prog_if = 0x00,
                                    .revision = 0x5a};

/// @brief A served device plus the client attached to it.
///
/// @details Serving happens on its own thread, as it does in the product, so
/// the tests exercise the same threading the transport ships with. Two
/// underlying conditions govern state shared with those callbacks. Lifetime:
/// state captured by work that may still run during teardown must outlive
/// this fixture, because the destructor joins the serving thread only after
/// such work has had every chance to run. Work submitted to the serving
/// thread after `stop_serving()` has joined it never runs, so it cannot
/// touch anything -- `DiscardsAskedWorkWhenServingHasStopped` relies on this
/// and declares its flag after the fixture. Synchronization: reading
/// non-atomic state a callback wrote still needs a happens-before with the
/// callback. `stop_serving()` supplies one through the join, as can a future
/// or another explicit synchronization; merely knowing that the thread has
/// stopped does not itself provide that ordering.
class ServedDevice {
public:
  ServedDevice()
      : socket_path_(std::format("/tmp/rj-vfu-test-{}-{}.sock", ::getpid(), ++instance_counter_)),
        device_("test-device", kTestId, &trace_) {
    std::filesystem::remove(socket_path_);
    built_ = host_.build();
    if (built_) {
      serving_ = std::jthread([this](std::stop_token stop) { (void)host_.run(stop); });
    }
  }

  ~ServedDevice() {
    serving_.request_stop();
    if (serving_.joinable()) {
      serving_.join();
    }
    host_.detach();
    std::filesystem::remove(socket_path_);
  }

  [[nodiscard]] bool built() const { return built_; }

  /// @brief Connect a client, retrying while the server reaches its accept loop.
  [[nodiscard]] bool attach(rocjitsu::test::VfioUserClient &client) {
    for (int attempt = 0; attempt < 200; ++attempt) {
      if (client.connect(socket_path_)) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
  }

  rocjitsu::ScratchPciDevice &device() { return device_; }

  rocjitsu::VfioDeviceHost &host() { return host_; }

  /// @brief Identify the serving thread, so work can assert it ran there.
  [[nodiscard]] std::thread::id serving_thread_id() const { return serving_.get_id(); }

  /// @brief Stop serving and join, without waiting for the destructor.
  void stop_serving() {
    serving_.request_stop();
    if (serving_.joinable()) {
      serving_.join();
    }
  }

private:
  static inline std::atomic<int> instance_counter_ = 0;

  rocjitsu::RegisterSymbols symbols_;
  rocjitsu::BarAccessTrace trace_{symbols_};
  std::string socket_path_;
  rocjitsu::ScratchPciDevice device_;
  rocjitsu::VfioDeviceHost host_{socket_path_, device_};
  std::jthread serving_;
  bool built_ = false;
};

TEST(VfioDeviceHost, ServesAClientAndReportsItsBusShape) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  uint32_t regions = 0;
  uint32_t irqs = 0;
  ASSERT_TRUE(client.device_info(regions, irqs));
  EXPECT_EQ(regions, VFU_PCI_DEV_NUM_REGIONS);

  uint64_t size = 0;
  uint32_t flags = 0;
  ASSERT_TRUE(client.region_info(VFU_PCI_DEV_BAR0_REGION_IDX, size, flags));
  EXPECT_EQ(size, rocjitsu::ScratchPciDevice::kBarSize);
}

// The identity a device declares has to survive the trip through configuration
// space, which is what a guest driver matches on when it decides to bind.
TEST(VfioDeviceHost, PresentsTheDeclaredIdentityInConfigSpace) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  std::array<std::byte, 12> header{};
  ASSERT_TRUE(client.region_read(VFU_PCI_DEV_CFG_REGION_IDX, 0, header));

  const auto byte_at = [&header](std::size_t index) {
    return static_cast<unsigned>(std::to_integer<uint8_t>(header[index]));
  };
  EXPECT_EQ(byte_at(0) | (byte_at(1) << 8), kTestId.vendor);
  EXPECT_EQ(byte_at(2) | (byte_at(3) << 8), kTestId.device);
  EXPECT_EQ(byte_at(8), kTestId.revision) << "revision must survive the pinned library ignoring it";
  EXPECT_EQ(byte_at(9), kTestId.prog_if);
  EXPECT_EQ(byte_at(10), kTestId.subcls);
  EXPECT_EQ(byte_at(11), kTestId.cls) << "a guest driver binds on the class code";
}

TEST(VfioDeviceHost, PublishesASecondGenerationPcieEndpointCapability) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  std::array<std::byte, PCI_CFG_SPACE_SIZE> config{};
  ASSERT_TRUE(client.region_read(VFU_PCI_DEV_CFG_REGION_IDX, 0, config));

  uint8_t capability = std::to_integer<uint8_t>(config[PCI_CAPABILITY_LIST]);
  while (capability != 0 && std::to_integer<uint8_t>(config[capability]) != PCI_CAP_ID_EXP) {
    capability = std::to_integer<uint8_t>(config[capability + PCI_CAP_LIST_NEXT]);
  }
  ASSERT_NE(capability, 0) << "an Express function must identify itself to the guest as one";

  pxcap pcie{};
  std::memcpy(&pcie, config.data() + capability, sizeof(pcie));
  EXPECT_EQ(pcie.pxcaps.ver, 2u) << "Device Capabilities 2 is defined from PCIe version 2";
  EXPECT_EQ(pcie.pxcaps.dpt, PCI_EXP_TYPE_ENDPOINT);
  EXPECT_EQ(pcie.pxdcap2.aocs32, 0u);
  EXPECT_EQ(pcie.pxdcap2.aocs64, 0u)
      << "the transport must not invent AtomicOp completion for an ordinary device";
}

TEST(VfioDeviceHost, PublishesGpuAtomicCompletionCapabilities) {
  const std::string socket_path =
      std::format("/tmp/rj-vfu-test-atomics-{}.sock", static_cast<int>(::getpid()));
  std::filesystem::remove(socket_path);

  rocjitsu::config::KfdDeviceConfig identity;
  identity.local_mem_size = 64 * 1024 * 1024;
  identity.gfx_target_version = 120500;
  rocjitsu::config::PciDeviceConfig pci;
  pci.vram_aperture_bytes = rocjitsu::GpuPciDevice::kMinMemoryBarBytes;
  pci.doorbell_aperture_bytes = rocjitsu::GpuPciDevice::kMinMemoryBarBytes;
  rocjitsu::GpuPciDevice gpu("atomics", rocjitsu::gpu_pci_spec_from_config(identity, pci), nullptr);
  ASSERT_TRUE(gpu.usable());

  rocjitsu::VfioDeviceHost host(socket_path, gpu);
  ASSERT_TRUE(host.build());
  std::jthread serving([&host](std::stop_token stop) { (void)host.run(stop); });
  rocjitsu::test::VfioUserClient client;
  bool attached = false;
  for (int attempt = 0; attempt < 200 && !attached; ++attempt) {
    attached = client.connect(socket_path);
    if (!attached) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  ASSERT_TRUE(attached);

  std::array<std::byte, PCI_CFG_SPACE_SIZE> config{};
  ASSERT_TRUE(client.region_read(VFU_PCI_DEV_CFG_REGION_IDX, 0, config));
  uint8_t capability = std::to_integer<uint8_t>(config[PCI_CAPABILITY_LIST]);
  while (capability != 0 && std::to_integer<uint8_t>(config[capability]) != PCI_CAP_ID_EXP) {
    capability = std::to_integer<uint8_t>(config[capability + PCI_CAP_LIST_NEXT]);
  }
  ASSERT_NE(capability, 0);

  pxcap pcie{};
  std::memcpy(&pcie, config.data() + capability, sizeof(pcie));
  EXPECT_EQ(pcie.pxdcap2.aocs32, 1u);
  EXPECT_EQ(pcie.pxdcap2.aocs64, 1u);

  serving.request_stop();
  serving.join();
  host.detach();
  std::filesystem::remove(socket_path);
}

TEST(VfioDeviceHost, CarriesABarWriteAndReadBackToTheDevice) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  constexpr uint32_t kWritten = 0xdeadbeef;
  const auto written = std::bit_cast<std::array<std::byte, 4>>(kWritten);
  ASSERT_TRUE(client.region_write(VFU_PCI_DEV_BAR0_REGION_IDX, 0x40, written));

  std::array<std::byte, 4> read{};
  ASSERT_TRUE(client.region_read(VFU_PCI_DEV_BAR0_REGION_IDX, 0x40, read));
  EXPECT_EQ(std::bit_cast<uint32_t>(read), kWritten);
}

// A device rejects an access it cannot service, and the client has to see that
// as an error rather than as a successful read of stale bytes.
TEST(VfioDeviceHost, ReportsADeviceRejectionAsAnError) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  std::array<std::byte, 4> read{};
  EXPECT_FALSE(
      client.region_read(VFU_PCI_DEV_BAR0_REGION_IDX, rocjitsu::ScratchPciDevice::kBarSize, read))
      << "an access past the end of the BAR must fail";
}

TEST(VfioDeviceHost, AnnouncesAWindowTheClientSharesMappably) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  constexpr uint64_t kWindowSize = 0x2000;
  const int backing = ::memfd_create("rj-vfu-test-window", 0);
  ASSERT_GE(backing, 0);
  ASSERT_EQ(::ftruncate(backing, kWindowSize), 0);

  ASSERT_TRUE(client.dma_map(0x100000, kWindowSize, backing, 0));
  EXPECT_EQ(served.device().mapped_regions(), 1u);

  ASSERT_TRUE(client.dma_unmap(0x100000, kWindowSize));
  EXPECT_EQ(served.device().mapped_regions(), 0u);
  ::close(backing);
}

TEST(VfioDeviceHost, PerformsTypedAtomicAccessOnlyOnLiveMappedGuestMemory) {
  ServedDevice served;
  ASSERT_TRUE(served.built());

  constexpr uint64_t kGuestAddress = 0x180000;
  constexpr uint64_t kWindowSize = 0x1000;
  EXPECT_EQ(served.host().atomic_load(kGuestAddress, sizeof(uint64_t)).outcome,
            simdojo::DmaAccessOutcome::Unavailable);
  EXPECT_EQ(served.host().atomic_store(kGuestAddress + 1, sizeof(uint64_t), 1),
            simdojo::DmaAccessOutcome::Malformed);

  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));
  const int backing = ::memfd_create("rj-vfu-test-atomic-window", 0);
  ASSERT_GE(backing, 0);
  ASSERT_EQ(::ftruncate(backing, kWindowSize), 0);
  void *mapping = ::mmap(nullptr, kWindowSize, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  auto *value = static_cast<uint64_t *>(mapping);
  std::atomic_ref<uint64_t>(*value).store(7, std::memory_order_release);
  ASSERT_TRUE(client.dma_map(kGuestAddress, kWindowSize, backing, 0));

  const simdojo::DmaAtomicLoadResult loaded =
      served.host().atomic_load(kGuestAddress, sizeof(uint64_t));
  EXPECT_EQ(loaded.outcome, simdojo::DmaAccessOutcome::Complete);
  EXPECT_EQ(loaded.value, 7u);
  const simdojo::DmaAtomicLoadResult loaded16 =
      served.host().atomic_load(kGuestAddress, sizeof(uint16_t));
  EXPECT_EQ(loaded16.outcome, simdojo::DmaAccessOutcome::Complete);
  EXPECT_EQ(loaded16.value, 7u);
  EXPECT_EQ(served.host().atomic_store(kGuestAddress, sizeof(uint64_t), 11),
            simdojo::DmaAccessOutcome::Complete);
  EXPECT_EQ(std::atomic_ref<uint64_t>(*value).load(std::memory_order_acquire), 11u);

  const simdojo::DmaAtomicCompareExchangeResult exchanged =
      served.host().compare_exchange(kGuestAddress, sizeof(uint64_t), 11, 13);
  EXPECT_EQ(exchanged.outcome, simdojo::DmaAccessOutcome::Complete);
  EXPECT_EQ(exchanged.observed, 11u);
  EXPECT_TRUE(exchanged.exchanged);
  EXPECT_EQ(std::atomic_ref<uint64_t>(*value).load(std::memory_order_acquire), 13u);

  const simdojo::DmaAtomicCompareExchangeResult mismatch =
      served.host().compare_exchange(kGuestAddress, sizeof(uint64_t), 11, 17);
  EXPECT_EQ(mismatch.outcome, simdojo::DmaAccessOutcome::Complete);
  EXPECT_EQ(mismatch.observed, 13u);
  EXPECT_FALSE(mismatch.exchanged);
  EXPECT_EQ(served.host().atomic_load(kGuestAddress + kWindowSize, sizeof(uint64_t)).outcome,
            simdojo::DmaAccessOutcome::Faulted);

  ASSERT_TRUE(client.dma_unmap(kGuestAddress, kWindowSize));
  ASSERT_EQ(::munmap(mapping, kWindowSize), 0);
  ::close(backing);
}

// The transport serves only windows it can map. One shared without a descriptor
// is accepted at the protocol level -- there is no way to refuse a single window
// -- and then withheld from the device, so the device never holds a window whose
// every access would fail.
TEST(VfioDeviceHost, WithholdsAWindowSharedWithoutADescriptor) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  ASSERT_TRUE(client.dma_map(0x200000, 0x1000, -1, 0));

  EXPECT_EQ(served.device().mapped_regions(), 0u)
      << "the device must not be told about a window it could never read";
}

TEST(VfioDeviceHost, FailedReadAcrossAnUnreachableWindowLeavesTheDestinationUnchanged) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  constexpr uint64_t kGuestAddress = 0x280000;
  constexpr std::size_t kWindowSize = 0x1000;
  constexpr std::size_t kTransferSize = 16;
  const int backing = ::memfd_create("rj-vfu-test-atomic-read", 0);
  ASSERT_GE(backing, 0);
  ASSERT_EQ(::ftruncate(backing, kWindowSize), 0);
  void *mapping = ::mmap(nullptr, kWindowSize, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  std::memset(mapping, 0xa5, kWindowSize);

  ASSERT_TRUE(client.dma_map(kGuestAddress, kWindowSize, backing, 0));
  ASSERT_TRUE(client.dma_map(kGuestAddress + kWindowSize, kWindowSize, -1, 0));

  std::array<std::byte, kTransferSize> destination{};
  destination.fill(std::byte{0x5a});
  const std::array<std::byte, kTransferSize> before = destination;
  EXPECT_EQ(
      served.host().read_outcome(kGuestAddress + kWindowSize - kTransferSize / 2, destination),
      simdojo::DmaAccessOutcome::Faulted);
  EXPECT_EQ(destination, before)
      << "a failed physical read must not expose the prefix from the reachable window";

  ASSERT_EQ(::munmap(mapping, kWindowSize), 0);
  ::close(backing);
}

TEST(VfioDeviceHost, FailedWriteAcrossAnUnreachableWindowLeavesGuestMemoryUnchanged) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  constexpr uint64_t kGuestAddress = 0x2a0000;
  constexpr std::size_t kWindowSize = 0x1000;
  constexpr std::size_t kTransferSize = 16;
  const int backing = ::memfd_create("rj-vfu-test-atomic-write", 0);
  ASSERT_GE(backing, 0);
  ASSERT_EQ(::ftruncate(backing, kWindowSize), 0);
  void *mapping = ::mmap(nullptr, kWindowSize, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  std::memset(mapping, 0x3c, kWindowSize);

  ASSERT_TRUE(client.dma_map(kGuestAddress, kWindowSize, backing, 0));
  ASSERT_TRUE(client.dma_map(kGuestAddress + kWindowSize, kWindowSize, -1, 0));

  const auto *bytes = static_cast<const std::byte *>(mapping);
  const std::array<std::byte, kTransferSize / 2> before = {
      bytes[kWindowSize - 8], bytes[kWindowSize - 7], bytes[kWindowSize - 6],
      bytes[kWindowSize - 5], bytes[kWindowSize - 4], bytes[kWindowSize - 3],
      bytes[kWindowSize - 2], bytes[kWindowSize - 1]};
  std::array<std::byte, kTransferSize> source{};
  source.fill(std::byte{0xc3});
  EXPECT_EQ(served.host().write_outcome(kGuestAddress + kWindowSize - kTransferSize / 2, source),
            simdojo::DmaAccessOutcome::Faulted);
  EXPECT_TRUE(
      std::ranges::equal(before, std::span(bytes + kWindowSize - before.size(), before.size())))
      << "a failed physical write must not modify the reachable window's prefix";

  ASSERT_EQ(::munmap(mapping, kWindowSize), 0);
  ::close(backing);
}

TEST(VfioDeviceHost, ReadsAcrossHundredsOfMappedWindows) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  constexpr uint64_t kGuestAddress = 0x400000;
  constexpr std::size_t kWindowSize = 0x1000;
  // One more than the former streaming threshold keeps highly fragmented
  // requests on the same all-or-nothing aggregate path as ordinary requests.
  constexpr std::size_t kWindowCount = 257;
  constexpr std::size_t kTransferSize = kWindowSize * kWindowCount;
  const int backing = ::memfd_create("rj-vfu-test-many-windows", 0);
  ASSERT_GE(backing, 0);
  ASSERT_EQ(::ftruncate(backing, kTransferSize), 0);
  void *mapping = ::mmap(nullptr, kTransferSize, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  std::memset(mapping, 0x69, kTransferSize);

  for (std::size_t index = 0; index < kWindowCount; ++index) {
    ASSERT_TRUE(client.dma_map(kGuestAddress + index * kWindowSize, kWindowSize, backing,
                               index * kWindowSize));
  }

  std::vector<std::byte> destination(kTransferSize);
  ASSERT_EQ(served.host().read_outcome(kGuestAddress, destination),
            simdojo::DmaAccessOutcome::Complete);
  EXPECT_TRUE(
      std::ranges::all_of(destination, [](std::byte value) { return value == std::byte{0x69}; }));

  ASSERT_EQ(::munmap(mapping, kTransferSize), 0);
  ::close(backing);
}

TEST(VfioDeviceHost, KeepsCountWhenTheSameWindowIsSharedTwice) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  constexpr uint64_t kWindowSize = 0x1000;
  const int backing = ::memfd_create("rj-vfu-test-dup", 0);
  ASSERT_GE(backing, 0);
  ASSERT_EQ(::ftruncate(backing, kWindowSize), 0);

  ASSERT_TRUE(client.dma_map(0x300000, kWindowSize, backing, 0));
  (void)client.dma_map(0x300000, kWindowSize, backing, 0);

  EXPECT_EQ(served.device().mapped_regions(), 1u)
      << "one logical window must produce one device mapping";
  ::close(backing);
}

// A device whose BARs all trap can be served again: nothing of it outlives the
// connection, so a VMM may be restarted against a running server.
TEST(VfioDeviceHostLifecycle, ServesAnotherClientWhenNothingWasShared) {
  ServedDevice served;
  ASSERT_TRUE(served.built());

  uint64_t first_generation = 0;
  {
    rocjitsu::test::VfioUserClient first;
    ASSERT_TRUE(served.attach(first));
    auto written = std::bit_cast<std::array<std::byte, 4>>(uint32_t{0xa5a5a5a5});
    ASSERT_TRUE(first.region_write(VFU_PCI_DEV_BAR0_REGION_IDX, 0, written));
    first_generation = served.device().transport_session_generation();
    ASSERT_NE(first_generation, 0u);
  }

  rocjitsu::test::VfioUserClient second;
  ASSERT_TRUE(served.attach(second)) << "a trapped-only device must accept a replacement client";
  std::array<std::byte, 4> read{};
  EXPECT_TRUE(second.region_read(VFU_PCI_DEV_BAR0_REGION_IDX, 0, read));
  EXPECT_NE(served.device().transport_session_generation(), first_generation)
      << "the second client inherited the first client's transport generation";
}

TEST(VfioDeviceHostLifecycle, PublishesCompletedDisconnectsAfterTrappedClientSessions) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  EXPECT_EQ(served.host().client_snapshot().completed_disconnects, 0u);
  EXPECT_FALSE(served.host().client_snapshot().attached);
  EXPECT_FALSE(served.host().client_snapshot().disconnect_in_progress);

  for (uint64_t expected_disconnects = 1; expected_disconnects <= 2; ++expected_disconnects) {
    {
      rocjitsu::test::VfioUserClient client;
      ASSERT_TRUE(served.attach(client));
      uint32_t regions = 0;
      uint32_t irqs = 0;
      ASSERT_TRUE(client.device_info(regions, irqs));
      const rocjitsu::VfioDeviceHost::ClientSnapshot attached = served.host().client_snapshot();
      EXPECT_TRUE(attached.attached);
      EXPECT_FALSE(attached.disconnect_in_progress);
      EXPECT_EQ(attached.completed_disconnects, expected_disconnects - 1);
    }

    rocjitsu::VfioDeviceHost::ClientSnapshot departed = served.host().client_snapshot();
    for (int attempt = 0; attempt < 500 && departed.completed_disconnects < expected_disconnects;
         ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      departed = served.host().client_snapshot();
    }
    EXPECT_EQ(departed.completed_disconnects, expected_disconnects);
    EXPECT_FALSE(departed.attached);
    EXPECT_FALSE(departed.disconnect_in_progress);
  }

  served.stop_serving();
  EXPECT_EQ(served.host().client_snapshot().completed_disconnects, 2u)
      << "a requested serving stop must not count as another client disconnect";
}

/// @brief A trapped-only device that records the state at its first reset.
class ResetObservingDevice final : public simdojo::PciDevice {
public:
  struct Observation {
    simdojo::ResetKind kind;
    uint64_t active_generation;
    simdojo::PciTransportSession::State old_session_state;
  };

  explicit ResetObservingDevice(std::shared_future<void> reset_release = {})
      : simdojo::PciDevice("reset-observer", kTestId), reset_release_(std::move(reset_release)) {}

  [[nodiscard]] std::vector<simdojo::BarSpec> bars() const override {
    simdojo::BarSpec bar;
    bar.index = 0;
    bar.size = 4096;
    bar.mem = true;
    return {bar};
  }

  [[nodiscard]] int64_t bar_access(int, std::span<std::byte> bytes, uint64_t, bool write) override {
    if (!write)
      std::ranges::fill(bytes, std::byte{0});
    return static_cast<int64_t>(bytes.size());
  }

  void dma_map(const simdojo::DmaRegion &) override {}
  void dma_unmap(const simdojo::DmaRegion &) override {}

  void reset(simdojo::ResetKind kind) override {
    if (recorded_.exchange(true))
      return;
    // Cleanup may reset after a failed setup, before the test has captured a session.
    first_reset_.set_value({.kind = kind,
                            .active_generation = transport_session_generation(),
                            .old_session_state = watched_session_ != nullptr
                                                     ? watched_session_->state()
                                                     : simdojo::PciTransportSession::State::Open});
    if (reset_release_.valid())
      reset_release_.wait();
  }

  [[nodiscard]] std::shared_ptr<simdojo::PciTransportSession> current_session() const {
    return transport_session();
  }

  void watch_session(std::shared_ptr<simdojo::PciTransportSession> session) {
    watched_session_ = std::move(session);
  }

  [[nodiscard]] std::future<Observation> first_reset() { return first_reset_.get_future(); }

private:
  std::shared_ptr<simdojo::PciTransportSession> watched_session_;
  std::promise<Observation> first_reset_;
  std::atomic<bool> recorded_ = false;
  std::shared_future<void> reset_release_;
};

class ScopedResetRelease {
public:
  explicit ScopedResetRelease(std::promise<void> &promise) : promise_(promise) {}
  ~ScopedResetRelease() { release(); }

  ScopedResetRelease(const ScopedResetRelease &) = delete;
  ScopedResetRelease &operator=(const ScopedResetRelease &) = delete;

  void release() {
    if (released_)
      return;
    promise_.set_value();
    released_ = true;
  }

private:
  std::promise<void> &promise_;
  bool released_ = false;
};

TEST(VfioDeviceHostLifecycle, KeepsClosedSnapshotWhileDisconnectResetIsSlow) {
  rocjitsu::test::ScopedTempDirectory runtime("rj-vfu-slow-reset-");
  const std::string socket_path = std::format("{}/gpu.sock", runtime.path());
  std::promise<void> reset_release;
  ResetObservingDevice device(reset_release.get_future().share());
  std::future<ResetObservingDevice::Observation> reset_started = device.first_reset();
  rocjitsu::VfioDeviceHost host(socket_path, device);
  ASSERT_TRUE(host.build());
  std::jthread serving([&](std::stop_token stop) { (void)host.run(stop); });
  // Constructed after the serving thread so even a failed assertion releases
  // the reset before the thread's destructor joins it.
  ScopedResetRelease release_on_exit(reset_release);

  {
    rocjitsu::test::VfioUserClient client;
    bool connected = false;
    for (int attempt = 0; attempt < 200 && !connected; ++attempt) {
      connected = client.connect(socket_path);
      if (!connected)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(connected);
    uint32_t regions = 0;
    uint32_t irqs = 0;
    ASSERT_TRUE(client.device_info(regions, irqs));
  }

  ASSERT_EQ(reset_started.wait_for(std::chrono::seconds(3)), std::future_status::ready);
  EXPECT_EQ(reset_started.get().kind, simdojo::ResetKind::LostConnection);
  const rocjitsu::VfioDeviceHost::ClientSnapshot draining = host.client_snapshot();
  EXPECT_FALSE(draining.attached);
  EXPECT_TRUE(draining.disconnect_in_progress);
  EXPECT_EQ(draining.completed_disconnects, 0u);

  // Stay past the node's five-second grace: socket closure and reset
  // completion are distinct events, and a slow drain does not reopen a socket.
  std::this_thread::sleep_for(std::chrono::milliseconds(5'200));
  const rocjitsu::VfioDeviceHost::ClientSnapshot still_draining = host.client_snapshot();
  EXPECT_FALSE(still_draining.attached);
  EXPECT_TRUE(still_draining.disconnect_in_progress);
  EXPECT_EQ(still_draining.completed_disconnects, 0u);

  release_on_exit.release();
  rocjitsu::VfioDeviceHost::ClientSnapshot completed = host.client_snapshot();
  for (int attempt = 0; attempt < 500 && completed.completed_disconnects == 0; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    completed = host.client_snapshot();
  }
  EXPECT_FALSE(completed.attached);
  EXPECT_FALSE(completed.disconnect_in_progress);
  EXPECT_EQ(completed.completed_disconnects, 1u);
  serving.request_stop();
  serving.join();
  host.detach();
}

// The library normally invokes its reset callback before returning ENOTCONN.
// This injects the raw errno path without that callback, while the first client
// is still connected. The child keeps a bad serving loop or blocked client
// request from hanging the whole test executable.
int exercise_raw_enotconn(const std::string &socket_path) {
  ResetObservingDevice device;
  std::future<ResetObservingDevice::Observation> first_reset = device.first_reset();
  rocjitsu::VfioDeviceHost host(socket_path, device);
  if (!host.build())
    return 1;
  rocjitsu::VfioDeviceHost::ServeResult serve_result =
      rocjitsu::VfioDeviceHost::ServeResult::Failed;
  std::jthread serving([&](std::stop_token stop) { serve_result = host.run(stop); });
  rocjitsu::test::VfioUserClient client;
  const int scenario_status = [&]() -> int {
    bool attached = false;
    for (int attempt = 0; attempt < 200 && !attached; ++attempt) {
      attached = client.connect(socket_path);
      if (!attached)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!attached)
      return 2;

    uint32_t regions = 0;
    uint32_t irqs = 0;
    if (!client.device_info(regions, irqs))
      return 3;
    std::shared_ptr<simdojo::PciTransportSession> old_session = device.current_session();
    if (old_session == nullptr || old_session->state() != simdojo::PciTransportSession::State::Open)
      return 4;
    device.watch_session(old_session);

    run_ctx_fault_state.store(RunCtxFaultState::Armed);
    const bool request_answered = client.device_info(regions, irqs);
    if (!request_answered)
      return 5;
    if (first_reset.wait_for(std::chrono::seconds(3)) != std::future_status::ready)
      return 6;
    const ResetObservingDevice::Observation observed = first_reset.get();
    if (run_ctx_fault_state.load() != RunCtxFaultState::Fired)
      return 7;
    if (observed.kind != simdojo::ResetKind::LostConnection)
      return 8;
    if (observed.active_generation != 0 ||
        observed.old_session_state != simdojo::PciTransportSession::State::Revoked)
      return 9;
    if (device.transport_session_generation() != 0 || old_session->acquire())
      return 10;
    rocjitsu::VfioDeviceHost::ClientSnapshot snapshot = host.client_snapshot();
    for (int attempt = 0; attempt < 200 && snapshot.completed_disconnects == 0; ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      snapshot = host.client_snapshot();
    }
    if (snapshot.attached || snapshot.disconnect_in_progress || snapshot.completed_disconnects != 1)
      return 12;
    return 0;
  }();

  // Keep the client open until all pre-detach observations are complete.
  run_ctx_fault_state.store(RunCtxFaultState::Idle);
  serving.request_stop();
  serving.join();
  host.detach();
  if (scenario_status != 0)
    return scenario_status;
  return serve_result == rocjitsu::VfioDeviceHost::ServeResult::Stopped ? 0 : 11;
}

TEST(VfioDeviceHostLifecycle, RawEnotconnRevokesAndResetsBeforeDetach) {
  const std::string socket_path =
      std::format("/tmp/rj-vfu-test-enotconn-{}.sock", static_cast<int>(::getpid()));
  std::filesystem::remove(socket_path);

  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0)
    ::_exit(exercise_raw_enotconn(socket_path));

  ChildProcessGuard child_guard(child);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t waited = ::waitpid(child, &status, WNOHANG);
    if (waited == child) {
      child_guard.release();
      std::filesystem::remove(socket_path);
      ASSERT_TRUE(WIFEXITED(status));
      EXPECT_EQ(WEXITSTATUS(status), 0)
          << "raw ENOTCONN did not reset and revoke the active session before detach; stage "
          << WEXITSTATUS(status);
      return;
    }
    ASSERT_TRUE(waited == 0 || (waited == -1 && errno == EINTR));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::filesystem::remove(socket_path);
  FAIL() << "raw ENOTCONN serving scenario did not finish within ten seconds";
}

// A device that shared video memory by descriptor cannot take that mapping back
// when the client goes away, so serving ends rather than handing a second
// client memory the first can still reach.
TEST(VfioDeviceHostLifecycle, StopsServingAfterASharedMemoryClientDisconnects) {
  const std::string socket_path =
      std::format("/tmp/rj-vfu-test-gpu-{}.sock", static_cast<int>(::getpid()));
  std::filesystem::remove(socket_path);

  rocjitsu::config::KfdDeviceConfig identity;
  identity.vendor_id = 0x1002;
  identity.device_id = 0x1250;
  // Only this target has an IP discovery profile, and a device without one
  // refuses to become usable rather than describing itself as another part.
  identity.gfx_target_version = 120500;
  identity.local_mem_size = 8ULL * 1024 * 1024;
  rocjitsu::GpuPciDevice gpu("gpu", rocjitsu::gpu_pci_spec_from_config(identity, {}), nullptr);
  ASSERT_TRUE(gpu.usable());

  rocjitsu::VfioDeviceHost host(socket_path, gpu);
  ASSERT_TRUE(host.build());

  std::atomic<bool> finished = false;
  auto result = rocjitsu::VfioDeviceHost::ServeResult::Failed;
  std::jthread serving([&](std::stop_token stop) {
    result = host.run(stop);
    finished = true;
  });

  {
    rocjitsu::test::VfioUserClient client;
    bool connected = false;
    for (int attempt = 0; attempt < 200 && !connected; ++attempt) {
      connected = client.connect(socket_path);
      if (!connected) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    ASSERT_TRUE(connected);
    uint32_t regions = 0;
    uint32_t irqs = 0;
    ASSERT_TRUE(client.device_info(regions, irqs));

    // What the guest actually sees, rather than what the device asked for. The
    // capability only reaches it if vfu_setup_device_nr_irqs() ran for the kind
    // the plan chose, and nothing else covers that path: the device-side tests
    // stop at interrupts() and plan_interrupts(). The device advertises
    // messages now, so the pin it used to offer must be gone as well -- a guest
    // that finds both would be free to choose the one this device cannot raise.
    uint32_t msix_vectors = 0;
    ASSERT_TRUE(client.irq_info(VFIO_PCI_MSIX_IRQ_INDEX, msix_vectors));
    EXPECT_EQ(msix_vectors, 1u)
        << "the guest is offered no message vector, so its driver's pci_alloc_irq_vectors() "
           "fails and the probe fails with it";

    uint32_t intx_vectors = 0;
    ASSERT_TRUE(client.irq_info(VFIO_PCI_INTX_IRQ_INDEX, intx_vectors));
    EXPECT_EQ(intx_vectors, 0u) << "the legacy pin outlived the switch to messages";
  }

  // No stop is requested: the disconnect alone must end serving.
  for (int attempt = 0; attempt < 500 && !finished.load(); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(finished.load()) << "serving must end on its own once the client is gone";
  serving.request_stop();
  serving.join();
  host.detach();
  std::filesystem::remove(socket_path);

  EXPECT_EQ(result, rocjitsu::VfioDeviceHost::ServeResult::Stopped)
      << "ending because the sole client left is orderly, not a failure";
}

// The device refuses a bus shape the transport would reject; the two must also
// agree on what is acceptable, or a device reports itself fine and then fails
// while the transport is being built around it.
TEST(VfioDeviceHostLifecycle, BuildsTheSmallestBusShapeTheDeviceAccepts) {
  const std::string socket_path =
      std::format("/tmp/rj-vfu-test-floor-{}.sock", static_cast<int>(::getpid()));
  std::filesystem::remove(socket_path);

  rocjitsu::config::KfdDeviceConfig identity;
  identity.local_mem_size = 64 * 1024 * 1024;
  identity.gfx_target_version = 120500;
  rocjitsu::config::PciDeviceConfig pci;
  pci.vram_aperture_bytes = rocjitsu::GpuPciDevice::kMinMemoryBarBytes;
  pci.doorbell_aperture_bytes = rocjitsu::GpuPciDevice::kMinMemoryBarBytes;

  rocjitsu::GpuPciDevice gpu("floor", rocjitsu::gpu_pci_spec_from_config(identity, pci), nullptr);
  ASSERT_TRUE(gpu.usable()) << "the PCI minimum must be acceptable to the device";

  rocjitsu::VfioDeviceHost host(socket_path, gpu);
  EXPECT_TRUE(host.build()) << "and to the transport, or the two disagree";
  host.detach();
  std::filesystem::remove(socket_path);
}

/// @brief A minimal MSI-X device that raises on demand.
///
/// @details The GPU device has nothing that triggers an interrupt yet -- that
/// is a later commit -- so the delivery path is exercised through a device
/// whose only job is to raise one. It advertises the same capability the GPU
/// does, so the transport sets up the same way.
class RaisingDevice : public simdojo::PciDevice {
public:
  RaisingDevice() : simdojo::PciDevice("raiser", kTestId) {}

  [[nodiscard]] std::vector<simdojo::BarSpec> bars() const override {
    simdojo::BarSpec msix;
    msix.index = 0;
    msix.size = 8 * 1024;
    msix.mem = true;
    return {msix};
  }

  [[nodiscard]] simdojo::InterruptSpec interrupts() const override {
    return {.kind = simdojo::InterruptKind::MsiX,
            .vectors = 1,
            .table_bar = 0,
            .table_offset = 0,
            .pending_offset = 4 * 1024};
  }

  [[nodiscard]] int64_t bar_access(int, std::span<std::byte> buf, uint64_t, bool write) override {
    if (!write) {
      std::ranges::fill(buf, std::byte{0});
    }
    return static_cast<int64_t>(buf.size());
  }

  void dma_map(const simdojo::DmaRegion &) override {}
  void dma_unmap(const simdojo::DmaRegion &) override {}
  void reset(simdojo::ResetKind) override {}

  /// @brief Raise vector zero, as an interrupt source eventually will.
  [[nodiscard]] bool raise() {
    simdojo::PciTransportSession::OperationLease transport = acquire_transport();
    return transport && transport.trigger(0);
  }
};

// The capability bytes only say the device *can* be signalled. This checks that
// it actually is: the client arms a vector with an eventfd, the device raises,
// and the descriptor the client handed over is what moves. Without it the test
// above still passes if vfu_setup_device_nr_irqs() and VfioDeviceHost::trigger()
// stop being connected to each other.
TEST(VfioDeviceHostLifecycle, DeliversAMessageToTheVectorTheClientArmed) {
  const std::string socket_path =
      std::format("/tmp/rj-vfu-test-raise-{}.sock", static_cast<int>(::getpid()));
  std::filesystem::remove(socket_path);

  RaisingDevice device;
  rocjitsu::VfioDeviceHost host(socket_path, device);
  ASSERT_TRUE(host.build());
  std::jthread serving([&host](std::stop_token stop) { (void)host.run(stop); });

  rocjitsu::test::VfioUserClient client;
  bool attached = false;
  for (int attempt = 0; attempt < 200 && !attached; ++attempt) {
    attached = client.connect(socket_path);
    if (!attached) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  ASSERT_TRUE(attached);

  const int signalled = ::eventfd(0, EFD_NONBLOCK);
  ASSERT_GE(signalled, 0);
  ASSERT_TRUE(client.arm_irq(VFIO_PCI_MSIX_IRQ_INDEX, signalled))
      << "the server refused to arm the vector, so nothing could be delivered";

  ASSERT_TRUE(device.raise());

  uint64_t count = 0;
  bool delivered = false;
  for (int attempt = 0; attempt < 200 && !delivered; ++attempt) {
    delivered = ::read(signalled, &count, sizeof(count)) == sizeof(count);
    if (!delivered) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  EXPECT_TRUE(delivered) << "the device raised and the client's eventfd never moved";
  EXPECT_EQ(count, 1u);

  ::close(signalled);
  serving.request_stop();
  serving.join();
  host.detach();
  std::filesystem::remove(socket_path);
}

// A server-thread request is allowed to invoke a device action that re-enters
// the transport. Interrupt delivery is the smallest complete example: the
// request already runs inside the libvfio-user serialization lock, then the
// device reaches the host's IRQ sink. Run the scenario in a child so a lost
// recursive-lock guarantee becomes a bounded test failure instead of wedging
// the test process during serving-thread destruction.
TEST(VfioDeviceHostLifecycle, ServingThreadRequestCanReenterTransportForInterrupt) {
  const std::string socket_path =
      std::format("/tmp/rj-vfu-test-request-raise-{}.sock", static_cast<int>(::getpid()));
  std::filesystem::remove(socket_path);

  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    RaisingDevice device;
    rocjitsu::VfioDeviceHost host(socket_path, device);
    if (!host.build())
      ::_exit(2);
    std::jthread serving([&host](std::stop_token stop) { (void)host.run(stop); });

    rocjitsu::test::VfioUserClient client;
    bool attached = false;
    for (int attempt = 0; attempt < 200 && !attached; ++attempt) {
      attached = client.connect(socket_path);
      if (!attached)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!attached)
      ::_exit(3);

    const int signalled = ::eventfd(0, EFD_NONBLOCK);
    if (signalled < 0)
      ::_exit(4);
    if (!client.arm_irq(VFIO_PCI_MSIX_IRQ_INDEX, signalled))
      ::_exit(5);
    if (!host.ask_serving_thread([&device] {
          if (!device.raise())
            ::_exit(6);
        })) {
      ::_exit(7);
    }

    pollfd wait = {.fd = signalled, .events = POLLIN, .revents = 0};
    int ready = 0;
    do {
      ready = ::poll(&wait, 1, 5'000);
    } while (ready < 0 && errno == EINTR);
    if (ready != 1)
      ::_exit(8);

    uint64_t count = 0;
    if (::read(signalled, &count, sizeof(count)) != sizeof(count) || count != 1)
      ::_exit(9);
    ::close(signalled);
    serving.request_stop();
    serving.join();
    host.detach();
    std::filesystem::remove(socket_path);
    ::_exit(EXIT_SUCCESS);
  }

  ChildProcessGuard child_guard(child);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t waited = ::waitpid(child, &status, WNOHANG);
    if (waited == child) {
      child_guard.release();
      std::filesystem::remove(socket_path);
      ASSERT_TRUE(WIFEXITED(status));
      EXPECT_EQ(WEXITSTATUS(status), EXIT_SUCCESS)
          << "serving-thread interrupt delivery did not complete";
      return;
    }
    ASSERT_TRUE(waited == 0 || (waited == -1 && errno == EINTR));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::filesystem::remove(socket_path);
  FAIL() << "serving-thread interrupt delivery deadlocked while re-entering the transport";
}

// The message capability is three little-endian dwords and every field in it is
// packed: the vector count is stored one less than it is, and the two offsets
// are stored in units of eight bytes with the BAR index tucked into the three
// bits below them. Nothing downstream rejects a bad packing -- a guest simply
// looks for its table at whatever address comes out -- so the encoding is
// checked here rather than by booting one and reading dmesg.
TEST(VfioDeviceHostLifecycle, EncodesTheMessageCapabilityAGuestCanFollow) {
  const std::string socket_path =
      std::format("/tmp/rj-vfu-test-msix-{}.sock", static_cast<int>(::getpid()));
  std::filesystem::remove(socket_path);

  rocjitsu::config::KfdDeviceConfig identity;
  identity.gfx_target_version = 120500;
  identity.local_mem_size = 64 * 1024 * 1024;
  rocjitsu::GpuPciDevice gpu("msix", rocjitsu::gpu_pci_spec_from_config(identity, {}), nullptr);
  ASSERT_TRUE(gpu.usable());

  rocjitsu::VfioDeviceHost host(socket_path, gpu);
  ASSERT_TRUE(host.build());
  std::jthread serving([&host](std::stop_token stop) { (void)host.run(stop); });
  rocjitsu::test::VfioUserClient client;
  bool attached = false;
  for (int attempt = 0; attempt < 200 && !attached; ++attempt) {
    attached = client.connect(socket_path);
    if (!attached) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  ASSERT_TRUE(attached);

  // A failed read throws rather than recording a non-fatal expectation: these
  // return a value, so ASSERT_ is unavailable here, and continuing would walk
  // the capability list through zeros and report a wrong pointer rather than
  // the read that never happened.
  const auto read_byte = [&client](uint64_t at) {
    std::array<std::byte, 1> one{};
    if (!client.region_read(VFU_PCI_DEV_CFG_REGION_IDX, at, one)) {
      throw std::runtime_error(std::format("cannot read config byte at {:#x}", at));
    }
    return std::to_integer<uint8_t>(one[0]);
  };
  const auto read_dword = [&client](uint64_t at) {
    std::array<std::byte, 4> four{};
    if (!client.region_read(VFU_PCI_DEV_CFG_REGION_IDX, at, four)) {
      throw std::runtime_error(std::format("cannot read config dword at {:#x}", at));
    }
    return std::bit_cast<uint32_t>(four);
  };

  // A capability added after the device is realized still writes itself and
  // still writes the pointer at 0x34; the one thing left clear is this bit, and
  // a guest reads it before it reads the pointer. So this, rather than finding
  // the capability, is what says the ordering in build() held.
  std::array<std::byte, 2> status{};
  ASSERT_TRUE(client.region_read(VFU_PCI_DEV_CFG_REGION_IDX, 0x06, status));
  ASSERT_NE(std::bit_cast<uint16_t>(status) & 0x10u, 0u)
      << "the capability list is not advertised, so a guest never walks it";

  // The point of the whole capability is to avoid a pin, whose delivery costs
  // the guest every BAR mapping it holds. Nothing else asserts what actually
  // reaches configuration space.
  EXPECT_EQ(read_byte(0x3d), 0u)
      << "a pin as well would put the client's mmap-disabling path back in reach";

  // Walk the capability list the way a guest does, from the pointer at 0x34.
  uint64_t at = read_byte(0x34);
  uint64_t found = 0;
  for (int hop = 0; hop < 48 && at >= 0x40; ++hop) {
    if (read_byte(at) == PCI_CAP_ID_MSIX) {
      found = at;
      break;
    }
    at = read_byte(at + 1);
  }
  ASSERT_NE(found, 0u) << "no message capability is published for a guest to find";

  const auto control = static_cast<uint16_t>(read_dword(found) >> 16);
  const uint32_t table = read_dword(found + 4);
  const uint32_t pending = read_dword(found + 8);

  EXPECT_EQ((control & 0x7ffu) + 1, rocjitsu::GpuPciDevice::kMsixVectors)
      << "the table size is stored one less than the count";
  EXPECT_EQ(table & 0x7u, static_cast<uint32_t>(rocjitsu::GpuPciDevice::kMsixBar));
  EXPECT_EQ(table & ~0x7u, rocjitsu::GpuPciDevice::kMsixTableOffset);
  EXPECT_EQ(pending & 0x7u, static_cast<uint32_t>(rocjitsu::GpuPciDevice::kMsixBar));
  EXPECT_EQ(pending & ~0x7u, rocjitsu::GpuPciDevice::kMsixPendingOffset);

  serving.request_stop();
  if (serving.joinable()) {
    serving.join();
  }
  host.detach();
  std::filesystem::remove(socket_path);
}

} // namespace

// The handoff the request signal depends on. Nothing else links an outside
// request to the device: the delivery tests call deliver_interrupt() directly,
// so without this the only untested part is the step that actually gets that
// call onto the thread allowed to make it.
// The serving loop needs a process, a socket and a connected client before it
// runs a line, so nothing that exercises it exercises only this mapping -- and
// the mapping is what silently breaks. A signal routed to the wrong arm, or the
// request arm deleted, leaves every other test passing.
TEST(VfioServerSignals, MapsEachHandledSignalToItsAction) {
  EXPECT_EQ(rocjitsu::action_for_signal(SIGINT), rocjitsu::ServerSignalAction::Stop);
  EXPECT_EQ(rocjitsu::action_for_signal(SIGTERM), rocjitsu::ServerSignalAction::Stop);
  EXPECT_EQ(rocjitsu::action_for_signal(SIGUSR1), rocjitsu::ServerSignalAction::DeliverInterrupt)
      << "the interrupt request is the one arm with no other coverage at all";

  // A timed-out wait returns negative, and anything else is not this server's.
  EXPECT_EQ(rocjitsu::action_for_signal(-1), rocjitsu::ServerSignalAction::KeepServing);
  EXPECT_EQ(rocjitsu::action_for_signal(SIGUSR2), rocjitsu::ServerSignalAction::KeepServing)
      << "a signal this server does not handle must not stop it or fire an interrupt";
}

namespace {

// Keep the machine small enough that a four-function server can run in the
// ordinary test suite. The target and bus identity come from the gfx1250
// profile; the one-XCD topology is sufficient for direct PCI transport tests.
std::string compact_vfio_server_config(uint32_t gpu_count) {
  return std::string(R"({"num_threads":1,"cpu_dispatch_threads":)") + std::to_string(gpu_count) +
         R"(,"async_helper_threads":0,
    "exec_mode":"functional",
    "vm":{"arch":"cdna5","target":"gfx1250","gpu":{"num_gpus":)" +
         std::to_string(gpu_count) + R"(,"device":{
      "gpu_id":1250,"gfx_target_version":120500,"vendor_id":4098,
      "device_id":30145,"unique_id":1250,"local_mem_size":67108864,
      "simd_count":4,"num_shader_engines":1,"num_shader_arrays_per_engine":1,
      "num_cu_per_sh":1,"simd_per_cu":4,"max_waves_per_simd":16,
      "wave_front_size":32,"max_slots_scratch_cu":32,"lds_size_kb":320,
      "num_sdma_engines":1,"num_sdma_queues_per_engine":2
    },"pci":{"vram_aperture_bytes":16777216}}},
    "topology":{"root":{"name":"soc","type":"soc","children":[
      {"name":"vram","type":"gpu_memory"},
      {"name":"xcd0","type":"xcd","children":[
        {"name":"l2","type":"l2_cache"},
        {"name":"cp","type":"command_processor"},
        {"name":"se0","type":"shader_engine","children":[
          {"name":"cu0","type":"compute_unit","config":[
            {"key":"num_wf_slots","value":"64"},
            {"key":"sgprs_per_wf","value":"128"},
            {"key":"vgprs_per_wf","value":"1024"},
            {"key":"lds_size_kb","value":"320"}]}]}]}]},
      "links":[
        {"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},
        {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10}
      ]}}
  )";
}

std::string compact_manifest_vfio_server_config() {
  std::string config = compact_vfio_server_config(2);
  const std::string replication = R"("num_gpus":2,)";
  const auto replication_at = config.find(replication);
  if (replication_at == std::string::npos)
    throw std::runtime_error("compact VFIO fixture lost its legacy replication field");
  config.erase(replication_at, replication.size());
  config.insert(1, R"("local_node":"selected","system":{"nodes":[
    {"id":"other","gpus":[
      {"id":"other_gpu","gpu_id":1252,"unique_id":1252,
       "location_id":768,"drm_render_minor":128}]},
    {"id":"selected","gpus":[
      {"id":"first","gpu_id":1251,"unique_id":1251,
       "location_id":1024,"drm_render_minor":129},
      {"id":"second","gpu_id":1250,"unique_id":1250,
       "location_id":768,"drm_render_minor":128}]}
  ]},)");
  return config;
}

bool compact_vfio_config_has_usable_functions(const std::string &path, uint32_t count) {
  util::StringDiagnostic diagnostic;
  auto loaded =
      rocjitsu::config::load_config(path, rocjitsu::kEmbeddedSchema, diagnostic.emitter());
  if (loaded.failed())
    throw std::runtime_error(diagnostic.message());
  if (loaded.value().num_gpus != count || loaded.value().devices.size() != count)
    return false;
  for (const auto &identity : loaded.value().devices) {
    rocjitsu::GpuPciDevice device(
        "probe", rocjitsu::gpu_pci_spec_from_config(identity, loaded.value().pci), nullptr);
    if (!device.usable())
      return false;
  }
  return true;
}

std::optional<int> wait_for_child_bounded(pid_t child, std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t waited = ::waitpid(child, &status, WNOHANG);
    if (waited == child)
      return status;
    if (waited < 0 && errno != EINTR)
      return std::nullopt;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return std::nullopt;
}

int poll_for_pipe_event(int fd, int timeout_ms) {
  pollfd wait{.fd = fd, .events = POLLIN | POLLHUP, .revents = 0};
  int result = 0;
  do {
    result = ::poll(&wait, 1, timeout_ms);
  } while (result < 0 && errno == EINTR);
  return result;
}

bool receive_one_ready_byte(int fd) {
  if (poll_for_pipe_event(fd, 30'000) != 1)
    return false;
  std::array<uint8_t, 2> bytes{};
  if (::read(fd, bytes.data(), bytes.size()) != 1 || bytes[0] != 1)
    return false;
  // Readiness is one event for the node, then EOF.
  return poll_for_pipe_event(fd, 2'000) == 1 && ::read(fd, bytes.data(), bytes.size()) == 0;
}

// Runs in a child so a stalled vfio-user exchange has a process deadline.
// Every function must answer independently while all clients are connected.
enum class MultiClientDeparture {
  StayConnected,
  DisconnectLast,
  DisconnectAll,
  DisconnectWithMissingPeer
};

int exercise_multi_function_clients(const std::vector<std::string> &socket_paths,
                                    MultiClientDeparture departure, int checked_fd,
                                    int release_fd) {
  std::vector<std::unique_ptr<rocjitsu::test::VfioUserClient>> clients;
  const std::size_t clients_to_connect =
      socket_paths.size() - (departure == MultiClientDeparture::DisconnectWithMissingPeer);
  for (std::size_t i = 0; i < clients_to_connect; ++i) {
    auto client = std::make_unique<rocjitsu::test::VfioUserClient>();
    if (!client->connect(socket_paths[i]))
      return static_cast<int>(10 + i);
    uint32_t regions = 0;
    uint32_t irqs = 0;
    if (!client->device_info(regions, irqs) || regions != VFU_PCI_DEV_NUM_REGIONS)
      return static_cast<int>(20 + i);
    std::array<std::byte, 4> pci_id{};
    if (!client->region_read(VFU_PCI_DEV_CFG_REGION_IDX, 0, pci_id))
      return static_cast<int>(30 + i);
    const auto byte_at = [&pci_id](std::size_t index) {
      return std::to_integer<uint8_t>(pci_id[index]);
    };
    if ((byte_at(0) | (byte_at(1) << 8)) != 0x1002 || (byte_at(2) | (byte_at(3) << 8)) != 30145)
      return static_cast<int>(40 + i);
    clients.push_back(std::move(client));
  }

  constexpr uint32_t register_region =
      VFU_PCI_DEV_BAR0_REGION_IDX + rocjitsu::GpuPciDevice::kRegisterBar;
  constexpr uint64_t index_offset = rocjitsu::byte_offset_of(rocjitsu::MmioRegister::MmIndex);
  for (std::size_t i = 0; i < clients.size(); ++i) {
    const uint32_t marker = 0x5a125000u + static_cast<uint32_t>(i);
    const auto bytes = std::bit_cast<std::array<std::byte, 4>>(marker);
    if (!clients[i]->region_write(register_region, index_offset, bytes))
      return static_cast<int>(50 + i);
  }
  for (std::size_t i = 0; i < clients.size(); ++i) {
    std::array<std::byte, 4> read{};
    if (!clients[i]->region_read(register_region, index_offset, read) ||
        std::bit_cast<uint32_t>(read) != 0x5a125000u + i)
      return static_cast<int>(60 + i);
  }

  if (departure == MultiClientDeparture::DisconnectLast)
    clients.back().reset();
  if (departure == MultiClientDeparture::DisconnectWithMissingPeer)
    clients.front().reset();
  if (departure == MultiClientDeparture::DisconnectAll) {
    for (std::unique_ptr<rocjitsu::test::VfioUserClient> &client : clients) {
      client.reset();
      // QEMU closes separately connected functions one at a time.
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  const uint8_t checked = 1;
  if (::write(checked_fd, &checked, sizeof(checked)) != sizeof(checked))
    return 70;
  if (poll_for_pipe_event(release_fd, 60'000) != 1)
    return 71;
  uint8_t release = 0;
  if (::read(release_fd, &release, sizeof(release)) != 0)
    return 72;
  return 0;
}

void run_multi_function_server_scenario(uint32_t gpu_count, MultiClientDeparture departure,
                                        const std::string &config_text = {}) {
  rocjitsu::test::ScopedTempFile config("rj-vfu-multi-config-");
  config.write(config_text.empty() ? compact_vfio_server_config(gpu_count) : config_text);
  ASSERT_TRUE(compact_vfio_config_has_usable_functions(config.path(), gpu_count));

  rocjitsu::test::ScopedTempDirectory runtime("rj-vfu-multi-");
  std::vector<std::string> socket_paths;
  for (uint32_t i = 0; i < gpu_count; ++i)
    socket_paths.push_back(std::format("{}/gpu{}.sock", runtime.path(), i));

  int ready_pipe[2];
  ASSERT_EQ(::pipe(ready_pipe), 0) << strerror(errno);
  ScopedFd ready_read(ready_pipe[0]);
  ScopedFd ready_write(ready_pipe[1]);
  const pid_t server = ::fork();
  ASSERT_GE(server, 0);
  if (server == 0) {
    ready_read.reset();
    ::_exit(rocjitsu::run_vfio_server(config.path(), socket_paths, ready_write.get()));
  }
  ChildProcessGuard server_guard(server);
  ready_write.reset();
  ASSERT_TRUE(receive_one_ready_byte(ready_read.get()))
      << "the multi-function server did not publish one readiness byte after binding every socket";
  ready_read.reset();
  for (const auto &path : socket_paths)
    ASSERT_TRUE(std::filesystem::is_socket(path)) << path;

  int checked_pipe[2];
  ASSERT_EQ(::pipe(checked_pipe), 0) << strerror(errno);
  ScopedFd checked_read(checked_pipe[0]);
  ScopedFd checked_write(checked_pipe[1]);
  int release_pipe[2];
  ASSERT_EQ(::pipe(release_pipe), 0) << strerror(errno);
  ScopedFd release_read(release_pipe[0]);
  ScopedFd release_write(release_pipe[1]);
  const pid_t client = ::fork();
  ASSERT_GE(client, 0);
  if (client == 0) {
    checked_read.reset();
    release_write.reset();
    ::_exit(exercise_multi_function_clients(socket_paths, departure, checked_write.get(),
                                            release_read.get()));
  }
  ChildProcessGuard client_guard(client);
  checked_write.reset();
  release_read.reset();
  ASSERT_EQ(poll_for_pipe_event(checked_read.get(), 15'000), 1)
      << "the client could not exercise every function within fifteen seconds";
  uint8_t checked = 0;
  if (::read(checked_read.get(), &checked, sizeof(checked)) != sizeof(checked) || checked != 1) {
    const auto status = wait_for_child_bounded(client, std::chrono::seconds(5));
    if (status.has_value())
      client_guard.release();
    FAIL() << "a client failed before every function answered, child status "
           << (status.has_value() && WIFEXITED(*status) ? WEXITSTATUS(*status) : -1);
  }
  checked_read.reset();

  if (departure == MultiClientDeparture::DisconnectAll) {
    const auto early_status = wait_for_child_bounded(server, std::chrono::seconds(6));
    if (early_status.has_value()) {
      server_guard.release();
      FAIL() << "the server exited before launcher shutdown after every function disconnected; "
                "child status "
             << (WIFEXITED(*early_status) ? WEXITSTATUS(*early_status) : -1);
    }
    ASSERT_EQ(::kill(server, 0), 0) << "the server did not remain alive for launcher shutdown";
  }
  if (departure != MultiClientDeparture::DisconnectLast &&
      departure != MultiClientDeparture::DisconnectWithMissingPeer)
    ASSERT_EQ(::kill(server, SIGTERM), 0) << strerror(errno);
  const auto server_status = wait_for_child_bounded(server, std::chrono::seconds(15));
  ASSERT_TRUE(server_status.has_value())
      << "the function group did not stop within fifteen seconds";
  server_guard.release();
  ASSERT_TRUE(WIFEXITED(*server_status));
  if (departure == MultiClientDeparture::DisconnectLast ||
      departure == MultiClientDeparture::DisconnectWithMissingPeer) {
    EXPECT_NE(WEXITSTATUS(*server_status), 0)
        << "losing a descriptor-backed function must fail the whole node";
  } else {
    EXPECT_EQ(WEXITSTATUS(*server_status), 0) << "SIGTERM must cleanly stop all four functions";
  }
  for (const auto &path : socket_paths)
    EXPECT_FALSE(std::filesystem::exists(path)) << path << " remained after group teardown";

  release_write.reset();
  const auto client_status = wait_for_child_bounded(client, std::chrono::seconds(5));
  ASSERT_TRUE(client_status.has_value());
  client_guard.release();
  ASSERT_TRUE(WIFEXITED(*client_status));
  EXPECT_EQ(WEXITSTATUS(*client_status), 0);
}

} // namespace

TEST(VfioServer, ServesTwoIndependentFunctionsAndStopsWhenOneDisconnects) {
  run_multi_function_server_scenario(2, MultiClientDeparture::DisconnectLast);
}

TEST(VfioServer, ServesSelectedManifestFunctionsInDeclaredOrder) {
  const std::string manifest = compact_manifest_vfio_server_config();
  rocjitsu::test::ScopedTempFile config("rj-vfu-manifest-config-");
  config.write(manifest);
  util::StringDiagnostic diagnostic;
  auto loaded =
      rocjitsu::config::load_config(config.path(), rocjitsu::kEmbeddedSchema, diagnostic.emitter());
  ASSERT_FALSE(loaded.failed()) << diagnostic.message();
  ASSERT_EQ(loaded.value().devices.size(), 2u);
  EXPECT_EQ(loaded.value().devices[0].gpu_id, 1251u);
  EXPECT_EQ(loaded.value().devices[0].drm_render_minor, 129u);
  EXPECT_EQ(loaded.value().devices[1].gpu_id, 1250u);
  EXPECT_EQ(loaded.value().devices[1].drm_render_minor, 128u);

  run_multi_function_server_scenario(2, MultiClientDeparture::StayConnected, manifest);
}

TEST(VfioServer, FailsWhenPeerNeverAttachesAfterDisconnect) {
  run_multi_function_server_scenario(2, MultiClientDeparture::DisconnectWithMissingPeer);
}

TEST(VfioServer, ServesFourIndependentFunctionsUntilCleanShutdown) {
  run_multi_function_server_scenario(4, MultiClientDeparture::StayConnected);
}

TEST(VfioServer, WaitsForLauncherAfterTwoFunctionsDisconnect) {
  run_multi_function_server_scenario(2, MultiClientDeparture::DisconnectAll);
}

TEST(VfioServer, WaitsForLauncherAfterFourFunctionsDisconnect) {
  run_multi_function_server_scenario(4, MultiClientDeparture::DisconnectAll);
}

TEST(VfioServer, CleansTheFirstListenerWhenTheSecondBindFails) {
  rocjitsu::test::ScopedTempFile config("rj-vfu-multi-config-");
  config.write(compact_vfio_server_config(2));
  ASSERT_TRUE(compact_vfio_config_has_usable_functions(config.path(), 2));

  rocjitsu::test::ScopedTempDirectory runtime("rj-vfu-multi-fail-");
  const std::string first_socket = runtime.path() + "/gpu0.sock";
  // An existing directory cannot become a socket. It reaches host.build()
  // after the first host has already bound, unlike duplicate-path preflight.
  const std::vector<std::string> socket_paths = {first_socket, runtime.path()};
  int ready_pipe[2];
  ASSERT_EQ(::pipe(ready_pipe), 0) << strerror(errno);
  ScopedFd ready_read(ready_pipe[0]);
  ScopedFd ready_write(ready_pipe[1]);
  const pid_t server = ::fork();
  ASSERT_GE(server, 0);
  if (server == 0) {
    ready_read.reset();
    ::_exit(rocjitsu::run_vfio_server(config.path(), socket_paths, ready_write.get()));
  }
  ChildProcessGuard server_guard(server);
  ready_write.reset();
  ASSERT_EQ(poll_for_pipe_event(ready_read.get(), 30'000), 1)
      << "the failed second bind did not close readiness within thirty seconds";
  std::array<uint8_t, 2> ready{};
  EXPECT_EQ(::read(ready_read.get(), ready.data(), ready.size()), 0)
      << "the server reported readiness before every function was bound";
  ready_read.reset();

  const auto status = wait_for_child_bounded(server, std::chrono::seconds(15));
  ASSERT_TRUE(status.has_value()) << "the failed second bind did not stop the server";
  server_guard.release();
  ASSERT_TRUE(WIFEXITED(*status));
  EXPECT_NE(WEXITSTATUS(*status), 0);
  EXPECT_FALSE(std::filesystem::exists(first_socket))
      << "the first listener survived the failure to bind the second";
}

// A server that cannot serve must say so and exit, not hang. The engine runs on
// its own thread and finishes only when asked to, so any early return that
// forgets to ask joins a thread that never ends -- and the process then sits
// there having already reported the failure, which a supervisor reads as a
// healthy server in front of a dead socket.
//
// A directory is the cheapest unbindable path: it exists, so the socket cannot
// be created there, and the failure lands in build() after the engine thread is
// already running. That is the exact window a bare `return 1` left open.
TEST(VfioServer, ReturnsRatherThanHangingWhenTheSocketCannotBeBound) {
  const std::string config = RJ_VFU_TEST_CONFIG_PATH;
  ASSERT_TRUE(std::filesystem::exists(config)) << "no config to build a machine from: " << config;

  // A child can be killed and reaped if the server hangs. A timed-out
  // std::launch::async future would block in its destructor and wedge the test.
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    const int result =
        rocjitsu::run_vfio_server(config, std::filesystem::temp_directory_path().string());
    ::_exit(result == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
  }

  ChildProcessGuard child_guard(child);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t waited = ::waitpid(child, &status, WNOHANG);
    if (waited == child) {
      child_guard.release();
      ASSERT_TRUE(WIFEXITED(status));
      EXPECT_NE(WEXITSTATUS(status), EXIT_SUCCESS)
          << "a server that cannot bind its socket must report failure";
      return;
    }
    ASSERT_TRUE(waited == 0 || (waited == -1 && errno == EINTR));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  FAIL() << "the server did not return after failing to bind; the engine thread was never asked "
            "to exit";
}

TEST(VfioServer, ReportsReadinessAfterTheSocketIsListening) {
  const std::string config = RJ_VFU_TEST_CONFIG_PATH;
  ASSERT_TRUE(std::filesystem::exists(config)) << "no config to build a machine from: " << config;
  const std::string socket_path = std::format("/tmp/rj-vfu-server-ready-{}.sock", ::getpid());
  std::filesystem::remove(socket_path);

  int ready_pipe[2];
  ASSERT_EQ(::pipe(ready_pipe), 0) << strerror(errno);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::close(ready_pipe[0]);
    const int result = rocjitsu::run_vfio_server(config, socket_path, ready_pipe[1]);
    ::_exit(result == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
  }

  ChildProcessGuard child_guard(child);
  ::close(ready_pipe[1]);
  pollfd ready_poll{.fd = ready_pipe[0], .events = POLLIN | POLLHUP, .revents = 0};
  int poll_result = 0;
  do {
    poll_result = ::poll(&ready_poll, 1, 60'000);
  } while (poll_result < 0 && errno == EINTR);
  ASSERT_EQ(poll_result, 1) << "the server did not report readiness";

  uint8_t ready = 0;
  ssize_t ready_bytes = 0;
  do {
    ready_bytes = ::read(ready_pipe[0], &ready, sizeof(ready));
  } while (ready_bytes < 0 && errno == EINTR);
  ::close(ready_pipe[0]);
  ASSERT_EQ(ready_bytes, static_cast<ssize_t>(sizeof(ready)));
  EXPECT_EQ(ready, 1);
  EXPECT_TRUE(std::filesystem::is_socket(std::filesystem::symlink_status(socket_path)));

  ASSERT_EQ(::kill(child, SIGTERM), 0) << strerror(errno);
  int status = 0;
  pid_t waited = 0;
  do {
    waited = ::waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  child_guard.release();
  EXPECT_EQ(waited, child);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), EXIT_SUCCESS);
  std::filesystem::remove(socket_path);
}

TEST(VfioServer, PropagatesAnEngineFailureAfterReadiness) {
  const std::string config = RJ_VFU_TEST_CONFIG_PATH;
  ASSERT_TRUE(std::filesystem::exists(config)) << "no config to build a machine from: " << config;
  const std::string socket_path = std::format("/tmp/rj-vfu-engine-failure-{}.sock", ::getpid());
  std::filesystem::remove(socket_path);

  int ready_pipe[2];
  ASSERT_EQ(::pipe(ready_pipe), 0) << strerror(errno);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::close(ready_pipe[0]);
    const int result =
        rocjitsu::run_vfio_server_with_engine_exit_for_test(config, socket_path, ready_pipe[1], 23);
    ::_exit(result);
  }

  ChildProcessGuard child_guard(child);
  ::close(ready_pipe[1]);
  pollfd ready_poll{.fd = ready_pipe[0], .events = POLLIN | POLLHUP, .revents = 0};
  int poll_result = 0;
  do {
    poll_result = ::poll(&ready_poll, 1, 60'000);
  } while (poll_result < 0 && errno == EINTR);
  ASSERT_EQ(poll_result, 1) << "the server did not report readiness";

  uint8_t ready = 0;
  ssize_t ready_bytes = 0;
  do {
    ready_bytes = ::read(ready_pipe[0], &ready, sizeof(ready));
  } while (ready_bytes < 0 && errno == EINTR);
  ::close(ready_pipe[0]);
  ASSERT_EQ(ready_bytes, static_cast<ssize_t>(sizeof(ready)));
  EXPECT_EQ(ready, 1);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t waited = ::waitpid(child, &status, WNOHANG);
    if (waited == child) {
      child_guard.release();
      ASSERT_TRUE(WIFEXITED(status));
      EXPECT_EQ(WEXITSTATUS(status), 23)
          << "the simulation failure status was not propagated by the server";
      std::filesystem::remove(socket_path);
      return;
    }
    ASSERT_TRUE(waited == 0 || (waited == -1 && errno == EINTR));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  FAIL() << "the server remained alive after its simulation engine failed";
}

// An empty target is a caller's mistake, and the serving thread is the wrong
// place to discover it: invoking one throws bad_function_call there, which is
// reported as work that ran and failed -- indistinguishable from work that
// could not be done -- while the single slot is spent on nothing.
TEST(VfioDeviceHost, RefusesEmptyWorkRatherThanThrowingOnTheServingThread) {
  ServedDevice served;
  ASSERT_TRUE(served.built());

  EXPECT_FALSE(served.host().ask_serving_thread({})) << "an empty target was accepted";

  // And the slot is still free, so a real request is not lost behind it. No
  // state is captured: the acceptance alone is the claim under test, and a
  // captured flag could never be read back reliably, because a request that
  // is still queued when serving stops is discarded by contract.
  EXPECT_TRUE(served.host().ask_serving_thread([] {}))
      << "the refused request consumed the one outstanding slot";
}

TEST(VfioDeviceHost, RunsAskedWorkOnTheServingThread) {
  // Declared before the fixture so the promise outlives the join in its
  // destructor: if the wait below times out, the test returns with the
  // request still pending, and the serving thread may call set_value()
  // after the test body has ended.
  std::promise<std::thread::id> ran_on;
  ServedDevice served;
  ASSERT_TRUE(served.built());

  std::future<std::thread::id> where = ran_on.get_future();
  ASSERT_TRUE(served.host().ask_serving_thread(
      [&ran_on] { ran_on.set_value(std::this_thread::get_id()); }));

  // Bounded by one transport poll, per the documented contract; the generous
  // wait here is for a loaded machine, not for the contract.
  ASSERT_EQ(where.wait_for(std::chrono::seconds(10)), std::future_status::ready)
      << "asked work never ran";
  EXPECT_EQ(where.get(), served.serving_thread_id())
      << "asked work must run on the serving thread, not the caller's";
}

// One outstanding request, refused rather than dropped. The earlier version kept
// a 64-deep queue and discarded silently past that, which gave a caller no way to
// tell that its request had been thrown away.
TEST(VfioDeviceHost, RefusesASecondRequestWhileOneIsOutstanding) {
  // Declared before the fixture so destruction order is the reverse of what
  // it would be otherwise: everything the callbacks touch outlives the join
  // in the fixture destructor, on every return path.
  std::promise<void> running;
  std::future<void> is_running = running.get_future();
  std::promise<void> release;
  std::future<void> may_finish = release.get_future();
  bool second_ran = false;

  ServedDevice served;
  ASSERT_TRUE(served.built());

  // Block the first request inside the serving thread so it stays outstanding
  // while the second is offered.
  ASSERT_TRUE(served.host().ask_serving_thread([&running, &may_finish] {
    running.set_value();
    (void)may_finish.wait_for(std::chrono::seconds(10));
  }));
  ASSERT_EQ(is_running.wait_for(std::chrono::seconds(10)), std::future_status::ready);

  EXPECT_FALSE(served.host().ask_serving_thread([&second_ran] { second_ran = true; }))
      << "a second request must be refused while one is outstanding";
  release.set_value();
  // Join before reading anything the callbacks wrote, so the write above is
  // visible here and the bounded wait in the first callback has ended.
  served.stop_serving();
  EXPECT_FALSE(second_ran) << "a refused request must not run";
}

// A request that throws must not take the process with it: it runs on the serving
// thread, where an escaping exception would terminate rather than propagate.
TEST(VfioDeviceHost, SurvivesAThrowingRequest) {
  // Both promises are declared before the fixture so they outlive the join
  // in its destructor: on a timed-out wait, or a rejected retry loop, the
  // test returns with a request still pending, and the serving thread may
  // call set_value() after the test body has ended.
  std::promise<void> threw;
  std::promise<void> ran_after;
  ServedDevice served;
  ASSERT_TRUE(served.built());

  std::future<void> did_throw = threw.get_future();
  ASSERT_TRUE(served.host().ask_serving_thread([&threw] {
    threw.set_value();
    throw std::runtime_error("request failed");
  }));
  ASSERT_EQ(did_throw.wait_for(std::chrono::seconds(10)), std::future_status::ready);

  // Still serving afterwards, which is the point: the request failed, the device
  // did not. Retried because the throwing request signalled before it threw, so
  // it is legitimately still outstanding -- and therefore still refusing -- for
  // the moment it takes the serving thread to unwind and clear it.
  std::future<void> after = ran_after.get_future();
  bool accepted = false;
  for (int attempt = 0; attempt < 200 && !accepted; ++attempt) {
    accepted = served.host().ask_serving_thread([&ran_after] { ran_after.set_value(); });
    if (!accepted) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  ASSERT_TRUE(accepted) << "the host never accepted a request again after one threw";
  EXPECT_EQ(after.wait_for(std::chrono::seconds(10)), std::future_status::ready)
      << "serving stopped after a request threw";
}

// Documented: a request may never run. Accepting one after serving has stopped
// must not leave a caller waiting for work that has nothing left to run it.
TEST(VfioDeviceHost, DiscardsAskedWorkWhenServingHasStopped) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  served.stop_serving();

  bool ran = false;
  // Accepted or refused is not the contract here; running is. Nothing is left to
  // drain the request, so it must simply never execute.
  (void)served.host().ask_serving_thread([&ran] { ran = true; });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_FALSE(ran) << "work ran with no serving thread to run it";
}
