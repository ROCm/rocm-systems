// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"
#include "rocjitsu/vm/amdgpu/pci/scratch_pci_device.h"
#include "rocjitsu/vmm/vfu/vfio_device_host.h"
#include "rocjitsu/vmm/vfu/vfio_server.h"
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
}

#include <sys/eventfd.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

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
/// the tests exercise the same threading the transport ships with.
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

  /// @brief The DMA engine the device uses, as the transport provides it.
  /// @details Data-path assertions go through this so the tested surface is
  /// the abstract engine, not the concrete host.
  simdojo::DmaEngine &dma() { return host_; }

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

  {
    rocjitsu::test::VfioUserClient first;
    ASSERT_TRUE(served.attach(first));
    auto written = std::bit_cast<std::array<std::byte, 4>>(uint32_t{0xa5a5a5a5});
    ASSERT_TRUE(first.region_write(VFU_PCI_DEV_BAR0_REGION_IDX, 0, written));
  }

  rocjitsu::test::VfioUserClient second;
  ASSERT_TRUE(served.attach(second)) << "a trapped-only device must accept a replacement client";
  std::array<std::byte, 4> read{};
  EXPECT_TRUE(second.region_read(VFU_PCI_DEV_BAR0_REGION_IDX, 0, read));
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
  [[nodiscard]] bool raise() { return irq_ != nullptr && irq_->trigger(0); }
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

// An empty target is a caller's mistake, and the serving thread is the wrong
// place to discover it: invoking one throws bad_function_call there, which is
// reported as work that ran and failed -- indistinguishable from work that
// could not be done -- while the single slot is spent on nothing.
TEST(VfioDeviceHost, RefusesEmptyWorkRatherThanThrowingOnTheServingThread) {
  ServedDevice served;
  ASSERT_TRUE(served.built());

  EXPECT_FALSE(served.host().ask_serving_thread({})) << "an empty target was accepted";

  // And the slot is still free, so a real request is not lost behind it.
  std::atomic<bool> ran = false;
  EXPECT_TRUE(served.host().ask_serving_thread([&ran] { ran = true; }))
      << "the refused request consumed the one outstanding slot";
}

TEST(VfioDeviceHost, RunsAskedWorkOnTheServingThread) {
  ServedDevice served;
  ASSERT_TRUE(served.built());

  std::promise<std::thread::id> ran_on;
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
  ServedDevice served;
  ASSERT_TRUE(served.built());

  // Block the first request inside the serving thread so it stays outstanding
  // while the second is offered.
  std::promise<void> running;
  std::future<void> is_running = running.get_future();
  std::promise<void> release;
  std::future<void> may_finish = release.get_future();
  ASSERT_TRUE(served.host().ask_serving_thread([&running, &may_finish] {
    running.set_value();
    (void)may_finish.wait_for(std::chrono::seconds(10));
  }));
  ASSERT_EQ(is_running.wait_for(std::chrono::seconds(10)), std::future_status::ready);

  bool second_ran = false;
  EXPECT_FALSE(served.host().ask_serving_thread([&second_ran] { second_ran = true; }))
      << "a second request must be refused while one is outstanding";
  release.set_value();
  EXPECT_FALSE(second_ran) << "a refused request must not run";
}

// A request that throws must not take the process with it: it runs on the serving
// thread, where an escaping exception would terminate rather than propagate.
TEST(VfioDeviceHost, SurvivesAThrowingRequest) {
  ServedDevice served;
  ASSERT_TRUE(served.built());

  std::promise<void> threw;
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
  std::promise<void> ran_after;
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

// ---------------------------------------------------------------------------
// Device-facing DMA data-path coverage.
//
// The tests above stop at protocol callbacks: a window is mapped, the device
// is told, the count changes. None of them moves a byte through
// DmaEngine::read()/write(), which is the path a real device leans on for
// command buffers and completion records. These tests share windows through
// the protocol client, drive the engine the way the device does, and verify
// the data against the backing files independently, so a symmetric
// addressing error cannot hide inside a write/read round trip.
// ---------------------------------------------------------------------------

namespace {

constexpr uint64_t kSingleWindowIova = 0x10000000;
constexpr uint64_t kCrossRegistrationIova = 0x20000000;
constexpr uint64_t kStreamingIova = 0x40000000;
constexpr uint64_t kGapIova = 0x60000000;
constexpr uint64_t kProtectionIova = 0x70000000;
constexpr uint64_t kMultiSegmentIova = 0x80000000;
constexpr uint64_t kReconnectIova = 0x90000000;
constexpr std::size_t kBoundaryHalfBytes = 64;
constexpr std::size_t kStreamingRegionCount = 300;

/// @brief A page-aligned, zero-filled anonymous file of @p pages pages.
class BackingFile {
public:
  explicit BackingFile(uint64_t bytes) {
    const long page_size = ::sysconf(_SC_PAGESIZE);
    const uint64_t rounded = (bytes + static_cast<uint64_t>(page_size) - 1) /
                             static_cast<uint64_t>(page_size) * page_size;
    fd_ = ::memfd_create("rj-vfu-dma-test", 0);
    if (fd_ < 0 || ::ftruncate(fd_, static_cast<off_t>(rounded)) != 0) {
      ADD_FAILURE() << "cannot create a " << rounded << "-byte memfd";
    }
  }
  ~BackingFile() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }
  BackingFile(const BackingFile &) = delete;
  BackingFile &operator=(const BackingFile &) = delete;
  [[nodiscard]] int fd() const { return fd_; }
  [[nodiscard]] uint64_t page_size() const {
    return static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
  }

private:
  int fd_ = -1;
};

/// @brief Bytes whose value depends on both @p seed and position.
/// @details A repeating chunk of any size cannot match by accident, so a
/// transfer that lands at the wrong offset or repeats one segment fails the
/// comparison.
std::vector<std::byte> byte_pattern(std::size_t length, uint8_t seed) {
  std::vector<std::byte> bytes(length);
  for (std::size_t i = 0; i < length; ++i) {
    bytes[i] = static_cast<std::byte>(static_cast<uint8_t>(seed + i * 131));
  }
  return bytes;
}

/// @brief Read exactly @p dst.size() bytes at @p offset, retrying short reads.
bool read_all_at(int fd, uint64_t offset, std::span<std::byte> dst) {
  std::size_t done = 0;
  while (done < dst.size()) {
    const ssize_t got =
        ::pread(fd, dst.data() + done, dst.size() - done, static_cast<off_t>(offset + done));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (got == 0) {
      return false;
    }
    done += static_cast<std::size_t>(got);
  }
  return true;
}

/// @brief Write exactly @p src.size() bytes at @p offset, retrying short writes.
bool write_all_at(int fd, uint64_t offset, std::span<const std::byte> src) {
  std::size_t done = 0;
  while (done < src.size()) {
    const ssize_t put =
        ::pwrite(fd, src.data() + done, src.size() - done, static_cast<off_t>(offset + done));
    if (put < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    done += static_cast<std::size_t>(put);
  }
  return true;
}

} // namespace

// One registration, one transfer: the whole range is covered by a single
// scatter-gather entry, so the copy goes through the direct one-entry path.
// The unaligned start and odd length catch an implementation that drops the
// entry's offset or length, and the file is inspected directly so a
// read/write addressing error cannot cancel itself out.
TEST(VfioDeviceHostDma, TransfersWithinOneRegisteredWindow) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  BackingFile backing(0x1000);
  const uint64_t page_size = backing.page_size();
  ASSERT_TRUE(client.dma_map(kSingleWindowIova, page_size, backing.fd(), 0));

  constexpr uint64_t kStart = 127;
  constexpr std::size_t kLength = 513;
  const uint64_t tail = kStart + kLength;
  // Fill the whole file with sentinels first: a transfer that starts or ends
  // at the wrong place shows up as a mismatched sentinel rather than as an
  // invisible change to never-written zeros.
  ASSERT_TRUE(write_all_at(backing.fd(), 0, byte_pattern(kStart, 0x77)));
  ASSERT_TRUE(write_all_at(backing.fd(), tail, byte_pattern(page_size - tail, 0x88)));
  const std::vector<std::byte> sentinel_before = byte_pattern(kStart, 0x77);
  const std::vector<std::byte> sentinel_after = byte_pattern(page_size - tail, 0x88);

  const std::vector<std::byte> source = byte_pattern(kLength, 0x31);

  EXPECT_TRUE(served.dma().write(kSingleWindowIova + kStart, source));

  std::vector<std::byte> in_file(kLength);
  ASSERT_TRUE(read_all_at(backing.fd(), kStart, in_file));
  EXPECT_EQ(in_file, source) << "the write did not land in the backing file";

  std::vector<std::byte> before(kStart);
  ASSERT_TRUE(read_all_at(backing.fd(), 0, before));
  EXPECT_EQ(before, sentinel_before) << "bytes before the write were touched";

  std::vector<std::byte> after(page_size - tail);
  ASSERT_TRUE(read_all_at(backing.fd(), tail, after));
  EXPECT_EQ(after, sentinel_after) << "bytes after the write were touched";

  std::vector<std::byte> read_back(kLength);
  EXPECT_TRUE(served.dma().read(kSingleWindowIova + kStart, read_back));
  EXPECT_EQ(read_back, source);
}

// Two registrations side by side: the transfer crosses their boundary, so the
// library reports two scatter-gather entries and the host must copy each into
// its own registration. Losing an entry, using one entry's file offset for
// the other, or mishandling the split length all fail the per-file checks.
TEST(VfioDeviceHostDma, SplitsATransferAtRegistrationBoundaries) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  BackingFile first_page(0x1000);
  BackingFile second_page(0x1000);
  const uint64_t page_size = first_page.page_size();
  ASSERT_TRUE(client.dma_map(kCrossRegistrationIova, page_size, first_page.fd(), 0));
  ASSERT_TRUE(client.dma_map(kCrossRegistrationIova + page_size, page_size, second_page.fd(), 0));

  const std::vector<std::byte> first_sentinel = byte_pattern(page_size, 0x11);
  const std::vector<std::byte> second_sentinel = byte_pattern(page_size, 0x22);
  ASSERT_TRUE(write_all_at(first_page.fd(), 0, first_sentinel));
  ASSERT_TRUE(write_all_at(second_page.fd(), 0, second_sentinel));

  const std::vector<std::byte> source = byte_pattern(kBoundaryHalfBytes * 2, 0x99);
  EXPECT_TRUE(served.dma().write(kCrossRegistrationIova + page_size - kBoundaryHalfBytes, source));

  std::vector<std::byte> first_tail(kBoundaryHalfBytes);
  ASSERT_TRUE(read_all_at(first_page.fd(), page_size - kBoundaryHalfBytes, first_tail));
  std::vector<std::byte> expected_first(source.begin(), source.begin() + kBoundaryHalfBytes);
  EXPECT_EQ(first_tail, expected_first) << "the first registration's half is wrong";

  std::vector<std::byte> second_head(kBoundaryHalfBytes);
  ASSERT_TRUE(read_all_at(second_page.fd(), 0, second_head));
  std::vector<std::byte> expected_second(source.begin() + kBoundaryHalfBytes, source.end());
  EXPECT_EQ(second_head, expected_second) << "the second registration's half is wrong";

  std::vector<std::byte> first_head(page_size - kBoundaryHalfBytes);
  ASSERT_TRUE(read_all_at(first_page.fd(), 0, first_head));
  EXPECT_EQ(first_head,
            std::vector<std::byte>(first_sentinel.begin(),
                                   first_sentinel.begin() + (page_size - kBoundaryHalfBytes)))
      << "the untouched start of the first registration changed";

  std::vector<std::byte> second_tail(page_size - kBoundaryHalfBytes);
  ASSERT_TRUE(read_all_at(second_page.fd(), kBoundaryHalfBytes, second_tail));
  EXPECT_EQ(second_tail, std::vector<std::byte>(second_sentinel.begin() + kBoundaryHalfBytes,
                                                second_sentinel.end()))
      << "the untouched end of the second registration changed";

  const std::vector<std::byte> read_pattern = byte_pattern(kBoundaryHalfBytes * 2, 0xAA);
  ASSERT_TRUE(write_all_at(first_page.fd(), page_size - kBoundaryHalfBytes,
                           {read_pattern.begin(), read_pattern.begin() + kBoundaryHalfBytes}));
  ASSERT_TRUE(write_all_at(second_page.fd(), 0,
                           {read_pattern.begin() + kBoundaryHalfBytes, read_pattern.end()}));
  std::vector<std::byte> read_back(kBoundaryHalfBytes * 2);
  EXPECT_TRUE(
      served.dma().read(kCrossRegistrationIova + page_size - kBoundaryHalfBytes, read_back));
  EXPECT_EQ(read_back, read_pattern);
}

// A client-side IOMMU can reflect one large range as hundreds of page-sized
// windows. Beyond the host's scatter-gather ceiling the transfer must stream
// registration by registration instead of failing, so this test registers 300
// adjacent windows -- above the 256-entry limit -- and moves the whole range
// in both directions, comparing every byte of every page.
TEST(VfioDeviceHostDma, StreamsAcrossMoreThanTheScatterGatherLimit) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  const uint64_t page_size = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
  const uint64_t total = kStreamingRegionCount * page_size;
  BackingFile backing(total);
  for (uint64_t i = 0; i < kStreamingRegionCount; ++i) {
    ASSERT_TRUE(
        client.dma_map(kStreamingIova + i * page_size, page_size, backing.fd(), i * page_size));
  }
  ASSERT_EQ(served.device().mapped_regions(), kStreamingRegionCount);

  const std::vector<std::byte> source = byte_pattern(total, 0x5a);
  EXPECT_TRUE(served.dma().write(kStreamingIova, source));

  std::vector<std::byte> in_file(total);
  ASSERT_TRUE(read_all_at(backing.fd(), 0, in_file));
  EXPECT_EQ(in_file, source) << "the streaming write lost or misplaced data";

  const std::vector<std::byte> second = byte_pattern(total, 0xc3);
  ASSERT_TRUE(write_all_at(backing.fd(), 0, second));
  std::vector<std::byte> read_back(total);
  EXPECT_TRUE(served.dma().read(kStreamingIova, read_back));
  EXPECT_EQ(read_back, second) << "the streaming read lost or misplaced data";
}

// A hole between registrations is not memory the device may touch, whichever
// side of the boundary a transfer starts from. The contract only promises
// failure for the whole transfer, not that buffers stay untouched, so the
// checks are on the return value alone.
TEST(VfioDeviceHostDma, RejectsTransfersThroughAnUnmappedGap) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  BackingFile backing(3 * 0x1000);
  const uint64_t page_size = backing.page_size();
  ASSERT_TRUE(client.dma_map(kGapIova, page_size, backing.fd(), 0));
  ASSERT_TRUE(client.dma_map(kGapIova + 2 * page_size, page_size, backing.fd(), 2 * page_size));

  std::vector<std::byte> buffer(kBoundaryHalfBytes, std::byte{0});
  EXPECT_FALSE(served.dma().read(kGapIova + page_size, buffer))
      << "a read wholly inside the gap must fail";
  EXPECT_FALSE(served.dma().write(kGapIova + page_size, buffer))
      << "a write wholly inside the gap must fail";

  std::vector<std::byte> crossing(kBoundaryHalfBytes * 2, std::byte{0});
  EXPECT_FALSE(served.dma().read(kGapIova + page_size - kBoundaryHalfBytes, crossing))
      << "a read crossing into the gap must fail";
  EXPECT_FALSE(served.dma().write(kGapIova + page_size - kBoundaryHalfBytes, crossing))
      << "a write crossing into the gap must fail";
}

// The transport must enforce the direction a window was shared with. The
// pinned library checks the write direction only: a write into a read-only
// window fails, while a read from a write-only window is not rejected. The
// test asserts the enforced direction and verifies against the backing file
// that a rejected write changed nothing and a permitted one landed.
TEST(VfioDeviceHostDma, EnforcesWriteProtection) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  BackingFile read_only(0x1000);
  BackingFile read_write(0x1000);
  const uint64_t page_size = read_only.page_size();
  ASSERT_TRUE(client.dma_map(kProtectionIova, page_size, read_only.fd(), 0,
                             rocjitsu::test::DmaProtection::ReadOnly));
  ASSERT_TRUE(client.dma_map(kProtectionIova + page_size, page_size, read_write.fd(), 0,
                             rocjitsu::test::DmaProtection::ReadWrite));

  const std::vector<std::byte> initial = byte_pattern(page_size, 0x44);
  ASSERT_TRUE(write_all_at(read_only.fd(), 0, initial));

  std::vector<std::byte> read_back(page_size);
  EXPECT_TRUE(served.dma().read(kProtectionIova, read_back));
  EXPECT_EQ(read_back, initial) << "reading a read-only window returned wrong bytes";

  const std::vector<std::byte> rejected = byte_pattern(page_size, 0x66);
  EXPECT_FALSE(served.dma().write(kProtectionIova, rejected))
      << "a write to a read-only window must fail";

  std::vector<std::byte> unchanged(page_size);
  ASSERT_TRUE(read_all_at(read_only.fd(), 0, unchanged));
  EXPECT_EQ(unchanged, initial) << "the rejected write changed the backing file";

  const std::vector<std::byte> accepted = byte_pattern(page_size, 0x77);
  EXPECT_TRUE(served.dma().write(kProtectionIova + page_size, accepted));

  std::vector<std::byte> landed(page_size);
  ASSERT_TRUE(read_all_at(read_write.fd(), 0, landed));
  EXPECT_EQ(landed, accepted) << "the permitted write did not land";

  std::vector<std::byte> rw_read_back(page_size);
  EXPECT_TRUE(served.dma().read(kProtectionIova + page_size, rw_read_back));
  EXPECT_EQ(rw_read_back, accepted);
}

// The multi-entry path again, with the window set released afterwards. The
// unmap checks are lifecycle coverage -- the device reports no mapped
// windows once the client withdraws them -- not evidence about any internal
// library call. The library exposes no way to observe that from outside, so
// the data checks are what carry the regression value here.
TEST(VfioDeviceHostDma, CompletesMultiSegmentTransfersAndReleasesTheWindowSet) {
  ServedDevice served;
  ASSERT_TRUE(served.built());
  rocjitsu::test::VfioUserClient client;
  ASSERT_TRUE(served.attach(client));

  BackingFile first_page(0x1000);
  BackingFile second_page(0x1000);
  const uint64_t page_size = first_page.page_size();
  ASSERT_TRUE(client.dma_map(kMultiSegmentIova, page_size, first_page.fd(), 0));
  ASSERT_TRUE(client.dma_map(kMultiSegmentIova + page_size, page_size, second_page.fd(), 0));

  const std::vector<std::byte> source = byte_pattern(kBoundaryHalfBytes * 2, 0xb1);
  EXPECT_TRUE(served.dma().write(kMultiSegmentIova + page_size - kBoundaryHalfBytes, source));

  std::vector<std::byte> first_tail(kBoundaryHalfBytes);
  ASSERT_TRUE(read_all_at(first_page.fd(), page_size - kBoundaryHalfBytes, first_tail));
  EXPECT_EQ(first_tail,
            std::vector<std::byte>(source.begin(), source.begin() + kBoundaryHalfBytes));
  std::vector<std::byte> second_head(kBoundaryHalfBytes);
  ASSERT_TRUE(read_all_at(second_page.fd(), 0, second_head));
  EXPECT_EQ(second_head, std::vector<std::byte>(source.begin() + kBoundaryHalfBytes, source.end()));

  const std::vector<std::byte> read_pattern = byte_pattern(kBoundaryHalfBytes * 2, 0x4d);
  ASSERT_TRUE(write_all_at(first_page.fd(), page_size - kBoundaryHalfBytes,
                           {read_pattern.begin(), read_pattern.begin() + kBoundaryHalfBytes}));
  ASSERT_TRUE(write_all_at(second_page.fd(), 0,
                           {read_pattern.begin() + kBoundaryHalfBytes, read_pattern.end()}));
  std::vector<std::byte> read_back(kBoundaryHalfBytes * 2);
  EXPECT_TRUE(served.dma().read(kMultiSegmentIova + page_size - kBoundaryHalfBytes, read_back));
  EXPECT_EQ(read_back, read_pattern);

  EXPECT_TRUE(client.dma_unmap(kMultiSegmentIova, page_size));
  EXPECT_TRUE(client.dma_unmap(kMultiSegmentIova + page_size, page_size));
  EXPECT_EQ(served.device().mapped_regions(), 0u)
      << "withdrawn windows must leave the device with none";
}

// A client that goes away without unmapping takes its windows with it: the
// transport clears its records when the connection ends, and a replacement
// client starts from a clean slate at the same addresses. Both clients'
// transfers are verified against their backing files, not just round-tripped.
TEST(VfioDeviceHostLifecycle, ForgetsGuestWindowsBeforeServingAnotherClient) {
  ServedDevice served;
  ASSERT_TRUE(served.built());

  {
    rocjitsu::test::VfioUserClient first;
    ASSERT_TRUE(served.attach(first));
    BackingFile first_backing(0x1000);
    const uint64_t page_size = first_backing.page_size();
    ASSERT_TRUE(first.dma_map(kReconnectIova, page_size, first_backing.fd(), 0));

    const std::vector<std::byte> first_source = byte_pattern(0x100, 0x3c);
    ASSERT_TRUE(served.dma().write(kReconnectIova, first_source));

    std::vector<std::byte> first_in_file(0x100);
    ASSERT_TRUE(read_all_at(first_backing.fd(), 0, first_in_file));
    EXPECT_EQ(first_in_file, first_source)
        << "the first client's write did not land before disconnect";

    std::vector<std::byte> first_read_back(0x100);
    ASSERT_TRUE(served.dma().read(kReconnectIova, first_read_back));
    EXPECT_EQ(first_read_back, first_source);
  }

  rocjitsu::test::VfioUserClient second;
  ASSERT_TRUE(served.attach(second)) << "the server must accept a replacement client";
  EXPECT_EQ(served.device().mapped_regions(), 0u)
      << "the disconnected client's windows must be gone";

  std::vector<std::byte> stale(4, std::byte{0});
  EXPECT_FALSE(served.dma().read(kReconnectIova, stale))
      << "the old window must not be readable by the new client";

  BackingFile second_backing(0x1000);
  const uint64_t page_size = second_backing.page_size();
  ASSERT_TRUE(second.dma_map(kReconnectIova, page_size, second_backing.fd(), 0));

  const std::vector<std::byte> second_source = byte_pattern(0x100, 0x7e);
  ASSERT_TRUE(served.dma().write(kReconnectIova, second_source));

  std::vector<std::byte> second_in_file(0x100);
  ASSERT_TRUE(read_all_at(second_backing.fd(), 0, second_in_file));
  EXPECT_EQ(second_in_file, second_source)
      << "the second client's write did not land at the reused address";

  std::vector<std::byte> second_read_back(0x100);
  ASSERT_TRUE(served.dma().read(kReconnectIova, second_read_back));
  EXPECT_EQ(second_read_back, second_source);
}
