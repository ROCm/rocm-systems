// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"
#include "rocjitsu/vm/amdgpu/pci/scratch_pci_device.h"
#include "rocjitsu/vmm/vfu/vfio_device_host.h"
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
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <thread>

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

} // namespace
