// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the public amdsmi_get_gpu_cpu_links() API (amd_smi.cc).
//
// Unlike amdsmi_set_npm_limit()/amdsmi_get_npm_info(), this function takes an
// amdsmi_processor_handle, not an amdsmi_node_handle -- there is no
// amdsmi_test_register_node_handle()-style hook to fabricate one, since a
// processor_handle can only be vended by real device enumeration inside
// AMDSmiSystem. The null-argument checks below need no device at all, so
// they run unconditionally; the buffer/round-trip checks need a real
// processor_handle and skip gracefully when no GPU is present, the same
// convention GetNpmInfoCurrentNodePowerRoundTrip/GetPowerInfoNullInfoIsInval
// use in amdsmi_npm_limit_test.cc. None of this fabricates node/board state,
// so SKIP_IF_WSL_BACKEND_ACTIVE does not apply here. amdsmi_get_gpu_cpu_links()
// does short-circuit to AMDSMI_STATUS_NOT_SUPPORTED for a WSL-backed device
// (ENABLE_WSL_BACKEND builds only); the handle-based tests below treat that
// as a skip rather than asserting unconditional SUCCESS.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "amd_smi/amdsmi.h"

namespace {

// Small RAII wrapper around amdsmi_init()/amdsmi_shut_down() so each test is
// self-contained regardless of test execution order.
class ScopedAmdSmiInit {
 public:
  ScopedAmdSmiInit() { status_ = amdsmi_init(AMDSMI_INIT_AMD_GPUS); }
  ~ScopedAmdSmiInit() {
    if (status_ == AMDSMI_STATUS_SUCCESS) {
      amdsmi_shut_down();
    }
  }
  amdsmi_status_t status() const { return status_; }

 private:
  amdsmi_status_t status_;
};

// Returns the first real GPU processor handle found on this system, or
// nullptr if none is discoverable.
amdsmi_processor_handle FindFirstGpuProcessor() {
  amdsmi_socket_handle sockets[16];
  uint32_t socket_count = 16;
  if (amdsmi_get_socket_handles(&socket_count, sockets) != AMDSMI_STATUS_SUCCESS) {
    return nullptr;
  }
  for (uint32_t s = 0; s < socket_count; ++s) {
    amdsmi_processor_handle procs[16];
    uint32_t proc_count = 16;
    if (amdsmi_get_processor_handles(sockets[s], &proc_count, procs) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    if (proc_count > 0) {
      return procs[0];
    }
  }
  return nullptr;
}

}  // namespace

// Driverless runners (e.g. the ASAN CI container) cannot init; any other init
// failure is still a test failure.
#define REQUIRE_AMDSMI_INIT(init)                             \
  do {                                                        \
    if ((init).status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) { \
      GTEST_SKIP() << "amdgpu driver not loaded";             \
    }                                                         \
    ASSERT_EQ((init).status(), AMDSMI_STATUS_SUCCESS);        \
  } while (0)

TEST(GpuUnit, GetGpuCpuLinksNullProcessorHandleIsInval) {
  ScopedAmdSmiInit init;
  REQUIRE_AMDSMI_INIT(init);

  uint32_t num_cpu_links = 0;
  EXPECT_EQ(amdsmi_get_gpu_cpu_links(nullptr, &num_cpu_links, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetGpuCpuLinksNullNumCpuLinksIsInval) {
  ScopedAmdSmiInit init;
  REQUIRE_AMDSMI_INIT(init);

  // amdsmi_get_gpu_cpu_links() checks num_cpu_links before ever touching
  // processor_handle, so this is exercised the same way regardless of
  // whether a real handle is available on this system.
  EXPECT_EQ(amdsmi_get_gpu_cpu_links(nullptr, nullptr, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, GetGpuCpuLinksNullBufferQueriesCountOnly) {
  ScopedAmdSmiInit init;
  REQUIRE_AMDSMI_INIT(init);

  amdsmi_processor_handle gpu = FindFirstGpuProcessor();
  if (gpu == nullptr) {
    GTEST_SKIP_("No GPU processor discovered on this system");
  }

  // A NULL cpu_links buffer must be accepted as a count-only query,
  // regardless of the capacity value passed in num_cpu_links.
  uint32_t num_cpu_links = 7;
  amdsmi_status_t status = amdsmi_get_gpu_cpu_links(gpu, &num_cpu_links, nullptr);
  if (status == AMDSMI_STATUS_NOT_SUPPORTED) {
    GTEST_SKIP_("GPU link discovery not supported on this device (e.g. WSL-backed device)");
  }
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  // num_cpu_links now holds the real count; no crash, no uninitialized read.
}

TEST(GpuUnit, GetGpuCpuLinksZeroCapacityBufferWritesNothing) {
  ScopedAmdSmiInit init;
  REQUIRE_AMDSMI_INIT(init);

  amdsmi_processor_handle gpu = FindFirstGpuProcessor();
  if (gpu == nullptr) {
    GTEST_SKIP_("No GPU processor discovered on this system");
  }

  uint32_t total = 0;
  amdsmi_status_t status = amdsmi_get_gpu_cpu_links(gpu, &total, nullptr);
  if (status == AMDSMI_STATUS_NOT_SUPPORTED) {
    GTEST_SKIP_("GPU link discovery not supported on this device (e.g. WSL-backed device)");
  }
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  amdsmi_cpu_link_t sentinel{};
  sentinel.package_id = 0xDEADBEEF;
  amdsmi_cpu_link_t probe = sentinel;
  uint32_t zero_capacity = 0;
  EXPECT_EQ(amdsmi_get_gpu_cpu_links(gpu, &zero_capacity, &probe), AMDSMI_STATUS_SUCCESS);

  // Nothing written despite a non-NULL buffer, because capacity was 0.
  EXPECT_EQ(probe.package_id, sentinel.package_id);
  // The real total is still reported back to the caller.
  EXPECT_EQ(zero_capacity, total);
}

TEST(GpuUnit, GetGpuCpuLinksTwoCallIdiomFillsExactCount) {
  ScopedAmdSmiInit init;
  REQUIRE_AMDSMI_INIT(init);

  amdsmi_processor_handle gpu = FindFirstGpuProcessor();
  if (gpu == nullptr) {
    GTEST_SKIP_("No GPU processor discovered on this system");
  }

  uint32_t num_cpu_links = 0;
  amdsmi_status_t status = amdsmi_get_gpu_cpu_links(gpu, &num_cpu_links, nullptr);
  if (status == AMDSMI_STATUS_NOT_SUPPORTED) {
    GTEST_SKIP_("GPU link discovery not supported on this device (e.g. WSL-backed device)");
  }
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  if (num_cpu_links == 0) {
    GTEST_SKIP_("This GPU has no CPU link to report");
  }

  std::vector<amdsmi_cpu_link_t> cpu_links(num_cpu_links);
  uint32_t capacity = num_cpu_links;
  ASSERT_EQ(amdsmi_get_gpu_cpu_links(gpu, &capacity, cpu_links.data()), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(capacity, num_cpu_links);
  for (const auto& link : cpu_links) {
    EXPECT_GE(link.link_type, AMDSMI_LINK_TYPE_INTERNAL);
    EXPECT_LE(link.link_type, AMDSMI_LINK_TYPE_XNUMA);
  }
}
