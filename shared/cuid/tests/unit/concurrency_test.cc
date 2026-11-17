// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Every other test here is single-threaded, so a sanitizer run over the suite
// cannot see a data race. This one drives the public API from several threads;
// its real value is under -fsanitize=thread (with setarch -R to disable ASLR).

#include "unit/concurrency_test.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#include "include/amd_cuid.h"

namespace {

constexpr int kThreads = 8;
constexpr int kIterations = 25;

// Every status the workers are allowed to see. Anything else means the API
// returned something undefined under concurrency.
bool acceptable(amdcuid_status_t status) {
  switch (status) {
    case AMDCUID_STATUS_SUCCESS:
    case AMDCUID_STATUS_DEVICE_NOT_FOUND:
    case AMDCUID_STATUS_UNSUPPORTED:
    case AMDCUID_STATUS_INSUFFICIENT_SIZE:
    case AMDCUID_STATUS_PERMISSION_DENIED:
    case AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND:
    case AMDCUID_STATUS_FILE_NOT_FOUND:
      return true;
    default:
      return false;
  }
}

}  // namespace

TestConcurrentApi::TestConcurrentApi() {
  SetTitle("Concurrent Public API Use");
  SetDescription(
      "Drive amdcuid_get_all_handles, amdcuid_query_device_property and "
      "amdcuid_refresh from several threads at once. Guards the "
      "CuidDeviceManager locking; run under ThreadSanitizer to detect races.");
}

void TestConcurrentApi::SetUp() {}

void TestConcurrentApi::Run() {
  std::atomic<int> bad_status{0};
  std::atomic<int> observed{0};

  auto worker = [&](int id) {
    for (int i = 0; i < kIterations; ++i) {
      uint32_t count = 0;
      amdcuid_status_t status = amdcuid_get_all_handles(nullptr, &count);
      if (!acceptable(status)) {
        ++bad_status;
        continue;
      }

      if (status == AMDCUID_STATUS_SUCCESS && count > 0) {
        std::vector<amdcuid_id_t> handles(count);
        uint32_t fetched = count;
        if (amdcuid_get_all_handles(handles.data(), &fetched) == AMDCUID_STATUS_SUCCESS) {
          ++observed;
          const uint32_t examine = (fetched < 4) ? fetched : 4;
          for (uint32_t h = 0; h < examine; ++h) {
            amdcuid_id_t derived{};
            uint32_t len = sizeof(derived);
            if (!acceptable(amdcuid_query_device_property(handles[h], AMDCUID_QUERY_DERIVED_CUID,
                                                          &derived, &len))) {
              ++bad_status;
            }
            // Size-query form: data == nullptr must still return a defined
            // status rather than an uninitialized one.
            uint32_t path_len = 0;
            if (!acceptable(amdcuid_query_device_property(handles[h], AMDCUID_QUERY_DEVICE_PATH,
                                                          nullptr, &path_len))) {
              ++bad_status;
            }
          }
        }
      }

      // Force a rediscovery underneath the readers. This is what makes the
      // device list get replaced while other threads are walking it.
      if ((i % 5) == (id % 5)) {
        if (!acceptable(amdcuid_refresh())) {
          ++bad_status;
        }
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back(worker, i);
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(bad_status.load(), 0) << "public API returned an unexpected status under concurrency";

  IF_VERB(1) {
    printf("  %d threads x %d iterations, %d successful enumerations\n", kThreads, kIterations,
           observed.load());
  }
}

void TestConcurrentApi::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestConcurrentApi::DisplayResults() const { TestBase::DisplayResults(); }
void TestConcurrentApi::Close() {}
