// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "functional/device_handles_test.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <set>
#include <string>

#include "src/cuid_util.h"

namespace {

// A GPU with no serial has only an auxiliary identity, which needs the host's
// machine identity. Without one it has no identity at all, even for root.
bool HasNoIdentity(const ColdLookup& gpu) {
  uint64_t probe = 0;
  return gpu.bdf_status == AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND &&
         CuidUtilities::make_fallback_fingerprint(CuidUtilities::AuxiliaryInput{}, probe) !=
             AMDCUID_STATUS_SUCCESS;
}

// Compare by-name lookups against enumeration.
std::string CuidForBdf(const std::vector<amdcuid_id_t>& handles, const std::string& bdf) {
  for (const amdcuid_id_t& handle : handles) {
    char buf[64] = {0};
    uint32_t length = sizeof(buf);
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_BDF, buf, &length) !=
        AMDCUID_STATUS_SUCCESS) {
      continue;
    }
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    length = sizeof(type);
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE, &type, &length) !=
            AMDCUID_STATUS_SUCCESS ||
        type != AMDCUID_DEVICE_TYPE_GPU) {
      continue;
    }
    if (bdf == buf) {
      const char* text = amdcuid_id_to_string(handle);
      return text ? text : "";
    }
  }
  return "";
}

// Warm lookups use the names published by the library.
struct Enumerated {
  std::string bdf;
  std::string dev_path;
  std::string cuid;
};

std::vector<Enumerated> EnumeratedGpus(const std::vector<amdcuid_id_t>& handles) {
  std::vector<Enumerated> gpus;
  for (const amdcuid_id_t& handle : handles) {
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    uint32_t length = sizeof(type);
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE, &type, &length) !=
            AMDCUID_STATUS_SUCCESS ||
        type != AMDCUID_DEVICE_TYPE_GPU) {
      continue;
    }
    Enumerated gpu;
    char buf[256] = {0};
    length = sizeof(buf);
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_BDF, buf, &length) ==
        AMDCUID_STATUS_SUCCESS) {
      gpu.bdf = buf;
    }
    buf[0] = '\0';
    length = sizeof(buf);
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_PATH, buf, &length) ==
        AMDCUID_STATUS_SUCCESS) {
      gpu.dev_path = buf;
    }
    const char* text = amdcuid_id_to_string(handle);
    gpu.cuid = text ? text : "";
    gpus.push_back(gpu);
  }
  return gpus;
}

}  // namespace

// ---------------------------------------------------------------------------
// TestGetAllHandles
// ---------------------------------------------------------------------------

TestGetAllHandles::TestGetAllHandles() {
  SetTitle("Get All Handles");
  SetDescription(
      "Verify every enumerated handle converts to a distinct, non-empty "
      "identifier, and that every GPU sysfs presents is enumerated.");
}

void TestGetAllHandles::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping handle validation.";
  }

  std::set<std::string> seen;
  for (const auto& handle : device_handles_) {
    const char* id_str = amdcuid_id_to_string(handle);
    ASSERT_NE(id_str, nullptr);
    EXPECT_GT(strlen(id_str), 0u);
    EXPECT_TRUE(seen.insert(id_str).second) << "two components share the identifier " << id_str;
    IF_VERB(1) { printf("  Handle: %s\n", id_str); }
  }

  // Every enumerated GPU must exist, and every GPU sysfs presents must be
  // enumerated, for any caller: a GPU's identity is the driver's world-readable
  // cuid_derived or a temporary CUID keyed by the machine-id.
  std::set<std::string> present;
  for (const ColdLookup& gpu : ColdLookupEnvironment::results()) present.insert(gpu.bdf);

  const std::vector<Enumerated> gpus = EnumeratedGpus(device_handles_);
  for (const Enumerated& gpu : gpus) {
    if (gpu.bdf.empty()) continue;
    EXPECT_EQ(present.count(gpu.bdf), 1u)
        << "enumeration reports a GPU at " << gpu.bdf << " that sysfs does not present";
  }

  for (const ColdLookup& gpu : ColdLookupEnvironment::results()) {
    if (HasNoIdentity(gpu)) continue;
    EXPECT_FALSE(CuidForBdf(device_handles_, gpu.bdf).empty())
        << "sysfs presents " << gpu.card_path << " at " << gpu.bdf
        << " but enumeration returned no GPU handle for it";
  }
}

// ---------------------------------------------------------------------------
// TestGetHandleByBDF
// ---------------------------------------------------------------------------

TestGetHandleByBDF::TestGetHandleByBDF() {
  SetTitle("Get Handle By BDF");
  SetDescription(
      "Verify amdcuid_get_handle_by_bdf resolves every GPU the machine has, "
      "to the same identifier enumeration reports, and rejects a BDF that is "
      "not present.");
}

void TestGetHandleByBDF::Run() {
  const std::vector<Enumerated> gpus = EnumeratedGpus(device_handles_);
  if (gpus.empty()) {
    GTEST_SKIP() << "No GPU enumerated on this machine; skipping.";
  }

  for (const Enumerated& gpu : gpus) {
    if (gpu.bdf.empty()) continue;
    amdcuid_id_t handle = {};
    const amdcuid_status_t status =
        amdcuid_get_handle_by_bdf(gpu.bdf.c_str(), AMDCUID_DEVICE_TYPE_GPU, &handle);
    ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS)
        << gpu.bdf << " was enumerated as a GPU but cannot be looked up by its own BDF";

    const char* id_str = amdcuid_id_to_string(handle);
    ASSERT_NE(id_str, nullptr);
    EXPECT_EQ(gpu.cuid, id_str)
        << gpu.bdf << ": the by-BDF lookup and enumeration disagree about the identifier";
    IF_VERB(1) { printf("  Handle for %s: %s\n", gpu.bdf.c_str(), id_str); }
  }

  // This valid BDF is expected to be absent on the test host.
  amdcuid_id_t absent = {};
  EXPECT_EQ(amdcuid_get_handle_by_bdf("ffff:ff:1f.7", AMDCUID_DEVICE_TYPE_GPU, &absent),
            AMDCUID_STATUS_DEVICE_NOT_FOUND);
}

// ---------------------------------------------------------------------------
// TestGetHandleByDevPath
// ---------------------------------------------------------------------------

TestGetHandleByDevPath::TestGetHandleByDevPath() {
  SetTitle("Get Handle By Device Path");
  SetDescription(
      "Verify amdcuid_get_handle_by_dev_path resolves both the sysfs card "
      "path and the /dev render node of every GPU, to the same identifier.");
}

void TestGetHandleByDevPath::Run() {
  const std::vector<Enumerated> gpus = EnumeratedGpus(device_handles_);
  if (gpus.empty()) {
    GTEST_SKIP() << "No GPU enumerated on this machine; skipping.";
  }

  for (const Enumerated& gpu : gpus) {
    if (gpu.dev_path.empty()) continue;

    // Round-trip the library's own path spelling.
    amdcuid_id_t handle = {};
    ASSERT_EQ(
        amdcuid_get_handle_by_dev_path(gpu.dev_path.c_str(), AMDCUID_DEVICE_TYPE_GPU, &handle),
        AMDCUID_STATUS_SUCCESS)
        << gpu.dev_path << " is the path the library itself published for " << gpu.bdf;

    const char* id_str = amdcuid_id_to_string(handle);
    ASSERT_NE(id_str, nullptr);
    EXPECT_EQ(gpu.cuid, id_str) << gpu.dev_path;
    IF_VERB(1) { printf("  Handle for %s: %s\n", gpu.dev_path.c_str(), id_str); }
  }

  amdcuid_id_t absent = {};
  EXPECT_NE(amdcuid_get_handle_by_dev_path("/sys/class/drm/card4294967295", AMDCUID_DEVICE_TYPE_GPU,
                                           &absent),
            AMDCUID_STATUS_SUCCESS);
}

// ---------------------------------------------------------------------------
// TestGetHandleByFD
// ---------------------------------------------------------------------------

TestGetHandleByFD::TestGetHandleByFD() {
  SetTitle("Get Handle By File Descriptor");
  SetDescription(
      "Verify amdcuid_get_handle_by_fd resolves an open render node to the "
      "same identifier its path and its BDF resolve to.");
}

void TestGetHandleByFD::Run() {
  const std::vector<Enumerated> gpus = EnumeratedGpus(device_handles_);
  if (gpus.empty()) {
    GTEST_SKIP() << "No GPU enumerated on this machine; skipping.";
  }

  size_t checked = 0;
  for (const Enumerated& gpu : gpus) {
    std::string node_path;
    for (const ColdLookup& sysfs_gpu : ColdLookupEnvironment::results()) {
      if (sysfs_gpu.bdf == gpu.bdf) {
        node_path = sysfs_gpu.node_path;
        break;
      }
    }
    if (node_path.empty()) continue;

    const int fd = open(node_path.c_str(), O_RDONLY);
    if (fd < 0) {
      // Not a skip for the whole test: another GPU's node may still be
      // openable, and /dev/dri permissions vary by distro.
      IF_VERB(1) { printf("  Cannot open %s; skipping it\n", node_path.c_str()); }
      continue;
    }

    amdcuid_id_t handle = {};
    const amdcuid_status_t status = amdcuid_get_handle_by_fd(fd, AMDCUID_DEVICE_TYPE_GPU, &handle);
    close(fd);
    ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS) << node_path;

    const char* id_str = amdcuid_id_to_string(handle);
    ASSERT_NE(id_str, nullptr);
    EXPECT_EQ(gpu.cuid, id_str)
        << gpu.bdf << ": the by-fd lookup and enumeration disagree about the identifier";
    IF_VERB(1) { printf("  Handle for fd on %s: %s\n", node_path.c_str(), id_str); }
    ++checked;
  }

  if (checked == 0) {
    GTEST_SKIP() << "No render node could be opened; skipping.";
  }
}

// ---------------------------------------------------------------------------
// TestColdHandleLookup
// ---------------------------------------------------------------------------

TestColdHandleLookup::TestColdHandleLookup() {
  SetTitle("Cold Handle Lookup");
  SetDescription(
      "Verify the by-name lookups resolve a GPU on a library that has not "
      "enumerated anything, which is how amd-smi calls them.");
}

void TestColdHandleLookup::Run() {
  const auto& gpus = ColdLookupEnvironment::results();
  if (gpus.empty()) {
    GTEST_SKIP() << "No DRM card node on this machine; skipping.";
  }

  for (const ColdLookup& gpu : gpus) {
    if (HasNoIdentity(gpu)) {
      EXPECT_EQ(gpu.card_status, AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND) << gpu.card_path;
      continue;
    }
    // Recorded before the device manager was populated; other tests here are
    // answered from its cache without reaching discovery.
    EXPECT_EQ(gpu.bdf_status, AMDCUID_STATUS_SUCCESS)
        << "cold amdcuid_get_handle_by_bdf(" << gpu.bdf
        << ") failed; this is the call amd-smi makes for every GPU, and it makes no other "
           "call first";
    EXPECT_EQ(gpu.card_status, AMDCUID_STATUS_SUCCESS)
        << "cold amdcuid_get_handle_by_dev_path(" << gpu.card_path << ") failed";

    // amd-smi looks a device up by BDF and then queries it. A cold path that
    // indexes the device under a different value than the handle it returns
    // only shows when the library must derive (no driver attributes);
    // otherwise both sides read the same driver-published value.
    if (gpu.bdf_status == AMDCUID_STATUS_SUCCESS) {
      EXPECT_EQ(gpu.bdf_query_status, AMDCUID_STATUS_SUCCESS)
          << gpu.bdf << ": the cold by-BDF lookup returned a handle that no property query can use";
      if (gpu.bdf_query_status == AMDCUID_STATUS_SUCCESS) {
        EXPECT_EQ(gpu.bdf_query_type, AMDCUID_DEVICE_TYPE_GPU) << gpu.bdf;
      }
    }

    if (gpu.bdf_status == AMDCUID_STATUS_SUCCESS && gpu.card_status == AMDCUID_STATUS_SUCCESS) {
      EXPECT_EQ(gpu.bdf_cuid, gpu.card_cuid)
          << gpu.bdf << ": the cold by-BDF and by-path lookups disagree";
    }

    if (!gpu.node_path.empty()) {
      EXPECT_EQ(gpu.node_status, AMDCUID_STATUS_SUCCESS)
          << "cold amdcuid_get_handle_by_dev_path(" << gpu.node_path << ") failed";
      if (gpu.node_status == AMDCUID_STATUS_SUCCESS && gpu.bdf_status == AMDCUID_STATUS_SUCCESS) {
        EXPECT_EQ(gpu.bdf_cuid, gpu.node_cuid)
            << gpu.bdf << ": the cold by-BDF and by-node lookups disagree";
      }
    }

    IF_VERB(1) {
      printf("  Cold [%s] bdf=%d card=%d node=%d cuid=%s\n", gpu.bdf.c_str(),
             static_cast<int>(gpu.bdf_status), static_cast<int>(gpu.card_status),
             static_cast<int>(gpu.node_status), gpu.bdf_cuid.c_str());
    }
  }
}
