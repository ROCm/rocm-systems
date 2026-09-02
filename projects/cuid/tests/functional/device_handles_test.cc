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

namespace {

// What the library reports for every handle it currently holds, keyed by BDF,
// so a by-name lookup can be checked against the enumeration rather than
// against nothing.
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

// Everything the library currently holds, in the terms it publishes them: a
// warm test must ask by the names the library itself hands out, because those
// are the ones a caller has.
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

  // Two components sharing one identifier is the failure the whole scheme
  // exists to prevent, so the suite should say so rather than only checking
  // that each string is non-empty.
  std::set<std::string> seen;
  for (const auto& handle : device_handles_) {
    const char* id_str = amdcuid_id_to_string(handle);
    ASSERT_NE(id_str, nullptr);
    EXPECT_GT(strlen(id_str), 0u);
    EXPECT_TRUE(seen.insert(id_str).second) << "two components share the identifier " << id_str;
    IF_VERB(1) { printf("  Handle: %s\n", id_str); }
  }

  // Every enumerated GPU must be one the machine really has. A phantom entry
  // is an identity with no component behind it, which an inventory cannot tell
  // from a component that has gone away.
  //
  // The converse -- every card enumerated -- holds only for root. A device
  // whose fingerprint has to come out of PCIe configuration space, such as a
  // BMC display controller with no Device Serial Number capability, cannot be
  // identified by an unprivileged process at all, so it is legitimately absent
  // from an unprivileged enumeration.
  std::set<std::string> present;
  for (const ColdLookup& gpu : ColdLookupEnvironment::results()) present.insert(gpu.bdf);

  const std::vector<Enumerated> gpus = EnumeratedGpus(device_handles_);
  for (const Enumerated& gpu : gpus) {
    if (gpu.bdf.empty()) continue;
    EXPECT_EQ(present.count(gpu.bdf), 1u)
        << "enumeration reports a GPU at " << gpu.bdf << " that sysfs does not present";
  }

  if (geteuid() == 0) {
    for (const ColdLookup& gpu : ColdLookupEnvironment::results()) {
      EXPECT_FALSE(CuidForBdf(device_handles_, gpu.bdf).empty())
          << "sysfs presents " << gpu.card_path << " at " << gpu.bdf
          << " but enumeration returned no GPU handle for it";
    }
  } else if (!present.empty()) {
    EXPECT_FALSE(gpus.empty())
        << "the machine has " << present.size()
        << " DRM card node(s) and enumeration produced no GPU at all";
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

  // The negative case, so a lookup that answered SUCCESS for anything at all
  // would still fail this test. ffff: is outside the PCI domain range.
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

    // The path the library published for this device through
    // AMDCUID_QUERY_DEVICE_PATH. A caller that reads a path out of the library
    // and hands it straight back must get the same device: this is the round
    // trip, and it is the weakest thing the lookup can be asked to do.
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
  EXPECT_NE(amdcuid_get_handle_by_dev_path("/sys/class/drm/card4294967295",
                                           AMDCUID_DEVICE_TYPE_GPU, &absent),
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
    // Recorded before the first test ran, when the device manager was still
    // empty; every other test in this file runs after enumeration has filled
    // it, and is answered from that cache without reaching discovery.
    EXPECT_EQ(gpu.bdf_status, AMDCUID_STATUS_SUCCESS)
        << "cold amdcuid_get_handle_by_bdf(" << gpu.bdf
        << ") failed; this is the call amd-smi makes for every GPU, and it makes no other "
           "call first";
    EXPECT_EQ(gpu.card_status, AMDCUID_STATUS_SUCCESS)
        << "cold amdcuid_get_handle_by_dev_path(" << gpu.card_path << ") failed";

    // A handle that resolves and cannot then be used is the worse failure, and
    // it is the one amd-smi hits: it looks a device up by BDF and then queries
    // the component type and the auxiliary flag on what comes back.
    //
    // Note what this can and cannot catch. The defect it was written for --
    // the cold path indexing the device under a different value than the one
    // it handed out -- only appears when the library has to derive: no CUID
    // attributes in the driver and no entry in the record store. Where the
    // driver publishes a value, or the store already holds one, both
    // derivations return that same recorded value and agree by accident. So
    // this assertion is vacuous on a node with either, and load-bearing on a
    // fresh one, which is the node AMD SMI ships CUID to before the kernel
    // series lands.
    if (gpu.bdf_status == AMDCUID_STATUS_SUCCESS) {
      EXPECT_EQ(gpu.bdf_query_status, AMDCUID_STATUS_SUCCESS)
          << gpu.bdf
          << ": the cold by-BDF lookup returned a handle that no property query can use";
      if (gpu.bdf_query_status == AMDCUID_STATUS_SUCCESS) {
        EXPECT_EQ(gpu.bdf_query_type, AMDCUID_DEVICE_TYPE_GPU) << gpu.bdf;
      }
    }

    if (gpu.bdf_status == AMDCUID_STATUS_SUCCESS &&
        gpu.card_status == AMDCUID_STATUS_SUCCESS) {
      EXPECT_EQ(gpu.bdf_cuid, gpu.card_cuid)
          << gpu.bdf << ": the cold by-BDF and by-path lookups disagree";
    }

    if (!gpu.node_path.empty()) {
      EXPECT_EQ(gpu.node_status, AMDCUID_STATUS_SUCCESS)
          << "cold amdcuid_get_handle_by_dev_path(" << gpu.node_path << ") failed";
      if (gpu.node_status == AMDCUID_STATUS_SUCCESS &&
          gpu.bdf_status == AMDCUID_STATUS_SUCCESS) {
        EXPECT_EQ(gpu.bdf_cuid, gpu.node_cuid)
            << gpu.bdf << ": the cold by-BDF and by-node lookups disagree";
      }
    }

    // Deliberately not compared against what enumeration reports now. A derived
    // CUID is a function of the node seed, the rekey tests in this suite
    // replace that seed, and the cold values were captured before the first
    // test ran. The three cold answers were all taken at the same moment, so
    // they are comparable with each other and with nothing else.

    IF_VERB(1) {
      printf("  Cold [%s] bdf=%d card=%d node=%d cuid=%s\n", gpu.bdf.c_str(),
             static_cast<int>(gpu.bdf_status), static_cast<int>(gpu.card_status),
             static_cast<int>(gpu.node_status), gpu.bdf_cuid.c_str());
    }
  }
}
