// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_TEST_COMMON_H_
#define CUID_TEST_COMMON_H_

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "include/amd_cuid.h"

struct CUIDTstGlobals {
  uint32_t verbosity = 0;
  bool dont_fail = false;
};

extern CUIDTstGlobals sCUIDGlvalues;

// Conditionally execute a block at or above the given verbosity level.
#define IF_VERB(V) if (sCUIDGlvalues.verbosity >= (V))

// Assert that ret == AMDCUID_STATUS_SUCCESS, unless dont_fail is set.
#define CHK_ERR_ASRT(RET)                       \
  do {                                          \
    if (!sCUIDGlvalues.dont_fail) {             \
      ASSERT_EQ(AMDCUID_STATUS_SUCCESS, (RET)); \
    }                                           \
  } while (0)

// Sysfs GPU and lookup results captured before TestBase::SetUp() enumerates.
// Only the first lookup starts cold; subsequent calls share its manager state.
struct ColdLookup {
  std::string bdf;        // 0000:63:00.0
  std::string card_path;  // /sys/class/drm/card1
  std::string node_path;  // /dev/dri/renderD128, empty where there is none
  amdcuid_status_t bdf_status = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  amdcuid_status_t card_status = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  amdcuid_status_t node_status = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  std::string bdf_cuid;
  std::string card_cuid;
  std::string node_cuid;
  // Query immediately after lookup to detect a handle missing from the index.
  amdcuid_status_t bdf_query_status = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  amdcuid_device_type_t bdf_query_type = AMDCUID_DEVICE_TYPE_NONE;
};

// Runs its SetUp() before the first test, which is the only moment in the
// process when the library is still cold.
class ColdLookupEnvironment : public ::testing::Environment {
 public:
  void SetUp() override;

  // The GPUs found in sysfs, with the cold answers recorded against each.
  // Empty when the machine has no DRM card node.
  static const std::vector<ColdLookup>& results();
};

void ProcessCmdline(CUIDTstGlobals* globals, int argc, char** argv);

// The temporary CUID of `handle`'s component, rebuilt from the properties the
// library publishes for it and this host's machine-id. False when a property
// it needs cannot be queried or there is no machine-id.
bool RebuiltTemporaryCuid(const amdcuid_id_t& handle, amdcuid_id_t& out);

// Whether `handle` is a GPU partition, which has no BDF of its own, so its
// temporary CUID cannot be rebuilt from what the library reports for it.
// partition_cuid_test.cc checks those against their parent instead.
bool GpuPartition(const amdcuid_id_t& handle);

// For a temporary handle, that AMDCUID_QUERY_PRIMARY_CUID is refused with
// UNSUPPORTED and leaves the buffer alone (recorded as gtest failures), and
// true. False for a handle that is not temporary.
bool TemporaryWithoutPrimary(const amdcuid_id_t& handle);

// A fresh directory under $TMPDIR (else /tmp), removed with everything in it
// when the object goes out of scope, including after a failed ASSERT. path()
// is empty if it could not be created.
class ScopedTempDir {
 public:
  explicit ScopedTempDir(const char* prefix);
  ~ScopedTempDir();
  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

#endif  // CUID_TEST_COMMON_H_
