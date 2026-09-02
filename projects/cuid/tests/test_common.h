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
#include "src/hmac.h"

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

// Put the node's provisioned seed back after a test that has to replace it.
//
// The key store is the machine's, not the test's: every derived CUID for a
// component the kernel does not answer for is a function of it.
//
// The original is re-applied through amdcuid_set_hash_key(), so the store and
// the process's in-memory key agree again. Where there was no key to begin with
// the store is removed; the in-memory key cannot be un-set through the public
// API, so this only helps a test that was going to change it anyway.
class KeyStoreGuard {
 public:
  KeyStoreGuard() {
    path_ = cuid_hmac().get_key_file_path();
    std::FILE* f = std::fopen(path_.c_str(), "rb");
    if (!f) return;
    present_ = std::fread(key_, 1, sizeof(key_), f) == sizeof(key_);
    std::fclose(f);
  }
  ~KeyStoreGuard() {
    if (present_) {
      amdcuid_set_hash_key(key_);
    } else {
      std::remove(path_.c_str());
    }
  }
  KeyStoreGuard(const KeyStoreGuard&) = delete;
  KeyStoreGuard& operator=(const KeyStoreGuard&) = delete;

 private:
  std::string path_;
  uint8_t key_[key_length] = {0};
  bool present_ = false;
};

// Give the run its own CUID record store: one fresh directory per process,
// removed on the way out.
//
// The library keeps its records in root-owned /var/lib/amdcuid and honours
// $AMDCUID_RECORD_DIR only when euid is not 0, so an unprivileged process
// cannot steer where a root-privileged refresh writes primary CUIDs and raw
// hardware fingerprints. An unprivileged suite therefore has nowhere to write
// unless it brings its own directory, and any shared path would both depend on
// what a previous run left behind and overwrite the node's real records. Under
// root the override is ignored and this does
// nothing, leaving the privileged tests on the real path.
class RecordStoreEnvironment : public ::testing::Environment {
 public:
  void SetUp() override;
  void TearDown() override;

  // Empty when this process is root, i.e. when the real store is in use.
  const std::string& dir() const { return dir_; }

 private:
  std::string dir_;
};

// One GPU as the machine presents it in sysfs, and what each by-name lookup
// answered for it while the library had not enumerated anything yet.
//
// Coldness is the whole point. amdcuid_get_handle_by_bdf() and
// amdcuid_get_handle_by_dev_path() both check the device manager first, so once
// anything in the process has called amdcuid_get_all_handles() they answer out
// of that cache and never reach the discovery path underneath. Every test in
// this suite runs after TestBase::SetUp() has enumerated, so the discovery path
// had no coverage at all -- and it was broken: it resolved a sysfs path through
// real_dev_path_from_fd(), which reads st_rdev, which is zero for a directory.
// amd-smi calls amdcuid_get_handle_by_bdf() on a cold library and nothing else,
// so amdsmi_get_gpu_cuid_info() returned NOT_SUPPORTED for every GPU.
struct ColdLookup {
  std::string bdf;         // 0000:63:00.0
  std::string card_path;   // /sys/class/drm/card1
  std::string node_path;   // /dev/dri/renderD128, empty where there is none
  amdcuid_status_t bdf_status = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  amdcuid_status_t card_status = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  amdcuid_status_t node_status = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  std::string bdf_cuid;
  std::string card_cuid;
  std::string node_cuid;
  // What a property query on the handle the cold by-BDF lookup returned
  // answered. A handle that resolves and then cannot be used is worse than one
  // that does not resolve: amdcuid_get_handle_by_bdf() returned the right
  // derived CUID while every amdcuid_query_device_property() on it said
  // DEVICE_NOT_FOUND, because the device had gone into the manager's list
  // without going into the CUID index it is looked up through.
  amdcuid_status_t bdf_query_status = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  amdcuid_device_type_t bdf_query_type = AMDCUID_DEVICE_TYPE_NONE;
};

// Runs its SetUp() before the first test, which is the only moment in the
// process when the library is still cold. Register it after
// RecordStoreEnvironment, whose SetUp() resolves paths but enumerates nothing.
class ColdLookupEnvironment : public ::testing::Environment {
 public:
  void SetUp() override;

  // The GPUs found in sysfs, with the cold answers recorded against each.
  // Empty when the machine has no DRM card node.
  static const std::vector<ColdLookup>& results();
};

void ProcessCmdline(CUIDTstGlobals* globals, int argc, char** argv);

#endif  // CUID_TEST_COMMON_H_
