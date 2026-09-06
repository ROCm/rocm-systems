// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// amdcuid_get_key_info() answers whether a node is provisioned and whether it
// carries the same seed as another, so the three states of the key store have
// to be three different answers. "Unprovisioned" is a successful answer, and
// conflating it with "the store is not a key" or with "a seed is provisioned
// and this caller cannot see it" leaves an operator unable to act on any of
// them. The key file is 0600, so the last case is every unprivileged caller on
// a provisioned node.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "src/hmac.h"

namespace {

void WriteKeyFile(const std::string& path, size_t length) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  for (size_t i = 0; i < length; ++i) {
    out.put(static_cast<char>(i));
  }
}

// Point the next cuid_hmac at `path`. The constructor reads the environment, so
// this affects only objects built after the call, not the library's own key,
// which was loaded before main().
void SetKeyPath(const char* path) {
  if (path) {
    setenv("AMDCUID_HMAC_KEY_PATH", path, 1);
  } else {
    unsetenv("AMDCUID_HMAC_KEY_PATH");
  }
}

}  // namespace

TEST(cuidtstUnprivileged, KeyStoreStatesAreDistinguishable) {
  const char* saved = getenv("AMDCUID_HMAC_KEY_PATH");
  const std::string saved_path = saved ? saved : "";

  char tmpl[] = "/tmp/cuid_key_info_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  ASSERT_NE(dir, nullptr);
  const std::string root(dir);
  const std::string key_path = root + "/hmac_key.bin";

  // No key store: an ordinary state. The public fallback seed is in use,
  // derivation works, and the caller is told through is_using_default_key()
  // rather than through an error.
  {
    SetKeyPath(key_path.c_str());
    cuid_hmac hmac;
    EXPECT_EQ(hmac.key_store_status(), AMDCUID_STATUS_SUCCESS)
        << "an unprovisioned node is not a failure";
    EXPECT_TRUE(hmac.is_valid());
    EXPECT_TRUE(hmac.is_using_default_key());
  }

  // A provisioned key store.
  {
    WriteKeyFile(key_path, 32);
    SetKeyPath(key_path.c_str());
    cuid_hmac hmac;
    EXPECT_EQ(hmac.key_store_status(), AMDCUID_STATUS_SUCCESS);
    EXPECT_TRUE(hmac.is_valid());
    EXPECT_FALSE(hmac.is_using_default_key());
  }

  // A key store that is not a key. Corruption, and distinct from absence: a
  // short seed silently adopted would change every CUID on the machine.
  {
    WriteKeyFile(key_path, 16);
    SetKeyPath(key_path.c_str());
    cuid_hmac hmac;
    EXPECT_EQ(hmac.key_store_status(), AMDCUID_STATUS_KEY_ERROR);
    EXPECT_FALSE(hmac.is_valid());
  }

  // A key store this caller cannot read. Root reads it regardless, so the case
  // exists only for an unprivileged caller.
  {
    WriteKeyFile(key_path, 32);
    ASSERT_EQ(chmod(key_path.c_str(), 0), 0);
    SetKeyPath(key_path.c_str());
    if (geteuid() == 0) {
      GTEST_LOG_(INFO) << "Running as root; the unreadable-key-store case cannot arise.";
    } else {
      cuid_hmac hmac;
      EXPECT_EQ(hmac.key_store_status(), AMDCUID_STATUS_PERMISSION_DENIED)
          << "a provisioned node whose key this caller cannot read must not be "
             "reported as unprovisioned";
      // Derivation still falls back to the public seed, so this changes what is
      // reported and not what is computed.
      EXPECT_TRUE(hmac.is_valid());
    }
    chmod(key_path.c_str(), 0600);
  }

  std::remove(key_path.c_str());
  rmdir(root.c_str());
  SetKeyPath(saved_path.empty() ? nullptr : saved_path.c_str());
}

// And the public entry point must report the store's state rather than
// flattening it.
//
// Which store the library opened was decided before main(), so the probe below
// is trusted only when its fingerprint proves it read the same key; otherwise
// only the invariants that hold regardless are checked.
TEST(cuidtstUnprivileged, GetKeyInfoReportsTheStoreState) {
  amdcuid_key_info_t info = {};
  EXPECT_EQ(amdcuid_get_key_info(nullptr), AMDCUID_STATUS_INVALID_ARGUMENT);

  const amdcuid_status_t status = amdcuid_get_key_info(&info);
  ASSERT_TRUE(status == AMDCUID_STATUS_SUCCESS || status == AMDCUID_STATUS_PERMISSION_DENIED ||
              status == AMDCUID_STATUS_KEY_ERROR)
      << amdcuid_status_to_string(status);

  if (status != AMDCUID_STATUS_SUCCESS) {
    // The two failures are the two broken stores. "Unprovisioned" is not one of
    // them; it is info.provisioned on a successful call.
    return;
  }

  // A fingerprint is always reported on success. An unprovisioned node reports
  // the public fallback seed's digest; an all-zero one would compare equal
  // everywhere.
  bool any_set = false;
  for (size_t i = 0; i < sizeof(info.fingerprint); ++i) {
    if (info.fingerprint[i] != 0) any_set = true;
  }
  EXPECT_TRUE(any_set);

  cuid_hmac probe;
  uint8_t probe_fingerprint[AMDCUID_KEY_FINGERPRINT_SIZE] = {0};
  if (probe.key_fingerprint(probe_fingerprint) != AMDCUID_STATUS_SUCCESS ||
      memcmp(probe_fingerprint, info.fingerprint, sizeof(probe_fingerprint)) != 0) {
    GTEST_LOG_(INFO) << "Probe opened a different key store than the library did; "
                        "nothing to compare provisioning state against.";
    return;
  }
  EXPECT_EQ(info.provisioned, probe.is_using_default_key() ? 0 : 1);
}
