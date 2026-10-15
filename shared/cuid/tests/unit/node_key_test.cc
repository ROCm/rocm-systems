// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The keys amdcuid_set_hash_key() refuses, and who may set or read the key.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <string>

#include "include/amd_cuid.h"
#include "src/hmac.h"
#include "test_common.h"

TEST(cuidtstUnprivileged, PublicAndTrivialKeysAreRejected) {
  uint8_t key[key_length];
  for (const uint8_t fill : std::initializer_list<uint8_t>{0x00, 0x41, 0xff}) {
    std::memset(key, fill, sizeof(key));
    EXPECT_TRUE(CuidUtilities::is_rejected_key(key)) << int(fill);
  }
  for (const char* constant : {"AMD-CUID-DEFAULT-SEED-v1", "AMD-CUID-TEMP-KEY-v1"}) {
    std::memset(key, 0, sizeof(key));
    std::memcpy(key, constant, std::strlen(constant));
    EXPECT_TRUE(CuidUtilities::is_rejected_key(key)) << constant;
  }
  for (size_t i = 0; i < key_length; ++i) key[i] = static_cast<uint8_t>(0xA5 ^ i);
  EXPECT_TRUE(CuidUtilities::is_rejected_key(key));
  for (size_t i = 0; i < key_length; ++i) key[i] = static_cast<uint8_t>(i);
  EXPECT_TRUE(CuidUtilities::is_rejected_key(key));

  key[0] = 0x80;
  EXPECT_FALSE(CuidUtilities::is_rejected_key(key));
  std::memset(key, 0, sizeof(key));
  std::memcpy(key, "AMD-CUID-TEMP-KEY-v1", 20);
  key[31] = 1;
  EXPECT_FALSE(CuidUtilities::is_rejected_key(key));
}

// Root is never exercised here: a key this suite passes would re-key amdgpu.
TEST(cuidtstUnprivileged, SetHashKeyNeedsRoot) {
  if (geteuid() == 0) GTEST_SKIP() << "only an ordinary user can call it safely";
  uint8_t key[key_length];
  for (size_t i = 0; i < key_length; ++i) key[i] = static_cast<uint8_t>(0x40 + 3 * i);
  EXPECT_EQ(amdcuid_set_hash_key(key), AMDCUID_STATUS_PERMISSION_DENIED);
}

TEST(cuidtstUnprivileged, KeyInfoNeedsRoot) {
  if (geteuid() == 0) GTEST_SKIP() << "root may read the key";
  amdcuid_key_info_t info;
  std::memset(&info, 0xff, sizeof(info));
  EXPECT_EQ(amdcuid_get_key_info(&info), AMDCUID_STATUS_PERMISSION_DENIED);
  const amdcuid_key_info_t cleared{};
  EXPECT_EQ(std::memcmp(&info, &cleared, sizeof(info)), 0);
  EXPECT_EQ(amdcuid_get_key_info(nullptr), AMDCUID_STATUS_INVALID_ARGUMENT);
}

namespace {
// A device bound to `driver` (none when empty) with a cuid_seed of `bytes`.
void write_seed(const std::string& device, const std::string& bytes,
                const std::string& driver = "amdgpu") {
  mkdir(device.c_str(), 0700);
  if (!driver.empty()) {
    const std::string target = "../../../bus/pci/drivers/" + driver;
    EXPECT_TRUE(symlink(target.c_str(), (device + "/driver").c_str()) == 0 || errno == EEXIST);
  }
  std::ofstream(device + "/cuid_seed", std::ios::binary | std::ios::trunc) << bytes;
}

std::string fingerprint_of(const cuid_hmac& hmac) {
  amdcuid_key_info_t info{};
  if (hmac.get_key_info(&info) != AMDCUID_STATUS_SUCCESS || !info.provisioned) return "none";
  return std::string(reinterpret_cast<const char*>(info.fingerprint), sizeof(info.fingerprint));
}
}  // namespace

// A cuid_seed that is absent means no key; one that fails otherwise, or has
// the wrong length, is an error that keeps the key last read.
TEST(cuidtstUnprivileged, NodeKeyReadErrorIsNotAbsence) {
  const ScopedTempDir devices("cuid_seed_");
  ASSERT_FALSE(devices.path().empty());
  const std::string gpu = devices.path() + "/0000:03:00.0";
  const std::string key(key_length, 'k');
  write_seed(gpu, key);
  mkdir((devices.path() + "/0000:00:01.0").c_str(), 0700);

  cuid_hmac hmac;
  ASSERT_EQ(hmac.reload_key(devices.path()), AMDCUID_STATUS_SUCCESS);
  const std::string loaded = fingerprint_of(hmac);
  ASSERT_NE(loaded, "none");

  ASSERT_EQ(::remove((gpu + "/cuid_seed").c_str()), 0);
  ASSERT_EQ(mkdir((gpu + "/cuid_seed").c_str(), 0700), 0);
  EXPECT_EQ(hmac.reload_key(devices.path()), AMDCUID_STATUS_FILE_ERROR);
  EXPECT_EQ(fingerprint_of(hmac), loaded);

  ASSERT_EQ(::remove((gpu + "/cuid_seed").c_str()), 0);
  for (const auto& wrong : {std::string(), key.substr(1), key + "\n"}) {
    write_seed(gpu, wrong);
    EXPECT_EQ(hmac.reload_key(devices.path()), AMDCUID_STATUS_FILE_ERROR) << wrong.size();
    EXPECT_EQ(fingerprint_of(hmac), loaded) << wrong.size();
  }

  ASSERT_EQ(::remove((gpu + "/cuid_seed").c_str()), 0);
  EXPECT_EQ(hmac.reload_key(devices.path()), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(fingerprint_of(hmac), "none");

  write_seed(gpu, key);
  ASSERT_EQ(hmac.reload_key(devices.path()), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(hmac.reload_key(devices.path() + "/missing"), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(fingerprint_of(hmac), "none");

  if (geteuid() != 0) {
    ASSERT_EQ(hmac.reload_key(devices.path()), AMDCUID_STATUS_SUCCESS);
    ASSERT_EQ(chmod((gpu + "/cuid_seed").c_str(), 0), 0);
    EXPECT_EQ(hmac.reload_key(devices.path()), AMDCUID_STATUS_PERMISSION_DENIED);
    EXPECT_EQ(fingerprint_of(hmac), "none");
  }
}

// Only a scan that completed without a cuid_seed means amdgpu is absent.
TEST(cuidtstUnprivileged, CuidSeedScanErrorIsNotAbsence) {
  const ScopedTempDir devices("cuid_seed_");
  ASSERT_FALSE(devices.path().empty());
  std::string bdf;
  EXPECT_EQ(CuidUtilities::find_cuid_seed(devices.path() + "/missing", bdf),
            AMDCUID_STATUS_UNSUPPORTED);
  mkdir((devices.path() + "/0000:00:01.0").c_str(), 0700);
  EXPECT_EQ(CuidUtilities::find_cuid_seed(devices.path(), bdf), AMDCUID_STATUS_UNSUPPORTED);

  write_seed(devices.path() + "/0000:03:00.0", "");
  ASSERT_EQ(CuidUtilities::find_cuid_seed(devices.path(), bdf), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(bdf, "0000:03:00.0");

  const std::string file = devices.path() + "/0000:03:00.0/cuid_seed";
  EXPECT_EQ(CuidUtilities::find_cuid_seed(file, bdf), AMDCUID_STATUS_FILE_ERROR);

  if (geteuid() != 0) {
    ASSERT_EQ(chmod(devices.path().c_str(), 0), 0);
    EXPECT_EQ(CuidUtilities::find_cuid_seed(devices.path(), bdf), AMDCUID_STATUS_PERMISSION_DENIED);
    ASSERT_EQ(chmod(devices.path().c_str(), 0700), 0);
  }
}

// A device whose cuid_seed cannot be checked is not a device without one, but
// another device's cuid_seed still serves.
TEST(cuidtstUnprivileged, CuidSeedDeviceErrorIsNotAbsence) {
  const ScopedTempDir devices("cuid_seed_");
  ASSERT_FALSE(devices.path().empty());
  std::string bdf;
  std::ofstream(devices.path() + "/0000:00:01.0") << "";
  EXPECT_EQ(CuidUtilities::find_cuid_seed(devices.path(), bdf), AMDCUID_STATUS_UNSUPPORTED);

  const std::string loop = devices.path() + "/0000:00:02.0";
  ASSERT_EQ(symlink(loop.c_str(), loop.c_str()), 0);
  EXPECT_EQ(CuidUtilities::find_cuid_seed(devices.path(), bdf), AMDCUID_STATUS_FILE_ERROR);

  const std::string closed = devices.path() + "/0000:00:03.0";
  if (geteuid() != 0) {
    ASSERT_EQ(mkdir(closed.c_str(), 0), 0);
    EXPECT_EQ(CuidUtilities::find_cuid_seed(devices.path(), bdf), AMDCUID_STATUS_PERMISSION_DENIED);
  }

  write_seed(devices.path() + "/0000:63:00.0", "");
  ASSERT_EQ(CuidUtilities::find_cuid_seed(devices.path(), bdf), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(bdf, "0000:63:00.0");
  if (geteuid() != 0) ASSERT_EQ(chmod(closed.c_str(), 0700), 0);
}

// Only amdgpu holds the node key: a cuid_seed on a device bound to another
// driver, or to none, is neither read nor written.
TEST(cuidtstUnprivileged, CuidSeedOfAnotherDriverIsIgnored) {
  const ScopedTempDir devices("cuid_seed_");
  ASSERT_FALSE(devices.path().empty());
  const std::string foreign(key_length, 'f');
  write_seed(devices.path() + "/0000:07:00.0", foreign, "nvme");
  write_seed(devices.path() + "/0000:08:00.0", foreign, "");

  cuid_hmac hmac;
  EXPECT_EQ(hmac.reload_key(devices.path()), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(fingerprint_of(hmac), "none");
  std::string bdf;
  EXPECT_EQ(CuidUtilities::find_cuid_seed(devices.path(), bdf), AMDCUID_STATUS_UNSUPPORTED);

  const std::string key(key_length, 'k');
  write_seed(devices.path() + "/0000:03:00.0", key);
  ASSERT_EQ(hmac.reload_key(devices.path()), AMDCUID_STATUS_SUCCESS);
  cuid_hmac expected(reinterpret_cast<const char*>(key.data()), key.size());
  EXPECT_EQ(fingerprint_of(hmac), fingerprint_of(expected));
  ASSERT_EQ(CuidUtilities::find_cuid_seed(devices.path(), bdf), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(bdf, "0000:03:00.0");
}
