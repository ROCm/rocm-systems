// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// What the record generator does with a device it cannot derive a CUID for.
//
// It used to record the entry anyway, with derived_cuid left as the zero
// amdcuid_derived_id it had failed to fill in. Now that a device with no
// obtainable primary fails derivation as a matter of course, every such device
// on a node would be recorded under the same all-zero CUID.
//
// A device the record does not name is answered by the live lookup path. A
// device the record names wrongly is not.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "src/cuid_device.h"
#include "src/cuid_file.h"
#include "src/cuid_util.h"

namespace {

// A component with a good primary whose derivation fails. Platform is the one
// type generate_from_devices() fills in without downcasting to a concrete
// device class, so nothing here depends on hardware.
class UnderivableDevice : public CuidDevice {
 public:
  amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_PLATFORM; }
  amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const override {
    return CuidUtilities::generate_primary_cuid(0xD3AAABD406C5349Bull, 0, 0, 0, 0,
                                                AMDCUID_DEVICE_TYPE_PLATFORM, &id, false);
  }
  amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override {
    fingerprint = 1;
    return AMDCUID_STATUS_SUCCESS;
  }
  amdcuid_status_t get_derived_cuid(amdcuid_derived_id& id, cuid_hmac* /*hmac*/) const override {
    id = amdcuid_derived_id{};
    return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
  }
};

class ScratchRecord {
 public:
  ScratchRecord() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe) - single-threaded test
    const char* tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string((tmp && tmp[0]) ? tmp : "/tmp") + "/amdcuid_rec.XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const int fd = mkstemp(buf.data());
    if (fd >= 0) close(fd);
    path_ = buf.data();
  }
  ~ScratchRecord() {
    std::remove(path_.c_str());
    std::remove((path_ + ".lock").c_str());
  }
  ScratchRecord(const ScratchRecord&) = delete;
  ScratchRecord& operator=(const ScratchRecord&) = delete;
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

bool IsZero(const amdcuid_id_t& id) {
  const amdcuid_id_t zero = {{0}};
  return std::memcmp(id.bytes, zero.bytes, sizeof(id.bytes)) == 0;
}

}  // namespace

TEST(cuidtstUnprivileged, RecordOmitsDeviceWhoseDerivationFailed) {
  ScratchRecord record;
  ASSERT_FALSE(record.path().empty());

  std::vector<std::shared_ptr<CuidDevice>> devices;
  devices.push_back(std::make_shared<UnderivableDevice>());

  ASSERT_EQ(CuidFileGenerator::generate_unpriv_from_devices(devices, record.path()),
            AMDCUID_STATUS_SUCCESS);

  CuidFile written(record.path(), false);
  ASSERT_EQ(written.load(), AMDCUID_STATUS_SUCCESS);

  for (const auto& entry : written.get_entries()) {
    EXPECT_FALSE(IsZero(entry.derived_cuid))
        << "an all-zero derived CUID was recorded for a device whose derivation failed";
  }
  EXPECT_TRUE(written.get_entries().empty())
      << "the only device offered could not be derived, so the record must name nothing";
}

// Device indices stay dense: a skipped device must not leave a hole, or
// [PLATFORM:1] appears in a record with no [PLATFORM:0] and a consumer counting
// sections disagrees with one reading indices.
TEST(cuidtstUnprivileged, SkippedDeviceDoesNotConsumeAnIndex) {
  class DerivableDevice : public UnderivableDevice {
   public:
    amdcuid_status_t get_derived_cuid(amdcuid_derived_id& id, cuid_hmac* hmac) const override {
      amdcuid_primary_id primary = {};
      const amdcuid_status_t status = get_primary_cuid(primary);
      if (status != AMDCUID_STATUS_SUCCESS) return status;
      return CuidUtilities::generate_derived_cuid(&primary, &id, hmac);
    }
  };

  ScratchRecord record;
  ASSERT_FALSE(record.path().empty());

  std::vector<std::shared_ptr<CuidDevice>> devices;
  devices.push_back(std::make_shared<UnderivableDevice>());
  devices.push_back(std::make_shared<DerivableDevice>());

  ASSERT_EQ(CuidFileGenerator::generate_unpriv_from_devices(devices, record.path()),
            AMDCUID_STATUS_SUCCESS);

  CuidFile written(record.path(), false);
  ASSERT_EQ(written.load(), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(written.get_entries().size(), 1u);
  EXPECT_EQ(written.get_entries()[0].device_index, 0u);
  EXPECT_FALSE(IsZero(written.get_entries()[0].derived_cuid));
}
