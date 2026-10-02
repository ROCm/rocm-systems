// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The serial that goes into payload bits 0:63 is chosen by walking a fixed
// list of sources and stopping at the first that yields a NON-ZERO value. A
// source that reads back zero is an unimplemented capability or an unpopulated
// register and must be treated as absent: accepting it gives every affected
// component on the machine the same CUID.
//
// CuidGpu used to stop at the first source that existed instead. An all-zero
// sysfs unique_id returned without ever consulting the PCIe Device Serial
// Number, and an all-zero DSN returned HW_FINGERPRINT_NOT_FOUND without ever
// consulting the vendor-specific capability.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "src/cuid_gpu.h"

namespace {

// Lay down one fake sysfs attribute. The reader takes a full path, so nothing
// here needs a real device or a loaded driver.
void WriteAttribute(const std::string& path, const std::string& contents) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << contents;
}

}  // namespace

TEST(cuidtstUnprivileged, GpuUniqueIdZeroIsAbsentNotAnIdentity) {
  char tmpl[] = "/tmp/cuid_serial_precedence_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  ASSERT_NE(dir, nullptr);
  const std::string root(dir);
  const std::string attr = root + "/unique_id";

  // A serial the driver really published: the one case that stops the walk.
  {
    WriteAttribute(attr, "d3aaabd406c5349b\n");
    uint64_t fingerprint = 0;
    EXPECT_EQ(CuidGpu::read_unique_id(attr, fingerprint), AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(fingerprint, 0xd3aaabd406c5349bull);
  }

  // An all-zero attribute. amdgpu creates unique_id for ASICs that have the
  // register whether or not it is programmed, so this is the source existing
  // and yielding nothing; the caller must go on to the DSN.
  {
    WriteAttribute(attr, "0000000000000000\n");
    uint64_t fingerprint = 0xdeadbeef;
    EXPECT_NE(CuidGpu::read_unique_id(attr, fingerprint), AMDCUID_STATUS_SUCCESS)
        << "an all-zero unique_id must be treated as absent";
    EXPECT_EQ(fingerprint, 0u) << "a rejected source must not leave a value behind";
  }

  // "0x0" is the same absence written differently.
  {
    WriteAttribute(attr, "0x0\n");
    uint64_t fingerprint = 0xdeadbeef;
    EXPECT_NE(CuidGpu::read_unique_id(attr, fingerprint), AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(fingerprint, 0u);
  }

  // Present but empty: the driver created the attribute and wrote nothing.
  {
    WriteAttribute(attr, "");
    uint64_t fingerprint = 0xdeadbeef;
    EXPECT_NE(CuidGpu::read_unique_id(attr, fingerprint), AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(fingerprint, 0u);
  }

  // Present and not a number.
  {
    WriteAttribute(attr, "not-a-serial\n");
    uint64_t fingerprint = 0xdeadbeef;
    EXPECT_EQ(CuidGpu::read_unique_id(attr, fingerprint), AMDCUID_STATUS_INVALID_FORMAT);
    EXPECT_EQ(fingerprint, 0u);
  }

  // Absent: the ordinary case on an ASIC with no unique_id register.
  {
    uint64_t fingerprint = 0xdeadbeef;
    EXPECT_EQ(CuidGpu::read_unique_id(root + "/no_such_attribute", fingerprint),
              AMDCUID_STATUS_FILE_NOT_FOUND);
    EXPECT_EQ(fingerprint, 0u);
  }

  std::remove(attr.c_str());
  rmdir(root.c_str());
}

// With no source left there is no serial, and in particular no serial of zero.
// A BDF that names nothing has neither a DSN nor a vendor-specific capability,
// so the walk runs off the end of the list and must say so.
TEST(cuidtstUnprivileged, GpuConfigSpaceSerialReportsAbsenceNotZero) {
  uint64_t fingerprint = 0xdeadbeef;
  const amdcuid_status_t status = CuidGpu::read_config_space_serial("0000:ff:1f.7", fingerprint);
  EXPECT_NE(status, AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(fingerprint, 0u);
}
