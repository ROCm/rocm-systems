/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "unit/driver_cuid_test.h"

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <string>

#include "src/cuid_util.h"

namespace {

// Lay down one fake sysfs attribute file. The reader takes a full path, so
// nothing here needs a real device, a loaded driver, or privilege.
void WriteAttribute(const std::string& path, const std::string& contents) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << contents;
}

}  // namespace

TestDriverCuid::TestDriverCuid() {
  SetTitle("Driver CUID Reader");
  SetDescription(
      "Verify CuidUtilities::read_driver_cuid_from_path parses a "
      "driver-published UUID and reports absent, unreadable and malformed "
      "attributes as three distinct statuses.");
}

void TestDriverCuid::SetUp() {}

void TestDriverCuid::Run() {
  char tmpl[] = "/tmp/cuid_driver_test_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  ASSERT_NE(dir, nullptr);
  const std::string root(dir);

  const std::string secondary = root + "/cuid_secondary";
  const std::string primary = root + "/cuid_primary";
  const std::string garbage = root + "/cuid_garbage";

  // The attribute is present and well-formed. The kernel writes the UUID with
  // a trailing newline, which the reader must trim.
  {
    WriteAttribute(secondary, "d4abaad3-9b34-8c50-9800-028dcc084200\n");
    const uint8_t expected[16] = {0xd4, 0xab, 0xaa, 0xd3, 0x9b, 0x34, 0x8c, 0x50,
                                  0x98, 0x00, 0x02, 0x8d, 0xcc, 0x08, 0x42, 0x00};
    amdcuid_id_t id = {{0}};
    EXPECT_EQ(CuidUtilities::read_driver_cuid_from_path(secondary, &id), AMDCUID_STATUS_SUCCESS);
    for (int i = 0; i < 16; ++i) {
      EXPECT_EQ(id.bytes[i], expected[i]) << "Mismatch at byte " << i;
    }

    // A driver-published value is reported unchanged, never recomputed.
    uint8_t raw_bits[16] = {0};
    CuidUtilities::remove_UUIDv8_bits(&id, raw_bits);
    amdcuid_id_t reframed = {{0}};
    CuidUtilities::add_UUIDv8_bits(raw_bits, &reframed);
    for (int i = 0; i < 16; ++i) {
      EXPECT_EQ(reframed.bytes[i], id.bytes[i]) << "Round-trip mismatch at byte " << i;
    }
  }

  // The attribute is absent: the ordinary case on a driver that predates the
  // CUID interface, and the only outcome that lets the caller move on to the
  // later lookup stages. Not to be conflated with the two failures below.
  {
    amdcuid_id_t id = {{0}};
    EXPECT_EQ(CuidUtilities::read_driver_cuid_from_path(root + "/cuid_absent", &id),
              AMDCUID_STATUS_FILE_NOT_FOUND);
    // A whole missing device directory reads the same way.
    EXPECT_EQ(CuidUtilities::read_driver_cuid_from_path(root + "/no_such_dev/cuid_primary", &id),
              AMDCUID_STATUS_FILE_NOT_FOUND);
  }

  // The attribute is present but not readable, as cuid_primary is for an
  // unprivileged process. The kernel still holds the value, so this is not
  // "the driver publishes nothing".
  {
    WriteAttribute(primary, "d4abaad3-9b34-8c50-9800-028dcc084200\n");
    ASSERT_EQ(chmod(primary.c_str(), 0), 0);
    amdcuid_id_t id = {{0}};
    if (geteuid() == 0) {
      // Root bypasses the mode bits, so EACCES cannot be provoked this way.
      EXPECT_EQ(CuidUtilities::read_driver_cuid_from_path(primary, &id), AMDCUID_STATUS_SUCCESS);
    } else {
      EXPECT_EQ(CuidUtilities::read_driver_cuid_from_path(primary, &id),
                AMDCUID_STATUS_PERMISSION_DENIED);
    }
    chmod(primary.c_str(), 0600);
  }

  // The attribute is present and readable but does not hold a UUID.
  {
    const char* bad_values[] = {
        "",
        "\n",
        "not-a-uuid\n",
        "d4abaad3-9b34-8c50-9800-028dcc0842\n",      // too short
        "d4abaad3-9b34-8c50-9800-028dcc08420000\n",  // too long
        "zzzzzzzz-9b34-8c50-9800-028dcc084200\n",    // right length, not hex
        "d4abaad39b348c509800028dcc084200\n",        // unseparated
        // 36 characters with four hyphens in the wrong places: hyphens are
        // stripped wherever they fall, so a length check alone accepts these.
        "d4abaad-39b348-c50-9800-028dcc084200\n",
        "-d4abaad39b348c509800028dcc084200---\n",
        "d4abaad3-9b34-8c50-9800-028dcc0842-0\n",
    };
    for (const char* value : bad_values) {
      WriteAttribute(garbage, value);
      amdcuid_id_t id = {{0}};
      EXPECT_EQ(CuidUtilities::read_driver_cuid_from_path(garbage, &id),
                AMDCUID_STATUS_INVALID_FORMAT)
          << "Accepted malformed attribute: \"" << value << "\"";
    }
  }

  // Argument checks: the reader must not dereference a null output, and the
  // BDF wrapper must not build a path out of an empty BDF, which is what a
  // GIM-only device reaches the caller with.
  {
    EXPECT_EQ(CuidUtilities::read_driver_cuid_from_path(secondary, nullptr),
              AMDCUID_STATUS_INVALID_ARGUMENT);
    amdcuid_id_t id = {{0}};
    EXPECT_EQ(CuidUtilities::read_driver_cuid("", CuidUtilities::kDriverPrimaryAttribute, &id),
              AMDCUID_STATUS_INVALID_ARGUMENT);
  }

  unlink(secondary.c_str());
  unlink(primary.c_str());
  unlink(garbage.c_str());
  rmdir(root.c_str());
}

void TestDriverCuid::Close() {}
