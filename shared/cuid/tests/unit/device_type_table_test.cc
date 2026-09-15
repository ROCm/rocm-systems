// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The Component Type name tables.
//
// CuidUtilities::device_type_to_string() and CuidFile::string_to_device_type()
// are the two halves of how a component type survives a trip through the
// on-disk record, and save()'s output-order list is a third copy that filters
// as well as orders. All three stopped at the original five names long after
// the enum grew to sixteen values, silently dropping the rest.
//
// Asserted as a bijection over every valid value, end to end through save() and
// load(), rather than name by name against a transcription.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "include/amd_cuid.h"
#include "src/cuid_file.h"
#include "src/cuid_util.h"

namespace {

// The twelve named Component Types. 0xb-0xe are reserved and have no name.
const amdcuid_device_type_t kNamedTypes[] = {
    AMDCUID_DEVICE_TYPE_PLATFORM, AMDCUID_DEVICE_TYPE_CPU,     AMDCUID_DEVICE_TYPE_GPU,
    AMDCUID_DEVICE_TYPE_NIC,      AMDCUID_DEVICE_TYPE_NPU,     AMDCUID_DEVICE_TYPE_STORAGE,
    AMDCUID_DEVICE_TYPE_MEMORY,   AMDCUID_DEVICE_TYPE_GENPCIE, AMDCUID_DEVICE_TYPE_GENC,
    AMDCUID_DEVICE_TYPE_RACKTRAY, AMDCUID_DEVICE_TYPE_RACK,    AMDCUID_DEVICE_TYPE_OTHER,
};

class ScratchRecord {
 public:
  ScratchRecord() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe) - single-threaded test
    const char* tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string((tmp && tmp[0]) ? tmp : "/tmp") + "/amdcuid_types.XXXXXX";
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

// A derived CUID that is distinct per type. add_entry() folds entries carrying
// the same derived CUID together, so identical ones would not all survive.
amdcuid_id_t DistinctCuid(amdcuid_device_type_t type) {
  amdcuid_primary_id primary = {};
  CuidUtilities::generate_primary_cuid(0xD3AAABD406C50000ull + static_cast<unsigned>(type), 0, 0, 0,
                                       0, type, &primary, false);
  return primary.UUIDv8_representation;
}

}  // namespace

TEST(cuidtstUnprivileged, DeviceTypeNamesAreDistinctAndComplete) {
  std::set<std::string> names;
  for (const auto type : kNamedTypes) {
    const std::string name = CuidUtilities::device_type_to_string(type);
    EXPECT_NE(name, "UNKNOWN") << "0x" << std::hex << static_cast<unsigned>(type)
                               << " is a named Component Type with no name in the table";
    EXPECT_TRUE(names.insert(name).second) << "two Component Types share the name " << name;
    EXPECT_TRUE(amdcuid_device_type_is_valid(type)) << name;
  }
  EXPECT_EQ(names.size(), 12u) << "every named Component Type must have its own name";

  // The reserved range and the sentinel have no name, and nothing must invent
  // one: a record naming a reserved type is not a Platform.
  for (unsigned value = 0xb; value <= 0xe; ++value) {
    const auto type = static_cast<amdcuid_device_type_t>(value);
    EXPECT_EQ(CuidUtilities::device_type_to_string(type), "UNKNOWN") << std::hex << value;
    EXPECT_FALSE(amdcuid_device_type_is_valid(type)) << std::hex << value;
  }
  EXPECT_EQ(CuidUtilities::device_type_to_string(AMDCUID_DEVICE_TYPE_NONE), "UNKNOWN");
  EXPECT_FALSE(amdcuid_device_type_is_valid(AMDCUID_DEVICE_TYPE_NONE));
}

// Twelve entries in, the same twelve types out. string_to_device_type() is
// private, so this drives it the way the library does: save() writes the
// section header with device_type_to_string(), load() parses it back.
TEST(cuidtstUnprivileged, DeviceTypesRoundTripThroughTheRecord) {
  ScratchRecord record;
  ASSERT_FALSE(record.path().empty());

  {
    CuidFile out(record.path(), false);
    uint32_t index = 0;
    for (const auto type : kNamedTypes) {
      CuidFileEntry entry;
      entry.device_type = type;
      entry.device_index = index++;
      entry.derived_cuid = DistinctCuid(type);
      entry.last_update = 0;
      ASSERT_EQ(out.add_entry(entry), AMDCUID_STATUS_SUCCESS)
          << CuidUtilities::device_type_to_string(type);
    }
    ASSERT_EQ(out.save(), AMDCUID_STATUS_SUCCESS);
  }

  CuidFile in(record.path(), false);
  ASSERT_EQ(in.load(), AMDCUID_STATUS_SUCCESS);

  std::set<int> recovered;
  for (const auto& entry : in.get_entries()) {
    EXPECT_TRUE(amdcuid_device_type_is_valid(entry.device_type))
        << "a record entry came back as a type that is not a Component Type";
    recovered.insert(static_cast<int>(entry.device_type));
  }

  for (const auto type : kNamedTypes) {
    EXPECT_EQ(recovered.count(static_cast<int>(type)), 1u)
        << CuidUtilities::device_type_to_string(type)
        << " did not survive the round trip through the record";
  }
  EXPECT_EQ(in.get_entries().size(), 12u)
      << "an entry was dropped on save or merged on load; every type is written once";
}
