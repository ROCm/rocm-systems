// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The Component Type name table must name every valid type distinctly and
// invent no name for a reserved one.

#include <gtest/gtest.h>

#include <set>
#include <string>

#include "include/amd_cuid.h"
#include "src/cuid_util.h"

namespace {

// The twelve named Component Types. 0xb-0xe are reserved and have no name.
const amdcuid_device_type_t kNamedTypes[] = {
    AMDCUID_DEVICE_TYPE_PLATFORM, AMDCUID_DEVICE_TYPE_CPU,     AMDCUID_DEVICE_TYPE_GPU,
    AMDCUID_DEVICE_TYPE_NIC,      AMDCUID_DEVICE_TYPE_NPU,     AMDCUID_DEVICE_TYPE_STORAGE,
    AMDCUID_DEVICE_TYPE_MEMORY,   AMDCUID_DEVICE_TYPE_GENPCIE, AMDCUID_DEVICE_TYPE_GENC,
    AMDCUID_DEVICE_TYPE_RACKTRAY, AMDCUID_DEVICE_TYPE_RACK,    AMDCUID_DEVICE_TYPE_OTHER,
};

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
  // one: a reserved type is not a Platform.
  for (unsigned value = 0xb; value <= 0xe; ++value) {
    const auto type = static_cast<amdcuid_device_type_t>(value);
    EXPECT_EQ(CuidUtilities::device_type_to_string(type), "UNKNOWN") << std::hex << value;
    EXPECT_FALSE(amdcuid_device_type_is_valid(type)) << std::hex << value;
  }
  EXPECT_EQ(CuidUtilities::device_type_to_string(AMDCUID_DEVICE_TYPE_NONE), "UNKNOWN");
  EXPECT_FALSE(amdcuid_device_type_is_valid(AMDCUID_DEVICE_TYPE_NONE));
}
