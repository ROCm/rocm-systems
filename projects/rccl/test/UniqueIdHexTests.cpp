/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only tests for the ncclUniqueId hex encoding in common/UniqueIdHex.hpp.

#include <cstddef>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "common/UniqueIdHex.hpp"

namespace RcclUnitTesting {
namespace {

ncclUniqueId patternId() {
  ncclUniqueId id;
  unsigned char* bytes = reinterpret_cast<unsigned char*>(&id);
  for (size_t i = 0; i < sizeof(id); ++i) bytes[i] = static_cast<unsigned char>(i * 37 + 11);
  return id;
}

}  // namespace

TEST(UniqueIdHex, RoundTrips) {
  const ncclUniqueId id = patternId();
  const std::string hex = uniqueIdToHex(id);
  ASSERT_EQ(hex.size(), sizeof(ncclUniqueId) * 2);
  ncclUniqueId decoded{};
  ASSERT_TRUE(uniqueIdFromHex(hex, decoded));
  EXPECT_EQ(std::memcmp(&id, &decoded, sizeof(id)), 0);
}

TEST(UniqueIdHex, RejectsWrongLength) {
  const std::string hex = uniqueIdToHex(patternId());
  ncclUniqueId decoded;
  EXPECT_FALSE(uniqueIdFromHex("", decoded));
  EXPECT_FALSE(uniqueIdFromHex(hex.substr(1), decoded));
  EXPECT_FALSE(uniqueIdFromHex(hex + "00", decoded));
}

TEST(UniqueIdHex, RejectsPairsThatAreNotTwoHexDigits) {
  const std::string hex = uniqueIdToHex(patternId());
  for (const char* pair : {"-1", "+1", "a.", " a", "0x", "zz"}) {
    std::string bad = hex;
    bad.replace(10, 2, pair);
    ncclUniqueId decoded;
    EXPECT_FALSE(uniqueIdFromHex(bad, decoded)) << "pair \"" << pair << "\"";
  }
}

}  // namespace RcclUnitTesting
