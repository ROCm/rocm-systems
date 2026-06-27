// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/inline_vector.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

namespace {

void construct_empty_vector() {
  util::inline_vector<int, 4> values;
  (void)values;
}

void construct_three_value_vector() {
  util::inline_vector<int, 4> values;
  values.push_back(1);
  values.push_back(2);
  values.push_back(3);
}

TEST(InlineVectorHistogramTest, AggregatesDestructionSizesByConstructionSite) {
  util::reset_inline_vector_histograms_for_testing();

  construct_empty_vector();
  construct_three_value_vector();
  construct_three_value_vector();

  std::ostringstream os;
  util::dump_inline_vector_histograms(os);
  const std::string report = os.str();

  EXPECT_NE(report.find("\"function\":"), std::string::npos);
  EXPECT_NE(report.find("construct_empty_vector"), std::string::npos);
  EXPECT_NE(report.find("construct_three_value_vector"), std::string::npos);
  EXPECT_NE(report.find("\"size\":0"), std::string::npos);
  EXPECT_NE(report.find("\"size\":3"), std::string::npos);
  EXPECT_NE(report.find("\"count\":1"), std::string::npos);
  EXPECT_NE(report.find("\"count\":2"), std::string::npos);
}

} // namespace
