/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipAPICoverageUtils.h"

class BasicAPIStats {
 public:
  int number_of_called_apis;
  int number_of_not_called_apis;
  int number_of_deprecated_apis;
  int total_number_of_api_calls;
  int total_number_of_test_cases;
  int total_number_of_apis;
  float tests_coverage_percentage;
  BasicAPIStats(std::vector<HipAPIGroup>& hip_api_groups);
  float getLowCoverageLimit() const;
  float getMediumCoverageLimit() const;
};

void generateXMLReportFiles(std::vector<HipAPI>& hip_apis, std::vector<HipAPIGroup>& hip_api_groups,
                            const std::string& work_directory);
void generateHTMLReportFiles(std::vector<HipAPI>& hip_apis,
                             std::vector<HipAPIGroup>& hip_api_groups,
                             std::string tests_root_directory, std::string hipApiHeaderFile,
                             std::string hip_rtc_header_file, const std::string& work_directory);
