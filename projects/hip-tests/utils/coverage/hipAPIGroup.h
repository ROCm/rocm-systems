/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipAPI.h"
#include <iomanip>

class HipAPIGroup {
  friend bool operator==(const HipAPIGroup& l_hip_api_group, const HipAPIGroup& r_hip_api_group);

 public:
  HipAPIGroup(std::string group_name, std::vector<HipAPI>& hip_apis);
  std::string getName() const;
  int getTotalNumberOfAPIs() const;
  int getTotalNumberOfCalls() const;
  int getTotalNumberOfTestCases() const;
  int getNumberOfCalledAPIs() const;
  int getNumberOfNotCalledAPIs() const;
  int getNumberOfDeprecatedAPIs() const;
  float getPercentageOfCalledAPIs() const;
  std::string getBasicStatsXML() const;
  std::string getBasicStatsHTML() const;
  std::string createHTMLReport() const;
  bool isDeprecated() const;

 private:
  std::string group_name;
  int total_number_of_apis;
  int number_of_api_calls;
  float percentage_of_called_apis;
  int number_of_test_cases;
  std::string parent_group_name;
  std::vector<HipAPI> called_apis;
  std::vector<HipAPI> not_called_apis;
  std::vector<HipAPI> deprecated_apis;
};
