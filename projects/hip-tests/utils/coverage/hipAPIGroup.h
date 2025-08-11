/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
