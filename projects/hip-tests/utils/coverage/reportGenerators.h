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

void generateXMLReportFiles(std::vector<HipAPI>& hip_apis,
                            std::vector<HipAPIGroup>& hip_api_groups);
void generateHTMLReportFiles(std::vector<HipAPI>& hip_apis,
                             std::vector<HipAPIGroup>& hip_api_groups,
                             std::string tests_root_directory, std::string hipApiHeaderFile,
                             std::string hip_rtc_header_file);
