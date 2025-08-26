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

#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>

/*
Helper class used to store information in what file has HIP API been detected,
and on what line of code in that file.
*/
class FileOccurrence {
 public:
  std::string file_name;
  int line_number;
  FileOccurrence(std::string file_name, int line_number);
};

/*
Helper class used to store information in what file has the API Test Case been detected,
and on what line of code in that file.
*/
class TestCaseOccurrence : public FileOccurrence {
  friend bool operator==(const TestCaseOccurrence& l_test, const TestCaseOccurrence& r_test);
  friend bool operator<(const TestCaseOccurrence& l_test, const TestCaseOccurrence& r_test);

 public:
  std::string test_case_name;
  TestCaseOccurrence(std::string test_case_name, std::string file_name, int line_number);
};

/*
Class used to store infromation about each HIP API. All information
is related to the API name, number of calls from test .cc files,
and its status of deprecation.
*/
class HipAPI {
  friend bool operator==(const HipAPI& l_hip_api, const HipAPI& r_hip_api);
  friend bool operator<(const HipAPI& l_hip_api, const HipAPI& r_hip_api);

 public:
  HipAPI(std::string api_name, bool deprecated_flag, std::string api_group_name,
         std::string file_restriction = "");
  std::string getName() const;
  std::string getGroupName() const;
  int getNumberOfCalls() const;
  std::vector<TestCaseOccurrence> getTestCases() const;
  void addFileOccurrence(FileOccurrence file_occurence);
  void addTestCase(TestCaseOccurrence test_case);
  bool isDeprecated() const;
  std::string getBasicStatsXML() const;
  std::string createHTMLReport() const;
  std::string getFileRestriction() const;
  std::vector<std::string> device_groups;

 private:
  std::string api_name;
  std::string file_restriction;
  int number_of_calls;
  bool deprecated;
  std::string api_group_name;
  std::vector<FileOccurrence> file_occurrences;
  std::vector<TestCaseOccurrence> test_cases;
};
