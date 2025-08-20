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

#include "reportGenerators.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Please provide the path to the cloned HIP/include/ directory as an argument! "
                 "Only one argument supported."
              << std::endl;
    std::cout << "\tExample: ./generateHipAPICoverage /workspace/user1/HIP/include/" << std::endl;
    return -1;
  }
  std::string hip_include_path = argv[1];
  /*
  Relative paths to all needed files, as it is expected that the application
  is called from the HIP/tests/catch/coverage directory.
  */
  std::string hip_api_header_file{
      findAbsolutePathOfFile(hip_include_path + "/hip/hip_runtime_api.h")};
  std::string hip_rtc_header_file{findAbsolutePathOfFile(hip_include_path + "/hip/hiprtc.h")};
  std::string tests_root_directory{findAbsolutePathOfFile("../../catch")};
  std::string device_api_file{"device_api_list.txt"};

  std::vector<std::string> api_group_names;
  // Extract all HIP API declarations from the HIP API header file.
  std::vector<HipAPI> hip_apis{extractHipAPIs(hip_api_header_file, api_group_names, false)};
  std::cout << "Number of detected HIP APIs from " << hip_api_header_file << ": " << hip_apis.size()
            << std::endl;

  api_group_names.push_back("Runtime Compilation");
  std::vector<HipAPI> hip_rtc_apis{extractHipAPIs(hip_rtc_header_file, api_group_names, true)};
  std::cout << "Number of detected HIP APIs from " << hip_rtc_header_file << ": "
            << hip_rtc_apis.size() << std::endl;
  hip_apis.insert(hip_apis.end(), hip_rtc_apis.begin(), hip_rtc_apis.end());

  std::vector<HipAPI> device_apis{extractDeviceAPIs(device_api_file, api_group_names)};
  std::cout << "Number of detected device APIs from " << device_api_file << ": "
            << device_apis.size() << std::endl;
  hip_apis.insert(hip_apis.end(), device_apis.begin(), device_apis.end());

  // Extract all test module .cc files that shall be used for API searching.
  std::cout << "Searching for HIP API calls in source files within " << tests_root_directory << "."
            << std::endl;
  std::vector<std::string> test_module_files{extractTestModuleFiles(tests_root_directory)};

  // Search for each HIP API in the extracted test .cc files.
  for (HipAPI& hip_api : hip_apis) {
    searchForAPI(hip_api, test_module_files);
  }

  std::vector<HipAPIGroup> hip_api_groups;
  for (auto const& api_group_name : api_group_names) {
    HipAPIGroup hip_api_group{api_group_name, hip_apis};
    // Avoid having duplicated groups.
    if (std::find(hip_api_groups.begin(), hip_api_groups.end(), hip_api_group) ==
        hip_api_groups.end()) {
      hip_api_groups.push_back(hip_api_group);
    }
  }

  std::cout << "Generating XML report files." << std::endl;
  generateXMLReportFiles(hip_apis, hip_api_groups);
  std::cout << "Generating HTML report files." << std::endl;
  generateHTMLReportFiles(hip_apis, hip_api_groups, tests_root_directory, hip_api_header_file,
                          hip_rtc_header_file);

  return 0;
}
