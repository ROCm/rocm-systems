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

#include <fstream>
#include <filesystem>
#include "hipAPIGroup.h"

void findAPICallInFile(HipAPI& hip_api, std::string test_module_file);
void findAPITestCaseInFileByDoxygen(HipAPI& hip_api, std::string test_module_file);
void findAPITestCaseInFileByAPIName(HipAPI& hip_api, std::string test_module_file);
void searchForAPI(HipAPI& hip_api, std::vector<std::string>& test_module_files);
std::vector<HipAPI> extractHipAPIs(std::string& hip_api_header_file,
                                   std::vector<std::string>& api_group_names, bool start_groups);
std::vector<HipAPI> extractDeviceAPIs(std::string& apis_list_file,
                                      std::vector<std::string>& api_group_names);
std::vector<std::string> extractTestModuleFiles(std::string& tests_root_directory);
std::string findAbsolutePathOfFile(std::string file_path);
