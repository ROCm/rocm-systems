/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
