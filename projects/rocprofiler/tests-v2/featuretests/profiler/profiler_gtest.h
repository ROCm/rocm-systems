/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef TESTS_FEATURETESTS_PROFILER_GTESTS_APPS_PROFILER_GTEST_H_
#define TESTS_FEATURETESTS_PROFILER_GTESTS_APPS_PROFILER_GTEST_H_

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <functional>
#include <thread>
#include <cassert>
#include <chrono>
#include <memory>
#include <stdexcept>

#include "utils/test_utils.h"

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis Implementation of a Parser class for Profiler output
 * Parses pre-saved golden output for kernel info and saves them in a vector
 * Executes appliaction(passed as param:app_name) and saves parsed kernel info
 * in a vector.
 * Subsequent tests can use this to parse different applications
 */
/* --------------------------------------------------------------------------*/

class ApplicationParser : public ::testing::Test {
 protected:
  virtual void SetUp(const char* app_name) { SetApplicationEnv(app_name); }
  virtual void TearDown() {}

  //!< saves lines of profiler output
  std::vector<std::string> output_lines;

 public:
  //!< Sets application enviornment by seting HSA_TOOLS_LIB.
  void SetApplicationEnv(const char* app_name);

  //!< Parses kernel-info from a pre-saved golden out files
  // and saves them in a vector.
  void GetKernelInfoForGoldenOutput(const char* app_name, std::string filename,
                                    std::vector<profiler_kernel_info_t>* kernel_info_output);

  //!< Parses kernel-info after running profiler against curent application
  // and saves them in a vector.
  void GetKernelInfoForRunningApplication(std::vector<profiler_kernel_info_t>* kernel_info_output);

 private:
  //!< Runs a given appllication and saves profiler output.
  // These output lines can be letter passed for kernel informations
  // i.e: kernel_names
  void ProcessApplication(std::stringstream& ss);

  //!< Parses kernel info fields from given input
  // i.e: kernel_names, kernel_duration
  void ParseKernelInfoFields(const std::string& s,
                             std::vector<profiler_kernel_info_t>* kernel_info_output);
};

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis Implementation of a ProfilerTest
 * Subsequent tests can use this to parse different applications
 */
/* --------------------------------------------------------------------------*/

class ProfilerTest : public ApplicationParser {
 protected:
  virtual void SetUp(const char* app_name) { ApplicationParser::SetUp(app_name); }
};

// /* --------------------------------------------------------------------------*/
// /**
//  * @Synopsis Base class for plugin tests.
//  * The file test will check wether certain filenames are created.
//  * Currently, file plugin tests only from build as they need to create files.
//  */
// /* --------------------------------------------------------------------------*/

class PluginTests : public ::testing::Test {
 public:
  //!< Sets application environment by seting rocprofv2.
  void RunApplication(const char* app_name, const char* appParams);

 private:
  //!< Runs a given appllication with the hsa activity.
  void ProcessApplication(std::stringstream& ss);
};
class FilePluginTest : public PluginTests {
 public:
  //!< Checks wether a file beginning with "filename" exists in "directory"
  static bool hasFileInDir(const std::string& filename, const char* directory);
};

class PerfettoPluginTest : public PluginTests {
 public:
  //!< Checks wether a file beginning with "filename" and ending with "pftrace" exists in
  //!< "directory"
  static bool hasFileInDir(const std::string& filename, const char* directory);
};

class CTFPluginTest : public FilePluginTest {
 public:
  static bool hasMetadataInDir(const char* directory);
};

#endif  // TESTS_FEATURETESTS_PROFILER_GTESTS_APPS_PROFILER_GTEST_H_
