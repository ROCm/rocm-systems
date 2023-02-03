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
#include "gtests/apps/profiler_gtest.h"
#include "utils/test_utils.h"

/**
 * Sets application enviornment by seting HSA_TOOLS_LIB.
 */
void ApplicationParser::SetApplicationEnv(const char* app_name) {
  std::string app_path = GetRunningPath("tests/featuretests/profiler/runFeatureTests");

  std::stringstream counter_path;
  counter_path << app_path << "tests/featuretests/profiler/gtests/apps/goldentraces/input.txt";
  setenv("COUNTERS_PATH", counter_path.str().c_str(), true);

  std::stringstream hsa_tools_lib_path;
  hsa_tools_lib_path << app_path << "librocprofiler_tool.so";
  setenv("LD_PRELOAD", hsa_tools_lib_path.str().c_str(), true);

  std::stringstream os;
  os << app_path << "tests/featuretests/profiler/gtests/apps/" << app_name;

  ProcessApplication(os);
}

/**
 * Parses kernel-info after running profiler against curent application
 * and saves them in a vector.
 */
void ApplicationParser::GetKernelInfoForRunningApplication(
    std::vector<KernelInfo>* kernel_info_output) {
  KernelInfo kinfo;
  for (std::string line : output_lines) {
    if (std::regex_match(line, std::regex("(dispatch)(.*)"))) {
      int spos = line.find("[");
      int epos = line.find("]", spos);
      std::string sub = line.substr(spos + 1, epos - spos - 1);
      kinfo.dispatch_id = sub;
      kernel_info_output->push_back(kinfo);

      // Kernel-Name
      size_t found = line.find("kernel-name");
      if (found != std::string::npos) {
        int spos = found;
        int epos = line.find(")", spos);
        int length = std::string("kernel-name").length();
        std::string sub = line.substr(spos + length + 1, epos - spos - length - 1);

        kinfo.kernel_name = sub;
        kernel_info_output->push_back(kinfo);
      }
      // Start-Time
      found = line.find("start_time");
      if (found != std::string::npos) {
        int spos = found;
        int epos = line.find(",", spos);
        int length = std::string("start_time").length();
        std::string sub = line.substr(spos + length + 1, epos - spos - length - 1);
        kinfo.start_time = sub;
        kernel_info_output->push_back(kinfo);
      }
      // End-Time
      found = line.find("end_time");
      if (found != std::string::npos) {
        int spos = line.find(",", found);
        int epos = line.find(")", spos);
        std::string sub = line.substr(spos + 1, epos - spos - 1);
        kinfo.end_time = sub;
        kernel_info_output->push_back(kinfo);
      }
    }
  }
}

/**
 * Parses kernel-names from a pre-saved golden out files
 * and saves them in a vector.
 */
void ApplicationParser::GetKernelInfoForGoldenOutput(const char* app_name, std::string file_name,
                                                     std::vector<KernelInfo>* kernel_info_output) {
  std::string entry;
  std::string path = GetRunningPath("runFeatureTests");
  entry = path.append("gtests/apps/goldentraces/") + file_name;
  // parse kernel info fields for golden output
  ParseKernelInfoFields(entry, kernel_info_output);
}

/**
 * Runs a given appllication and saves profiler output.
 * These output lines can be letter passed for kernel informations
 * i.e: kernel_names
 */
void ApplicationParser::ProcessApplication(std::stringstream& ss) {
  FILE* handle = popen(ss.str().c_str(), "r");
  ASSERT_NE(handle, nullptr);

  char* ln{NULL};
  std::string temp{""};
  size_t len{0};

  while (getline(&ln, &len, handle) != -1) {
    temp = temp + std::string(ln);
  }

  free(ln);
  size_t pos{0};
  std::string delimiter{"\n"};
  while ((pos = temp.find(delimiter)) != std::string::npos) {
    output_lines.push_back(temp.substr(0, pos));
    temp.erase(0, pos + delimiter.length());
  }

  pclose(handle);
}

/**
 * Parses kernel-info for golden output file
 * and saves them in a vector.
 */
void ApplicationParser::ParseKernelInfoFields(const std::string& s,
                                              std::vector<KernelInfo>* kernel_info_output) {
  std::string line;
  KernelInfo kinfo;

  std::ifstream golden_file(s);
  while (!golden_file.eof()) {
    getline(golden_file, line);
    if (std::regex_match(line, std::regex("(dispatch)(.*)"))) {
      int spos = line.find("[");
      int epos = line.find("]", spos);
      std::string sub = line.substr(spos + 1, epos - spos - 1);
      kinfo.dispatch_id = sub;
      kernel_info_output->push_back(kinfo);

      // Kernel-Name
      size_t found = line.find("kernel-name");
      if (found != std::string::npos) {
        int spos = found;
        int epos = line.find(")", spos);
        int length = std::string("kernel-name").length();
        std::string sub = line.substr(spos + length + 1, epos - spos - length - 1);

        kinfo.kernel_name = sub;
        kernel_info_output->push_back(kinfo);
      }
      // Start-Time
      found = line.find("start_time");
      if (found != std::string::npos) {
        int spos = found;
        int epos = line.find(",", spos);
        int length = std::string("start_time").length();
        std::string sub = line.substr(spos + length + 1, epos - spos - length - 1);
        kinfo.start_time = sub;
        kernel_info_output->push_back(kinfo);
      }
      // End-Time
      found = line.find("end_time");
      if (found != std::string::npos) {
        int spos = line.find(",", found);
        int epos = line.find(")", spos);
        std::string sub = line.substr(spos + 1, epos - spos - 1);
        kinfo.end_time = sub;
        kernel_info_output->push_back(kinfo);
      }
    }
  }
  golden_file.close();
}
