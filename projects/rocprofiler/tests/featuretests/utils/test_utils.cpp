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
#include "test_utils.h"

namespace rocmtools {
namespace tests {
namespace utility {

// This function returns the running path of executable
std::string GetRunningPath(std::string string_to_erase) {
  std::string path;
  char *real_path;
  Dl_info dl_info;

  if (0 != dladdr(reinterpret_cast<void *>(main), &dl_info)) {
    std::string to_erase = string_to_erase;
    path = dl_info.dli_fname;
    real_path = realpath(path.c_str(), NULL);
    if (real_path == nullptr) {
      throw(std::string("Error! in extracting real path"));
    }
    path.clear();  // reset path
    path.append(real_path);

    size_t pos = path.find(to_erase);
    if (pos != std::string::npos) path.erase(pos, to_erase.length());
  } else {
    throw(std::string("Error! in extracting real path"));
  }
  return path;
}

// This function returns number of cores
// available in system
int GetNumberOfCores() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  const int num_cpu_cores = std::count(
      std::istream_iterator<std::string>(cpuinfo),
      std::istream_iterator<std::string>(), std::string("processor"));
  return num_cpu_cores;
}

}  // namespace utility
}  // namespace tests
}  // namespace rocmtools
