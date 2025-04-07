
/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "common/rdc_utils.h"

#include <arpa/inet.h>
#include <assert.h>
#include <netinet/in.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace amd {
namespace rdc {

bool FileExists(char const* filename) {
  struct stat buf;
  return (stat(filename, &buf) == 0);
}

int ReadFile(std::string path, std::string* retStr, bool chop_newline) {
  std::stringstream ss;
  int ret = 0;

  assert(retStr != nullptr);

  std::ifstream fs;
  fs.open(path);

  if (!fs.is_open()) {
    ret = errno;
    errno = 0;
    return ret;
  }
  ss << fs.rdbuf();
  fs.close();

  *retStr = ss.str();

  if (chop_newline) {
    retStr->erase(std::remove(retStr->begin(), retStr->end(), '\n'), retStr->end());
  }
  return ret;
}

int ReadFile(const char* path, std::string* retStr, bool chop_newline) {
  assert(path != nullptr);
  assert(retStr != nullptr);

  std::string file_path(path);

  return amd::rdc::ReadFile(file_path, retStr, chop_newline);
}

bool IsNumber(const std::string& s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

bool IsIP(const std::string& s) {
  struct sockaddr_in sa;
  int result = inet_pton(AF_INET, s.c_str(), &sa);
  // inet_pton returns 1 on success
  return result == 1;
}

}  // namespace rdc
}  // namespace amd
