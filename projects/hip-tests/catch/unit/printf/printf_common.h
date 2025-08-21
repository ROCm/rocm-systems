/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once
#ifdef __linux__
#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>

struct CaptureStream {
  int saved_fd;
  int orig_fd;
  int temp_fd;

  char tempname[13] = "mytestXXXXXX";

  explicit CaptureStream(FILE* original) {
    orig_fd = fileno(original);
    saved_fd = dup(orig_fd);

    if ((temp_fd = mkstemp(tempname)) == -1) {
      error(0, errno, "Error");
      assert(false);
    }

    fflush(nullptr);
    if (dup2(temp_fd, orig_fd) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
    if (close(temp_fd) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
  }

  void restoreStream() {
    if (saved_fd == -1) return;
    fflush(nullptr);
    if (dup2(saved_fd, orig_fd) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
    if (close(saved_fd) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
    saved_fd = -1;
  }

  const char* getTempFilename() { return (const char*)tempname; }

  std::ifstream getCapturedData() {
    restoreStream();
    std::ifstream temp(tempname);
    return temp;
  }

  ~CaptureStream() {
    restoreStream();
    if (remove(tempname) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
  }
  std::string gulp(std::ifstream& input) {
    std::string retval;
    input.seekg(0, std::ios_base::end);
    retval.resize(input.tellg());
    input.seekg(0, std::ios_base::beg);
    input.read(&retval[0], retval.size());
    input.close();
    return retval;
  }
};
extern const char* msg_short;
extern const char* msg_long1;
extern const char* msg_long2;

__device__ const char* msg_short_dev = "Carpe diem.";
__device__ const char* msg_long1_dev =
    "Lorem ipsum dolor sit amet, consectetur nullam. In mollis imperdiet nibh nec ullamcorper.";  // NOLINT
__device__ const char* msg_long2_dev =
    "Curabitur nec metus sit amet augue vehicula ultrices ut id leo. Lorem ipsum dolor sit amet, "
    "consectetur adipiscing elit amet.";  // NOLINT
#endif
