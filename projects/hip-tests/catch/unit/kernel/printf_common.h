/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef _STRESSTEST_PRINTF_COMMON_H_
#define _STRESSTEST_PRINTF_COMMON_H_

#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>

class CaptureStream {
 public:
  CaptureStream() {
    if (pipe2(pipe, 0) == -1) {
      error(0, errno, "Error");
      assert(false);
    }

    orig_fd = dup(fileno(stdout));
    if (orig_fd == -1) {
      error(0, errno, "Error");
      assert(false);
    }
  }

  ~CaptureStream() {
    close(orig_fd);
    close(pipe[READ]);
    close(pipe[WRITE]);
  }

  void beginCapture() {
    fflush(stdout);
    if (dup2(pipe[WRITE], fileno(stdout)) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
  }

  void endCapture() {
    // End Capture
    fflush(stdout);
    if (dup2(orig_fd, fileno(stdout)) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
    result.clear();

    // Get string
    const int bufSize = 2048;
    char buf[bufSize];
    int bytesRead = 0;
    bytesRead = read(pipe[READ], &buf, bufSize);
    
    if (bytesRead < 0) {
      error(0, errno, "Error reading from pipe");
      assert(false);
      result.clear();
    } else {
      result = std::string(buf, bytesRead);
    }
  }

  std::string getCapturedData() { return result; }

 private:
  enum PIPES { READ, WRITE };
  int pipe[2] = {0, 0};
  int orig_fd;
  std::string result;
};

#endif  // _STRESSTEST_PRINTF_COMMON_H_
