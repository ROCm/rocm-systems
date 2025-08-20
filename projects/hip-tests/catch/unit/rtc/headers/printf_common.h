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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef CATCH_UNIT_RTC_HEADERS_PRINTF_COMMON_H_
#define CATCH_UNIT_RTC_HEADERS_PRINTF_COMMON_H_
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <error.h>
#include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#if defined(_WIN32)
class CaptureStream {
 private:
  FILE* stream;
  int fdPipe[2];
  int fd;

  static constexpr size_t bufferSize = 25 * 1024 * 1024;

 public:
  explicit CaptureStream(FILE* original) {
    stream = original;

    if (pipe(fdPipe, bufferSize, O_TEXT) != 0) {
      fprintf(stderr, "pipe(3) failed with error %d\n", errno);
      assert(false);
    }

    if ((fd = dup(fileno(stream))) == -1) {
      fprintf(stderr, "dup(1) failed with error %d\n", errno);
      assert(false);
    }
  }

  ~CaptureStream() {
    close(fd);
    close(fdPipe[1]);
    close(fdPipe[0]);
  }

  void Begin() {
    fflush(stream);

    if (dup2(fdPipe[1], fileno(stream)) == -1) {
      fprintf(stderr, "dup2(2) failed with error %d\n", errno);
      assert(false);
    }

    setvbuf(stream, NULL, _IONBF, 0);
  }

  void End() {
    if (dup2(fd, fileno(stream)) == -1) {
      fprintf(stderr, "dup2(2) failed with error %d\n", errno);
      assert(false);
    }
  }

  std::string getData() {
    std::string data;
    data.resize(bufferSize);

    int numRead = read(fdPipe[0], const_cast<char*>(data.c_str()), bufferSize);
    data[numRead] = '\0';

    data.resize(strlen(data.c_str()));
    data.shrink_to_fit();

    return data;
  }
};
#else
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
  }

  void Begin() {
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

  void End() {
    fflush(nullptr);
    if (dup2(saved_fd, orig_fd) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
    if (close(saved_fd) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
  }

  std::string getData() {
    std::ifstream tmpFileStream(tempname);
    std::stringstream strStream;
    strStream << tmpFileStream.rdbuf();
    return strStream.str();
  }

  ~CaptureStream() {
    if (remove(tempname) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
  }

  // Truncate the file up to size if we don't want too long log
  void Truncate(size_t size) {
    struct stat sb;
    memset(&sb, 0, sizeof(sb));
    if (::stat(tempname, &sb) == -1) {
      std::cout << "failed lstat " << tempname;
      std::cout << "with error: " << ::strerror(errno) << std::endl;
      return;
    }
    if (sb.st_size > size) {
      if (::truncate(tempname, static_cast<off_t>(size)) == -1) {
        std::cout << "failed truncate " << tempname;
        std::cout << "with error: " << ::strerror(errno) << std::endl;
        return;
      }
    }
  }
};
#endif

#define DECLARE_DATA()                                                                             \
  const char* msg_short = "Carpe diem.";                                                           \
  const char* msg_long1 =                                                                          \
      "Lorem ipsum dolor sit amet, consectetur nullam. "                                           \
      "In mollis imperdiet nibh nec ullamcorper.";                                                 \
  const char* msg_long2 =                                                                          \
      "Curabitur nec metus sit amet augue vehicula "                                               \
      "ultrices ut id leo. Lorem ipsum dolor sit amet, "                                           \
      "consectetur adipiscing elit amet.";

#endif  // CATCH_UNIT_RTC_HEADERS_PRINTF_COMMON_H_
