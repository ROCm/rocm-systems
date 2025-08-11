/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "rdc_lib/RdcLogger.h"

#include <stdlib.h>
#include <string.h>

#include <chrono>  // NOLINT
#include <iomanip>
#include <iostream>
#include <sstream>

#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcLogger::RdcLogger(std::ostream& os) : os_(os) {
  char* verbose = getenv("RDC_LOG");
  if (verbose == nullptr) {
    log_level_ = RDC_ERROR;
  } else if (strcmp(verbose, "DEBUG") == 0) {
    log_level_ = RDC_DEBUG;
  } else if (strcmp(verbose, "INFO") == 0) {
    log_level_ = RDC_INFO;
  } else {
    log_level_ = RDC_ERROR;
  }
}

std::string RdcLogger::get_log_header(uint32_t severity, const char* file, int line) {
  std::stringstream strstream;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
  strstream << std::fixed << std::setprecision(3) << (ms / 1000.0) << " ";
  if (severity == RDC_DEBUG) {
    strstream << "DEBUG ";
  } else if (severity == RDC_INFO) {
    strstream << "INFO ";
  } else {
    strstream << "ERROR ";
  }

  //  extract out the file path as it may be very long.
  if (file != nullptr) {
    std::string file_str(file);
    auto found = file_str.find_last_of("/");
    if (found != std::string::npos) {
      file_str = file_str.substr(found + 1);
    }
    strstream << file_str << "(" << line << "): ";
  }

  return strstream.str();
}

}  // namespace rdc
}  // namespace amd
