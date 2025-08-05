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
#ifndef INCLUDE_RDC_LIB_RDCLOGGER_H_
#define INCLUDE_RDC_LIB_RDCLOGGER_H_
#include <chrono>  // NOLINT
#include <iostream>
#include <string>

#define RDC_ERROR 0
#define RDC_INFO 1
#define RDC_DEBUG 2

#define RDC_LOG(debug_level, msg)                                                             \
  do {                                                                                        \
    auto& logger = amd::rdc::RdcLogger::getLogger();                                          \
    if (logger.should_log((debug_level))) {                                                   \
      logger.get_ostream() << logger.get_log_header((debug_level), __FILE__, __LINE__) << msg \
                           << std::endl;                                                      \
    }                                                                                         \
  } while (0)

namespace amd {
namespace rdc {
class RdcLogger {
 public:
  explicit RdcLogger(std::ostream& os);

  static RdcLogger& getLogger() {
    static RdcLogger logger(std::cout);
    return logger;
  }

  bool should_log(uint32_t severity) { return log_level_ >= severity; }

  std::ostream& get_ostream() { return os_; }

  std::string get_log_header(uint32_t severity, const char* file, int line);

 private:
  std::ostream& os_;
  uint32_t log_level_;
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCLOGGER_H_
