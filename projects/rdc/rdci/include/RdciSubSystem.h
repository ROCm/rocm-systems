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
#ifndef RDCI_INCLUDE_RDCISUBSYSTEM_H_
#define RDCI_INCLUDE_RDCISUBSYSTEM_H_

#include <memory>
#include <string>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

class RdciSubSystem {
 public:
  RdciSubSystem();
  virtual void parse_cmd_opts(int argc, char** argv) = 0;
  virtual void connect();

  virtual void process() = 0;
  virtual ~RdciSubSystem();

  bool is_json_output() const;

 protected:
  void set_json_output(bool is_json);
  std::vector<std::string> split_string(const std::string& s, char delimiter) const;
  void show_common_usage() const;
  rdc_handle_t rdc_handle_;
  std::string ip_port_;

  bool use_auth_;
  std::string config_test_;
  std::string root_ca_;
  std::string client_cert_;
  std::string client_key_;

 private:
  bool is_json_output_;
};

typedef std::shared_ptr<RdciSubSystem> RdciSubSystemPtr;

}  // namespace rdc
}  // namespace amd

#endif  // RDCI_INCLUDE_RDCISUBSYSTEM_H_
