/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef RDCI_INCLUDE_RDCIHEALTHSUBSYSTEM_H_
#define RDCI_INCLUDE_RDCIHEALTHSUBSYSTEM_H_
#include <signal.h>

#include <string>

#include "RdciSubSystem.h"

namespace amd {
namespace rdc {

class RdciHealthSubSystem : public RdciSubSystem {
 public:
  RdciHealthSubSystem();
  ~RdciHealthSubSystem();
  void parse_cmd_opts(int argc, char** argv) override;
  void process() override;

 private:
  void show_help() const;

  void get_watches() const;
  void set_watches() const;
  void health_check() const;
  void health_clear() const;

  std::string health_string(rdc_health_result_t health) const;
  std::string component_string(rdc_health_system_t component) const;
  void output_errstr(const std::string& input) const;
  unsigned int handle_one_component(rdc_health_response_t &response,
                                    unsigned int start_index,
                                    uint32_t gpu_index,
                                    rdc_health_system_t component,
                                    rdc_health_result_t &component_health,
                                    std::vector<std::string> &err_str) const;
  unsigned int handle_one_gpu(rdc_health_response_t &response,
                              unsigned int start_index,
                              uint32_t gpu_index) const;

  enum OPERATIONS {
    HEALTH_UNKNOWN = 0,
    HEALTH_HELP,
    HEALTH_FETCH,
    HEALTH_SET,
    HEALTH_CHECK,
    HEALTH_CLEAR,
  } health_ops_;

  rdc_gpu_group_t      group_id_;
  unsigned int         components_;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDCI_INCLUDE_RDCIHEALTHSUBSYSTEM_H_
