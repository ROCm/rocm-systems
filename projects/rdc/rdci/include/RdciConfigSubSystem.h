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
#ifndef RDCI_INCLUDE_RDCICONFIGSUBSYSTEM_H_
#define RDCI_INCLUDE_RDCICONFIGSUBSYSTEM_H_

#include "RdciSubSystem.h"

namespace amd {
namespace rdc {

class RdciConfigSubSystem : public RdciSubSystem {
 public:
  RdciConfigSubSystem();
  ~RdciConfigSubSystem() override;
  void parse_cmd_opts(int argc, char** argv) override;
  void process() override;
  typedef enum {
    CONFIG_COMMAND_NONE = 0,
    CONFIG_COMMAND_SET,
    CONFIG_COMMAND_GET,
    CONFIG_COMMAND_CLEAR,
    CONFIG_COMMAND_HELP,
  } config_command_type_t;

 private:
  void show_help() const;
  void display_config_settings(rdc_config_setting_list_t& rdc_configs_list);
  rdc_status_t rdc_group_field_find(rdc_handle_t p_rdc_handle, const char* field_group_name,
                                    rdc_field_grp_t* out_field_group_id);
  config_command_type_t config_cmd_;
  static constexpr rdc_field_grp_t JOB_FIELD_ID = 1;
  uint32_t group_id_;
  uint32_t power_limit_;
  uint64_t gfx_max_clock_;
  uint64_t memory_max_clock_;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDCI_INCLUDE_RDCICONFIGSUBSYSTEM_H_
