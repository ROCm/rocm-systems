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
#ifndef RDCI_INCLUDE_RDCIDMONSUBSYSTEM_H_
#define RDCI_INCLUDE_RDCIDMONSUBSYSTEM_H_
#include <signal.h>

#include <map>
#include <vector>

#include "RdciSubSystem.h"

namespace amd {
namespace rdc {

class RdciDmonSubSystem : public RdciSubSystem {
 public:
  RdciDmonSubSystem();
  ~RdciDmonSubSystem();
  void parse_cmd_opts(int argc, char** argv) override;
  void process() override;

 private:
  void show_help() const;
  void show_field_usage() const;
  void clean_up();

  // Need to resolve gpu indexes after process is called
  void resolve_gpu_indexes();

  void create_temp_group();
  void create_temp_field_group();

  enum OPERATIONS {
    DMON_UNKNOWN = 0,
    DMON_HELP,
    DMON_LIST_FIELDS,
    DMON_LIST_ALL_FIELDS,
    DMON_MONITOR
  } dmon_ops_;

  enum OPTIONS {
    OPTIONS_UNKNOWN = 0,
    OPTIONS_COUNT,
    OPTIONS_DELAY,
    OPTIONS_FIELD_GROUP_ID,
    OPTIONS_GROUP_ID
  };

  std::map<OPTIONS, uint32_t> options_;
  std::vector<rdc_field_t> field_ids_;
  std::string raw_gpu_indexes_;
  std::vector<uint32_t> gpu_indexes_;
  bool need_cleanup_;
  uint64_t latest_time_stamp_;
  bool show_timpstamps_;
  static volatile sig_atomic_t is_terminating_;
  static void set_terminating(int sig);
};

}  // namespace rdc
}  // namespace amd

#endif  // RDCI_INCLUDE_RDCIDMONSUBSYSTEM_H_
