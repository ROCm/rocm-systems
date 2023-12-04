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
#ifndef RDCI_INCLUDE_RDCISTATSSUBSYSTEM_H_
#define RDCI_INCLUDE_RDCISTATSSUBSYSTEM_H_
#include <signal.h>

#include <string>

#include "RdciSubSystem.h"

namespace amd {
namespace rdc {

class RdciStatsSubSystem : public RdciSubSystem {
 public:
  RdciStatsSubSystem();
  ~RdciStatsSubSystem();
  void parse_cmd_opts(int argc, char** argv) override;
  void process() override;

 private:
  void show_help() const;
  void show_job_stats(const rdc_gpu_usage_info_t& gpu_info) const;
  void show_job_stats_json(const rdc_gpu_usage_info_t& gpu_info) const;

  enum OPERATIONS {
    STATS_UNKNOWN = 0,
    STATS_HELP,
    STATS_START_RECORDING,
    STATS_STOP_RECORDING,
    STATS_DISPLAY,
    STATS_REMOVE,
    STATS_REMOVE_ALL
  } stats_ops_;

  std::string job_id_;
  uint32_t group_id_;
  bool is_verbose_ = false;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDCI_INCLUDE_RDCISTATSSUBSYSTEM_H_
