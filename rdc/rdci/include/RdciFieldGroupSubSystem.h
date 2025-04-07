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
#ifndef RDCI_INCLUDE_RDCIFIELDGROUPSUBSYSTEM_H_
#define RDCI_INCLUDE_RDCIFIELDGROUPSUBSYSTEM_H_

#include <string>

#include "RdciSubSystem.h"

namespace amd {
namespace rdc {

class RdciFieldGroupSubSystem : public RdciSubSystem {
 public:
  RdciFieldGroupSubSystem();
  void parse_cmd_opts(int argc, char** argv) override;
  void process() override;

 private:
  void show_help() const;

  enum OPERATIONS {
    FIELD_GROUP_UNKNOWN = 0,
    FIELD_GROUP_HELP,
    FIELD_GROUP_CREATE,
    FIELD_GROUP_DELETE,
    FIELD_GROUP_LIST,
    FIELD_GROUP_INFO
  } field_group_ops_;

  bool is_group_set_;
  uint32_t group_id_;
  std::string group_name_;
  std::string field_ids_;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDCI_INCLUDE_RDCIFIELDGROUPSUBSYSTEM_H_
