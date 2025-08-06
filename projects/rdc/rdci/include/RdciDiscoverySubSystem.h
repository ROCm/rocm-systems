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
#ifndef RDCI_INCLUDE_RDCIDISCOVERYSUBSYSTEM_H_
#define RDCI_INCLUDE_RDCIDISCOVERYSUBSYSTEM_H_

#include "RdciSubSystem.h"

namespace amd {
namespace rdc {

class RdciDiscoverySubSystem : public RdciSubSystem {
 public:
  RdciDiscoverySubSystem();
  void parse_cmd_opts(int argc, char** argv) override;
  void process() override;

 private:
  bool show_help_;
  void show_help() const;
  bool is_list_;
  bool is_partition_;
  void show_attributes();
  void show_attributes_with_partitions();
  bool show_version_;
  void show_version();
};

}  // namespace rdc
}  // namespace amd

#endif  // RDCI_INCLUDE_RDCIDISCOVERYSUBSYSTEM_H_
