/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <gtest/gtest.h>

#include <vector>
#include "core/hardware/hsa_info.h"

TEST(WhenTestingAgentInfoGetterSetters, TestRunsSuccessfully) {
  Agent::AgentInfo agent_info = Agent::AgentInfo();
  char gpu_name[] = "gfx10";
  agent_info.setName(gpu_name);
  agent_info.setIndex(0);
  agent_info.setType(hsa_device_type_t::HSA_DEVICE_TYPE_GPU);

  EXPECT_EQ(agent_info.getName(), gpu_name);
  EXPECT_EQ(agent_info.getIndex(), 0);
  EXPECT_EQ(agent_info.getType(), hsa_device_type_t::HSA_DEVICE_TYPE_GPU);

  Agent::CounterHardwareInfo hw_info(0, "GRBM");
  EXPECT_TRUE(getHardwareInfo(0, "GRBM", &hw_info));
}
