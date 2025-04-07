// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/fwd.h>

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/types.hpp"

#include <vector>

namespace rocprofiler
{
namespace pc_sampling
{
namespace ioctl
{
using rocp_pcs_cfgs_vec_t = std::vector<rocprofiler_pc_sampling_configuration_t>;

rocprofiler_status_t
ioctl_query_pcs_configs(const rocprofiler_agent_t* agent, rocp_pcs_cfgs_vec_t& rocp_configs);

rocprofiler_status_t
ioctl_pcs_create(const rocprofiler_agent_t*       agent,
                 rocprofiler_pc_sampling_method_t method,
                 rocprofiler_pc_sampling_unit_t   unit,
                 uint64_t                         interval,
                 uint32_t*                        ioctl_pcs_id);

int
get_kfd_fd();
}  // namespace ioctl
}  // namespace pc_sampling
}  // namespace rocprofiler
