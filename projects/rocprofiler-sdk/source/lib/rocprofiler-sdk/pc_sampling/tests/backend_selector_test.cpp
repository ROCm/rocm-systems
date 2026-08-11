#include "lib/rocprofiler-sdk/pc_sampling/kernel/selector.hpp"

#include <gtest/gtest.h>

#include <cstdlib>

namespace
{
TEST(pc_sampling_kernel_iface, default_is_kfd)
{
    unsetenv("ROCPROFILER_KERNEL_IFACE");
    EXPECT_EQ(rocprofiler::pc_sampling::kernel::parse_kernel_iface_mode(),
              rocprofiler::pc_sampling::kernel::kernel_iface_mode_t::kfd);
}

TEST(pc_sampling_kernel_iface, parse_kgd)
{
    setenv("ROCPROFILER_KERNEL_IFACE", "kgd", 1);
    EXPECT_EQ(rocprofiler::pc_sampling::kernel::parse_kernel_iface_mode(),
              rocprofiler::pc_sampling::kernel::kernel_iface_mode_t::kgd);
    unsetenv("ROCPROFILER_KERNEL_IFACE");
}

TEST(pc_sampling_kernel_iface, parse_auto)
{
    setenv("ROCPROFILER_KERNEL_IFACE", "auto", 1);
    EXPECT_EQ(rocprofiler::pc_sampling::kernel::parse_kernel_iface_mode(),
              rocprofiler::pc_sampling::kernel::kernel_iface_mode_t::auto_select);
    unsetenv("ROCPROFILER_KERNEL_IFACE");
}
}  // namespace
