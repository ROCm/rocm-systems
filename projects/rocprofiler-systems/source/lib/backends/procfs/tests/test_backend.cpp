// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "backends/procfs/backend.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

using namespace rocprofsys::backends::procfs;

class procfs_backend_test : public ::testing::Test
{
protected:
    size_t cpu_count = static_cast<size_t>(std::max(0L, sysconf(_SC_NPROCESSORS_ONLN)));
};

TEST_F(procfs_backend_test, reads_proc_stat)
{
    backend drv(cpu_count);

    auto jiffies = drv.read_proc_stat();

    EXPECT_FALSE(jiffies.empty());
    for(const auto& [cpu_id, data] : jiffies)
    {
        EXPECT_GT(data.total(), 0u);
    }
}

TEST_F(procfs_backend_test, reads_rusage)
{
    backend drv(cpu_count);

    auto snap = drv.read_rusage();

    EXPECT_GT(snap.page_rss, 0);
    EXPECT_GT(snap.virt_mem, 0);
}

TEST_F(procfs_backend_test, socket_topology_nonempty)
{
    backend drv(cpu_count);

    const auto& topology = drv.get_socket_topology();

    EXPECT_FALSE(topology.empty());
    EXPECT_GE(drv.get_socket_count(), 1u);

    size_t total_cpus = 0;
    for(const auto& [socket_id, cpus] : topology)
    {
        total_cpus += cpus.size();
    }
    EXPECT_EQ(total_cpus, cpu_count);
}

TEST_F(procfs_backend_test, repeated_reads_return_valid_data)
{
    backend drv(cpu_count);

    auto first  = drv.read_proc_stat();
    auto second = drv.read_proc_stat();

    EXPECT_FALSE(first.empty());
    EXPECT_FALSE(second.empty());
    EXPECT_EQ(first.size(), second.size());

    for(const auto& [cpu_id, data] : second)
    {
        EXPECT_GE(data.total(), first.at(cpu_id).total());
    }
}

TEST_F(procfs_backend_test, large_cpu_count_does_not_crash)
{
    EXPECT_NO_THROW({ backend drv(1024); });
}
