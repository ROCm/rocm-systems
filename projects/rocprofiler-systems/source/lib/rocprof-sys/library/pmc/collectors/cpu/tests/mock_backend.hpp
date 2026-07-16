// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "backends/procfs/backend.hpp"

#include <gmock/gmock.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>

namespace rocprofsys::backends::procfs::testing
{

static constexpr std::size_t DEFAULT_CPU_COUNT    = 4;
static constexpr std::size_t DEFAULT_SOCKET_COUNT = 2;
static constexpr float       DEFAULT_CPU_FREQ_MHZ = 2000.0F;

static constexpr std::int64_t KB_TO_BYTES = 1024;
static constexpr std::int64_t MB_TO_BYTES = KB_TO_BYTES * KB_TO_BYTES;

static constexpr cpu_jiffies DEFAULT_JIFFIES{ /* user    */ std::uint64_t{ 200 },
                                              /* nice    */ std::uint64_t{ 10 },
                                              /* system  */ std::uint64_t{ 150 },
                                              /* idle    */ std::uint64_t{ 9500 },
                                              /* iowait  */ std::uint64_t{ 50 },
                                              /* irq     */ std::uint64_t{ 30 },
                                              /* softirq */ std::uint64_t{ 60 } };

static constexpr rusage_snapshot DEFAULT_RUSAGE{
    /* page_rss         */ std::int64_t{ 50 } * MB_TO_BYTES,
    /* virt_mem         */ std::int64_t{ 200 } * MB_TO_BYTES,
    /* peak_rss         */ std::int64_t{ 60 } * MB_TO_BYTES,
    /* context_switches */ std::int64_t{ 1000 },
    /* page_faults      */ std::int64_t{ 500 },
    /* user_mode_time   */ std::int64_t{ 5'000'000 },
    /* kernel_mode_time */ std::int64_t{ 1'000'000 },
};

/**
 * @brief Mock implementation of procfs backend for unit testing.
 *
 * Used by device and collector tests to inject synthetic CPU data
 * without touching the filesystem.
 */
class mock_backend
{
public:
    MOCK_METHOD((std::map<size_t, cpu_jiffies>), read_proc_stat, ());
    MOCK_METHOD((std::map<size_t, float>), read_cpu_frequencies, ());
    MOCK_METHOD(rusage_snapshot, read_rusage, ());
    MOCK_METHOD((const socket_topology_t&), get_socket_topology, (), (const));
    MOCK_METHOD(size_t, get_socket_count, (), (const));

    /**
     * @brief Set up default mock behaviors.
     *
     * Configures: 4 CPUs with 2000 MHz, moderate jiffies, basic process usage.
     */
    void set_up_defaults()
    {
        using ::testing::Return;

        std::map<std::size_t, cpu_jiffies> default_jiffies;
        std::map<std::size_t, float>       default_freqs;
        for(std::size_t i = 0; i < DEFAULT_CPU_COUNT; ++i)
        {
            default_jiffies[i] = DEFAULT_JIFFIES;
            default_freqs[i]   = DEFAULT_CPU_FREQ_MHZ;
        }

        ON_CALL(*this, read_proc_stat()).WillByDefault(Return(default_jiffies));
        ON_CALL(*this, read_cpu_frequencies()).WillByDefault(Return(default_freqs));
        ON_CALL(*this, read_rusage()).WillByDefault(Return(DEFAULT_RUSAGE));

        static socket_topology_t default_topology = { { 0, { 0, 1 } }, { 1, { 2, 3 } } };
        ON_CALL(*this, get_socket_topology())
            .WillByDefault(::testing::ReturnRef(default_topology));
        ON_CALL(*this, get_socket_count()).WillByDefault(Return(DEFAULT_SOCKET_COUNT));
    }
};

using strict_mock_backend = ::testing::StrictMock<mock_backend>;

/**
 * @brief Factory for creating and injecting mock backend instances in tests.
 */
struct mock_backend_factory
{
    using backend_t = mock_backend;

    static std::shared_ptr<backend_t> s_mock_backend;

    static std::shared_ptr<backend_t> create_backend(
        [[maybe_unused]] size_t cpu_count = 0)
    {
        return s_mock_backend;
    }

    static void set_mock_backend(std::shared_ptr<backend_t> backend)
    {
        s_mock_backend = std::move(backend);
    }
};

inline std::shared_ptr<mock_backend> mock_backend_factory::s_mock_backend = nullptr;

}  // namespace rocprofsys::backends::procfs::testing
