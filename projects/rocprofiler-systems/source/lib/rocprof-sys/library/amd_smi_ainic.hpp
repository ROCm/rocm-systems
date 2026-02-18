// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/defines.hpp"
#include "core/gpu_metrics.hpp"
#include "core/state.hpp"
#include "library/thread_data.hpp"

#if ROCPROFSYS_USE_ROCM > 0
#    include "core/amd_smi.hpp"
#    include <amd_smi/amdsmi.h>
#endif

#include <chrono>
#include <cstdint>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <ratio>
#include <thread>
#include <tuple>
#include <type_traits>

#include "library/ainic_stats.hpp"

namespace rocprofsys
{
namespace amd_smi
{

void
nic_setup();

void
nic_config();

void
nic_sample();

struct nic_data
{
    using timestamp_t = int64_t;

    explicit nic_data(uint32_t nic_index, const std::string& nic);

    static std::vector<nic_data>&   get_initial();
    static std::vector<std::string> nic_vec;
    const std::string&              get_nic() const;
    static bool                     setup();

    void sample();

    static void post_process(size_t nic_index);

    static ai_nic_stats_collector nic_stats_collector;

    timestamp_t m_ts = 0;

private:
    std::string _nic;
    uint32_t    _nic_index        = 0;
    uint64_t    _rx_rdma_cnp_pkts = 0;
    uint64_t    _tx_rdma_cnp_pkts = 0;
    uint64_t    _rx_ucast_bytes   = 0;
    uint64_t    _tx_ucast_bytes   = 0;
    uint64_t    _rx_ucast_pkts    = 0;
    uint64_t    _tx_ucast_pkts    = 0;
};

}  // namespace amd_smi

}  // namespace rocprofsys
