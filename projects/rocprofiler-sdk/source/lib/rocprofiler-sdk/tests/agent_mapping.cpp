// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Pairing of rocprofiler agent records with HSA agents. Pure integer matching:
// no HSA runtime, no DXG thunk and no GPU is involved, so this runs anywhere.

#include "lib/rocprofiler-sdk/agent_mapping.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace
{
using rocprofiler::agent::compute_agent_mapping;
using rocprofiler::agent::mapping_agent_view;
using rocprofiler::agent::mapping_hsa_view;
using rocprofiler::agent::mapping_policy;

mapping_agent_view
rocp_cpu(int64_t logical_node_id, uint32_t node_id)
{
    return mapping_agent_view{logical_node_id, node_id, false};
}

mapping_agent_view
rocp_gpu(int64_t logical_node_id, uint32_t node_id)
{
    return mapping_agent_view{logical_node_id, node_id, true};
}

mapping_hsa_view
hsa_cpu(uint32_t driver_node_id)
{
    return mapping_hsa_view{driver_node_id, true, false};
}

mapping_hsa_view
hsa_gpu(uint32_t driver_node_id)
{
    return mapping_hsa_view{driver_node_id, true, true};
}
}  // namespace

// --- bare metal ------------------------------------------------------------

// On bare metal rocprofiler and HSA both walk the KFD sysfs tree, so the dense
// logical ordinal and the driver node id agree and every agent pairs up.
TEST(agent_mapping, bare_metal_maps_every_agent)
{
    const auto rocp =
        std::vector<mapping_agent_view>{rocp_cpu(0, 0), rocp_gpu(1, 1), rocp_gpu(2, 2)};
    const auto hsa = std::vector<mapping_hsa_view>{hsa_cpu(0), hsa_gpu(1), hsa_gpu(2)};

    const auto mapping = compute_agent_mapping(rocp, hsa, mapping_policy::strict);

    ASSERT_EQ(mapping.pairs.size(), 3);
    EXPECT_TRUE(mapping.complete());
    for(const auto& itr : mapping.pairs)
    {
        EXPECT_EQ(itr.rocp_index, itr.hsa_index);
        EXPECT_EQ(itr.key, hsa.at(itr.hsa_index).driver_node_id);
    }
}

// The strict key is rocprofiler's logical ordinal, not node_id. Preserved
// verbatim from the pre-WSL behaviour, so pin it down: a record whose node_id
// disagrees with its logical ordinal must still match on the logical ordinal.
TEST(agent_mapping, bare_metal_keys_on_the_logical_ordinal)
{
    const auto rocp = std::vector<mapping_agent_view>{rocp_cpu(0, 90), rocp_gpu(1, 91)};
    const auto hsa  = std::vector<mapping_hsa_view>{hsa_cpu(0), hsa_gpu(1)};

    const auto mapping = compute_agent_mapping(rocp, hsa, mapping_policy::strict);

    ASSERT_EQ(mapping.pairs.size(), 2);
    EXPECT_TRUE(mapping.complete());
}

// An HSA GPU with no rocprofiler counterpart is an internal inconsistency on
// bare metal; the caller turns an incomplete mapping into a fatal.
TEST(agent_mapping, bare_metal_reports_an_unmatched_hsa_gpu)
{
    const auto rocp = std::vector<mapping_agent_view>{rocp_cpu(0, 0), rocp_gpu(1, 1)};
    const auto hsa  = std::vector<mapping_hsa_view>{hsa_cpu(0), hsa_gpu(1), hsa_gpu(2)};

    const auto mapping = compute_agent_mapping(rocp, hsa, mapping_policy::strict);

    EXPECT_EQ(mapping.pairs.size(), 2);
    EXPECT_FALSE(mapping.complete());
    ASSERT_EQ(mapping.unmatched_gpu_node_ids.size(), 1);
    EXPECT_EQ(mapping.unmatched_gpu_node_ids.front(), 2);
}

// --- WSL -------------------------------------------------------------------

// The blocker case: the DXG thunk described one of the two adapters well
// enough to publish, HSA reports both. The published GPU must still pair, and
// the other must surface as unmatched rather than derailing the whole mapping.
TEST(agent_mapping, wsl_publishes_one_of_two_gpus_without_losing_the_other)
{
    const auto rocp = std::vector<mapping_agent_view>{rocp_cpu(0, 0), rocp_gpu(1, 2)};
    const auto hsa  = std::vector<mapping_hsa_view>{hsa_cpu(0), hsa_gpu(1), hsa_gpu(2)};

    const auto mapping = compute_agent_mapping(rocp, hsa, mapping_policy::wsl);

    ASSERT_EQ(mapping.pairs.size(), 2);
    EXPECT_FALSE(mapping.complete());

    // the CPU and the GPU that KMT node 2 describes
    EXPECT_EQ(mapping.pairs.at(0).key, 0);
    EXPECT_EQ(mapping.pairs.at(1).key, 2);
    EXPECT_EQ(mapping.pairs.at(1).rocp_index, 1);
    EXPECT_EQ(mapping.pairs.at(1).hsa_index, 2);

    ASSERT_EQ(mapping.unmatched_gpu_node_ids.size(), 1);
    EXPECT_EQ(mapping.unmatched_gpu_node_ids.front(), 1);
    EXPECT_TRUE(mapping.unmatched_other_node_ids.empty());
}

// A dense-ordinal match would have paired the published GPU with the wrong HSA
// agent above; make the difference explicit.
TEST(agent_mapping, wsl_keys_on_the_kmt_node_id_not_the_dense_ordinal)
{
    const auto rocp =
        std::vector<mapping_agent_view>{rocp_cpu(0, 0), rocp_gpu(1, 3), rocp_gpu(2, 7)};
    const auto hsa = std::vector<mapping_hsa_view>{hsa_cpu(0), hsa_gpu(3), hsa_gpu(7)};

    const auto mapping = compute_agent_mapping(rocp, hsa, mapping_policy::wsl);

    ASSERT_EQ(mapping.pairs.size(), 3);
    EXPECT_TRUE(mapping.complete());
    EXPECT_EQ(mapping.pairs.at(1).key, 3);
    EXPECT_EQ(mapping.pairs.at(2).key, 7);

    // the same input under the bare-metal key finds neither GPU
    const auto strict = compute_agent_mapping(rocp, hsa, mapping_policy::strict);
    EXPECT_EQ(strict.pairs.size(), 1);
    EXPECT_EQ(strict.unmatched_gpu_node_ids.size(), 2);
}

// An old thunk publishes no GPU at all. Every HSA GPU is unmatched, the CPU
// still pairs and nothing about that is fatal.
TEST(agent_mapping, wsl_publishes_no_gpu_at_all)
{
    const auto rocp = std::vector<mapping_agent_view>{rocp_cpu(0, 0)};
    const auto hsa  = std::vector<mapping_hsa_view>{hsa_cpu(0), hsa_gpu(1)};

    const auto mapping = compute_agent_mapping(rocp, hsa, mapping_policy::wsl);

    ASSERT_EQ(mapping.pairs.size(), 1);
    EXPECT_EQ(mapping.pairs.front().key, 0);
    ASSERT_EQ(mapping.unmatched_gpu_node_ids.size(), 1);
    EXPECT_EQ(mapping.unmatched_gpu_node_ids.front(), 1);
}

// --- shared invariants -----------------------------------------------------

// Two HSA agents reporting the same driver node id must not both alias onto
// the one rocprofiler record: the second stays unmatched.
TEST(agent_mapping, a_rocprofiler_agent_is_claimed_at_most_once)
{
    const auto rocp = std::vector<mapping_agent_view>{rocp_gpu(1, 1)};
    const auto hsa  = std::vector<mapping_hsa_view>{hsa_gpu(1), hsa_gpu(1)};

    for(auto policy : {mapping_policy::strict, mapping_policy::wsl})
    {
        const auto mapping = compute_agent_mapping(rocp, hsa, policy);
        ASSERT_EQ(mapping.pairs.size(), 1);
        EXPECT_EQ(mapping.pairs.front().hsa_index, 0);
        EXPECT_EQ(mapping.unmatched_gpu_node_ids.size(), 1);
        EXPECT_FALSE(mapping.complete());
    }
}

// An HSA agent whose driver node id could not be read is unpairable under
// every policy and is reported separately from a genuine mismatch.
TEST(agent_mapping, an_unqueryable_hsa_agent_is_counted_not_matched)
{
    const auto rocp = std::vector<mapping_agent_view>{rocp_gpu(0, 0)};
    const auto hsa  = std::vector<mapping_hsa_view>{mapping_hsa_view{0, false, true}};

    for(auto policy : {mapping_policy::strict, mapping_policy::wsl})
    {
        const auto mapping = compute_agent_mapping(rocp, hsa, policy);
        EXPECT_TRUE(mapping.pairs.empty());
        EXPECT_EQ(mapping.unqueryable_count, 1);
        EXPECT_TRUE(mapping.unmatched_gpu_node_ids.empty());
        EXPECT_FALSE(mapping.complete());
    }
}

// A rocprofiler agent HSA does not report is not an error in either direction:
// nothing is unmatched because the mapping is driven by the HSA list.
TEST(agent_mapping, a_surplus_rocprofiler_agent_is_not_a_mismatch)
{
    const auto rocp = std::vector<mapping_agent_view>{rocp_cpu(0, 0), rocp_gpu(1, 1)};
    const auto hsa  = std::vector<mapping_hsa_view>{hsa_cpu(0)};

    for(auto policy : {mapping_policy::strict, mapping_policy::wsl})
    {
        const auto mapping = compute_agent_mapping(rocp, hsa, policy);
        EXPECT_EQ(mapping.pairs.size(), 1);
        EXPECT_TRUE(mapping.complete());
    }
}

TEST(agent_mapping, empty_inputs_are_a_complete_mapping)
{
    const auto mapping = compute_agent_mapping({}, {}, mapping_policy::strict);
    EXPECT_TRUE(mapping.pairs.empty());
    EXPECT_TRUE(mapping.complete());
}
