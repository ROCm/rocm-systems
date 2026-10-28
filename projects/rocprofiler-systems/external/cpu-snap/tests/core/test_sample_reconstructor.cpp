// Unit tests for cpu_snap::core::sample_reconstructor.

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <vector>

#include "cpu_snap/core/chunk_framer.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"

namespace
{

using cpu_snap::core::chunk_framer;
using cpu_snap::core::sample_reconstructor;

/**
 * Encodes a small known stream: two nodes forming a 2-frame stack and
 * one node forming a 1-frame stack, three SAMPLE records split across
 * two tids, a MAPS_INIT with one module, and a MAPS_DELTA adding a
 * second module partway through.
 */
std::vector<std::byte>
build_known_stream()
{
    std::vector<std::byte> bytes;
    auto                   sink = [&bytes](std::span<const std::byte> chunk) {
        bytes.insert(bytes.end(), chunk.begin(), chunk.end());
    };

    chunk_framer::config cfg;
    cfg.pid           = 4242;
    cfg.start_unix_ns = 1000;
    cfg.process_name  = "sample_reconstructor_test";

    chunk_framer framer(cfg, sink);
    framer.start();

    cpu_snap::core::maps_init_record init;
    init.wall_ns = 50;
    cpu_snap::core::module_info mod_a;
    mod_a.base_va = 0x1000;
    mod_a.size    = 0x100;
    mod_a.path    = "/lib/a.so";
    init.modules  = { mod_a };
    framer.write(init);

    cpu_snap::core::new_node_record node1;
    node1.node_id   = 1;
    node1.parent_id = 0;
    node1.ip        = 0x1010;
    framer.write(node1);

    cpu_snap::core::new_node_record node2;
    node2.node_id   = 2;
    node2.parent_id = 1;
    node2.ip        = 0x1020;
    framer.write(node2);

    cpu_snap::core::sample_record sample_tid100_a;
    sample_tid100_a.timestamp_ns = 1000;
    sample_tid100_a.tid          = 100;
    sample_tid100_a.leaf_node_id = 2;
    framer.write(sample_tid100_a);

    cpu_snap::core::maps_delta_record delta;
    delta.wall_ns   = 1500;
    delta.operation = cpu_snap::core::maps_delta_record::op::add;
    delta.module.base_va = 0x2000;
    delta.module.size    = 0x100;
    delta.module.path    = "/lib/b.so";
    framer.write(delta);

    cpu_snap::core::new_node_record node3;
    node3.node_id   = 3;
    node3.parent_id = 0;
    node3.ip        = 0x2010;
    framer.write(node3);

    cpu_snap::core::sample_record sample_tid200;
    sample_tid200.timestamp_ns = 2000;
    sample_tid200.tid          = 200;
    sample_tid200.leaf_node_id = 3;
    framer.write(sample_tid200);

    cpu_snap::core::sample_record sample_tid100_b;
    sample_tid100_b.timestamp_ns = 3000;
    sample_tid100_b.tid          = 100;
    sample_tid100_b.leaf_node_id = 1;
    framer.write(sample_tid100_b);

    framer.flush();
    return bytes;
}

}  // namespace

TEST(SampleReconstructor, RebuildsPerTidSamplesInEncodedOrder)
{
    const std::vector<std::byte> bytes = build_known_stream();

    sample_reconstructor reconstructor;
    reconstructor.feed(bytes);

    const auto& per_tid = reconstructor.samples();
    ASSERT_EQ(per_tid.count(100), 1U);
    ASSERT_EQ(per_tid.count(200), 1U);

    const auto& tid100 = per_tid.at(100);
    ASSERT_EQ(tid100.size(), 2U);

    EXPECT_EQ(tid100[0].timestamp_ns, 1000U);
    EXPECT_EQ(tid100[0].ip, 0x1020U);
    ASSERT_EQ(tid100[0].frames.size(), 2U);
    EXPECT_EQ(tid100[0].frames[0], 0x1020U);
    EXPECT_EQ(tid100[0].frames[1], 0x1010U);

    EXPECT_EQ(tid100[1].timestamp_ns, 3000U);
    EXPECT_EQ(tid100[1].ip, 0x1010U);
    ASSERT_EQ(tid100[1].frames.size(), 1U);
    EXPECT_EQ(tid100[1].frames[0], 0x1010U);

    const auto& tid200 = per_tid.at(200);
    ASSERT_EQ(tid200.size(), 1U);
    EXPECT_EQ(tid200[0].timestamp_ns, 2000U);
    EXPECT_EQ(tid200[0].ip, 0x2010U);

    EXPECT_EQ(reconstructor.counters().parse_errors, 0U);
}

TEST(SampleReconstructor, RebuildsModuleMapFromInitAndDelta)
{
    const std::vector<std::byte> bytes = build_known_stream();

    sample_reconstructor reconstructor;
    reconstructor.feed(bytes);

    const auto& modules = reconstructor.modules();

    const auto module_a = modules.resolve(0x1050, 1000);
    ASSERT_TRUE(module_a.has_value());
    EXPECT_EQ(module_a->path, "/lib/a.so");

    EXPECT_FALSE(modules.resolve(0x2050, 1000).has_value())
        << "module B is not added until wall_ns=1500";

    const auto module_b = modules.resolve(0x2050, 2000);
    ASSERT_TRUE(module_b.has_value());
    EXPECT_EQ(module_b->path, "/lib/b.so");
}

TEST(SampleReconstructor, FeedCanBeCalledIncrementally)
{
    const std::vector<std::byte> bytes = build_known_stream();
    ASSERT_GT(bytes.size(), 8U);
    const std::size_t split = bytes.size() / 2;

    sample_reconstructor reconstructor;
    reconstructor.feed(std::span<const std::byte>(bytes.data(), split));
    reconstructor.feed(std::span<const std::byte>(bytes.data() + split, bytes.size() - split));

    const auto& per_tid = reconstructor.samples();
    ASSERT_EQ(per_tid.count(100), 1U);
    EXPECT_EQ(per_tid.at(100).size(), 2U);
    ASSERT_EQ(per_tid.count(200), 1U);
    EXPECT_EQ(per_tid.at(200).size(), 1U);
}

TEST(SampleReconstructor, CountsSamplesWithUnresolvableStack)
{
    std::vector<std::byte> bytes;
    chunk_framer           framer(chunk_framer::config{},
                        [&bytes](std::span<const std::byte> chunk) {
                            bytes.insert(bytes.end(), chunk.begin(), chunk.end());
                        });
    framer.start();

    cpu_snap::core::new_node_record node;
    node.node_id = 1;
    node.ip      = 0x1010;
    framer.write(node);

    cpu_snap::core::sample_record known;
    known.tid          = 100;
    known.leaf_node_id = 1;
    framer.write(known);

    cpu_snap::core::sample_record unknown;
    unknown.tid          = 100;
    unknown.leaf_node_id = 99;
    framer.write(unknown);
    framer.flush();

    sample_reconstructor reconstructor;
    reconstructor.feed(bytes);

    EXPECT_EQ(reconstructor.samples().at(100).size(), 1U);
    EXPECT_EQ(reconstructor.unresolved_samples(), 1U);
}
