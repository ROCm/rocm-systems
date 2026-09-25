// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Unit tests for discover_stream_dispatch_log_gpus: support gate is the GFX12-only
// dispatch_log_stream_format node (retired GFX9 dispatch_log_format must NOT count).
#include <gtest/gtest.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <cstdio>
#include <string>
#include "lib/rocprofiler-sdk/kfd/kfd_profiler.hpp"
namespace
{
using rocprofiler::kfd::discover_stream_dispatch_log_gpus;
struct fake_topology  // owns a throwaway "<root>/nodes" tree; rm -rf on destruction
{
    std::string root, nodes;
    fake_topology()
    {
        char tmpl[] = "/tmp/kfd_topo_test_XXXXXX";
        root        = mkdtemp(tmpl) ? tmpl : "";
        EXPECT_FALSE(root.empty()) << "mkdtemp failed";
        nodes = root + "/nodes";
        mkdir(nodes.c_str(), 0700);
    }
    ~fake_topology()
    {
        if(!root.empty())
        {
            [[maybe_unused]] auto _rc = system(("rm -rf '" + root + "'").c_str());
        }
    }
    fake_topology(const fake_topology&) = delete;
    fake_topology& operator=(const fake_topology&) = delete;
    fake_topology(fake_topology&&)                 = delete;
    fake_topology& operator=(fake_topology&&) = delete;
    // gpu_id 0 = CPU-only node; flags create the legacy/stream format nodes.
    void add_node(const std::string& name, uint32_t gpu_id, bool has_legacy, bool has_stream) const
    {
        std::string dir = nodes + "/" + name;
        mkdir(dir.c_str(), 0700);
        write_file(dir + "/gpu_id", std::to_string(gpu_id));
        if(has_legacy) write_file(dir + "/dispatch_log_format", "legacy");
        if(has_stream) write_file(dir + "/dispatch_log_stream_format", "stream");
    }
    static void write_file(const std::string& path, const std::string& body)
    {
        FILE* f = fopen(path.c_str(), "w");
        ASSERT_NE(f, nullptr) << "fopen " << path;
        fwrite(body.data(), 1, body.size(), f);
        fclose(f);
    }
};
}  // namespace
// CPU (gpu_id 0) skipped even w/ stream node; GFX9 legacy-only rejected; stream/both accepted.
TEST(TopologyDiscovery, discovers_only_stream_capable_gpus)
{
    struct node
    {
        const char* name;
        uint32_t    gpu_id;
        bool        legacy, stream, supported;
        const char* label;
    };
    const node    rows[] = {{"0", 0, false, true, false, "CPU with stream is skipped"},
                         {"1", 0x1111, true, false, false, "GFX9 legacy-only"},
                         {"3", 0x2222, true, true, true, "both formats"},
                         {"4", 0x3333, false, true, true, "stream-only"}};
    fake_topology topo;
    for(const auto& r : rows)
        topo.add_node(r.name, r.gpu_id, r.legacy, r.stream);
    auto ids = discover_stream_dispatch_log_gpus(topo.nodes.c_str());
    EXPECT_EQ(ids.size(), 2u);
    for(const auto& r : rows)
        EXPECT_EQ(ids.count(r.gpu_id) == 1, r.supported) << r.label;
}
// Missing topology root -> empty (falls back to HSA timestamps).
TEST(TopologyDiscovery, missing_topology_root_yields_empty)
{
    auto ids = discover_stream_dispatch_log_gpus("/nonexistent/kfd/topology/nodes");
    EXPECT_TRUE(ids.empty());
}
