// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Unit tests for unified_memory_processor_t.
//
// The processor is templated on its agent_manager and output_file_registry
// collaborators (defaults bind to the production types). These tests
// instantiate the alternative specialization
//   unified_memory_processor_t<MockAgentManager, MockOutputFileRegistry>
// so that collaborator interactions can be verified with GMock and the
// processor body runs without any KFD/HSA/rocprofiler-sdk dependencies.
//
// Coverage rationale: the production class is otherwise only exercised by
// integration tests that require XNACK-capable hardware (gfx950 MI300X / MI325X).
// These tests lock in the behavior of the Copilot review pass fixes so a
// future refactor cannot silently regress them.
//

// Include the implementation directly so the mock specialization
// unified_memory_processor_t<MockAgentManager, MockOutputFileRegistry>
// can be instantiated below.
#include "core/trace_cache/unified_memory_processor.inl"
#include "filesystem.hpp"
#include "mock_unified_memory_collaborators.hpp"

#include <nlohmann/json.hpp>
#include <timemory/settings.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using rocprofsys::agent;
using rocprofsys::agent_type;
using rocprofsys::output_format;
using rocprofsys::trace_cache::kfd_sample;
using rocprofsys::trace_cache::migration_stats;
using rocprofsys::trace_cache::unified_memory_processor_t;
using rocprofsys::trace_cache::detail::kTriggerTable;
using rocprofsys::trace_cache::test::make_cpu_agent;
using rocprofsys::trace_cache::test::make_gpu_agent;
using rocprofsys::trace_cache::test::make_kfd_page_fault_sample;
using rocprofsys::trace_cache::test::make_kfd_page_migrate_sample;
using rocprofsys::trace_cache::test::make_kfd_page_migrate_sample_raw_args;
using rocprofsys::trace_cache::test::MockAgentManager;
using rocprofsys::trace_cache::test::MockOutputFileRegistry;

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::Throw;

// Explicit instantiation of the mock specialization. The .inl above provides
// the method bodies; this gives us a single set of out-of-line definitions
// that the rest of the test TU can link against.
namespace rocprofsys
{
namespace trace_cache
{
template class unified_memory_processor_t<test::MockAgentManager,
                                          test::MockOutputFileRegistry>;
}  // namespace trace_cache
}  // namespace rocprofsys

namespace
{

using TestProcessor =
    unified_memory_processor_t<MockAgentManager, MockOutputFileRegistry>;

// Captures every register_file(path, format) call made by finalize_processing
// so individual tests can inspect what was registered (path + format).
struct registered_entry
{
    std::string   path;
    output_format format;
};

// RAII guard: snapshot tim::settings::use_output_suffix() on construction,
// force-enable it for the test body, restore on destruction. Exception-safe.
// Held by the fixture as unique_ptr, so copy/move are never reached.
struct ScopedUseOutputSuffix
{
    const bool previous;
    explicit ScopedUseOutputSuffix(bool desired)
    : previous(tim::settings::use_output_suffix())
    {
        tim::settings::use_output_suffix() = desired;
    }
    ~ScopedUseOutputSuffix() { tim::settings::use_output_suffix() = previous; }
};

class UnifiedMemoryProcessorTest : public ::testing::Test
{
protected:
    static constexpr int      kPid  = 12345;
    static constexpr uint32_t kCpu0 = 0;
    static constexpr uint32_t kGpu1 = 1;
    static constexpr uint32_t kGpu2 = 2;

    void SetUp() override
    {
        // The processor relies on timemory's use_output_suffix() setting to
        // decide whether the tag (PID) gets stitched into the output path.
        // ScopedUseOutputSuffix snapshots and restores it exception-safely.
        m_suffix_guard = std::make_unique<ScopedUseOutputSuffix>(true);

        // Snapshot HSA_XNACK at the same point the processor's constructor
        // reads it (a few lines below) so the assertion in JsonReportsXnackFlag
        // compares against the same value the processor captured — not a
        // fresh re-read in the test body, which could observe a different
        // value if anything mutated the env between SetUp and the test body.
        // Mirrors the production idiom in unified_memory_processor.inl ctor.
        const char* xnack      = std::getenv("HSA_XNACK");
        expected_xnack_enabled = (xnack != nullptr && std::strcmp(xnack, "1") == 0);

        // CPU at node 0, two GPUs at nodes 1 and 2.
        agents = { make_cpu_agent(kCpu0, "AMD CPU"), make_gpu_agent(kGpu1, "gfx950"),
                   make_gpu_agent(kGpu2, "gfx950") };

        agent_mgr = std::make_shared<NiceMock<MockAgentManager>>();
        ON_CALL(*agent_mgr, get_agents()).WillByDefault(Return(agents));

        // Default the CPU lookup to return the CPU agent. Individual tests
        // can override this with EXPECT_CALL.
        ON_CALL(*agent_mgr, get_agent_by_type_index(_, agent_type::CPU))
            .WillByDefault(
                [this](size_t, agent_type) -> const agent& { return *agents[0]; });

        registry = std::make_unique<NiceMock<MockOutputFileRegistry>>();
        ON_CALL(*registry, register_file(_, _))
            .WillByDefault([this](std::string p, output_format f) {
                registered.push_back({ std::move(p), f });
            });

        // Unique tmp dir so parallel test invocations don't collide.
        char        tmpl[] = "/tmp/rocprofsys_um_test_XXXXXX";
        const char* d      = mkdtemp(tmpl);
        ASSERT_NE(d, nullptr) << "mkdtemp failed";
        tmp_dir = d;

        processor = std::make_unique<TestProcessor>(agent_mgr, kPid, tmp_dir, *registry);
    }

    void TearDown() override
    {
        processor.reset();
        if(!tmp_dir.empty())
        {
            std::error_code ec;
            test_common::fs::remove_all(tmp_dir, ec);
        }
        // m_suffix_guard is restored by the fixture destructor (RAII).
    }

    // Locate the JSON file that finalize_processing() registered, parse it,
    // and return the resulting nlohmann::json. Returns std::nullopt (and
    // marks the test as failed via ADD_FAILURE) if no JSON file was
    // registered, it could not be opened, or parsing threw — callers should
    // guard with ASSERT_TRUE(j.has_value()) before dereferencing.
    std::optional<nlohmann::json> read_json_output() const
    {
        std::string json_path;
        for(const auto& e : registered)
            if(e.format == output_format::json) json_path = e.path;
        if(json_path.empty())
        {
            ADD_FAILURE() << "JSON file was not registered";
            return std::nullopt;
        }
        std::ifstream f(json_path);
        if(!f.is_open())
        {
            ADD_FAILURE() << "JSON file missing on disk: " << json_path;
            return std::nullopt;
        }
        // allow_exceptions=false → parse errors yield a discarded value
        // instead of throwing.
        auto j = nlohmann::json::parse(f, /*cb=*/nullptr, /*allow_exceptions=*/false);
        if(j.is_discarded())
        {
            ADD_FAILURE() << "JSON parse failed for " << json_path;
            return std::nullopt;
        }
        return j;
    }

    // Feed a single H→D page-migrate sample (kCpu0 → kGpu1, device_id=0)
    // whose payload size (.value field) is overridden to v. Used by the
    // float-sanitization tests to probe the size-cast guard without re-stating
    // the migrate-builder boilerplate. Direction is in the name so callers
    // know which JSON bucket ("host_to_device") receives the recorded entry.
    void feed_h2d_migrate_with_value(double v)
    {
        auto s  = make_kfd_page_migrate_sample(kCpu0, kGpu1, /*size=*/0,
                                               /*duration=*/100, /*device_id=*/0);
        s.value = v;
        // handle() never throws on this path (the size guard rejects without
        // throwing); call it directly rather than wrapping in EXPECT_NO_THROW.
        processor->handle(s);
    }

    std::shared_ptr<NiceMock<MockAgentManager>>       agent_mgr;
    std::vector<std::shared_ptr<agent>>               agents;
    std::unique_ptr<NiceMock<MockOutputFileRegistry>> registry;
    std::string                                       tmp_dir;
    std::vector<registered_entry>                     registered;
    std::unique_ptr<TestProcessor>                    processor;
    bool                                              expected_xnack_enabled = false;

private:
    std::unique_ptr<ScopedUseOutputSuffix> m_suffix_guard;
};

// ──────────────────────────────────────────────────────────────────────
// 1) migration_stats arithmetic
//
// Pure-data invariants — no processor instance needed.
// ──────────────────────────────────────────────────────────────────────
TEST(MigrationStats, ArithmeticAndSentinels)
{
    migration_stats s;
    EXPECT_EQ(s.count, 0u);
    EXPECT_EQ(s.total_size_bytes, 0u);
    EXPECT_EQ(s.max_size_bytes, 0u);
    EXPECT_EQ(s.min_size_bytes, std::numeric_limits<uint64_t>::max());  // sentinel
    EXPECT_DOUBLE_EQ(s.avg_size_bytes(), 0.0);
    EXPECT_DOUBLE_EQ(s.bandwidth_gbps(), 0.0);  // zero-division guard

    s.add_migration(/*size=*/1000, /*duration_ns=*/100);
    s.add_migration(/*size=*/3000, /*duration_ns=*/300);

    EXPECT_EQ(s.count, 2u);
    EXPECT_EQ(s.total_size_bytes, 4000u);
    EXPECT_EQ(s.total_time_ns, 400u);
    EXPECT_EQ(s.min_size_bytes, 1000u);
    EXPECT_EQ(s.max_size_bytes, 3000u);
    EXPECT_DOUBLE_EQ(s.avg_size_bytes(), 2000.0);
    EXPECT_DOUBLE_EQ(s.bandwidth_gbps(), 10.0);  // 4000 bytes / 400 ns = 10 GB/s
}

// ──────────────────────────────────────────────────────────────────────
// 2) JSON schema always emits all 3 directions
//
// Locks in the Copilot #1 fix: the JSON producer must emit
// host_to_device, device_to_host, device_to_device unconditionally so the
// validator and downstream consumers can rely on a stable schema.
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, JsonSchemaAlwaysEmitsAllDirections)
{
    // Single H→D migration so the device map has exactly one entry.
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, /*size=*/4096,
                                                   /*duration=*/1000,
                                                   /*device_id=*/0));
    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    ASSERT_TRUE(j.contains("devices"));
    ASSERT_EQ(j["devices"].size(), 1u);
    auto const& migrations = j["devices"][0]["migrations"];

    for(const char* dir : { "host_to_device", "device_to_host", "device_to_device" })
    {
        ASSERT_TRUE(migrations.contains(dir)) << "missing direction key: " << dir;
        for(const char* k :
            { "count", "total_size_bytes", "min_size_bytes", "max_size_bytes",
              "avg_size_bytes", "total_time_ns", "bandwidth_gbps" })
        {
            EXPECT_TRUE(migrations[dir].contains(k)) << dir << " missing stat: " << k;
        }
    }

    // Empty directions must report zero counts (not be omitted).
    EXPECT_EQ(migrations["device_to_host"]["count"], 0u);
    EXPECT_EQ(migrations["device_to_device"]["count"], 0u);
    EXPECT_EQ(migrations["host_to_device"]["count"], 1u);
}

// ──────────────────────────────────────────────────────────────────────
// 3) classify_direction topology
//
// Verifies that direction classification depends on the agent type
// associated with each numeric node id (the March-20 fix that replaced
// brittle string-prefix matching with a node→type lookup).
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, ClassifyDirectionFromTopology)
{
    // CPU(0) → GPU(1) is host_to_device
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));
    // GPU(1) → CPU(0) is device_to_host
    processor->handle(make_kfd_page_migrate_sample(kGpu1, kCpu0, 2048, 200, /*dev=*/0));
    // GPU(1) → GPU(2) is device_to_device
    processor->handle(make_kfd_page_migrate_sample(kGpu1, kGpu2, 4096, 400, /*dev=*/0));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& m = (*j_opt)["devices"][0]["migrations"];
    EXPECT_EQ(m["host_to_device"]["count"], 1u);
    EXPECT_EQ(m["host_to_device"]["total_size_bytes"], 1024u);
    EXPECT_EQ(m["device_to_host"]["count"], 1u);
    EXPECT_EQ(m["device_to_host"]["total_size_bytes"], 2048u);
    EXPECT_EQ(m["device_to_device"]["count"], 1u);
    EXPECT_EQ(m["device_to_device"]["total_size_bytes"], 4096u);
}

// ──────────────────────────────────────────────────────────────────────
// 4) Host→device migrations bucket by destination GPU under producer semantics
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest,
       HostToDeviceMigrationsBucketByDestinationGpuUnderProducerSemantics)
{
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/kCpu0,
                                                   agent_type::CPU));
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu2, 2048, 200, /*dev=*/kCpu0,
                                                   agent_type::CPU));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    ASSERT_EQ(j["devices"].size(), 2u);

    bool saw_gpu1 = false;
    bool saw_gpu2 = false;
    for(auto const& dev : j["devices"])
    {
        auto        device_id = dev["device_id"].get<uint32_t>();
        auto const& h2d       = dev["migrations"]["host_to_device"];

        if(device_id == kGpu1)
        {
            EXPECT_EQ(h2d["count"], 1u);
            EXPECT_EQ(h2d["total_size_bytes"], 1024u);
            saw_gpu1 = true;
        }
        else if(device_id == kGpu2)
        {
            EXPECT_EQ(h2d["count"], 1u);
            EXPECT_EQ(h2d["total_size_bytes"], 2048u);
            saw_gpu2 = true;
        }
        else
        {
            ADD_FAILURE() << "unexpected GPU bucket id: " << device_id;
        }
    }

    EXPECT_TRUE(saw_gpu1);
    EXPECT_TRUE(saw_gpu2);
}

// ──────────────────────────────────────────────────────────────────────
// 5) extract_gpu_name resolves real name or falls back
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, ExtractGpuNameResolvesOrFallsBack)
{
    agents[2]->name.clear();
    processor = std::make_unique<TestProcessor>(agent_mgr, kPid, tmp_dir, *registry);

    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));

    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu2, 1024, 100, /*dev=*/0));
    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    ASSERT_EQ(j["devices"].size(), 2u);

    bool saw_resolved = false;
    bool saw_fallback = false;
    for(auto const& dev : j["devices"])
    {
        std::string name = dev["device_name"];
        if(dev["device_id"] == kGpu1)
        {
            EXPECT_THAT(name, HasSubstr("gfx950")) << "name=" << name;
            saw_resolved = true;
        }
        else if(dev["device_id"] == kGpu2)
        {
            EXPECT_THAT(name, ::testing::StartsWith("GPU 2")) << "name=" << name;
            saw_fallback = true;
        }
    }
    EXPECT_TRUE(saw_resolved);
    EXPECT_TRUE(saw_fallback);
}

// ──────────────────────────────────────────────────────────────────────
// 5) Agent lookup throws → handle() does not throw, falls back to "CPU {id}"
//
// Locks in Copilot #3 (the throw guard around get_agent_by_type_index).
// Production saw out-of-range failures during agent resolution; the
// processor must catch and continue with a fallback CPU label.
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, AgentLookupThrowFallsBackSafely)
{
    // First migrate event for device_id=42 will trigger the agent lookup;
    // make it throw and verify handle() does not propagate the exception.
    EXPECT_CALL(*agent_mgr, get_agent_by_type_index(42, agent_type::CPU))
        .WillOnce(Throw(std::out_of_range{ "no such agent" }));

    auto sample =
        make_kfd_page_migrate_sample(kCpu0, kGpu1, /*size=*/1024,
                                     /*duration=*/100, /*device_id=*/42, agent_type::CPU);
    EXPECT_NO_THROW(processor->handle(sample));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    ASSERT_EQ(j["devices"].size(), 1u);
    std::string name = j["devices"][0]["device_name"];
    // Fallback CPU label is "CPU 42"; full label looks like "gfx950 (via CPU 42)".
    EXPECT_THAT(name, HasSubstr("CPU 42"));
}

// ──────────────────────────────────────────────────────────────────────
// 6) PID-suffixed paths registered
//
// Locks in Copilot #5 — output filenames must be PID-suffixed so that
// concurrent processes don't clobber each other's reports.
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, PidSuffixedPathsRegistered)
{
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));
    processor->finalize_processing();

    bool saw_txt  = false;
    bool saw_json = false;
    for(const auto& e : registered)
    {
        EXPECT_THAT(e.path, HasSubstr("unified_memory"));
        EXPECT_THAT(e.path, HasSubstr(std::to_string(kPid)));
        if(e.format == output_format::text)
        {
            EXPECT_THAT(e.path, ::testing::EndsWith(".txt"));
            saw_txt = true;
        }
        else if(e.format == output_format::json)
        {
            EXPECT_THAT(e.path, ::testing::EndsWith(".json"));
            saw_json = true;
        }
    }
    EXPECT_TRUE(saw_txt) << "text file not registered";
    EXPECT_TRUE(saw_json) << "json file not registered";
}

// ──────────────────────────────────────────────────────────────────────
// 7) Faults-only output is emitted
//
// Locks in Copilot #2 — when the workload generates only page faults
// (e.g. XNACK on but no migrations triggered), the processor must still
// produce the .txt and .json reports rather than skipping output.
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, FaultsOnlyEmitsOutput)
{
    processor->handle(make_kfd_page_fault_sample(/*agent=*/kGpu1, /*read=*/true));
    processor->handle(make_kfd_page_fault_sample(/*agent=*/kGpu1, /*read=*/false));
    processor->handle(make_kfd_page_fault_sample(/*agent=*/kGpu2, /*read=*/true));

    processor->finalize_processing();

    // Two register_file calls: one for .txt, one for .json
    EXPECT_EQ(registered.size(), 2u);

    bool saw_txt  = false;
    bool saw_json = false;
    for(const auto& e : registered)
    {
        EXPECT_TRUE(test_common::fs::exists(e.path)) << "missing file: " << e.path;
        if(e.format == output_format::text) saw_txt = true;
        if(e.format == output_format::json) saw_json = true;
    }
    EXPECT_TRUE(saw_txt);
    EXPECT_TRUE(saw_json);

    // Sanity-check the JSON: 3 total faults, no devices, schema still valid.
    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    EXPECT_EQ(j["summary"]["total_page_faults"], 3u);
    EXPECT_EQ(j["devices"].size(), 0u);
}

// ──────────────────────────────────────────────────────────────────────
// 8) Malformed args_str → migrate event is silently skipped
//
// Indirect coverage of parse_agent_ids_from_args (which delegates to
// process_arguments_string in core/common_types.hpp). When the args_str
// does not parse as a "<idx>;;<type>;;<name>;;<value>;;" 4-tuple stream,
// the event must be ignored without recording a bogus migration.
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, MalformedArgsStringSkipsEvent)
{
    // Co-feed one valid page-fault sample so finalize_processing() takes the
    // emit-output branch (faults > 0). This makes the JSON observable, so the
    // assertion below tests *specifically* that the malformed migrate was
    // skipped — not just that the whole processor saw no events (which would
    // also cause registered.size() == 0 for unrelated reasons).
    processor->handle(make_kfd_page_fault_sample(/*agent=*/kGpu1, /*read=*/true));

    processor->handle(
        make_kfd_page_migrate_sample_raw_args("totally_wrong_format_no_markers_here"));

    processor->finalize_processing();

    // Output IS produced (because of the page fault), but the malformed
    // migrate must not appear as a device entry — and, more directly, must
    // not have incremented any migration_trigger counter (page faults do
    // not touch triggers; only page migrates do).
    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    EXPECT_EQ(j["summary"]["total_page_faults"], 1u);
    EXPECT_EQ(j["devices"].size(), 0u)
        << "Malformed migrate event was incorrectly recorded as a device";
    // Iterate the shared trigger table so adding a new trigger naturally
    // extends this assertion via one edit at the table in the header.
    auto const& triggers = j["summary"]["migration_triggers"];
    for(const auto& row : kTriggerTable)
    {
        EXPECT_EQ(triggers[row.json_key], 0u)
            << "Malformed migrate event incorrectly classified as trigger: "
            << row.json_key;
    }
}

// ──────────────────────────────────────────────────────────────────────
// 9) kTriggerTable covers all four KFD trigger names + unknown
//
// handle_page_migrate's trigger lookup walks kTriggerTable and bumps the
// matching migration_trigger_stats counter (falling through to the
// sentinel "unknown" row when no name matches). The JSON
// "summary.migration_triggers" object is the observable; iterating the
// same table in the test keeps the production/test pairing in lockstep.
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, TriggerTableCoversAllKfdNames)
{
    // Drive both feed and assertion from the shared kTriggerTable so the
    // single source of truth covers production AND test. The sentinel row
    // (kfd_name == nullptr) is fed an arbitrary unrecognized name to land
    // in the "unknown" bucket.
    constexpr const char* kSentinelFedName = "SOMETHING_ELSE";
    for(const auto& row : kTriggerTable)
    {
        const char* name = (row.kfd_name != nullptr) ? row.kfd_name : kSentinelFedName;
        processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, /*size=*/1024,
                                                       /*duration=*/100, /*device_id=*/0,
                                                       agent_type::CPU, name));
    }

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& triggers = (*j_opt)["summary"]["migration_triggers"];
    for(const auto& row : kTriggerTable)
    {
        EXPECT_EQ(triggers[row.json_key], 1u)
            << "trigger key not incremented: " << row.json_key;
    }
}

// ──────────────────────────────────────────────────────────────────────
// 10) xnack_enabled is reported in the JSON summary
//
// The processor reads HSA_XNACK in its constructor; SetUp() snapshots the
// same env var into expected_xnack_enabled at the same point, so this test
// asserts against a value captured at the same time as the processor's —
// no risk of observing a different value if anything mutated the env
// between SetUp and the test body.
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, JsonReportsXnackFlag)
{
    processor->handle(make_kfd_page_migrate_sample(kCpu0, kGpu1, 1024, 100, /*dev=*/0));
    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& summary = (*j_opt)["summary"];
    ASSERT_TRUE(summary.contains("xnack_enabled"));
    ASSERT_TRUE(summary["xnack_enabled"].is_boolean());
    EXPECT_EQ(summary["xnack_enabled"].get<bool>(), expected_xnack_enabled);
}

// ──────────────────────────────────────────────────────────────────────
// 11) Float sanitization: NaN / inf / negative / out-of-range size
//
// Locks in the float→uint64 boundary guard. None of these inputs may
// trigger UB on the cast, and all should produce size_bytes == 0 in
// the recorded migration.
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, FloatSanitizationProducesZeroSize)
{
    // Inputs that the production guard must reject. Mix of UB-on-cast values
    // (NaN/±inf/2^64) and benign boundary values (negative, zero) — all share
    // the same observable: recorded as count++ with size_bytes=0.
    const std::vector<double> rejected_values = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -1.0,
        0.0,                                                        // > 0 guard
        static_cast<double>(std::numeric_limits<uint64_t>::max()),  // = 2^64 (UB if cast)
    };

    for(double v : rejected_values)
        feed_h2d_migrate_with_value(v);

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& h2d = (*j_opt)["devices"][0]["migrations"]["host_to_device"];
    EXPECT_EQ(h2d["count"], rejected_values.size());
    // All four size aggregators must be zero. Without explicit min/max checks,
    // a regression that left min_size_bytes at its uint64-max sentinel (or
    // failed to reset max) would still pass the count+total assertion.
    EXPECT_EQ(h2d["total_size_bytes"], 0u)
        << "no rejected value should have contributed nonzero bytes";
    EXPECT_EQ(h2d["min_size_bytes"], 0u)
        << "sentinel not overwritten by add_migration(0)";
    EXPECT_EQ(h2d["max_size_bytes"], 0u);
}

// ──────────────────────────────────────────────────────────────────────
// 11b) Float sanitization — just-below-boundary value IS accepted
//
// Complements FloatSanitizationProducesZeroSize: probes the *good* side of
// the kMaxSafeUint64 boundary so a regression that flips the guard to
// strictly-less-than-something-smaller can't slip through. Uses 2^53 — the
// largest power-of-two that's exactly representable as both double and
// uint64_t — so the round-trip is loss-free.
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, FloatJustBelowBoundaryIsAccepted)
{
    // Largest power-of-two exactly representable as both double and uint64_t
    // (= 2^DBL_MANT_DIG on IEEE-754 — survives the float→uint64 round-trip
    // bit-for-bit by definition of the mantissa width). Strictly less than
    // kMaxSafeUint64, so the production guard's strict-less-than check accepts
    // it.
    static constexpr uint64_t kLargestExactDoubleUint =
        1ULL << std::numeric_limits<double>::digits;

    feed_h2d_migrate_with_value(static_cast<double>(kLargestExactDoubleUint));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& h2d = (*j_opt)["devices"][0]["migrations"]["host_to_device"];
    EXPECT_EQ(h2d["count"], 1u);
    // For a single sample, every size aggregator must equal the input — and
    // each must serialize as an unsigned integer, not a double, so a future
    // refactor that switches the JSON type is caught (uint→double could
    // silently still compare equal at 2^53 via implicit conversion).
    for(const char* k : { "total_size_bytes", "min_size_bytes", "max_size_bytes" })
    {
        ASSERT_TRUE(h2d[k].is_number_unsigned()) << "key=" << k;
        EXPECT_EQ(h2d[k].get<uint64_t>(), kLargestExactDoubleUint) << "key=" << k;
    }
}

// ──────────────────────────────────────────────────────────────────────
// 13) Node-id values exceeding uint32_t are rejected by parse_node_id_pair
//
// from_chars writes uint32_t directly, so any decimal string whose value
// exceeds uint32_t::max yields errc::result_out_of_range and the event
// is skipped. Verified indirectly by inspecting that no migration was
// recorded.
// ──────────────────────────────────────────────────────────────────────
TEST_F(UnifiedMemoryProcessorTest, NodeIdsExceedingUint32AreRejected)
{
    // Co-feed a fault so finalize emits the JSON.
    processor->handle(make_kfd_page_fault_sample(kGpu1, /*read=*/true));

    // args_str carries a src_agent node id > UINT32_MAX so parse_node_id_pair
    // rejects it via std::errc::result_out_of_range.
    processor->handle(make_kfd_page_migrate_sample_raw_args(
        "0;;uint64_t;;start_address;;0x0;;"
        "1;;uint64_t;;end_address;;0x1000;;"
        "2;;string;;src_agent;;9999999999;;"  // > UINT32_MAX
        "3;;string;;dst_agent;;1;;"));

    processor->finalize_processing();

    auto j_opt = read_json_output();
    ASSERT_TRUE(j_opt.has_value());
    auto const& j = *j_opt;
    EXPECT_EQ(j["devices"].size(), 0u);
}

}  // namespace
