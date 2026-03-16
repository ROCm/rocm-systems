// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "rocprof-sys/library/rocprofiler-sdk/marker_writer.hpp"

#include "core/trace_cache/sample_type.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using rocprofsys::rocprofiler_sdk::annotation_entry;
using rocprofsys::trace_cache::region_sample;

using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::NiceMock;
using ::testing::StrEq;

MATCHER_P2(IsAnnotation, key, value, "")
{
    return std::string(arg.key) == key && std::holds_alternative<uint64_t>(arg.value) &&
           std::get<uint64_t>(arg.value) == static_cast<uint64_t>(value);
}

class mock_api
{
public:
    MOCK_METHOD(void, push_timemory, (std::string_view name));
    MOCK_METHOD(void, pop_timemory, (std::string_view name));
    MOCK_METHOD(void, push_perfetto_ts,
                (const char* name, uint64_t ts, uint64_t flow_id,
                 const std::vector<annotation_entry>& annotations));
    MOCK_METHOD(void, pop_perfetto_ts,
                (const char* name, uint64_t ts,
                 const std::vector<annotation_entry>& annotations));
    MOCK_METHOD(void, cache_init, ());
    MOCK_METHOD(void, store_region, (const region_sample& sample));
    MOCK_METHOD(void, add_thread_info, (uint64_t thread_id));
};

struct mock_marker_policy
{
    static inline std::unique_ptr<NiceMock<mock_api>> api{};

    static void reset() { api = std::make_unique<NiceMock<mock_api>>(); }

    static void push_timemory(std::string_view name) { api->push_timemory(name); }
    static void pop_timemory(std::string_view name) { api->pop_timemory(name); }

    static void push_perfetto_ts(const char* name, uint64_t ts, uint64_t flow_id,
                                 const std::vector<annotation_entry>& annotations)
    {
        api->push_perfetto_ts(name, ts, flow_id, annotations);
    }

    static void pop_perfetto_ts(const char* name, uint64_t ts,
                                const std::vector<annotation_entry>& annotations)
    {
        api->pop_perfetto_ts(name, ts, annotations);
    }

    static void cache_init() { api->cache_init(); }
    static void store_region(const region_sample& sample) { api->store_region(sample); }
    static void add_thread_info(uint64_t thread_id) { api->add_thread_info(thread_id); }
};

using mock_marker_writer = rocprofsys::rocprofiler_sdk::marker_writer<mock_marker_policy>;

rocprofiler_callback_tracing_record_t
make_dummy_record()
{
    rocprofiler_callback_tracing_record_t record{};
    record.thread_id                     = 42;
    record.correlation_id.internal       = 100;
    record.correlation_id.external.value = 200;
    return record;
}

class marker_writer_test : public ::testing::Test
{
protected:
    void SetUp() override { mock_marker_policy::reset(); }
    void TearDown() override { mock_marker_policy::api.reset(); }
};

}  // namespace

TEST_F(marker_writer_test, all_backends_with_annotations)
{
    auto& mock   = *mock_marker_policy::api;
    auto  record = make_dummy_record();

    EXPECT_CALL(mock, cache_init());
    EXPECT_CALL(mock, push_timemory(std::string_view("my_region")));
    EXPECT_CALL(mock, pop_timemory(std::string_view("my_region")));
    EXPECT_CALL(mock, push_perfetto_ts(StrEq("my_region"), 1000, 100,
                                       ElementsAre(IsAnnotation("begin_ns", 1000u),
                                                   IsAnnotation("stack_id", 100u))));
    EXPECT_CALL(mock, pop_perfetto_ts(StrEq("my_region"), 2000,
                                      ElementsAre(IsAnnotation("end_ns", 2000u))));
    EXPECT_CALL(mock, add_thread_info(42u));
    EXPECT_CALL(mock,
                store_region(AllOf(Field(&region_sample::thread_id, 42u),
                                   Field(&region_sample::start_timestamp, 1000u),
                                   Field(&region_sample::end_timestamp, 2000u),
                                   Field(&region_sample::correlation_id_internal, 100u),
                                   Field(&region_sample::correlation_id_ancestor, 200u),
                                   Field(&region_sample::args_str, "arg1=val1"),
                                   Field(&region_sample::name, "my_region"))));

    const mock_marker_writer writer(true, true, true);
    writer.write_begin("my_region");
    writer.write_end("my_region", 1000, 2000, "arg1=val1", record);
}

TEST_F(marker_writer_test, perfetto_disabled)
{
    auto& mock   = *mock_marker_policy::api;
    auto  record = make_dummy_record();

    EXPECT_CALL(mock, push_timemory(_));
    EXPECT_CALL(mock, pop_timemory(_));
    EXPECT_CALL(mock, push_perfetto_ts(_, _, _, _)).Times(0);
    EXPECT_CALL(mock, pop_perfetto_ts(_, _, _)).Times(0);
    EXPECT_CALL(mock, store_region(_));

    const mock_marker_writer writer(false, true, false);
    writer.write_begin("r");
    writer.write_end("r", 100, 200, "{}", record);
}

TEST_F(marker_writer_test, timemory_disabled_no_annotations)
{
    auto& mock   = *mock_marker_policy::api;
    auto  record = make_dummy_record();

    EXPECT_CALL(mock, push_timemory(_)).Times(0);
    EXPECT_CALL(mock, pop_timemory(_)).Times(0);
    EXPECT_CALL(mock, push_perfetto_ts(_, 100, _, IsEmpty()));
    EXPECT_CALL(mock, pop_perfetto_ts(_, 200, IsEmpty()));
    EXPECT_CALL(mock, store_region(_));

    const mock_marker_writer writer(true, false, false);
    writer.write_begin("r");
    writer.write_end("r", 100, 200, "{}", record);
}
