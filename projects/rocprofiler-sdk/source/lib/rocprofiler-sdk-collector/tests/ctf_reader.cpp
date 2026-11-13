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

#include "lib/rocprofiler-sdk-collector/ctf/reader.hpp"

#include "ctf_writer.hpp"
#include "scratch_dir.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace
{
namespace ctf  = rocprofiler::collector::ctf;
namespace fs   = std::filesystem;
namespace test = rocprofiler::collector::test;

using test::scratch_dir;

// Every field type the reader decodes, with TraceLogging's transport fields first.
const auto all_types_fields = std::string{R"(
		integer { size = 8; align = 8; signed = 0; encoding = none; base = 10; } ___ms_ActivityId_length;
		struct { } align(8) __ms_ActivityId_padding;
		integer { size = 8; align = 8; signed = 0; encoding = none; base = 16; } __ms_ActivityId[ ___ms_ActivityId_length ];
		integer { size = 16; align = 8; signed = 0; encoding = none; base = 10; } __name_length;
		struct { } align(8) _name_padding;
		integer { size = 8; align = 8; signed = 0; encoding = UTF8; base = 10; } _name[ __name_length ];
		integer { size = 64; align = 64; signed = 1; encoding = none; base = 10; } _value;
		floating_point { exp_dig = 11; mant_dig = 53; align = 64; } _scale;
		floating_point { exp_dig = 8; mant_dig = 24; align = 32; } _ratio;
		enum : integer { size = 8; align = 8; signed = 0; encoding = none; base = 10; } {
			"false" = 0,
			"true" = 1,
		} _flag;
		integer { size = 3; align = 1; signed = 1; encoding = none; base = 10; } _small;
		integer { size = 8; align = 8; signed = 0; encoding = UTF8; base = 10; } _tag[4];
		uint16_t _dims[3];
		string _note;
)"};

void
write_all_types(test::bit_writer* out, int64_t value, const std::string& name)
{
    test::put(out, 2, 8);  // ActivityId length
    test::put(out, 0xAA, 8);
    test::put(out, 0xBB, 8);
    test::put(out, name.size(), 16);
    test::put_text(out, name, false);
    test::put(out, static_cast<uint64_t>(value), 64, 64);
    auto scale = 2.5;
    auto bits  = uint64_t{};
    std::memcpy(&bits, &scale, sizeof(bits));
    test::put(out, bits, 64, 64);
    auto ratio  = 0.25F;
    auto bits32 = uint32_t{};
    std::memcpy(&bits32, &ratio, sizeof(bits32));
    test::put(out, bits32, 32, 32);
    test::put(out, 1, 8);              // flag
    test::put(out, 0b101, 3, 1);       // small = -3
    test::put_text(out, "ab", false);  // tag[4] = "ab\0\0"
    test::put(out, 0, 16);
    test::put(out, 1, 16);
    test::put(out, 2, 16);
    test::put(out, 3, 16);
    test::put_text(out, "note", true);
}

const auto simple_fields = std::string{
    "\t\tinteger { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _n;\n"};

std::string
simple_metadata(uint64_t vpid)
{
    return test::metadata_prologue(vpid) + test::metadata_event("p:simple", 0, simple_fields) +
           test::metadata_event("p:extended", 40, simple_fields);
}

test::test_event
simple_event(uint64_t id, uint64_t timestamp, uint32_t n, int32_t vtid = 1)
{
    return test::test_event{
        id, timestamp, 7, vtid, [n](test::bit_writer* out) { test::put(out, n, 32); }};
}

std::vector<ctf::trace_ptr>
load(const fs::path& root)
{
    auto traces = std::vector<ctf::trace_ptr>{};
    for(const auto& dir : ctf::find_traces(root.string()))
    {
        auto source = std::make_shared<ctf::trace>();
        auto error  = std::string{};
        EXPECT_TRUE(ctf::load_trace(dir, source.get(), &error)) << error;
        traces.push_back(source);
    }
    return traces;
}

struct read_result
{
    bool                               ok        = false;
    std::string                        error     = {};
    std::vector<ctf::trace_ptr>        traces    = {};  // the records point into these
    std::vector<ctf::event_record>     events    = {};
    std::vector<ctf::discarded_record> discarded = {};
    ctf::read_statistics               stats     = {};
};

read_result
read_all(const fs::path& root)
{
    auto result   = read_result{};
    result.traces = load(root);
    result.ok     = ctf::read_events(
        result.traces,
        [&result](const ctf::event_record& event) { result.events.push_back(event); },
        [&result](const ctf::discarded_record& record) { result.discarded.push_back(record); },
        &result.stats,
        &result.error);
    return result;
}

uint64_t
field_u64(const ctf::event_record& event, const char* name)
{
    const auto* field = ctf::find_field(event.fields, name);
    EXPECT_NE(field, nullptr) << name;
    return field != nullptr ? field->u64 : 0;
}
}  // namespace

TEST(ctf_reader, decodes_every_field_type)
{
    auto dir = scratch_dir{};
    test::write_file(dir.path / "metadata",
                     test::metadata_prologue(42) +
                         test::metadata_event("rocm_hip:hipTest_enter;k0;", 3, all_types_fields));
    auto packet  = test::test_packet{};
    packet.begin = 1000;
    packet.end   = 2000;
    packet.events.push_back(
        {3, 1100, 42, 43, [](test::bit_writer* out) { write_all_types(out, -5, "hip"); }});
    test::write_stream(dir.path / "channel0_0", {packet});

    auto result = read_all(dir.path);
    ASSERT_TRUE(result.ok) << result.error;
    ASSERT_EQ(result.events.size(), 1u);
    const auto& event = result.events[0];
    EXPECT_EQ(event.description->name, "rocm_hip:hipTest_enter;k0;");
    EXPECT_EQ(event.timestamp, 1100u);
    EXPECT_EQ(event.source->description.env.at("vpid"), "42");

    ASSERT_EQ(event.context.size(), 2u);
    EXPECT_EQ(event.context[0].name, "vpid");
    EXPECT_EQ(event.context[0].i64, 42);
    EXPECT_EQ(event.context[1].name, "vtid");
    EXPECT_EQ(event.context[1].i64, 43);

    // Lengths are folded into their sequences; empty padding structs add nothing.
    auto names = std::vector<std::string>{};
    for(const auto& itr : event.fields)
        names.push_back(itr.name);
    EXPECT_EQ(names,
              (std::vector<std::string>{"_ms_ActivityId",
                                        "name",
                                        "value",
                                        "scale",
                                        "ratio",
                                        "flag",
                                        "small",
                                        "tag",
                                        "dims",
                                        "note"}));

    const auto& fields = event.fields;
    EXPECT_EQ(fields[0].kind, ctf::value_kind::unsigned_array);
    EXPECT_EQ(fields[0].elements, (std::vector<uint64_t>{0xAA, 0xBB}));
    EXPECT_EQ(fields[1].kind, ctf::value_kind::string);
    EXPECT_EQ(fields[1].str, "hip");
    EXPECT_EQ(fields[2].kind, ctf::value_kind::signed_integer);
    EXPECT_EQ(fields[2].i64, -5);
    EXPECT_EQ(fields[3].kind, ctf::value_kind::floating_point);
    EXPECT_DOUBLE_EQ(fields[3].f64, 2.5);
    EXPECT_DOUBLE_EQ(fields[4].f64, 0.25);
    EXPECT_EQ(fields[5].u64, 1u);
    EXPECT_EQ(fields[5].enum_label, "true");
    EXPECT_EQ(fields[6].i64, -3);
    EXPECT_EQ(fields[7].str, "ab");
    EXPECT_EQ(fields[8].elements, (std::vector<uint64_t>{1, 2, 3}));
    EXPECT_EQ(fields[9].str, "note");
}

TEST(ctf_reader, reconstructs_timestamps)
{
    auto dir = scratch_dir{};
    test::write_file(dir.path / "metadata", simple_metadata(7));

    // Compact headers keep the low 27 bits: the reader carries the wrap of those bits
    // between events. A larger change, or an id above 30, needs the extended header.
    constexpr auto wrap   = uint64_t{1} << 27;
    auto           base   = uint64_t{5} * wrap - 100;
    auto           packet = test::test_packet{};
    packet.begin          = base;
    packet.end            = base + (10 * wrap);
    packet.events         = {simple_event(0, base + 10, 1),
                     simple_event(0, base + 150, 2),              // low bits wrapped
                     simple_event(0, base + 150, 3),              // same time
                     simple_event(0, base + (3 * wrap), 4),       // extended: large change
                     simple_event(40, base + (3 * wrap) + 5, 5),  // extended: id > 30
                     simple_event(0, base + (3 * wrap) + 6, 6)};
    test::write_stream(dir.path / "channel0_0", {packet});

    auto result = read_all(dir.path);
    ASSERT_TRUE(result.ok) << result.error;
    ASSERT_EQ(result.events.size(), 6u);
    auto expected = std::vector<uint64_t>{base + 10,
                                          base + 150,
                                          base + 150,
                                          base + (3 * wrap),
                                          base + (3 * wrap) + 5,
                                          base + (3 * wrap) + 6};
    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(result.events[i].timestamp, expected[i]) << i;
        EXPECT_EQ(field_u64(result.events[i], "n"), i + 1);
    }
    EXPECT_EQ(result.events[4].description->name, "p:extended");
}

TEST(ctf_reader, merges_streams_and_traces_by_time)
{
    auto dir = scratch_dir{};
    for(auto pid : {7, 8})
    {
        auto trace = dir.path / ("app-" + std::to_string(pid));
        fs::create_directories(trace / "index");
        test::write_file(trace / "metadata", simple_metadata(pid));
        auto first    = test::test_packet{};
        first.begin   = 100;
        first.end     = 900;
        first.events  = {simple_event(0, 100 + pid, (pid * 100) + 1),
                        simple_event(0, 300 + pid, (pid * 100) + 3)};
        auto second   = first;
        second.events = {simple_event(0, 200 + pid, (pid * 100) + 2),
                         simple_event(0, 400 + pid, (pid * 100) + 4)};
        test::write_stream(trace / "channel0_0", {first});
        test::write_stream(trace / "channel0_1", {second});
        test::write_file(trace / "index" / "channel0_0.idx", "not a stream");
    }

    auto result = read_all(dir.path);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.stats.traces, 2u);
    EXPECT_EQ(result.stats.streams, 4u);
    EXPECT_EQ(result.stats.packets, 4u);
    EXPECT_EQ(result.stats.events, 8u);
    auto order = std::vector<uint64_t>{};
    for(const auto& event : result.events)
        order.push_back(field_u64(event, "n"));
    EXPECT_EQ(order, (std::vector<uint64_t>{701, 801, 702, 802, 703, 803, 704, 804}));
}

TEST(ctf_reader, reports_discarded_events)
{
    auto dir = scratch_dir{};
    test::write_file(dir.path / "metadata", simple_metadata(7));

    // events_discarded is a running count, read at each packet.
    auto packets = std::vector<test::test_packet>(3);
    packets[0]   = {100, 200, 3, 64, {simple_event(0, 150, 1)}};
    packets[1]   = {300, 400, 3, 0, {simple_event(0, 350, 2)}};
    packets[2]   = {500, 600, 8, 16, {simple_event(0, 550, 3)}};
    test::write_stream(dir.path / "channel0_0", packets);

    auto result = read_all(dir.path);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.events.size(), 3u);
    EXPECT_EQ(result.stats.discarded, 8u);
    ASSERT_EQ(result.discarded.size(), 2u);
    EXPECT_EQ(result.discarded[0].count, 3u);
    EXPECT_EQ(result.discarded[0].begin, 100u);  // no earlier packet: from this one's start
    EXPECT_EQ(result.discarded[0].end, 200u);
    EXPECT_EQ(result.discarded[1].count, 5u);
    EXPECT_EQ(result.discarded[1].begin, 400u);  // after the previous packet
    EXPECT_EQ(result.discarded[1].end, 600u);
    EXPECT_EQ(result.discarded[1].source->description.env.at("vpid"), "7");
}

TEST(ctf_reader, reports_decoding_errors)
{
    auto dir = scratch_dir{};
    test::write_file(dir.path / "metadata", simple_metadata(7));

    // The event's field extends past the packet content.
    auto packet = test::test_packet{};
    packet.events.push_back({0, 10, 7, 1, [](test::bit_writer* out) { test::put(out, 1, 16); }});
    test::write_stream(dir.path / "channel0_0", {packet});
    auto result = read_all(dir.path);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("channel0_0"), std::string::npos) << result.error;
    EXPECT_NE(result.error.find("past the packet content"), std::string::npos) << result.error;

    // An event id the metadata does not declare.
    packet.events = {simple_event(9, 10, 1)};
    test::write_stream(dir.path / "channel0_0", {packet});
    result = read_all(dir.path);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("unknown event id 9"), std::string::npos) << result.error;

    // Not a CTF packet.
    test::write_file(dir.path / "channel0_0", std::string(64, 'x'));
    result = read_all(dir.path);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("bad packet magic"), std::string::npos) << result.error;
}

TEST(ctf_reader, reads_a_trace_without_streams)
{
    // A process that ended before writing an event: metadata without a stream, and empty
    // stream files.
    auto dir = scratch_dir{};
    test::write_file(dir.path / "metadata", R"(/* CTF 1.8 */
typealias integer { size = 32; align = 8; signed = false; } := uint32_t;
trace {
	major = 1;
	minor = 8;
	byte_order = le;
	packet.header := struct { uint32_t magic; };
};
)");
    test::write_file(dir.path / "channel0_0", "");

    auto result = read_all(dir.path);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.events.empty());
    EXPECT_EQ(result.stats.traces, 1u);
}

TEST(ctf_reader, finds_and_loads_traces)
{
    auto dir = scratch_dir{};
    fs::create_directories(dir.path / "ust" / "pid" / "b-2");
    fs::create_directories(dir.path / "ust" / "pid" / "a-1");
    fs::create_directories(dir.path / "ust" / "pid" / "empty");
    test::write_file(dir.path / "ust" / "pid" / "b-2" / "metadata", simple_metadata(2));
    test::write_file(dir.path / "ust" / "pid" / "a-1" / "metadata", "not metadata");

    auto found = ctf::find_traces(dir.path.string());
    ASSERT_EQ(found.size(), 2u);
    EXPECT_EQ(fs::path{found[0]}.filename(), "a-1");
    EXPECT_EQ(fs::path{found[1]}.filename(), "b-2");

    auto source = ctf::trace{};
    auto error  = std::string{};
    EXPECT_FALSE(ctf::load_trace(found[0], &source, &error));
    EXPECT_NE(error.find("a-1"), std::string::npos) << error;
    EXPECT_TRUE(ctf::load_trace(found[1], &source, &error)) << error;
    EXPECT_EQ(source.path, found[1]);
    EXPECT_TRUE(ctf::find_traces((dir.path / "missing").string()).empty());
}
