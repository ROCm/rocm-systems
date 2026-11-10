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

#include "lib/rocprofiler-sdk-collector/ctf/metadata.hpp"

#include "ctf_writer.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace
{
namespace ctf  = rocprofiler::collector::ctf;
namespace test = rocprofiler::collector::test;

// The fields of a TraceLogging event as LTTng-UST 2.13 declares them (rocm_hip probes).
const auto tracelogging_fields = std::string{R"(
		integer { size = 8; align = 8; signed = 0; encoding = none; base = 10; } ___ms_ActivityId_length;
		struct { } align(8) __ms_ActivityId_padding;
		integer { size = 8; align = 8; signed = 0; encoding = none; base = 16; } __ms_ActivityId[ ___ms_ActivityId_length ];
		integer { size = 16; align = 8; signed = 0; encoding = none; base = 10; } __result_length;
		struct { } align(8) _result_padding;
		integer { size = 8; align = 8; signed = 0; encoding = UTF8; base = 10; } _result[ __result_length ];
		enum : integer { size = 8; align = 8; signed = 0; encoding = none; base = 10; } {
			"false" = 0,
			"true" = 1,
		} _exception;
		floating_point { exp_dig = 11; mant_dig = 53; align = 64; } _scale;
		string _note;
)"};

std::string
synthetic_metadata()
{
    return test::metadata_prologue(42) +
           test::metadata_event("rocm_hip:hipApiName_exit;k0;", 0, tracelogging_fields);
}

ctf::trace_description
parse_ok(const std::string& text)
{
    auto trace = ctf::trace_description{};
    auto error = std::string{};
    EXPECT_TRUE(ctf::parse_metadata(text, &trace, &error)) << error;
    return trace;
}

std::string
parse_error(const std::string& text)
{
    auto trace = ctf::trace_description{};
    auto error = std::string{};
    EXPECT_FALSE(ctf::parse_metadata(text, &trace, &error));
    return error;
}

// Wraps text in metadata packets of at most chunk bytes of text each, as LTTng writes it.
std::string
packetize(const std::string& text, size_t chunk, size_t padding)
{
    auto out = test::bit_writer{};
    for(size_t offset = 0; offset < text.size(); offset += chunk)
    {
        auto part = text.substr(offset, chunk);
        test::put(&out, 0x75D11D57, 32);
        for(int i = 0; i < 16; ++i)
            test::put(&out, 0, 8);  // uuid
        test::put(&out, 0, 32);     // checksum
        test::put(&out, (37 + part.size()) * 8, 32);
        test::put(&out, (37 + part.size() + padding) * 8, 32);
        for(int i = 0; i < 5; ++i)
            test::put(&out, 0, 8);  // compression, encryption, checksum scheme, major, minor
        test::put_text(&out, part, false);
        for(size_t i = 0; i < padding; ++i)
            test::put(&out, 0, 8);
    }
    return std::string{out.bytes.begin(), out.bytes.end()};
}
}  // namespace

TEST(ctf_metadata, parses_trace_clock_and_stream)
{
    auto trace = parse_ok(synthetic_metadata());
    EXPECT_EQ(trace.major, 1u);
    EXPECT_EQ(trace.minor, 8u);
    EXPECT_EQ(trace.env.at("vpid"), "42");
    EXPECT_EQ(trace.env.at("procname"), "app");
    EXPECT_EQ(trace.clock.name, "monotonic");
    EXPECT_EQ(trace.clock.frequency, 1000000000u);
    EXPECT_EQ(trace.clock.offset, 1782026330110231331);

    ASSERT_TRUE(trace.packet_header);
    ASSERT_EQ(trace.packet_header->members.size(), 4u);
    EXPECT_EQ(trace.packet_header->members[1].type->kind, ctf::type_kind::array);
    EXPECT_EQ(trace.packet_header->members[1].type->length, 16u);

    ASSERT_EQ(trace.streams.count(0), 1u);
    const auto& stream = trace.streams.at(0);
    ASSERT_TRUE(stream.packet_context && stream.event_header && stream.event_context);
    // "unsigned long" is one (multi-word) alias.
    EXPECT_EQ(stream.packet_context->members[5].name, "events_discarded");
    EXPECT_EQ(stream.packet_context->members[5].type->size, 64u);
    EXPECT_TRUE(stream.packet_context->members[0].type->maps_clock);

    // The compact header: a 5-bit enumeration with a range, and a variant selected by it.
    const auto& header = *stream.event_header;
    EXPECT_EQ(header.alignment, 8u);
    ASSERT_EQ(header.members.size(), 2u);
    const auto& id = *header.members[0].type;
    EXPECT_EQ(id.kind, ctf::type_kind::enumeration);
    EXPECT_EQ(id.size, 5u);
    ASSERT_EQ(id.enumerators.size(), 2u);
    EXPECT_EQ(id.enumerators[0].label, "compact");
    EXPECT_EQ(id.enumerators[0].lo, 0);
    EXPECT_EQ(id.enumerators[0].hi, 30);
    EXPECT_EQ(id.enumerators[1].lo, 31);
    const auto& variant = *header.members[1].type;
    EXPECT_EQ(variant.kind, ctf::type_kind::variant);
    EXPECT_EQ(variant.tag, "id");
    ASSERT_EQ(variant.members.size(), 2u);
    EXPECT_EQ(variant.members[0].type->members[0].type->size, 27u);
    EXPECT_TRUE(variant.members[0].type->members[0].type->maps_clock);
}

TEST(ctf_metadata, parses_tracelogging_event_fields)
{
    auto trace = parse_ok(synthetic_metadata());
    ASSERT_EQ(trace.events.count(0), 1u);
    const auto& event = trace.events.at(0);
    EXPECT_EQ(event.name, "rocm_hip:hipApiName_exit;k0;");
    EXPECT_EQ(event.loglevel, 13);

    const auto& members = event.fields->members;
    ASSERT_EQ(members.size(), 9u);
    EXPECT_EQ(members[1].type->kind, ctf::type_kind::structure);  // padding
    EXPECT_TRUE(members[1].type->members.empty());
    EXPECT_EQ(members[1].type->alignment, 8u);
    EXPECT_EQ(members[2].type->kind, ctf::type_kind::sequence);
    EXPECT_EQ(members[2].type->length_name, "___ms_ActivityId_length");
    EXPECT_EQ(members[2].type->element->base, 16u);
    EXPECT_TRUE(members[5].type->element->is_text);
    EXPECT_EQ(members[6].type->kind, ctf::type_kind::enumeration);
    EXPECT_EQ(members[6].type->enumerators[1].label, "true");
    EXPECT_EQ(members[7].type->kind, ctf::type_kind::floating_point);
    EXPECT_EQ(members[7].type->size, 64u);
    EXPECT_EQ(members[7].type->alignment, 64u);
    EXPECT_EQ(members[8].type->kind, ctf::type_kind::string);

    EXPECT_EQ(ctf::field_name(members[0].name), "__ms_ActivityId_length");
    EXPECT_EQ(ctf::field_name(members[5].name), "result");
    EXPECT_EQ(ctf::field_name("plain"), "plain");
}

TEST(ctf_metadata, removes_packet_framing)
{
    auto text   = synthetic_metadata();
    auto result = std::string{};
    auto error  = std::string{};

    // Packets with padding after their content, as LTTng writes them.
    ASSERT_TRUE(ctf::metadata_text(packetize(text, 1000, 13), &result, &error)) << error;
    EXPECT_EQ(result, text);

    // Plain text is taken as is.
    ASSERT_TRUE(ctf::metadata_text(text, &result, &error)) << error;
    EXPECT_EQ(result, text);

    // A packet that claims more than the file holds.
    auto broken = packetize(text, 1000, 0);
    broken.resize(broken.size() - 10);
    EXPECT_FALSE(ctf::metadata_text(broken, &result, &error));
    EXPECT_NE(error.find("out of range"), std::string::npos) << error;
}

TEST(ctf_metadata, rejects_what_it_does_not_support)
{
    auto base = synthetic_metadata();

    auto big_endian = base;
    big_endian.replace(big_endian.find("byte_order = le"), 15, "byte_order = be");
    EXPECT_NE(parse_error(big_endian).find("byte order be"), std::string::npos);

    auto version = base;
    version.replace(version.find("minor = 8"), 9, "minor = 9");
    EXPECT_NE(parse_error(version).find("CTF 1.9"), std::string::npos);

    auto two_clocks = base + "clock { name = \"other\"; freq = 1000; };\n";
    EXPECT_NE(parse_error(two_clocks).find("more than one clock"), std::string::npos);

    auto unknown = base + test::metadata_event("p:e", 5, "\t\tmystery_t _x;\n");
    auto error   = parse_error(unknown);
    EXPECT_NE(error.find("unknown type 'mystery_t'"), std::string::npos) << error;
    EXPECT_NE(error.find("metadata line "), std::string::npos) << error;

    EXPECT_NE(parse_error(base + "/* open").find("unterminated comment"), std::string::npos);
    EXPECT_NE(
        parse_error("/* CTF 1.8 */ typealias integer { size = 8; } := u8;").find("no trace block"),
        std::string::npos);
    EXPECT_NE(parse_error(base + "typealias integer { size = 65; } := huge;").find("1..64"),
              std::string::npos);
}
