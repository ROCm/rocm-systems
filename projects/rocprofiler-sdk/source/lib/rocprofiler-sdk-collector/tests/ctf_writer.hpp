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

#pragma once

// Writes small CTF traces for the reader and join tests: the metadata LTTng-UST 2.13 writes
// for a per-process session (compact event header, vpid/vtid context), and stream files
// of packets built bit by bit.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace collector
{
namespace test
{
// The types, trace, clock, stream and packet declarations; events are appended.
inline std::string
metadata_prologue(uint64_t vpid)
{
    return R"(/* CTF 1.8 */

typealias integer { size = 8; align = 8; signed = false; } := uint8_t;
typealias integer { size = 16; align = 8; signed = false; } := uint16_t;
typealias integer { size = 32; align = 8; signed = false; } := uint32_t;
typealias integer { size = 64; align = 8; signed = false; } := uint64_t;
typealias integer { size = 64; align = 8; signed = false; } := unsigned long;
typealias integer { size = 5; align = 1; signed = false; } := uint5_t;
typealias integer { size = 27; align = 1; signed = false; } := uint27_t;

trace {
	major = 1;
	minor = 8;
	uuid = "ee3e31ff-c64e-49df-9e6e-3c4122b54b11";
	byte_order = le;
	packet.header := struct {
		uint32_t magic;
		uint8_t  uuid[16];
		uint32_t stream_id;
		uint64_t stream_instance_id;
	};
};

env {
	domain = "ust";
	tracer_name = "lttng-ust";
	vpid = )" +
           std::to_string(vpid) +
           R"(;
	procname = "app";
};

clock {
	name = "monotonic";
	freq = 1000000000; /* Frequency, in Hz */
	offset = 1782026330110231331;
};

typealias integer {
	size = 27; align = 1; signed = false;
	map = clock.monotonic.value;
} := uint27_clock_monotonic_t;

typealias integer {
	size = 64; align = 8; signed = false;
	map = clock.monotonic.value;
} := uint64_clock_monotonic_t;

struct packet_context {
	uint64_clock_monotonic_t timestamp_begin;
	uint64_clock_monotonic_t timestamp_end;
	uint64_t content_size;
	uint64_t packet_size;
	uint64_t packet_seq_num;
	unsigned long events_discarded;
	uint32_t cpu_id;
};

struct event_header_compact {
	enum : uint5_t { compact = 0 ... 30, extended = 31 } id;
	variant <id> {
		struct {
			uint27_clock_monotonic_t timestamp;
		} compact;
		struct {
			uint32_t id;
			uint64_clock_monotonic_t timestamp;
		} extended;
	} v;
} align(8);

stream {
	id = 0;
	event.header := struct event_header_compact;
	packet.context := struct packet_context;
	event.context := struct {
		integer { size = 32; align = 8; signed = 1; encoding = none; base = 10; } _vpid;
		integer { size = 32; align = 8; signed = 1; encoding = none; base = 10; } _vtid;
	};
};

)";
}

inline std::string
metadata_event(const std::string& name, uint64_t id, const std::string& fields)
{
    return "event {\n\tname = \"" + name + "\";\n\tid = " + std::to_string(id) +
           ";\n\tstream_id = 0;\n\tloglevel = 13;\n\tfields := struct {\n" + fields +
           "\t};\n};\n\n";
}

// Little-endian bit stream.
struct bit_writer
{
    std::vector<uint8_t> bytes = {};
    uint64_t             bits  = 0;
};

inline void
align_to(bit_writer* out, uint32_t align)
{
    if(align > 1) out->bits = (out->bits + align - 1) / align * align;
    out->bytes.resize((out->bits + 7) / 8);
}

inline void
put(bit_writer* out, uint64_t value, uint32_t size, uint32_t align = 8)
{
    align_to(out, align);
    out->bytes.resize((out->bits + size + 7) / 8);
    for(uint32_t i = 0; i < size; ++i, ++out->bits)
        if(((value >> i) & 1U) != 0) out->bytes[out->bits / 8] |= uint8_t(1U << (out->bits % 8));
}

inline void
put_text(bit_writer* out, const std::string& text, bool terminate)
{
    for(auto ch : text)
        put(out, static_cast<uint8_t>(ch), 8);
    if(terminate) put(out, 0, 8);
}

inline void
patch_u64(bit_writer* out, uint64_t bit_offset, uint64_t value)
{
    for(int i = 0; i < 8; ++i)
        out->bytes[(bit_offset / 8) + i] = uint8_t(value >> (8 * i));
}

// One event of a stream: header, vpid/vtid context, and fields written by write_fields.
struct test_event
{
    uint64_t                         id        = 0;
    uint64_t                         timestamp = 0;
    int32_t                          vpid      = 0;
    int32_t                          vtid      = 0;
    std::function<void(bit_writer*)> fields    = {};
};

struct test_packet
{
    uint64_t                begin     = 0;  // timestamp_begin
    uint64_t                end       = 0;  // timestamp_end
    uint64_t                discarded = 0;  // running count
    uint32_t                padding   = 0;  // bytes after the content
    std::vector<test_event> events    = {};
    bool                    context   = true;  // events have the vpid/vtid context
};

// Appends a packet; the header of an event is compact when its id and the change of
// the clock's low 27 bits allow.
inline void
write_packet(bit_writer* out, const test_packet& packet)
{
    auto start = out->bits;
    put(out, 0xC1FC1FC1, 32);
    for(int i = 0; i < 16; ++i)
        put(out, 0, 8);
    put(out, 0, 32);  // stream_id
    put(out, 0, 64);  // stream_instance_id
    put(out, packet.begin, 64);
    put(out, packet.end, 64);
    auto content_size_at = out->bits;
    put(out, 0, 64);
    auto packet_size_at = out->bits;
    put(out, 0, 64);
    put(out, 0, 64);  // packet_seq_num
    put(out, packet.discarded, 64);
    put(out, 0, 32);  // cpu_id

    auto clock = packet.begin;
    for(const auto& event : packet.events)
    {
        align_to(out, 8);
        constexpr auto low_mask = (uint64_t{1} << 27) - 1;
        auto           compact =
            event.id < 31 && event.timestamp >= clock && event.timestamp - clock <= low_mask;
        if(compact)
        {
            put(out, event.id, 5, 1);
            put(out, event.timestamp & low_mask, 27, 1);
        }
        else
        {
            put(out, 31, 5, 1);
            put(out, event.id, 32);
            put(out, event.timestamp, 64);
        }
        clock = event.timestamp;
        if(packet.context)
        {
            put(out, static_cast<uint32_t>(event.vpid), 32);
            put(out, static_cast<uint32_t>(event.vtid), 32);
        }
        if(event.fields) event.fields(out);
    }
    align_to(out, 8);
    auto content = out->bits - start;
    out->bits += uint64_t{packet.padding} * 8;
    out->bytes.resize(out->bits / 8);
    patch_u64(out, content_size_at, content);
    patch_u64(out, packet_size_at, out->bits - start);
}

inline void
write_file(const std::filesystem::path& path, const std::string& contents)
{
    auto file = std::ofstream{path, std::ios::binary};
    file << contents;
}

inline void
write_stream(const std::filesystem::path& path, const std::vector<test_packet>& packets)
{
    auto out = bit_writer{};
    for(const auto& packet : packets)
        write_packet(&out, packet);
    write_file(path, std::string{out.bytes.begin(), out.bytes.end()});
}
}  // namespace test
}  // namespace collector
}  // namespace rocprofiler
