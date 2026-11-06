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

// This executable includes only the public C API and links the shared library.
// Synthetic input generation, ELF loading, and warmup are outside the timer.
#include "rocprof_trace_decoder/rocprof_trace_decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr uint64_t CodeObjectId = 1;
constexpr uint64_t LoadAddress = 0x100000;
constexpr uint64_t LoadSize = 0x10000;
constexpr uint64_t EntryAddress = LoadAddress + 0x100;
constexpr uint32_t Slots = 16;
constexpr uint32_t PayloadWords = 130;

void check(rocprofiler_thread_trace_decoder_status_t status, const char* operation)
{
    if (status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + rocprof_trace_decoder_get_status_string(status));
}

struct Handle
{
    rocprof_trace_decoder_handle_t value{};
    Handle() { check(rocprof_trace_decoder_create_handle(&value), "create_handle"); }
    ~Handle() { rocprof_trace_decoder_destroy_handle(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};

void write_le(std::vector<uint8_t>& bytes, size_t offset, uint64_t value, size_t count)
{
    for (size_t i = 0; i < count; ++i) bytes[offset + i] = static_cast<uint8_t>(value >> (i * 8));
}

void append_le(std::vector<uint8_t>& bytes, uint64_t value, size_t count)
{
    const size_t offset = bytes.size();
    bytes.resize(offset + count);
    write_le(bytes, offset, value, count);
}

std::vector<uint8_t> make_code_object()
{
    // ELF64 little-endian, gfx90a HSA code object v4. Only metadata is needed;
    // no code or disassembler is involved in these marker-only parses.
    const char names[] = "\0.sqtt_funcmap\0.shstrtab";
    const char metadata[] = "W:64\nP:1:point\nP:2:payload\nR:2:extra_payload_count=130\n";
    constexpr size_t ElfHeaderSize = 64;
    constexpr size_t SectionHeaderSize = 64;
    const size_t string_offset = ElfHeaderSize + sizeof(metadata);
    const size_t section_offset = (string_offset + sizeof(names) + 7) & ~size_t{7};
    std::vector<uint8_t> bytes(section_offset + 3 * SectionHeaderSize, 0);

    std::memcpy(bytes.data(), "\177ELF", 4);
    bytes[4] = 2;                // ELFCLASS64
    bytes[5] = 1;                // ELFDATA2LSB
    bytes[6] = 1;                // ELF version
    bytes[7] = 64;               // AMDGPU HSA
    bytes[8] = 2;                // HSA code object v4
    write_le(bytes, 16, 3, 2);   // ET_DYN
    write_le(bytes, 18, 224, 2); // EM_AMDGPU
    write_le(bytes, 20, 1, 4);   // ELF version
    write_le(bytes, 40, section_offset, 8);
    write_le(bytes, 48, 0x3f, 4); // gfx90a
    write_le(bytes, 52, ElfHeaderSize, 2);
    write_le(bytes, 54, 56, 2); // Program header size; no program headers.
    write_le(bytes, 58, SectionHeaderSize, 2);
    write_le(bytes, 60, 3, 2);
    write_le(bytes, 62, 2, 2); // Section-name table index.

    const size_t marker_section = section_offset + SectionHeaderSize;
    write_le(bytes, marker_section, 1, 4);
    write_le(bytes, marker_section + 4, 1, 4); // SHT_PROGBITS
    write_le(bytes, marker_section + 24, ElfHeaderSize, 8);
    write_le(bytes, marker_section + 32, sizeof(metadata), 8);
    write_le(bytes, marker_section + 48, 1, 8);
    const size_t string_section = section_offset + 2 * SectionHeaderSize;
    write_le(bytes, string_section, 1 + sizeof(".sqtt_funcmap"), 4);
    write_le(bytes, string_section + 4, 3, 4); // SHT_STRTAB
    write_le(bytes, string_section + 24, string_offset, 8);
    write_le(bytes, string_section + 32, sizeof(names), 8);
    write_le(bytes, string_section + 48, 1, 8);
    std::memcpy(bytes.data() + ElfHeaderSize, metadata, sizeof(metadata));
    std::memcpy(bytes.data() + string_offset, names, sizeof(names));
    return bytes;
}

void append_userdata2(std::vector<uint8_t>& bytes, uint32_t value)
{
    // gfx9 REG: token=2, ME=0, userdata register type=3, writes enabled.
    append_le(bytes, 2u | (1u << 7) | (3u << 10) | (1u << 15) | (uint64_t{0xC342} << 16) | (uint64_t{value} << 32), 8);
}

void append_code_object(std::vector<uint8_t>& bytes)
{
    rocprof_trace_decoder_instrument_enable_t enable{};
    enable.char2 = 'R';
    enable.char3 = 'O';
    enable.char4 = 'C';
    append_userdata2(bytes, enable.u32All);
    auto field = [&](rocprof_trace_decoder_codeobj_marker_type_t type, uint32_t value)
    {
        rocprof_trace_decoder_packet_header_t header{};
        header.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ;
        header.type = type;
        append_userdata2(bytes, header.u32All);
        append_userdata2(bytes, value);
    };
    field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO, static_cast<uint32_t>(CodeObjectId));
    field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI, static_cast<uint32_t>(CodeObjectId >> 32));
    field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO, static_cast<uint32_t>(LoadSize));
    field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI, static_cast<uint32_t>(LoadSize >> 32));
    field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO, static_cast<uint32_t>(LoadAddress));
    field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI, static_cast<uint32_t>(LoadAddress >> 32));
    rocprof_trace_decoder_codeobj_marker_tail_t tail{};
    tail.bFromStart = 1;
    field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL, tail.raw);

    // gfx9 REG_CS, ME=0, pipe=0, COMPUTE_PGM_LO=0xC / HI=0xD.
    append_le(bytes, 5u | (1u << 7) | (0xCu << 9) | ((EntryAddress >> 8) << 16), 6);
    append_le(bytes, 5u | (1u << 7) | (0xDu << 9), 6);
}

uint64_t location(uint32_t slot)
{
    // Four wave slots on each of four SIMDs, all on CU 0.
    return (uint64_t{slot % 4} << 10) | (uint64_t{slot / 4} << 14);
}

void append_shaderdata(std::vector<uint8_t>& bytes, uint32_t slot, uint32_t value)
{ append_le(bytes, 12u | (1u << 4) | location(slot) | (uint64_t{value} << 16), 6); }

// gfx12 tokens are nibble-aligned bit fields packed least-significant first.
class Gfx12Writer
{
public:
    explicit Gfx12Writer(std::vector<uint8_t>& out) : bytes(out) {}

    void put(uint64_t value, unsigned bits)
    {
        for (unsigned bit = 0; bit < bits; bit += 4)
        {
            const auto nibble = static_cast<uint8_t>((value >> bit) & 0xF);
            if (high)
                bytes.back() |= static_cast<uint8_t>(nibble << 4);
            else
                bytes.push_back(nibble);
            high = !high;
        }
    }

    // REG: header 0b1001, CS bit 15, regaddr [31:16], regdata [63:32].
    void reg(uint16_t address, uint32_t value, bool cs)
    {
        put(0b1001 | (uint64_t{cs} << 15) | (uint64_t{address} << 16) | (uint64_t{value} << 32), 64);
    }

    void code_object()
    {
        rocprof_trace_decoder_instrument_enable_t enable{};
        enable.char2 = 'R';
        enable.char3 = 'O';
        enable.char4 = 'C';
        reg(0xC342, enable.u32All, false);
        auto field = [&](rocprof_trace_decoder_codeobj_marker_type_t type, uint32_t value)
        {
            rocprof_trace_decoder_packet_header_t header{};
            header.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ;
            header.type = type;
            reg(0xC342, header.u32All, false);
            reg(0xC342, value, false);
        };
        field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO, static_cast<uint32_t>(CodeObjectId));
        field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI, static_cast<uint32_t>(CodeObjectId >> 32));
        field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO, static_cast<uint32_t>(LoadSize));
        field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI, static_cast<uint32_t>(LoadSize >> 32));
        field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO, static_cast<uint32_t>(LoadAddress));
        field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI, static_cast<uint32_t>(LoadAddress >> 32));
        rocprof_trace_decoder_codeobj_marker_tail_t tail{};
        tail.bFromStart = 1;
        field(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL, tail.raw);
        reg(0xC, static_cast<uint32_t>(EntryAddress >> 8), true); // COMPUTE_PGM_LO
        reg(0xD, 0, true);                                         // COMPUTE_PGM_HI
    }

    // Slots use WGP 0 of SA 0: SIMD in [1:0] of the slot, wave ID in [3:2].
    static uint64_t simd_wave(uint32_t slot, unsigned simd_bit, unsigned wave_bit)
    { return (uint64_t{slot % 4} << simd_bit) | (uint64_t{slot / 4} << wave_bit); }

    // WAVE_START: header 0b01100, simd [9:8], wid [19:15], count [31:25].
    void wave_start(uint32_t slot) { put(0b01100 | simd_wave(slot, 8, 15) | (uint64_t{8} << 25), 32); }
    // SHADER_DATA: header 0b0000110, tm [9:7], simd [12:11], wave [21:17], data [53:22].
    void shaderdata(uint32_t slot, uint32_t value)
    { put(0b0000110 | (uint64_t{1} << 7) | simd_wave(slot, 11, 17) | (uint64_t{value} << 22), 56); }
    // WAVE_END: header 0b10101, simd [10:9], wid [19:15].
    void wave_end(uint32_t slot) { put(0b10101 | simd_wave(slot, 9, 15), 20); }

private:
    std::vector<uint8_t>& bytes;
    bool high = false;
};

struct Trace
{
    std::vector<uint8_t> bytes;
    uint64_t markers = 0;
    uint64_t payload_words = 0;
    uint64_t checksum = 0;
};

Trace make_trace(bool gfx12, size_t target_bytes, uint32_t payload_words)
{
    Trace trace;
    trace.bytes.reserve(target_bytes + 4096);
    Gfx12Writer writer(trace.bytes);
    if (gfx12)
    {
        writer.put(0b0010001 | (uint64_t{4} << 7), 64); // RDNA header, trace version 4.
        writer.code_object();
    }
    else
    {
        rocprof_trace_decoder_gfx9_header_t header{};
        header.gfx9_version2 = 5;
        header.DSIMDM = 0xF;
        append_le(trace.bytes, header.raw, 8);
        append_code_object(trace.bytes);
    }
    auto shaderdata = [&](uint32_t slot, uint32_t value)
    {
        if (gfx12)
            writer.shaderdata(slot, value);
        else
            append_shaderdata(trace.bytes, slot, value);
    };
    for (uint32_t slot = 0; slot < Slots; ++slot)
    {
        if (gfx12)
            writer.wave_start(slot);
        else
            append_le(trace.bytes, 3u | (1u << 4) | location(slot) | (8u << 22), 4); // WAVE_START
    }

    const uint32_t id = payload_words == 0 ? 1 : 2;
    const size_t token_bytes = gfx12 ? 7 : 6;
    const size_t rounds = std::max<size_t>(1, target_bytes / (Slots * (payload_words + 1) * token_bytes));
    for (size_t round = 0; round < rounds; ++round)
    {
        for (uint32_t slot = 0; slot < Slots; ++slot)
        {
            shaderdata(slot, id << 2);
            trace.checksum += CodeObjectId + id + (id << 2);
        }
        // Interleave at word granularity so all sixteen payloads are in flight.
        for (uint32_t word = 0; word < payload_words; ++word)
            for (uint32_t slot = 0; slot < Slots; ++slot)
            {
                const uint32_t value = 0x9E3779B9u ^ (static_cast<uint32_t>(round) * 0x85EBCA6Bu) ^ (slot << 16) ^ word;
                shaderdata(slot, value);
                trace.checksum += value;
            }
    }
    for (uint32_t slot = 0; slot < Slots; ++slot)
    {
        if (gfx12)
            writer.wave_end(slot);
        else
            append_le(trace.bytes, 6u | (1u << 4) | location(slot), 2); // WAVE_END
    }
    if (gfx12) trace.bytes.resize(trace.bytes.size() + 16, 0); // NOP padding for the fast reader.
    trace.markers = rounds * Slots;
    trace.payload_words = trace.markers * payload_words;
    return trace;
}

struct Sink
{
    bool synthetic = true;
    uint64_t callbacks = 0;
    uint64_t markers = 0;
    uint64_t payload_words = 0;
    uint64_t checksum = 0;
    uint64_t shaderdata = 0;
};

rocprofiler_thread_trace_decoder_status_t consume(
    rocprofiler_thread_trace_decoder_record_type_t type, void* data, uint64_t count, void* userdata
)
{
    auto& sink = *static_cast<Sink*>(userdata);
    if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA)
    {
        sink.shaderdata += count;
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
    }
    if (type != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
    ++sink.callbacks;
    sink.markers += count;
    // Accumulate in locals: stores through `sink` may alias the record bytes and
    // would serialize the loop on store forwarding, charging the consumer's cost
    // to buffered delivery.
    uint64_t payload_words = 0;
    uint64_t checksum = 0;
    const bool synthetic = sink.synthetic;
    auto* cursor = static_cast<const uint8_t*>(data);
    for (uint64_t i = 0; i < count; ++i)
    {
        const auto& marker = *reinterpret_cast<const rocprofiler_thread_trace_decoder_marker_t*>(cursor);
        // Captured traces may legitimately end inside a payload.
        if (marker.size < sizeof(marker) || !marker.metadata || (marker.payload_count != 0 && !marker.payload) ||
            (synthetic && (marker.marker_flags & ROCPROFILER_THREAD_TRACE_DECODER_MARKER_FLAGS_INCOMPLETE) != 0))
            return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA;
        payload_words += marker.payload_count;
        checksum += marker.kernel_entry.code_object_id + marker.metadata->marker_id + marker.shaderdata.value;
        for (uint32_t word = 0; word < marker.payload_count; ++word) checksum += marker.payload[word];
        cursor += marker.size;
    }
    sink.payload_words += payload_words;
    sink.checksum += checksum;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

struct Options
{
    size_t mib = 32;
    double seconds = 0.25;
    uint64_t iterations = 0; // Zero selects adaptive repetition by elapsed time.
    std::string arch = "all";
    std::string pattern = "all";
    std::string delivery = "all";
    std::string trace;                                   // Captured .att file instead of synthetic patterns.
    std::vector<std::pair<uint64_t, std::string>> codeobjs; // id, path
    bool help = false;
};

std::vector<uint8_t> read_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h")
        {
            options.help = true;
            continue;
        }
        if (++i == argc) throw std::invalid_argument("Missing value for " + argument);
        const std::string value = argv[i];
        size_t used = 0;
        if (argument == "--mib")
        {
            const uint64_t mib = std::stoull(value, &used);
            if (used != value.size() || mib == 0 || mib > 1024)
                throw std::invalid_argument("--mib must be an integer in [1, 1024]");
            options.mib = static_cast<size_t>(mib);
        }
        else if (argument == "--seconds")
        {
            options.seconds = std::stod(value, &used);
            if (used != value.size() || !std::isfinite(options.seconds) || options.seconds <= 0)
                throw std::invalid_argument("--seconds must be positive and finite");
        }
        else if (argument == "--iterations")
        {
            options.iterations = std::stoull(value, &used);
            if (used != value.size() || options.iterations == 0 || value.front() == '-')
                throw std::invalid_argument("--iterations must be a positive integer");
        }
        else if (argument == "--pattern")
        {
            if (value != "all" && value != "points" && value != "payload130")
                throw std::invalid_argument("--pattern must be all, points, or payload130");
            options.pattern = value;
        }
        else if (argument == "--delivery")
        {
            if (value != "all" && value != "disabled" && value != "shaderdata" && value != "buffered" &&
                value != "immediate" && value != "both")
                throw std::invalid_argument(
                    "--delivery must be all, disabled, shaderdata, buffered, immediate, or both"
                );
            options.delivery = value;
        }
        else if (argument == "--arch")
        {
            if (value != "all" && value != "gfx9" && value != "gfx12")
                throw std::invalid_argument("--arch must be all, gfx9, or gfx12");
            options.arch = value;
        }
        else if (argument == "--trace")
            options.trace = value;
        else if (argument == "--codeobj")
        {
            const size_t colon = value.find(':');
            if (colon == std::string::npos || colon == 0) throw std::invalid_argument("--codeobj must be ID:PATH");
            options.codeobjs.emplace_back(std::stoull(value.substr(0, colon)), value.substr(colon + 1));
        }
        else
            throw std::invalid_argument("Unknown option: " + argument);
    }
    return options;
}

void verify(
    const Sink& sink, const Trace& trace, uint64_t iterations, bool disabled, bool immediate, bool shaderdata
)
{
    const uint64_t expected = disabled ? 0 : iterations;
    // Every header and payload word is one raw shaderdata record.
    const uint64_t raw_records = shaderdata ? (trace.markers + trace.payload_words) * iterations : 0;
    if (sink.markers != trace.markers * expected || sink.payload_words != trace.payload_words * expected ||
        sink.checksum != trace.checksum * expected || (disabled && sink.callbacks != 0) ||
        (immediate && sink.callbacks != sink.markers) || sink.shaderdata != raw_records)
        throw std::runtime_error("Consumer count/checksum mismatch");
}

void benchmark(Handle& handle, const Trace& trace, const char* pattern, const char* delivery, const Options& options)
{
    const bool synthetic = options.trace.empty();
    const bool immediate = std::strcmp(delivery, "immediate") == 0;
    const bool markers = immediate || std::strcmp(delivery, "buffered") == 0 || std::strcmp(delivery, "both") == 0;
    const bool shaderdata = std::strcmp(delivery, "shaderdata") == 0 || std::strcmp(delivery, "both") == 0;
    const bool disabled = !markers;
    std::vector<rocprofiler_thread_trace_decoder_record_request_t> requests;
    if (markers)
        requests.push_back(
            {sizeof(rocprofiler_thread_trace_decoder_record_request_t),
             ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER,
             immediate ? uint32_t{ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REQUEST_FLAGS_IMMEDIATE} : 0u}
        );
    if (shaderdata)
        requests.push_back(
            {sizeof(rocprofiler_thread_trace_decoder_record_request_t),
             ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA,
             0u}
        );
    rocprofiler_thread_trace_decoder_record_filter_t filter{sizeof(filter), requests.data(), requests.size()};
    check(rocprof_trace_decoder_set_record_filter(handle.value, &filter), "set_record_filter");

    auto parse = [&](Sink& sink)
    {
        sink.synthetic = synthetic;
        check(
            rocprof_trace_decoder_parse(handle.value, trace.bytes.data(), trace.bytes.size(), consume, &sink), "parse"
        );
    };
    Sink warmup;
    parse(warmup);
    if (synthetic) verify(warmup, trace, 1, disabled, immediate, shaderdata);

    Sink sink;
    uint64_t iterations = 0;
    const auto begin = std::chrono::steady_clock::now();
    double seconds = 0;
    do
    {
        parse(sink);
        ++iterations;
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    }
    while (options.iterations != 0 ? iterations < options.iterations : seconds < options.seconds);
    if (synthetic) verify(sink, trace, iterations, disabled, immediate, shaderdata);

    const double input_mbps = static_cast<double>(trace.bytes.size()) * iterations / seconds / 1.0e6;
    std::cout << std::left << std::setw(18) << pattern << std::setw(11) << delivery << std::right << std::setw(12)
              << trace.bytes.size() << std::setw(11) << iterations << std::fixed << std::setprecision(2)
              << std::setw(14) << input_mbps << std::setprecision(4) << std::setw(11) << seconds << std::setw(14)
              << sink.markers << std::setw(16) << sink.payload_words << std::setw(13) << sink.callbacks << "  "
              << std::setw(10) << sink.shaderdata << "  " << sink.checksum << '\n';
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options = parse_options(argc, argv);
        if (options.help)
        {
            std::cout << "Usage: marker_benchmark [--mib 32] [--seconds 0.25] [--iterations N]\n"
                         "                        [--arch all|gfx9|gfx12] [--pattern all|points|payload130]\n"
                         "                        [--delivery all|disabled|shaderdata|buffered|immediate|both]\n"
                         "                        [--trace FILE.att --codeobj ID:PATH ...]\n"
                         "--mib is approximate encoded input MiB per pattern (1 MiB = 1,048,576 bytes).\n"
                         "--seconds is minimum measured time per case; --iterations overrides it.\n"
                         "--trace replays a captured shader-engine trace with its code objects.\n";
            return 0;
        }

        const char* deliveries[] = {"disabled", "shaderdata", "buffered", "immediate", "both"};
        Handle handle;
        if (!options.trace.empty())
        {
            uint64_t base = 1ull << 40;
            for (const auto& [id, path] : options.codeobjs)
            {
                // Marker metadata is looked up by ID; the address only needs to be unique.
                const auto elf = read_file(path);
                check(
                    rocprof_trace_decoder_codeobj_load(handle.value, id, base, elf.size(), elf.data(), elf.size()),
                    "codeobj_load"
                );
                base += 1ull << 32;
            }
            Trace trace;
            trace.bytes = read_file(options.trace);
            std::cout << "Single-thread public C API; captured trace " << options.trace << ".\n"
                      << "pattern           delivery    input_bytes iterations    input_MB/s    seconds"
                         "       markers   payload_words    callbacks  shaderdata  checksum\n";
            for (const char* delivery : deliveries)
                if (options.delivery == "all" || options.delivery == delivery)
                    benchmark(handle, trace, "trace", delivery, options);
            return 0;
        }

        const auto elf = make_code_object();
        check(
            rocprof_trace_decoder_codeobj_load(
                handle.value, CodeObjectId, LoadAddress, LoadSize, elf.data(), elf.size()
            ),
            "codeobj_load"
        );
        std::cout << "Single-thread public C API; 16 interleaved wave slots; no packed clocks.\n"
                     "input_MB/s uses actual encoded SQTT bytes (1 MB = 1,000,000 bytes).\n"
                     "Setup and warmup excluded; disabled uses an empty record filter.\n"
                     "pattern           delivery    input_bytes iterations    input_MB/s    seconds"
                     "       markers   payload_words    callbacks  shaderdata  checksum\n";
        for (const char* arch : {"gfx9", "gfx12"})
            for (const char* pattern : {"points", "payload130"})
            {
                if (options.arch != "all" && options.arch != arch) continue;
                if (options.pattern != "all" && options.pattern != pattern) continue;
                const bool gfx12 = std::strcmp(arch, "gfx12") == 0;
                const auto trace = make_trace(
                    gfx12, options.mib * 1024 * 1024, std::strcmp(pattern, "points") == 0 ? 0 : PayloadWords
                );
                const std::string label = std::string(arch) + '-' + pattern;
                for (const char* delivery : deliveries)
                {
                    if (options.delivery != "all" && options.delivery != delivery) continue;
                    benchmark(handle, trace, label.c_str(), delivery, options);
                }
            }
    }
    catch (const std::exception& error)
    {
        std::cerr << "marker_benchmark: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
