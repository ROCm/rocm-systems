// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// att-bench: single-threaded rare-token scan benchmark for .att files.
//   0 - fast scan only, no reference checker.
//   1 - fast scan plus reference checker.
//
// The benchmark links the static library and reaches into internal headers
// because the public C API only exposes the all-or-nothing entry point
// rocprof_trace_decoder_parse_data().

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gfx10/token_types.h" // RdnaType enum
#include "gfx12/gfx12parser.h"  // gfx12::TokenGenerator (reference walker)
#include "gfx12/rare_scan.h"    // gfx12::rare_scan::scan_gfx12
#include "gfx9/gfx9token.h"     // gfx9::token_len_dict (for reference walker)
#include "gfx9/rare_scan.h"     // gfx9::rare_scan::scan_gfx9 (fast rare-only scanner)
#include "iterate_tokens.hpp"  // IterateTokens_internal (templated visitor)
#include "mi400/mi400parser.h" // mi400::TokenGenerator::RareToken
#include "mi400/rare_scan.h"   // mi400::rare_scan::scan_mi400 (fast rare-only scanner)
#include "stitch/stitch.hpp"   // Stitcher, ICodeServicer
#include "trace_parser.hpp"    // AnalyseBinary_internal, DetectArch_internal, TraceArch

namespace
{

bool g_check_rare_reference = true;

// ---------------------------------------------------------------------------
// File I/O (excluded from timing)
// ---------------------------------------------------------------------------
std::vector<uint8_t> slurp(const char* path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    auto                 size = static_cast<std::streamsize>(f.tellg());
    std::vector<uint8_t> buf(size);
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(buf.data()), size))
        throw std::runtime_error(std::string("short read on ") + path);
    return buf;
}

// ---------------------------------------------------------------------------
// Token-type name tables (mirrors the enum order in the source headers; the
// bench owns these so it doesn't have to grow a string export in the library)
// ---------------------------------------------------------------------------
const char* arch_name(TraceArch a)
{
    switch (a)
    {
        case TraceArch::GFX9: return "gfx9";
        case TraceArch::GFX10: return "gfx10";
        case TraceArch::GFX11: return "gfx11";
        case TraceArch::GFX12: return "gfx12";
        case TraceArch::MI400: return "mi400";
        default: return "unknown";
    }
}

// gfx10 / gfx11 / gfx12 / mi400 share the RdnaType enum (gfx10/token_types.h).
const char* rdna_name(int type)
{
    switch (static_cast<RdnaType>(type))
    {
        case RdnaType::UNKNOWN: return "UNKNOWN";
        case RdnaType::VALU_INST: return "VALU_INST";
        case RdnaType::IMM_ONE: return "IMM_ONE";
        case RdnaType::IMMEDIATE: return "IMMEDIATE";
        case RdnaType::WAVE_READY: return "WAVE_READY";
        case RdnaType::NEW_PC_GFX10: return "NEW_PC_GFX10";
        case RdnaType::WAVE_END: return "WAVE_END";
        case RdnaType::WAVE_START: return "WAVE_START";
        case RdnaType::WAVE_START_EXT: return "WAVE_START_EXT";
        case RdnaType::WAVE_ALLOC: return "WAVE_ALLOC";
        case RdnaType::SHADER_DATA: return "SHADER_DATA";
        case RdnaType::SHADER_DATA_SHORT: return "SHADER_DATA_SHORT";
        case RdnaType::UTIL_COUNTER_GFX10: return "UTIL_COUNTER_GFX10";
        case RdnaType::TIME: return "TIME";
        case RdnaType::NOP: return "NOP";
        case RdnaType::MISC_GFX10: return "MISC_GFX10";
        case RdnaType::EVENT: return "EVENT";
        case RdnaType::EVENT_SYNC: return "EVENT_SYNC";
        case RdnaType::REG: return "REG";
        case RdnaType::REG_INIT: return "REG_INIT";
        case RdnaType::TIMESTAMP: return "TIMESTAMP";
        case RdnaType::HEADER: return "HEADER";
        case RdnaType::INST: return "INST";
        case RdnaType::UTIL_COUNTER_GFX11: return "UTIL_COUNTER_GFX11";
        case RdnaType::EXEC_POPCOUNT1: return "EXEC_POPCOUNT1";
        case RdnaType::EXEC_POPCOUNT3: return "EXEC_POPCOUNT3";
        case RdnaType::NEW_PC_GFX12: return "NEW_PC_GFX12";
        case RdnaType::LDS_CONFIG: return "LDS_CONFIG";
        case RdnaType::MEDIUM_TIME: return "MEDIUM_TIME";
        default: return "?";
    }
}

// gfx9::sqtt_token_type_t (gfx9/gfx9token.h) — different enum space.
const char* gfx9_name(int type)
{
    static const char* names[] = {
        "MISC", "TIME",   "REG",       "WAVE_START",  "WAVE_ALLOC", "REG_CS",  "WAVE_END",   "EVENT",
        "EVENT_CS", "EVENT_GFX1", "INST", "INST_PC", "SHADERDATA", "ISSUE", "PERF", "REG_CS_PRIV"};
    if (type < 0 || type >= int(sizeof(names) / sizeof(names[0]))) return "?";
    return names[type];
}

const char* token_name(TraceArch arch, int type)
{
    return arch == TraceArch::GFX9 ? gfx9_name(type) : rdna_name(type);
}

// ---------------------------------------------------------------------------
// Stitcher / callback plumbing for mode 2 (waves, no stitch)
// ---------------------------------------------------------------------------
class ErroringCodeServicer : public ICodeServicer
{
public:
    assemblyLine GetInstruction(pcinfo_t, int) override { throw std::runtime_error("bench: ISA not provided"); }
};

// Stitcher::stitch is now virtual (see source/stitch/stitch.hpp); this
// override turns it into a no-op so mode 2 measures only token decode +
// wave-state reconstruction.
class NoopStitcher : public Stitcher
{
public:
    using Stitcher::Stitcher;
    void stitch(WaveDataInternal&) override { /* skip ISA stitching entirely */ }
};

// Discards every wave/occupancy/shader-data event the parser emits.
rocprofiler_thread_trace_decoder_status_t
discard_callback(rocprofiler_thread_trace_decoder_record_type_t, void*, uint64_t, void*)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------
struct Stats
{
    double  seconds = 0.0;
    size_t  bytes = 0;
    size_t  total_tokens = 0;
    TraceArch arch = TraceArch::UNKNOWN;
};

void print_throughput(const Stats& s, const char* mode)
{
    double mb = double(s.bytes) / (1024.0 * 1024.0);
    double mbps = s.seconds > 0 ? mb / s.seconds : 0.0;
    std::printf("  mode=%s arch=%s bytes=%zu time=%.3f s throughput=%.2f MB/s\n",
                mode,
                arch_name(s.arch),
                s.bytes,
                s.seconds,
                mbps);
}

int run_tokens(const std::vector<uint8_t>& buf, const char* path)
{
    // Flat array indexed by token type id. Both RdnaType and
    // gfx9::sqtt_token_type_t are small dense enums (< 64) so 256 slots
    // is comfortably enough and keeps the inner loop branch-free / cache-hot.
    constexpr size_t       kHistSize = 256;
    std::array<size_t, kHistSize> hist{};
    Stats                  s;
    s.bytes = buf.size();

    // Always capture full contents of EVENT / EVENT_SYNC / REG / REG_INIT
    // tokens — that's the actual point of tokens-mode for downstream consumers.
    // Pre-reserve so the timed loop never reallocates (these are uncommon but
    // can number in the tens of thousands on long traces).
    std::vector<mi400::TokenGenerator::RareToken> rare;
    rare.reserve(32 * 1024);

    auto t0 = std::chrono::steady_clock::now();
    s.arch = IterateTokens_internal(
        buf.data(),
        buf.size(),
        [&](int type) {
            if (unsigned(type) < kHistSize) ++hist[type];
        },
        &rare);
    auto t1 = std::chrono::steady_clock::now();
    s.seconds = std::chrono::duration<double>(t1 - t0).count();
    for (size_t c : hist) s.total_tokens += c;

    if (s.arch == TraceArch::UNKNOWN)
    {
        std::fprintf(stderr, "%s: failed to detect architecture from header\n", path);
        return 1;
    }

    std::printf("file: %s\n", path);
    print_throughput(s, "tokens");
    std::printf("  total_tokens=%zu\n", s.total_tokens);
    std::printf("  histogram (sorted by count):\n");
    std::vector<std::pair<int, size_t>> sorted;
    sorted.reserve(32);
    for (size_t i = 0; i < kHistSize; ++i)
        if (hist[i]) sorted.emplace_back(int(i), hist[i]);
    std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
    for (auto& [type, count] : sorted)
        std::printf("    %-22s %12zu\n", token_name(s.arch, type), count);

    // Per-type counts derived from the rare-token vector — should match the
    // histogram entries for those four types (sanity check that capture
    // preserved every event without dropping or duplicating).
    std::array<size_t, kHistSize> rare_hist{};
    for (auto& r : rare) ++rare_hist[r.type];
    std::printf("  rare-token capture: %zu total\n", rare.size());
    for (int t : {int(RdnaType::REG),
                  int(RdnaType::REG_INIT),
                  int(RdnaType::EVENT),
                  int(RdnaType::EVENT_SYNC)})
        if (rare_hist[t]) std::printf("    %-22s %12zu\n", rdna_name(t), rare_hist[t]);
    return 0;
}

int run_waves(const std::vector<uint8_t>& buf, const char* path)
{
    Stats s;
    s.bytes = buf.size();

    auto code_service = std::make_shared<ErroringCodeServicer>();
    NoopStitcher stitcher(code_service, &discard_callback, /*cbdata=*/nullptr);

    CppReturnInfo info{};
    s.arch = DetectArch_internal(buf.data(), buf.size());

    auto t0 = std::chrono::steady_clock::now();
    auto parser = AnalyseBinary_internal(info, buf.data(), buf.size(), /*gfx9_target_cu=*/-1, stitcher);
    auto t1 = std::chrono::steady_clock::now();
    s.seconds = std::chrono::duration<double>(t1 - t0).count();

    if (!parser)
    {
        std::fprintf(stderr, "%s: AnalyseBinary_internal returned null (unrecognised header?)\n", path);
        return 1;
    }

    std::printf("file: %s\n", path);
    print_throughput(s, "waves");
    std::printf("  packet_lost=%s realtime_freq=%lu\n",
                info.bPacketLost ? "yes" : "no",
                static_cast<unsigned long>(info.realtime_frequency));
    return 0;
}

int run_stitch(const std::vector<uint8_t>&, const char* path)
{
    std::fprintf(stderr, "%s: stitch mode is not yet implemented\n", path);
    return 2;
}

// Reference walker for gfx9: byte-for-byte, using a local copy of
// token_len_dict (mirroring gfx9token.cpp:29). Mirrors the spec from
// gfx9token.cpp:parseOne but skips Token{} construction and only emits
// captures for {REG, REG_CS, EVENT, EVENT_CS, REG_CS_PRIV}. Used solely as
// an independent oracle for the SIMD scanner — the duplicated array is the
// point: this walker must NOT share tables with the production scanner.
size_t gfx9_rare_reference(const uint8_t* buf, size_t size,
                           gfx9::rare_scan::RareToken* out, size_t out_cap)
{
    // Bits per token, indexed by nibble; mirrors gfx9::token_len_dict.
    static constexpr int len_bits[16] =
        {16, 64, 64, 32, 16, 48, 16, 16, 16, 16, 16, 64, 48, 32, 64, 48};
    constexpr uint16_t mask = (1u << 2) | (1u << 5) | (1u << 7) | (1u << 8) | (1u << 15);
    size_t bp = 0, n = 0;
    while (bp + 8 <= size && n < out_cap)
    {
        unsigned nibble = buf[bp] & 0x0F;
        unsigned bytes  = static_cast<unsigned>(len_bits[nibble]) / 8u;
        if ((mask >> nibble) & 1u)
        {
            uint64_t contents;
            std::memcpy(&contents, buf + bp, sizeof(contents));
            out[n++] = gfx9::rare_scan::RareToken{contents, nibble};
        }
        bp += bytes ? bytes : 2;
    }
    return n;
}

size_t gfx12_rare_reference(const uint8_t* buf, size_t size,
                            gfx12::rare_scan::RareToken* out, size_t out_cap)
{
    size_t n = 0;
    try
    {
        gfx12::TokenGenerator gen(buf, size, 0, 0);
        while (gen.nextValid() && n < out_cap)
        {
            auto token = gen.next();
            if (static_cast<unsigned>(token.type) < 5)
                out[n++] = gfx12::rare_scan::RareToken{token.contents, static_cast<uint32_t>(token.type)};
        }
    }
    catch (const std::exception&)
    {}
    return n;
}

// rare-mode: time the purpose-built rare-only scanner. mi400 uses
// mi400::rare_scan::scan_mi400, gfx9 uses gfx9::rare_scan::scan_gfx9.
// Cross-checks output against an independent reference walker.
int run_rare(const std::vector<uint8_t>& buf, const char* path)
{
    Stats s;
    s.bytes = buf.size();
    s.arch  = DetectArch_internal(buf.data(), buf.size());

    if (s.arch == TraceArch::MI400)
    {
        constexpr size_t                              kCap = 32 * 1024;
        std::vector<mi400::TokenGenerator::RareToken> fast(kCap);

        auto t0      = std::chrono::steady_clock::now();
        size_t n_fast = mi400::rare_scan::scan_mi400(buf.data(), buf.size(), fast.data(), fast.size());
        auto t1       = std::chrono::steady_clock::now();
        s.seconds     = std::chrono::duration<double>(t1 - t0).count();
        fast.resize(n_fast);

        std::vector<mi400::TokenGenerator::RareToken> ref;
        bool match = true;
        size_t mismatch_at = 0;
        if (g_check_rare_reference)
        {
            // Reference path: existing scan_types<true>. Not timed; cross-check only.
            ref.reserve(kCap);
            IterateTokens_internal(
                buf.data(), buf.size(), [](int) { /* discard */ }, &ref);

            match = (ref.size() == fast.size());
            if (match)
            {
                for (size_t i = 0; i < ref.size(); ++i)
                {
                    if (ref[i].type != fast[i].type || ref[i].contents != fast[i].contents)
                    {
                        match       = false;
                        mismatch_at = i;
                        break;
                    }
                }
            }
        }

        std::printf("file: %s\n", path);
        print_throughput(s, "rare");
        if (g_check_rare_reference)
            std::printf("  rare captured: fast=%zu  reference=%zu  match=%s\n",
                        fast.size(),
                        ref.size(),
                        match ? "yes" : "NO");
        else
            std::printf("  rare captured: fast=%zu  reference=skipped  match=skipped\n", fast.size());

        if (g_check_rare_reference && !match)
        {
            std::fprintf(stderr,
                         "  MISMATCH at index %zu: fast={type=%u contents=0x%016llx} ref={type=%u contents=0x%016llx}\n",
                         mismatch_at,
                         mismatch_at < fast.size() ? fast[mismatch_at].type : 0u,
                         mismatch_at < fast.size() ? (unsigned long long) fast[mismatch_at].contents : 0ull,
                         mismatch_at < ref.size() ? ref[mismatch_at].type : 0u,
                         mismatch_at < ref.size() ? (unsigned long long) ref[mismatch_at].contents : 0ull);
        }

        // Per-type breakdown (so the user can sanity-check vs. tokens-mode).
        std::array<size_t, 32> hist{};
        for (auto& r : fast) ++hist[r.type & 31];
        for (int t : {int(RdnaType::REG), int(RdnaType::REG_INIT), int(RdnaType::EVENT), int(RdnaType::EVENT_SYNC)})
            if (hist[t]) std::printf("    %-22s %12zu\n", rdna_name(t), hist[t]);

        return match ? 0 : 1;
    }
    else if (s.arch == TraceArch::GFX12)
    {
        constexpr size_t                        kCap = 128 * 1024;
        std::vector<gfx12::rare_scan::RareToken> fast(kCap);

        auto t0       = std::chrono::steady_clock::now();
        size_t n_fast = gfx12::rare_scan::scan_gfx12(buf.data(), buf.size(), fast.data(), fast.size());
        auto t1       = std::chrono::steady_clock::now();
        s.seconds     = std::chrono::duration<double>(t1 - t0).count();
        fast.resize(n_fast);

        std::vector<gfx12::rare_scan::RareToken> ref;
        bool   match       = true;
        size_t mismatch_at = 0;
        if (g_check_rare_reference)
        {
            ref.resize(kCap);
            size_t n_ref = gfx12_rare_reference(buf.data(), buf.size(), ref.data(), ref.size());
            ref.resize(n_ref);

            match = (ref.size() == fast.size());
            if (match)
            {
                for (size_t i = 0; i < ref.size(); ++i)
                {
                    if (ref[i].type != fast[i].type || ref[i].contents != fast[i].contents)
                    {
                        match       = false;
                        mismatch_at = i;
                        break;
                    }
                }
            }
        }

        std::printf("file: %s\n", path);
        print_throughput(s, "rare");
        if (g_check_rare_reference)
            std::printf("  rare captured: fast=%zu  reference=%zu  match=%s\n",
                        fast.size(),
                        ref.size(),
                        match ? "yes" : "NO");
        else
            std::printf("  rare captured: fast=%zu  reference=skipped  match=skipped\n", fast.size());
        if (g_check_rare_reference && !match)
        {
            std::fprintf(stderr,
                         "  MISMATCH at index %zu: fast={type=%u contents=0x%016llx} ref={type=%u contents=0x%016llx}\n",
                         mismatch_at,
                         mismatch_at < fast.size() ? fast[mismatch_at].type : 0u,
                         mismatch_at < fast.size() ? (unsigned long long) fast[mismatch_at].contents : 0ull,
                         mismatch_at < ref.size() ? ref[mismatch_at].type : 0u,
                         mismatch_at < ref.size() ? (unsigned long long) ref[mismatch_at].contents : 0ull);
        }

        std::array<size_t, 32> hist{};
        for (auto& r : fast) ++hist[r.type & 31];
        for (int t : {int(RdnaType::REG), int(RdnaType::REG_INIT), int(RdnaType::EVENT), int(RdnaType::EVENT_SYNC)})
            if (hist[t]) std::printf("    %-22s %12zu\n", rdna_name(t), hist[t]);

        return match ? 0 : 1;
    }
    else if (s.arch == TraceArch::GFX9)
    {
        // gfx9: skip the 8-byte header before scanning.
        constexpr size_t header = sizeof(rocprof_trace_decoder_gfx9_header_t);
        if (buf.size() < header)
        {
            std::fprintf(stderr, "%s: buffer smaller than gfx9 header\n", path);
            return 1;
        }
        const uint8_t* tokens     = buf.data() + header;
        const size_t   tokens_len = buf.size() - header;

        constexpr size_t                          kCap = 64 * 1024;
        std::vector<gfx9::rare_scan::RareToken>   fast(kCap);

        auto t0       = std::chrono::steady_clock::now();
        size_t n_fast = gfx9::rare_scan::scan_gfx9(tokens, tokens_len, fast.data(), fast.size());
        auto t1       = std::chrono::steady_clock::now();
        s.seconds     = std::chrono::duration<double>(t1 - t0).count();
        // Time only the scanner; we report MB/s against full file size for
        // apples-to-apples vs mi400 (the header is negligible).
        fast.resize(n_fast);

        std::vector<gfx9::rare_scan::RareToken> ref;
        bool   match       = true;
        size_t mismatch_at = 0;
        if (g_check_rare_reference)
        {
            // Reference: independent byte-walker over the same token region.
            ref.resize(kCap);
            size_t n_ref = gfx9_rare_reference(tokens, tokens_len, ref.data(), ref.size());
            ref.resize(n_ref);

            match = (ref.size() == fast.size());
            if (match)
            {
                for (size_t i = 0; i < ref.size(); ++i)
                {
                    if (ref[i].type != fast[i].type || ref[i].contents != fast[i].contents)
                    {
                        match       = false;
                        mismatch_at = i;
                        break;
                    }
                }
            }
        }

        std::printf("file: %s\n", path);
        print_throughput(s, "rare");
        if (g_check_rare_reference)
            std::printf("  rare captured: fast=%zu  reference=%zu  match=%s\n",
                        fast.size(),
                        ref.size(),
                        match ? "yes" : "NO");
        else
            std::printf("  rare captured: fast=%zu  reference=skipped  match=skipped\n", fast.size());
        if (g_check_rare_reference && !match)
        {
            std::fprintf(stderr,
                         "  MISMATCH at index %zu: fast={type=%u contents=0x%016llx} ref={type=%u contents=0x%016llx}\n",
                         mismatch_at,
                         mismatch_at < fast.size() ? fast[mismatch_at].type : 0u,
                         mismatch_at < fast.size() ? (unsigned long long) fast[mismatch_at].contents : 0ull,
                         mismatch_at < ref.size() ? ref[mismatch_at].type : 0u,
                         mismatch_at < ref.size() ? (unsigned long long) ref[mismatch_at].contents : 0ull);
        }

        std::array<size_t, 16> hist{};
        for (auto& r : fast) ++hist[r.type & 15];
        for (int t : {2, 5, 7, 8, 15})
            if (hist[t]) std::printf("    %-22s %12zu\n", gfx9_name(t), hist[t]);

        return match ? 0 : 1;
    }
    else
    {
        std::fprintf(stderr, "%s: rare mode supports mi400 / gfx12 / gfx9 only (detected %s)\n",
                     path, arch_name(s.arch));
        return 1;
    }
}

// Walks the slow next() path so we have access to Token::time. Reports the
// cycle window from first WAVE_START to last WAVE_END (mi400 only — uses the
// mi400 generator directly).
int run_spans(const std::vector<uint8_t>& buf, const char* path)
{
    auto arch = DetectArch_internal(buf.data(), buf.size());
    if (arch != TraceArch::MI400)
    {
        std::fprintf(stderr, "%s: spans mode currently only supports mi400 (detected %s)\n",
                     path, arch_name(arch));
        return 1;
    }

    int64_t first_start = -1;
    int64_t last_end    = -1;
    size_t  n_starts = 0, n_ends = 0;

    try
    {
        mi400::TokenGenerator gen(buf.data(), buf.size(), 0, 0);
        while (gen.nextValid())
        {
            gfx10::Token tok = gen.next();
            if (tok.type == RdnaType::WAVE_START)
            {
                if (first_start < 0) first_start = tok.time;
                ++n_starts;
            }
            else if (tok.type == RdnaType::WAVE_END)
            {
                last_end = tok.time;
                ++n_ends;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s: decode aborted (%s) — using events seen so far\n", path, e.what());
    }

    std::printf("file: %s\n", path);
    std::printf("  arch=%s\n", arch_name(arch));
    std::printf("  WAVE_START events: %zu  (first time = %lld cycles)\n",
                n_starts, static_cast<long long>(first_start));
    std::printf("  WAVE_END   events: %zu  (last  time = %lld cycles)\n",
                n_ends, static_cast<long long>(last_end));
    if (first_start >= 0 && last_end >= 0)
    {
        int64_t span = last_end - first_start;
        std::printf("  span (last_END - first_START) = %lld cycles\n",
                    static_cast<long long>(span));
    }
    else
    {
        std::printf("  span: n/a (missing WAVE_START or WAVE_END)\n");
    }
    return 0;
}

void usage(const char* argv0)
{
    std::fprintf(stderr,
                 "usage: %s <mode> <att-file> [<att-file>...]\n"
                 "  modes:\n"
                 "    0       fast rare-token scan, skip reference checker\n"
                 "    1       fast rare-token scan, run reference checker\n",
                 argv0);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        usage(argv[0]);
        return 64;
    }

    const std::string mode = argv[1];
    if (mode == "0")
        g_check_rare_reference = false;
    else if (mode == "1")
        g_check_rare_reference = true;
    else
    {
        std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        usage(argv[0]);
        return 64;
    }

    int worst = 0;
    for (int i = 2; i < argc; ++i)
    {
        try
        {
            auto buf = slurp(argv[i]);
            int  rc = run_rare(buf, argv[i]);
            if (rc > worst) worst = rc;
        }
        catch (const std::exception& e)
        {
            std::fprintf(stderr, "%s: %s\n", argv[i], e.what());
            worst = std::max(worst, 1);
        }
    }
    return worst;
}
