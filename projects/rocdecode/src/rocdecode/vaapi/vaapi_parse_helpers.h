/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

// File-local helper for parsing a comma-separated list of device indices,
// such as the value of ROCR_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES.
//
// The caller passes the result of std::getenv() directly. getenv returns
// a pointer into the process environment; mutating it is undefined
// behaviour (C11 §7.22.4.6). This helper never writes to the source
// buffer.
//
// Threat model
// ------------
// These env vars are user-controlled and read on a privileged decoder
// init path, so the parser is hardened against malformed input rather
// than trusting the runtime to filter it. Specifically:
//
//   * Buffer exhaustion — input is length-capped before any work
//     (kMaxEnvLength). strlen() terminates on the first NUL, so a
//     null-byte truncation attack cannot smuggle data past the cap.
//   * Integer overflow — per-token digit cap (kMaxTokenDigits) bounds
//     the manual digit accumulator well below INT_MAX, so no
//     atoi/strtol-style overflow is reachable.
//   * Charset smuggling — the single-pass scanner accepts only ASCII
//     digits ('0'-'9') and commas. Everything else is rejected on
//     first sight: whitespace, control characters (NUL-in-middle, BS,
//     BEL, ESC, DEL), high-bit bytes (>= 0x80, so UTF-8 payloads
//     including Unicode commas U+FF0C/U+060C, BOM, combining marks,
//     and homoglyphs), sign characters ('-', '+'), alternate-base
//     prefixes ('x', 'o', 'b'), radix points, and alternate delimiters
//     (';', ':', whitespace).
//   * Locale-dependent parsing — digit accumulation is manual ASCII
//     arithmetic; std::locale / std::isdigit / strtol are not invoked.
//   * Static / global state — the parser is a pure function: no strtok
//     save-slot, no shared buffer, no mutation of argv/environ. Safe to
//     call concurrently from multiple decoder init threads.
//   * TOCTOU on the environment — the parser reads through a pointer
//     obtained before the scan; a racing setenv() from another thread
//     cannot change what this call sees after strlen() returns.
//
// Hard-reject rules (return empty vector):
//   * nullptr or empty input
//   * length > kMaxEnvLength bytes
//   * any byte outside [0-9,]  (rejected immediately — single pass)
//   * any token longer than kMaxTokenDigits digits
//   * any token whose numeric value exceeds kMaxDeviceIndex
//
// Tolerated (silently accepted):
//   * empty tokens from leading/trailing/consecutive commas
//     (",0,1" / "0,1," / "0,,1" all yield {0,1})
//   * duplicate indices — deduplicated, preserving first-seen order
//     ("1,0,1,0" yields {1,0})
//
// The returned vector is not sorted; callers apply their own ordering.

namespace rocm_vaapi_parse_detail {
// Hard cap on the total env-var length we will parse. 8192 bytes holds
// ~1365 4-digit indices plus commas — comfortably above 1024 GPUs, which
// is itself ~16x any realistic AMD node (including extreme PCIe-riser
// mining / render rigs).
inline constexpr std::size_t kMaxEnvLength   = 8192;
// Per-token digit cap. 4 digits keeps the manual digit accumulator below
// INT_MAX independent of the numeric-range check below, giving us a
// belt-and-braces overflow guard.
inline constexpr std::size_t kMaxTokenDigits = 4;
// Numeric upper bound for a single device index. 1023 allows up to 1024
// GPUs per process, which is well beyond any realistic AMD topology yet
// small enough that an accidentally pasted 32-bit integer does not slip
// past the range check.
inline constexpr int         kMaxDeviceIndex = 1023;
}  // namespace rocm_vaapi_parse_detail

static inline std::vector<int> ParseVisibleDevicesCsv(const char* env) {
    std::vector<int> out;
    if (env == nullptr) {
        return out;
    }

    const std::size_t len = std::strlen(env);
    if (len == 0 || len > rocm_vaapi_parse_detail::kMaxEnvLength) {
        return out;
    }

    // Single-pass scan: validate charset and tokenise simultaneously.
    // At any point, we are either skipping commas (empty tokens) or
    // accumulating a digit run. Anything else rejects immediately.
    std::size_t pos = 0;
    while (pos < len) {
        // Skip commas — empty tokens are tolerated.
        if (env[pos] == ',') {
            ++pos;
            continue;
        }

        // Start of a digit token. First byte must be a digit.
        if (env[pos] < '0' || env[pos] > '9') {
            return {};
        }

        // Accumulate the digit run.
        int value = 0;
        std::size_t digits = 0;
        while (pos < len && env[pos] >= '0' && env[pos] <= '9') {
            ++digits;
            if (digits > rocm_vaapi_parse_detail::kMaxTokenDigits) {
                return {};
            }
            value = value * 10 + (env[pos] - '0');
            ++pos;
        }

        // After the digit run, the next byte must be a comma or end-of-input.
        if (pos < len && env[pos] != ',') {
            return {};
        }

        if (value > rocm_vaapi_parse_detail::kMaxDeviceIndex) {
            return {};
        }

        // Dedup via linear scan — efficient for realistic GPU counts
        // (typically 1-8), avoids <set> tree overhead.
        if (std::none_of(out.begin(), out.end(),
                         [value](int existing) { return existing == value; })) {
            out.push_back(value);
        }
    }
    return out;
}
