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

#pragma once

#include <cstddef>
#include <cstdint>

#include "mi400parser.h" // mi400::TokenGenerator::QuickToken

namespace mi400::quick_scan
{
// Purpose-built fast scanner that walks an mi400 SQTT token stream and
// captures only the rare-token cluster (REG, REG_INIT, EVENT, EVENT_SYNC)
// with full 64-bit contents. Skips everything else with a flat byte-length
// table — no visitor calls, no FIFO updates, no globaltime tracking, no
// Token{} construction.
//
// Writes up to `out_cap` entries into `out` in stream order; returns the
// number written. If the buffer would produce more than `out_cap` rare
// tokens, the excess are silently dropped (callers should size out_cap
// generously — observed counts are O(50) per 15 MB of trace).
//
// Single-threaded, no exceptions. The buffer is read up to position
// `size`; the windowed inner loop loads 8 bytes at a time, so a tail
// fallback handles the last <16 bytes safely.
size_t scan_mi400(const uint8_t* buf, size_t size, TokenGenerator::QuickToken* out, size_t out_cap);

} // namespace mi400::quick_scan
