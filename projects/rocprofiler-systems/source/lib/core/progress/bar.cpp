// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/progress/bar.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>

namespace rocprofsys::progress
{
namespace detail
{
namespace
{
// Eighth-block precision (UTF-8 bytes for U+2588 LEFT EIGHTH BLOCK family).
inline constexpr std::array<const char*, 9> BLOCK_CHARS = {
    " ",             // 0/8 (empty)
    "\xe2\x96\x8f",  // 1/8 U+258F
    "\xe2\x96\x8e",  // 2/8 U+258E
    "\xe2\x96\x8d",  // 3/8 U+258D
    "\xe2\x96\x8c",  // 4/8 U+258C
    "\xe2\x96\x8b",  // 5/8 U+258B
    "\xe2\x96\x8a",  // 6/8 U+258A
    "\xe2\x96\x89",  // 7/8 U+2589
    "\xe2\x96\x88"   // 8/8 U+2588 (full block)
};
inline constexpr std::size_t PARTS_PER_CELL = 8;
}  // namespace

std::string
format_bar(bar_style style, std::size_t width, double frac)
{
    frac = std::clamp(frac, 0.0, 1.0);

    std::string out;
    out.reserve(width * 4);  // worst case: 3 UTF-8 bytes per cell + brackets

    const auto filled = static_cast<std::size_t>(frac * static_cast<double>(width));
    const auto empty  = width - filled;

    switch(style)
    {
        case bar_style::ascii_brackets:
        {
            out.push_back('[');
            out.append(filled, '#');
            out.append(empty, '.');
            out.push_back(']');
            break;
        }
        case bar_style::ascii_arrow:
        {
            out.push_back('[');
            if(filled == 0)
                out.append(width, ' ');
            else if(filled >= width)
                out.append(width, '=');
            else
            {
                out.append(filled - 1, '=');
                out.push_back('>');
                out.append(width - filled, ' ');
            }
            out.push_back(']');
            break;
        }
        case bar_style::unicode_blocks:
        {
            // Clamp eighths to width*PARTS_PER_CELL so a floating-point
            // rounding edge cannot underflow trailing_pad (size_t).
            const auto max_eighths   = width * PARTS_PER_CELL;
            const auto total_eighths = std::min(
                static_cast<std::size_t>(frac * static_cast<double>(max_eighths)),
                max_eighths);
            const auto full         = total_eighths / PARTS_PER_CELL;
            const auto remainder    = total_eighths % PARTS_PER_CELL;
            const auto trailing_pad = width - full - (remainder > 0 ? 1 : 0);

            out.push_back('[');
            for(std::size_t i = 0; i < full; ++i)
                out.append(BLOCK_CHARS[PARTS_PER_CELL]);
            if(remainder > 0) out.append(BLOCK_CHARS[remainder]);
            out.append(trailing_pad, ' ');
            out.push_back(']');
            break;
        }
        case bar_style::unicode_dots:
        {
            // U+25CF (filled), U+25CB (empty). No surrounding brackets.
            for(std::size_t i = 0; i < filled; ++i)
                out.append("\xe2\x97\x8f");
            for(std::size_t i = 0; i < empty; ++i)
                out.append("\xe2\x97\x8b");
            break;
        }
        case bar_style::text_only: break;
    }
    return out;
}
}  // namespace detail

namespace
{
// Carriage return + ANSI "Erase In Line" (mode 2 = entire line). Together they
// move the cursor to column 0 and clear whatever was previously rendered there,
// so the next write starts on a clean line. Used between throttled redraws.
constexpr const char* ANSI_RESET_LINE = "\r\033[2K";

[[nodiscard]] std::int64_t
now_ns() noexcept
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

[[nodiscard]] bool
stream_is_tty(std::FILE* stream) noexcept
{
    if(stream == nullptr) return false;
    const int fd = ::fileno(stream);
    if(fd < 0) return false;
    return ::isatty(fd) != 0;
}
}  // namespace

// Sample visibility once at construction. Verbose level and the stream's
// TTY-ness are set during process startup and do not change during a
// post-processing run, so re-checking on every advance would be waste.
bar::bar(std::string label, std::uint64_t total, bar_options opts)
: m_label(std::move(label))
, m_opts(opts)
, m_total(total)
, m_visible(opts.enabled && opts.verbose >= 0 && stream_is_tty(opts.stream))
{}

// Idempotent: on_finish()'s CAS short-circuits if it was called already.
bar::~bar() { on_finish(); }

void
bar::on_set_total(std::uint64_t total) noexcept
{
    m_total.store(total, std::memory_order_relaxed);
}

void
bar::on_advance(std::uint64_t delta) noexcept
{
    if(delta == 0) return;
    const auto prev    = m_current.fetch_add(delta, std::memory_order_relaxed);
    const auto current = prev + delta;
    const auto total   = m_total.load(std::memory_order_relaxed);

    if(!m_visible) return;

    // Finalize as soon as we cross the total so the bar's last line lands
    // before any other code (e.g. file_output_message) writes to stderr.
    if(total > 0 && current >= total)
    {
        on_finish();
        return;
    }
    try_render();
}

void
bar::on_finish() noexcept
{
    bool expected = false;
    if(!m_finished.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        return;
    if(!m_visible) return;

    std::lock_guard<std::mutex> lock{ m_render_mtx };
    const auto                  total = m_total.load(std::memory_order_relaxed);
    if(total > 0) m_current.store(total, std::memory_order_relaxed);
    render_locked();
    std::fputc('\n', m_opts.stream);
    std::fflush(m_opts.stream);
}

void
bar::try_render() noexcept
{
    // Cheap pre-check: avoid even attempting the lock once finished.
    if(m_finished.load(std::memory_order_acquire)) return;

    // Acquire the render lock first so the throttle timestamp is only read
    // and updated by the single thread that will actually render. This avoids
    // burning the throttle slot when contention causes try_lock() to fail.
    std::unique_lock<std::mutex> lock{ m_render_mtx, std::try_to_lock };
    if(!lock.owns_lock()) return;

    // Re-check under the lock: on_finish() may have completed between the
    // pre-check and acquiring the lock. Without this, a partial bar would be
    // written after the final newline, leaving stray output on the next line.
    if(m_finished.load(std::memory_order_relaxed)) return;

    const auto now = now_ns();
    const auto interval =
        std::chrono::duration_cast<std::chrono::nanoseconds>(m_opts.tick).count();
    if(now - m_last_render_ns.load(std::memory_order_relaxed) < interval) return;
    m_last_render_ns.store(now, std::memory_order_relaxed);

    render_locked();
}

void
bar::render_locked() noexcept
{
    const auto total   = m_total.load(std::memory_order_relaxed);
    const auto current = std::min(m_current.load(std::memory_order_relaxed), total);

    const double frac =
        (total == 0) ? 0.0 : static_cast<double>(current) / static_cast<double>(total);
    const auto width   = static_cast<std::size_t>(m_opts.width);
    const auto bar_str = detail::format_bar(m_opts.style, width, frac);
    const auto sep     = bar_str.empty() ? "" : " ";

    std::fputs(ANSI_RESET_LINE, m_opts.stream);
    std::fprintf(m_opts.stream, "%s%s%s %3.0f%%", m_label.c_str(), sep, bar_str.c_str(),
                 frac * 100.0);
    std::fflush(m_opts.stream);
}

}  // namespace rocprofsys::progress
