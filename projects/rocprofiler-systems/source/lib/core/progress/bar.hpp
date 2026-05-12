// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace rocprofsys::progress
{

/**
 * @brief Visual style for the rendered bar.
 *
 * - `ascii_brackets` (default): `[######......]` - safe everywhere.
 * - `ascii_arrow`:              `[======>     ]` - leading edge marker.
 * - `unicode_blocks`:           `[##########  ]` using 1/8-cell sub-precision
 *                               via U+2588 etc. Requires a UTF-8 terminal.
 * - `unicode_dots`:             `●●●●●●○○○○○○` - no brackets, dot fill.
 *                               Requires a UTF-8 terminal.
 * - `text_only`:                no bar drawn, just label + percent + counts.
 */
enum class bar_style
{
    ascii_brackets,
    ascii_arrow,
    unicode_blocks,
    unicode_dots,
    text_only
};

/**
 * @brief Configuration for bar.
 *
 * @note `stream` is a `std::FILE*` rather than a `std::ostream&` so the
 *       renderer can emit raw control sequences (`\r\033[2K`) and use
 *       `printf`-style formatting cheaply.
 * @note `verbose` is injected by the caller (typically the application's
 *       configuration layer). The bar treats negative values as "quiet"
 *       and skips all rendering. Keeping the dependency external means the
 *       bar has no ties to the global config singleton, so it is unit-
 *       testable in isolation.
 */
struct bar_options
{
    std::uint32_t             width   = 40;
    std::chrono::milliseconds tick    = std::chrono::milliseconds{ 100 };
    std::FILE*                stream  = stderr;
    bool                      enabled = true;
    int                       verbose = 0;
    bar_style                 style   = bar_style::ascii_brackets;
};

namespace detail
{
/**
 * @brief Renders the bar fill for a given style at fractional fill @p frac.
 *
 * Pure / side-effect-free; exposed in the header so it can be unit-tested
 * without a TTY. @p frac is clamped to `[0, 1]` internally.
 */
[[nodiscard]] std::string
format_bar(bar_style style, std::size_t width, double frac);
}  // namespace detail

/**
 * @brief TTY-aware, throttled progress renderer.
 *
 * Producers report progress by calling `on_advance(delta)`. The bar is
 * thread-safe so multiple worker threads may share a single instance.
 * Typically a producer wraps the bar in a `progress::progress_callback`
 * lambda -> see `core/progress/callback.hpp`.
 *
 * Behaviour:
 * - When the configured @ref bar_options::stream is not a TTY, or when
 *   verbose level is below 0, or when @ref bar_options::enabled is false,
 *   the bar is silent - `on_*` calls update internal counters but never
 *   render.
 * - When visible, advances are throttled to one redraw per
 *   @ref bar_options::tick interval. Each redraw issues `\r\033[2K` to
 *   reset the current terminal line in place. Render is gated by
 *   `try_lock` so only one thread at a time formats output; other threads
 *   simply skip.
 * - `on_finish()` (or destruction) prints the final line followed by a
 *   newline so subsequent log lines are not clobbered.
 *
 * Thread-safety: `on_advance` is safe to call from multiple threads.
 */
class bar
{
public:
    bar(std::string label, std::uint64_t total, bar_options opts = {});
    ~bar();

    bar(const bar&)            = delete;
    bar& operator=(const bar&) = delete;
    bar(bar&&)                 = delete;
    bar& operator=(bar&&)      = delete;

    void on_set_total(std::uint64_t total) noexcept;
    void on_advance(std::uint64_t delta) noexcept;
    void on_finish() noexcept;

private:
    void try_render() noexcept;
    void render_locked() noexcept;

    std::string                m_label;
    bar_options                m_opts;
    std::atomic<std::uint64_t> m_total;
    std::atomic<std::uint64_t> m_current{ 0 };
    std::atomic<std::int64_t>  m_last_render_ns{ 0 };
    std::mutex                 m_render_mtx;
    std::atomic<bool>          m_finished{ false };
    const bool                 m_visible;
};

}  // namespace rocprofsys::progress
