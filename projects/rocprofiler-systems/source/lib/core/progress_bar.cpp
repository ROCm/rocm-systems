// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "progress_bar.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ostream>
#include <string>
#include <utility>

#include <spdlog/fmt/fmt.h>

namespace rocprofsys
{

// Unicode block elements: space, 1/8 .. 7/8, full block
static constexpr std::array<const char*, 9> BLOCK_CHARS = {
    " ", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589", "\u2588"
};

progress_bar::progress_bar(size_t bar_width, std::string prefix_text,
                           std::string postfix_text, size_t max_progress,
                           std::ostream& stream)
: m_bar_width(bar_width)
, m_prefix_text(std::move(prefix_text))
, m_postfix_text(std::move(postfix_text))
, m_max_progress(max_progress > 0 ? max_progress : 1)
, m_os(stream)
{}

progress_bar::~progress_bar()
{
    if(!is_completed()) mark_as_completed();
}

void
progress_bar::set_progress(size_t value)
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    auto                              clamped = std::min(value, m_max_progress);
    if(clamped == m_progress) return;
    m_progress = clamped;
    render();
    if(m_progress >= m_max_progress)
    {
        m_completed.store(true, std::memory_order_release);
        m_os << '\n';
    }
}

bool
progress_bar::is_completed() const noexcept
{
    return m_completed.load(std::memory_order_acquire);
}

void
progress_bar::mark_as_completed()
{
    const std::lock_guard<std::mutex> lock(m_mutex);
    if(m_completed.load(std::memory_order_acquire)) return;
    m_progress = m_max_progress;
    m_completed.store(true, std::memory_order_release);
    render();
    m_os << '\n';
}

void
progress_bar::render()
{
    const auto fraction = std::min(
        static_cast<double>(m_progress) / static_cast<double>(m_max_progress), 1.0);

    const auto total_eighths =
        static_cast<size_t>(fraction * static_cast<double>(m_bar_width * PARTS_PER_CELL));
    const auto full      = total_eighths / PARTS_PER_CELL;
    const auto remainder = total_eighths % PARTS_PER_CELL;
    const auto empty     = m_bar_width - full - (remainder > 0 ? 1 : 0);
    const auto percent   = static_cast<size_t>(fraction * 100.0);

    std::string bar_fill;
    bar_fill.reserve(m_bar_width * 4);
    for(size_t i = 0; i < full; ++i)
    {
        bar_fill += BLOCK_CHARS[PARTS_PER_CELL];
    }
    if(remainder > 0)
    {
        bar_fill += BLOCK_CHARS[remainder];
    }
    for(size_t i = 0; i < empty; ++i)
    {
        bar_fill += ' ';
    }

    m_os << fmt::format("\r{}[{}] {:>3}% {}", m_prefix_text, bar_fill, percent,
                        m_postfix_text);
    m_os.flush();
}

}  // namespace rocprofsys
