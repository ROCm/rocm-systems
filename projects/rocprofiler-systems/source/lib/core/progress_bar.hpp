// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <iosfwd>
#include <mutex>
#include <string>

namespace rocprofsys
{

/**
 * Thread-safe unicode block progress bar.
 *
 * Renders a bar like: Processing [████████▌       ] 53% output.proto
 * Uses \r for in-place updates, emits \n on completion.
 */
class progress_bar
{
public:
    /**
     * @param bar_width    Number of character cells for the bar fill area.
     * @param prefix_text  Text printed before the bar (e.g. "Processing ").
     * @param postfix_text Text printed after the percentage (e.g. filename).
     * @param max_progress Value at which the bar reaches 100%.
     * @param stream       Output stream for rendering (typically std::cerr).
     */
    progress_bar(size_t bar_width, std::string prefix_text, std::string postfix_text,
                 size_t max_progress, std::ostream& stream);

    ~progress_bar();

    progress_bar(const progress_bar&)            = delete;
    progress_bar& operator=(const progress_bar&) = delete;
    progress_bar(progress_bar&&)                 = delete;
    progress_bar& operator=(progress_bar&&)      = delete;

    void set_progress(size_t value);

    [[nodiscard]] bool is_completed() const noexcept;

    void mark_as_completed();

private:
    void render();

    static constexpr size_t PARTS_PER_CELL = 8;

    size_t            m_bar_width    = 0;
    std::string       m_prefix_text  = {};
    std::string       m_postfix_text = {};
    size_t            m_max_progress = 0;
    size_t            m_progress     = 0;
    std::atomic<bool> m_completed{ false };
    std::ostream&     m_os;
    std::mutex        m_mutex;
};

}  // namespace rocprofsys
