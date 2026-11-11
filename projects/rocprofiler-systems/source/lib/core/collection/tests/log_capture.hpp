// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "logger/logger.hpp"

#include <spdlog/common.h>
#include <spdlog/sinks/ostream_sink.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

namespace rocprofsys::collection::testing
{
/**
 * Captures what the project logger writes while it lives, at every level. Adds a sink to
 * the logger, so create and destroy it while no other thread logs.
 */
class log_capture
{
public:
    log_capture()
    : m_sink{ std::make_shared<spdlog::sinks::ostream_sink_mt>(m_lines) }
    , m_level{ logger_t::instance().level() }
    {
        m_sink->set_pattern("%l %v");
        logger_t::instance().sinks().push_back(m_sink);
        logger_t::instance().set_level(spdlog::level::trace);
    }

    ~log_capture()
    {
        auto& sinks = logger_t::instance().sinks();
        sinks.erase(std::ranges::remove(sinks, m_sink).begin(), sinks.end());
        logger_t::instance().set_level(m_level);
    }

    log_capture(const log_capture&)            = delete;
    log_capture& operator=(const log_capture&) = delete;
    log_capture(log_capture&&)                 = delete;
    log_capture& operator=(log_capture&&)      = delete;

    /** Number of captured lines of `level` ("warning", "error", ...) containing `text`.
     */
    [[nodiscard]] std::size_t count(std::string_view level, std::string_view text) const
    {
        std::istringstream lines{ m_lines.str() };
        std::size_t        found = 0;
        for(std::string line; std::getline(lines, line);)
        {
            if(line.starts_with(level) && line.find(text) != std::string::npos)
            {
                ++found;
            }
        }
        return found;
    }

private:
    std::ostringstream                              m_lines;
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> m_sink;
    spdlog::level::level_enum                       m_level;
};
}  // namespace rocprofsys::collection::testing
