// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__linux__)
#    include <unistd.h>
#endif

namespace rocprofsys
{
namespace testing
{
/**
 * @brief Trace event phase types
 */
enum class trace_phase : char
{
    begin       = 'B',  // Begin duration event
    end         = 'E',  // End duration event
    instant     = 'i',  // Instant event
    counter     = 'C',  // Counter event
    async_begin = 'b',  // Async begin
    async_end   = 'e',  // Async end
};

/**
 * @brief Represents an argument (key-value pair) for a trace event
 */
struct trace_arg
{
    std::string key;
    std::string value;

    trace_arg() = default;

    // Constructor for string_view
    trace_arg(std::string_view k, std::string_view v)
    : key(k)
    , value(v)
    {}

    // Constructor for const char* to avoid ambiguity
    trace_arg(std::string_view k, const char* v)
    : key(k)
    , value(v ? v : "")
    {}

    // Template constructor for all other types
    template <typename T,
              typename std::enable_if<
                  !std::is_convertible<T, std::string_view>::value &&
                      !std::is_same<typename std::decay<T>::type, const char*>::value &&
                      !std::is_same<typename std::decay<T>::type, char*>::value,
                  int>::type = 0>
    trace_arg(std::string_view k, T&& v)
    : key(k)
    , value(convert_value(std::forward<T>(v)))
    {}

    bool operator==(const trace_arg& other) const
    {
        return key == other.key && value == other.value;
    }

private:
    // Helper to convert values to strings
    template <typename T>
    static std::string convert_value(T&& v)
    {
        using DecayT = typename std::decay<T>::type;
        if constexpr(std::is_arithmetic<DecayT>::value)
        {
            return std::to_string(v);
        }
        else if constexpr(std::is_convertible<DecayT, std::string>::value)
        {
            return std::string(std::forward<T>(v));
        }
        else
        {
            // Fallback for other types - this will cause a compile error
            // if T is not convertible to string, which is intentional
            static_assert(std::is_arithmetic<DecayT>::value ||
                              std::is_convertible<DecayT, std::string>::value,
                          "Type must be arithmetic or convertible to string");
            return "";
        }
    }
};

/**
 * @brief Represents a single trace event captured by the mock sink
 */
struct trace_event
{
    std::string             name;
    std::string             category;
    trace_phase             phase;
    uint64_t                timestamp;
    std::optional<uint64_t> duration;
    std::vector<trace_arg>  args;
    uint64_t                thread_id;
    uint64_t                process_id;

    trace_event() = default;

    trace_event(std::string_view n, std::string_view c, trace_phase ph, uint64_t ts = 0,
                uint64_t tid = 0, uint64_t pid = 0)
    : name(n)
    , category(c)
    , phase(ph)
    , timestamp(ts)
    , thread_id(tid)
    , process_id(pid)
    {}

    /**
     * @brief Add an argument to this event
     */
    void add_arg(std::string_view key, std::string_view value)
    {
        args.emplace_back(key, value);
    }

    template <typename T>
    void add_arg(std::string_view key, const T& value)
    {
        args.emplace_back(key, value);
    }

    /**
     * @brief Get argument value by key
     */
    std::optional<std::string> get_arg(std::string_view key) const
    {
        auto it = std::find_if(args.begin(), args.end(),
                               [key](const trace_arg& arg) { return arg.key == key; });
        if(it != args.end()) return it->value;
        return std::nullopt;
    }

    /**
     * @brief Check if event has argument with given key
     */
    bool has_arg(std::string_view key) const
    {
        return std::any_of(args.begin(), args.end(),
                           [key](const trace_arg& arg) { return arg.key == key; });
    }
};

/**
 * @brief Mock trace sink for capturing and verifying trace events in unit tests
 *
 */
class mock_trace_sink
{
public:
    using clock_fn = std::function<uint64_t()>;

    mock_trace_sink()
    : m_clock(default_clock)
    , m_event_count(0)
    {}

    /**
     * @brief Reset all captured events and counters
     */
    void reset()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_events.clear();
        m_event_count.store(0);
    }

    /**
     * @brief Set custom clock function for timestamp generation
     */
    void set_clock(clock_fn clock)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_clock = std::move(clock);
    }

    /**
     * @brief Emit a begin duration event
     */
    void emit_begin(std::string_view name, std::string_view category)
    {
        trace_event evt(name, category, trace_phase::begin, get_timestamp(),
                        get_thread_id(), get_process_id());
        add_event(std::move(evt));
    }

    /**
     * @brief Emit an end duration event
     */
    void emit_end(std::string_view name, std::string_view category)
    {
        trace_event evt(name, category, trace_phase::end, get_timestamp(),
                        get_thread_id(), get_process_id());
        add_event(std::move(evt));
    }

    /**
     * @brief Emit an instant event
     */
    void emit_instant(std::string_view name, std::string_view category)
    {
        trace_event evt(name, category, trace_phase::instant, get_timestamp(),
                        get_thread_id(), get_process_id());
        add_event(std::move(evt));
    }

    /**
     * @brief Emit a complete event with duration
     */
    void emit_complete(std::string_view name, std::string_view category,
                       uint64_t duration_ns)
    {
        trace_event evt(name, category, trace_phase::begin, get_timestamp(),
                        get_thread_id(), get_process_id());
        evt.duration = duration_ns;
        add_event(std::move(evt));
    }

    /**
     * @brief Emit a begin event with arguments
     */
    template <typename... Args>
    void emit_begin_with_args(std::string_view name, std::string_view category,
                              Args&&... args)
    {
        trace_event evt(name, category, trace_phase::begin, get_timestamp(),
                        get_thread_id(), get_process_id());
        add_args_to_event(evt, std::forward<Args>(args)...);
        add_event(std::move(evt));
    }

    /**
     * @brief Emit an end event with arguments
     */
    template <typename... Args>
    void emit_end_with_args(std::string_view name, std::string_view category,
                            Args&&... args)
    {
        trace_event evt(name, category, trace_phase::end, get_timestamp(),
                        get_thread_id(), get_process_id());
        add_args_to_event(evt, std::forward<Args>(args)...);
        add_event(std::move(evt));
    }

    /**
     * @brief Get total number of captured events
     */
    int get_event_count() const { return m_event_count.load(); }

    /**
     * @brief Get all captured events
     */
    std::vector<trace_event> get_events() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_events;
    }

    /**
     * @brief Check if any event with given name exists
     */
    bool has_event(std::string_view name) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return std::any_of(m_events.begin(), m_events.end(),
                           [name](const trace_event& evt) { return evt.name == name; });
    }

    /**
     * @brief Check if any event with given category exists
     */
    bool has_category(std::string_view category) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return std::any_of(
            m_events.begin(), m_events.end(),
            [category](const trace_event& evt) { return evt.category == category; });
    }

    /**
     * @brief Get count of events with specific name
     */
    int count_events_by_name(std::string_view name) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return std::count_if(m_events.begin(), m_events.end(),
                             [name](const trace_event& evt) { return evt.name == name; });
    }

    /**
     * @brief Get count of events with specific phase
     */
    int count_events_by_phase(trace_phase phase) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return std::count_if(
            m_events.begin(), m_events.end(),
            [phase](const trace_event& evt) { return evt.phase == phase; });
    }

    /**
     * @brief Get events filtered by name
     */
    std::vector<trace_event> get_events_by_name(std::string_view name) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<trace_event>    result;
        std::copy_if(m_events.begin(), m_events.end(), std::back_inserter(result),
                     [name](const trace_event& evt) { return evt.name == name; });
        return result;
    }

    /**
     * @brief Get events filtered by category
     */
    std::vector<trace_event> get_events_by_category(std::string_view category) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<trace_event>    result;
        std::copy_if(
            m_events.begin(), m_events.end(), std::back_inserter(result),
            [category](const trace_event& evt) { return evt.category == category; });
        return result;
    }

    /**
     * @brief Get events filtered by phase
     */
    std::vector<trace_event> get_events_by_phase(trace_phase phase) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<trace_event>    result;
        std::copy_if(m_events.begin(), m_events.end(), std::back_inserter(result),
                     [phase](const trace_event& evt) { return evt.phase == phase; });
        return result;
    }

    /**
     * @brief Get first event (or nullopt if no events)
     */
    std::optional<trace_event> get_first_event() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if(m_events.empty()) return std::nullopt;
        return m_events.front();
    }

    /**
     * @brief Get last event (or nullopt if no events)
     */
    std::optional<trace_event> get_last_event() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if(m_events.empty()) return std::nullopt;
        return m_events.back();
    }

    /**
     * @brief Verify event sequence by names
     */
    bool verify_event_sequence(const std::vector<std::string>& expected_names) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if(m_events.size() != expected_names.size()) return false;

        for(size_t i = 0; i < m_events.size(); ++i)
        {
            if(m_events[i].name != expected_names[i]) return false;
        }
        return true;
    }

    /**
     * @brief Check if events are properly paired (begin/end)
     */
    bool verify_balanced_events() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::map<std::string, int>  balance;
        for(const auto& evt : m_events)
        {
            if(evt.phase == trace_phase::begin)
                balance[evt.name]++;
            else if(evt.phase == trace_phase::end)
                balance[evt.name]--;
        }

        return std::all_of(balance.begin(), balance.end(),
                           [](const auto& pair) { return pair.second == 0; });
    }

private:
    void add_event(trace_event&& evt)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_events.push_back(std::move(evt));
        m_event_count.fetch_add(1);
    }

    uint64_t get_timestamp()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_clock();
    }

    template <typename T1, typename T2, typename... Rest>
    void add_args_to_event(trace_event& evt, T1&& key, T2&& value, Rest&&... rest)
    {
        evt.add_arg(std::forward<T1>(key), std::forward<T2>(value));
        if constexpr(sizeof...(Rest) > 0)
        {
            add_args_to_event(evt, std::forward<Rest>(rest)...);
        }
    }

    static uint64_t default_clock()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    static uint64_t get_thread_id()
    {
        return std::hash<std::thread::id>{}(std::this_thread::get_id());
    }

    static uint64_t get_process_id()
    {
#if defined(__linux__)
        return static_cast<uint64_t>(getpid());
#else
        return 0;
#endif
    }

    std::vector<trace_event> m_events;
    std::atomic<int>         m_event_count;
    clock_fn                 m_clock;
    mutable std::mutex       m_mutex;
};

}  // namespace testing
}  // namespace rocprofsys
