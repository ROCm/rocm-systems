#ifndef HSA_PROFILER_HPP_
#define HSA_PROFILER_HPP_


#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>

namespace rocr {
class Profiler {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::microseconds;

    // Singleton accessor
    static Profiler& instance() {
        static Profiler singleton;
        return singleton;
    }

    // Disable copy and move

    Profiler(const Profiler&) = delete;

    Profiler& operator=(const Profiler&) = delete;

    void profile_start(const std::string& event_name) {
        auto now = Clock::now();
        get_event_times()[event_name] = now;
    }

    void profile_end(const std::string& event_name) {
        auto now = Clock::now();
        auto& times = get_event_times();
        auto it = times.find(event_name);
        if (it != times.end()) {
            auto duration = std::chrono::duration_cast<Duration>(now - it->second).count();
            get_events().push_back(Event{event_name, std::this_thread::get_id(), duration});
            times.erase(it);
        }
    }

    ~Profiler() {
        // Aggregate all thread-local events at process end
        std::vector<Event> all_events;
        for (auto collector : *get_collectors()) {
            for (const auto& event : *collector) {
                all_events.push_back(event);
            }
        }
        std::cout << "Profiling results:\n";
        for (const auto& event : all_events) {
            std::cout << "Event \"" << event.name << "\" [thread "
                      << event.thread_id << "] took "
                     << event.duration_us << " us.\n";
        }
    }

private:
    Profiler() = default;
    struct Event {
        std::string name;
        std::thread::id thread_id;
        long long duration_us;
    };
    // Per-thread storage
    static std::unordered_map<std::string, TimePoint>& get_event_times() {
        thread_local std::unordered_map<std::string, TimePoint> event_times;
        return event_times;
    }
    static std::vector<Event>& get_events() {
        thread_local std::vector<Event>* events = nullptr;
        if (!events) {
            events = new std::vector<Event>();
            get_collectors()->push_back(events);
        }
        return *events;
    }

    // Collects all per-thread event vectors for aggregation
    static std::vector<std::vector<Event>* >* get_collectors() {
        static auto* collectors = new std::vector<std::vector<Event>*>();
        return collectors;
    }
};

} // namespace amd


#define PROFILE_START(name) rocr::Profiler::instance().profile_start(name)
#define PROFILE_END(name)  rocr::Profiler::instance().profile_end(name)
#endif