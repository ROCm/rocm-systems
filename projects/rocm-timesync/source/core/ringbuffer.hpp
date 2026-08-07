#pragma once

#include <core/ipc.hpp>

#include <atomic>
#include "atomic.hpp"

namespace rocm_timesync
{

namespace ipc
{

struct ringbuffer_shm_t;
struct cursor_t;

struct ringbuffer_t
{
    ringbuffer_t() = default;

    // producer api
    int create(std::string name, uint8_t ring_order);
    void destroy();
    void publish(std::vector<event_t>& events);

    // consume api
    int attach(std::string name);
    void detach();
    void poll(const callback_t& callback);
    void consume(std::vector<event_t>&, int64_t wait_ms);
    void stop();

    // both
    uint64_t size() const;
    uint64_t length() const;
private:
    struct cursor_t {
        uint64_t read_idx;
    };

    ringbuffer_shm_t* rbuf;
    cursor_t cursor;
    std::string name;
    mutable std::atomic<bool> stop_requested{false};

    bool await(int64_t wait_ms);
    void do_consume(std::vector<event_t>& events);
};

} // namespace ipc
} // namespace rocm_timesync
