#pragma once

#include <core/ipc.hpp>

#include <atomic>
#include "atomic.hpp"

namespace rocm_timesync
{

namespace ipc
{

struct ringbuffer_shm_t;
struct ringbuffer_t
{
    ringbuffer_shm_t* rbuf; 
    std::string name;

    ringbuffer_t() = default;

    int create(std::string name, uint8_t ring_order);
    int attach(std::string name);
    void destroy();
    void detach();
    uint64_t size() const;
    uint64_t length() const;
    void publish(std::vector<event_t>& events);
    void poll(const callback_t& callback) const;

};

} // namespace ipc
} // namespace rocm_timesync
