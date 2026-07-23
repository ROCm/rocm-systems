#pragma once

#include <vector>
#include <atomic>
#include <cstdint>
#include <string>
#include <functional>

namespace rocm_timesync
{

namespace ipc
{

constexpr uint32_t MAGIC = 0x52545359;
constexpr uint32_t VERSION = 1;

struct channel_t;
struct event_t
{
    uint32_t gpu_id;
    uint64_t gpu_timestamp_ns;
    uint64_t system_timestamp_ns;
};

using callback_t = std::function<void(const event_t&)>;

// producer/consumer
uint64_t size(const channel_t* channel);

// producer
channel_t* create(std::string name, uint8_t ring_order);
void destroy(channel_t*);
void publish(channel_t* channel, std::vector<event_t>& events);

// consumer
channel_t* attach(std::string name);
void detach(channel_t*);
void poll(const channel_t* channel, const callback_t& callback);
void stop(channel_t* channel);

} // namespace ipc

} // namespace rocm_timesync

