#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace rocm_timesync
{

namespace ipc
{

constexpr uint32_t MAGIC = 0x52545359;
constexpr uint32_t VERSION = 1;

typedef struct 
{
    uint16_t gpu_id;
    uint64_t gpu_timestamp_ns;
    uint64_t system_timestamp_ns;
} event_t;

typedef struct 
{
    uint32_t magic;
    uint32_t version;
    uint64_t ring_size;
} header_t;

typedef struct
{
    alignas(64) std::atomic<uint64_t> write_idx;
    alignas(64) std::atomic<uint32_t> wait_seq;
    event_t events[];
} ringbuffer_t;

typedef struct 
{
    header_t header;
    ringbuffer_t rbuf;
} state_t;

typedef struct
{
    uint64_t read_idx;
} cursor_t;

// establish shared state
state_t* create(std::string name, uint8_t ring_order);
const state_t* attach(std::string name);
void destroy(state_t* state);

// ringbuffer produce/consume
void publish(state_t* state, event_t& event);
template <typename Callback>
bool poll(const state_t* state, cursor_t& cursor, Callback&& callback);
void test(const state_t* state, cursor_t& cursor);

} // namespace ipc

} // namespace rocm_timesync

