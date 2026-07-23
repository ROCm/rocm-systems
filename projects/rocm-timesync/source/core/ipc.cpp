#include <cassert>

#include <core/ipc.hpp>
#include "atomic.hpp"
#include "ringbuffer.hpp"

namespace rocm_timesync
{

namespace ipc
{

struct channel_t
{
    enum class mode_t
    {
        producer,
        consumer
    };

    // producer or consumer
    mode_t mode;

    // shared memory
    ringbuffer_t rbuf;
};

namespace
{

std::string make_shm_name(const std::string& name)
{
    return std::string("/rocm-timesync-" + name);
}

}

channel_t* create(std::string name, uint8_t ring_order)
{
    channel_t* channel = new channel_t();

    // create ringbuffer
    int ret = channel->rbuf.create(make_shm_name(name), ring_order);
    if (ret != 0) {
        delete channel;
        return nullptr;
    }

    channel->mode = channel_t::mode_t::producer;
    return channel;
}

void destroy(channel_t* channel)
{
    assert(channel->mode == channel_t::mode_t::producer);
    channel->rbuf.destroy();
}

channel_t* attach(std::string name)
{
    channel_t* channel = new channel_t();

    // attach ringbuffer
    int ret = channel->rbuf.attach(make_shm_name(name));
    if (ret != 0) {
        delete channel;
        return nullptr;
    }

    channel->mode = channel_t::mode_t::consumer;
    return channel;
}

void detach(channel_t* channel)
{
    assert(channel->mode == channel_t::mode_t::consumer);
    channel->rbuf.detach();
}

void publish(channel_t* channel, std::vector<event_t>& events)
{
    assert(channel->mode == channel_t::mode_t::producer);
    channel->rbuf.publish(events);
}

void poll(const channel_t* channel, const callback_t& callback)
{
    assert(channel->mode == channel_t::mode_t::consumer);
    channel->rbuf.poll(callback);
}

void stop(channel_t* channel)
{
    assert(channel->mode == channel_t::mode_t::consumer);
    channel->rbuf.stop();
}

uint64_t size(const channel_t* channel)
{
    return channel->rbuf.size();
}

} // namespace ipc
} // namespace rocm_timesync
