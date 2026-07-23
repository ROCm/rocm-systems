#include <thread>
#include <iostream>
#include <cassert>

#include <rocm-timesync/rocm_timesync.hpp>
#include <core/ipc.hpp>


namespace rocm_timesync
{

static ipc::channel_t* channel = nullptr;
static bool keep_running = true;
static std::thread streamer;

namespace
{

static std::string precision_to_name(ts_precision_t precision)
{
    if (precision == TIMESYNC_PRECISION_HIGH)
        return "hz_high";
    else
        return "hz_low";
}

} // namespace

int timesync_init(ts_precision_t precision)
{
    // TODO: establish DB connection

    // attach to ringbuffer
    channel = ipc::attach(precision_to_name(precision));
    if (channel == nullptr)
        return -ENODEV;

    // spawn thread
    streamer = std::thread([]() {
        while (keep_running) {
            std::cout << "ipc streamer running...\n";
            ipc::poll(channel, [](const ipc::event_t& event) {
                std::cout << "processed event with gpu_id:" << event.gpu_id <<
                    " and gpu_timestamp: " << event.gpu_timestamp_ns << std::endl;
            });
        }
    });

    return 0;
}

int timesync_deinit()
{
    if (channel == nullptr)
        return 0;

    // stop streaming thread
    keep_running = false;
    ipc::stop(channel);
    streamer.join();

    // detach from ringbuffer
    ipc::detach(channel);

    // TODO: destroy DB connection

    return 0;
}

int timesync_translate(uint32_t agent_kfd_gpu_id, uint64_t agent_timestamp, uint64_t *system_timestamp)
{
    std::cerr << "translate_time request for kfd gpu id: " << agent_kfd_gpu_id << std::endl << std::flush;
    return 0;
}

} // namespace rocm_timesync
