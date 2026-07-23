#include <iostream>
#include <thread>
#include <chrono>

#include <core/ipc.hpp>
#include "config.hpp"
#include "kfd.hpp"

#define PUBLISH_INTERVAL_MS 100

using namespace rocm_timesync;

struct p_channel_t {
    std::string name;
    uint32_t hz;
    uint32_t batch_size;
    ipc::channel_t* channel;
};

int main()
{
    auto cfg = LoadConfig("config.yml");

    std::vector<gpu_context_t> gpus;
    int st = kfd_enumerate_gpus(gpus);
    if (st != 0) {
        fprintf(stderr, "Failed to enumerate GPUs: %d\n", st);
        return 1;
    }

    // create a channel for each configured frequency interval
    std::vector<p_channel_t> channels;
    auto make_channel = [&channels](auto&& name, uint32_t hz, uint32_t order) {
        const uint32_t batch_size = std::max(
            1u,
            static_cast<uint32_t>(
                (static_cast<uint64_t>(hz) * PUBLISH_INTERVAL_MS) / 1000)
        );

        channels.push_back(p_channel_t{
            .name=name, 
            .hz=hz, 
            .batch_size=batch_size,
            .channel=ipc::create(name, order)
        });
    };

    make_channel("hz_high", cfg.hz_high.hz, cfg.hz_high.order);
    make_channel("hz_low", cfg.hz_low.hz, cfg.hz_low.order);

    // launch one thread per channel
    std::vector<std::thread> threads;
    for (auto& p_channel : channels) {
        threads.emplace_back([p_channel, &gpus] {
            ipc::channel_t* channel = p_channel.channel;
            std::vector<ipc::event_t> events;
            while (true) {
                for (const auto& gpu : gpus) {
                    crosststamp_t xstamp;
                    int st = kfd_get_crosststamp(gpu, xstamp);
                    if (st != 0) {
                        fprintf(stderr, "Failed to get crosststamp: %d\n", st);
                        continue;
                    }

                    events.push_back(ipc::event_t{gpu.kfd_gpu_id, xstamp.gpu_timestamp, xstamp.system_timestamp});
                    if (events.size() == p_channel.batch_size) {
                        ipc::publish(channel, events);
                        events.clear();
                    }

                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1000/p_channel.hz));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}
