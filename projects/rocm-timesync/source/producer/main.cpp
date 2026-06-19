#include <iostream>
#include <thread>
#include <chrono>

#include <core/ipc.hpp>
#include "config.hpp"


using namespace rocm_timesync;

int main()
{
    auto cfg = config::LoadConfig("config.yml");
    std::cout << "high:" << cfg.hz_precision_high << ", low: " << cfg.hz_precision_low << std::endl;
    std::cout << "ring_order: " << (int)cfg.ring_order << std::endl;

    auto* state = ipc::create("test0", cfg.ring_order);
    if (state == nullptr) {
        printf("could not create ringbuffer\n");
        return 1;
    }

    for (uint64_t i = 0; i < 100000; i++) {
        auto event = ipc::event_t{1,i,3};
        ipc::publish(state, event);
        event = ipc::event_t{2,i,3};
        ipc::publish(state, event);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}
