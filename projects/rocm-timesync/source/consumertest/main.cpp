#include <iostream>

#include <core/ipc.hpp>

using namespace rocm_timesync;

int main()
{
    auto* channel = ipc::attach("hz_high");
    if (channel == nullptr) {
        printf("could not attach ringbuffer\n");
        return 1;
    }

    std::cout << "attached to channel of size = " << ipc::size(channel) << std::endl;

    ipc::poll(channel, [](const ipc::event_t& event) {
        std::cout << "processed event with gpu_id:" << event.gpu_id <<
            " and gpu_timestamp: " << event.gpu_timestamp_ns << std::endl;
    });
    return 0;
}
