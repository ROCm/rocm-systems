#include <iostream>

#include <core/ipc.hpp>

using namespace rocm_timesync;

int main()
{
    auto* state = ipc::attach("test0");
    if (state == nullptr) {
        printf("could not attach ringbuffer\n");
        return 1;
    }

    std::cout << "attached to ringbuffer of size = " << state->header.ring_size << std::endl;

    auto cursor = ipc::cursor_t();
    ipc::test(state, cursor);
    return 0;
}
