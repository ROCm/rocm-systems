#include "rocprof_attach.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>

int
main(int argc, char* argv[])
{
    const auto* _attach_pid = argv[2];

    printf("[%s:%d] attach pid: %s\n", __FILE__, __LINE__, _attach_pid);

    auto pid = atoi(_attach_pid);
    attach(pid);

    std::cout << "Press ENTER to detach" << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    detach();

    return EXIT_SUCCESS;
}
