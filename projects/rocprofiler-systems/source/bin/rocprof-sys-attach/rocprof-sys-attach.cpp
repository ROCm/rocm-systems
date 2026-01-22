#include <cstdlib>
#include <iostream>
#include <limits>

#include <rocprofiler-sdk-rocattach/rocattach.h>

int
main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    const auto* _attach_pid = argv[2];

    std::cout << "[rocprof-sys-attach] Trying to attach to process " << _attach_pid
              << std::endl;

    auto pid    = atoi(_attach_pid);
    auto result = rocattach_attach(pid);
    if(result != ROCATTACH_STATUS_SUCCESS)
    {
        std::cerr << "[rocprof-sys-attach] Failed to attach to process " << pid
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[rocprof-sys-attach] Attached to process " << pid
              << ". Press ENTER to detach." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    result = rocattach_detach(pid);
    if(result != ROCATTACH_STATUS_SUCCESS)
    {
        std::cerr << "[rocprof-sys-attach] Failed to detach from process " << pid
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[rocprof-sys-attach] Detached from process " << pid << std::endl;

    return EXIT_SUCCESS;
}
