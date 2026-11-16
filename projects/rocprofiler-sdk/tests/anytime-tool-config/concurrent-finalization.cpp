#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <optional>
#include <string_view>
#include <thread>

namespace
{
rocprofiler_client_finalize_t          client_finalizer = nullptr;
std::optional<rocprofiler_client_id_t> client_id;
std::promise<void>                     finalizer_entered;
std::promise<void>                     release_finalizer;
std::atomic<unsigned>                  finalizer_calls{0};
std::atomic<bool>                      finalizer_finished{false};

int
initialize(rocprofiler_client_finalize_t finalize, void*)
{
    client_finalizer = finalize;
    return 0;
}

void
finalize(void*)
{
    ++finalizer_calls;
    client_finalizer(*client_id);
    finalizer_entered.set_value();
    release_finalizer.get_future().wait();
    finalizer_finished.store(true);
}

rocprofiler_tool_configure_result_t*
configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* identifier)
{
    client_id.emplace(*identifier);
    static auto result = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), initialize, finalize, nullptr};
    return &result;
}
}  // namespace

int
main(int argc, char** argv)
{
    const bool exit_while_finalizing =
        argc == 2 && std::string_view{argv[1]} == "--exit-while-finalizing";
    if(exit_while_finalizing)
    {
        std::atexit([] {
            if(!finalizer_finished.load())
            {
                std::cerr << "Process exit destroyed SDK state before finalization completed\n";
                std::_Exit(1);
            }
            std::cout << "Process exit waited for the active client\n";
        });
    }

    if(rocprofiler_force_configure(configure) != ROCPROFILER_STATUS_SUCCESS ||
       client_finalizer == nullptr)
        return 2;

    auto active = std::thread{[] { client_finalizer(*client_id); }};
    finalizer_entered.get_future().wait();
    if(exit_while_finalizing)
    {
        active.detach();
        std::thread{[] {
            std::this_thread::sleep_for(std::chrono::milliseconds{200});
            release_finalizer.set_value();
        }}.detach();
        std::exit(0);
    }

    auto caller_started = std::promise<void>{};
    auto concurrent     = std::async(std::launch::async, [&] {
        caller_started.set_value();
        client_finalizer(*client_id);
    });
    caller_started.get_future().wait();
    const bool returned_early =
        concurrent.wait_for(std::chrono::milliseconds{200}) == std::future_status::ready;

    release_finalizer.set_value();
    active.join();
    concurrent.get();
    client_finalizer(*client_id);

    if(returned_early || finalizer_calls.load() != 1)
    {
        std::cerr << "Concurrent finalization returned before its client completed\n";
        return 1;
    }
    std::cout << "Concurrent finalization waits for the active client\n";
    return 0;
}
