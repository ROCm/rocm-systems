#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace
{
void
require(bool condition, const char* message)
{
    if(!condition)
    {
        std::cerr << message << '\n';
        std::_Exit(EXIT_FAILURE);
    }
}

void
check(hsa_status_t status)
{
    if(status != HSA_STATUS_SUCCESS)
    {
        const char* message = nullptr;
        hsa_status_string(status, &message);
        std::cerr << "HSA status " << status << ": " << (message ? message : "unknown") << '\n';
        std::_Exit(EXIT_FAILURE);
    }
}

struct options
{
    size_t bytes      = 1 << 20;
    size_t iterations = 64;
    size_t threads    = 2;
    bool   profiling  = true;
    bool   peer       = false;
    bool   blit       = false;
    bool   benchmark  = false;
};

struct agents
{
    hsa_agent_t              cpu{};
    std::vector<hsa_agent_t> gpus;
};

hsa_status_t
collect_agent(hsa_agent_t agent, void* context)
{
    auto&             result = *static_cast<agents*>(context);
    hsa_device_type_t type{};
    check(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type));
    if(type == HSA_DEVICE_TYPE_CPU && !result.cpu.handle) result.cpu = agent;
    if(type == HSA_DEVICE_TYPE_GPU) result.gpus.push_back(agent);
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
collect_pool(hsa_amd_memory_pool_t pool, void* context)
{
    auto&             result = *static_cast<hsa_amd_memory_pool_t*>(context);
    hsa_amd_segment_t segment{};
    bool              allocatable = false;
    check(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));
    check(hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allocatable));
    if(!result.handle && segment == HSA_AMD_SEGMENT_GLOBAL && allocatable) result = pool;
    return HSA_STATUS_SUCCESS;
}

hsa_amd_memory_pool_t
pool_for(hsa_agent_t agent)
{
    hsa_amd_memory_pool_t pool{};
    check(hsa_amd_agent_iterate_memory_pools(agent, collect_pool, &pool));
    require(pool.handle != 0, "No allocatable global memory pool");
    return pool;
}

struct buffer
{
    void* data = nullptr;
    buffer(hsa_amd_memory_pool_t pool, size_t bytes, const std::vector<hsa_agent_t>& access)
    {
        check(hsa_amd_memory_pool_allocate(pool, bytes, 0, &data));
        check(hsa_amd_agents_allow_access(access.size(), access.data(), nullptr, data));
    }
    ~buffer() { hsa_amd_memory_pool_free(data); }
    buffer(const buffer&)            = delete;
    buffer& operator=(const buffer&) = delete;
};

struct signals
{
    std::array<hsa_signal_t, 4> values{};
    signals()
    {
        for(auto& signal : values)
            check(hsa_signal_create(1, 0, nullptr, &signal));
    }
    ~signals()
    {
        for(auto signal : values)
            hsa_signal_destroy(signal);
    }
};

struct route
{
    hsa_agent_t              source;
    hsa_agent_t              destination;
    hsa_amd_sdma_engine_id_t engine{};
};

route
make_route(hsa_agent_t source, hsa_agent_t destination, bool blit)
{
    route result{source, destination};
    if(!blit)
    {
        uint32_t available = 0;
        uint32_t preferred = 0;
        check(hsa_amd_memory_copy_engine_status(destination, source, &available));
        check(hsa_amd_memory_get_preferred_copy_engine(destination, source, &preferred));
        if(available == 0)
        {
            std::cerr << "SKIP: no SDMA engine for requested route\n";
            std::exit(77);
        }
        if(preferred & available) available &= preferred;
        result.engine = static_cast<hsa_amd_sdma_engine_id_t>(available & (~available + 1));
    }
    return result;
}

void
submit(const route& copy,
       void*        destination,
       const void*  source,
       size_t       bytes,
       hsa_signal_t dependency,
       hsa_signal_t completion,
       bool         blit)
{
    if(blit)
        check(hsa_amd_memory_async_copy(
            destination, copy.destination, source, copy.source, bytes, 1, &dependency, completion));
    else
        check(hsa_amd_memory_async_copy_on_engine(destination,
                                                  copy.destination,
                                                  source,
                                                  copy.source,
                                                  bytes,
                                                  1,
                                                  &dependency,
                                                  completion,
                                                  copy.engine,
                                                  true));
}

std::atomic<uint64_t> completed{0};
std::atomic<uint64_t> timestamp_checks{0};
std::atomic<uint64_t> elapsed_ns{0};

void
wait_and_check(hsa_signal_t signal, uint64_t timeout, bool timestamps)
{
    require(hsa_signal_wait_scacquire(
                signal, HSA_SIGNAL_CONDITION_EQ, 0, timeout, HSA_WAIT_STATE_ACTIVE) == 0,
            "Copy completion timed out");
    if(timestamps)
    {
        hsa_amd_profiling_async_copy_time_t time{};
        check(hsa_amd_profiling_get_async_copy_time(signal, &time));
        require(time.start > 0 && time.end > time.start,
                "Completed SDMA copy has invalid or non-positive profiling duration");
        ++timestamp_checks;
    }
    ++completed;
}

void
run(const options& config, const agents& devices, size_t worker, uint64_t timeout)
{
    const auto                     gpu_source      = devices.gpus.front();
    const auto                     gpu_destination = config.peer ? devices.gpus[1] : gpu_source;
    const std::vector<hsa_agent_t> access =
        config.peer ? std::vector<hsa_agent_t>{gpu_source, gpu_destination}
                    : std::vector<hsa_agent_t>{gpu_source};
    buffer  host_source(pool_for(devices.cpu), config.bytes, access);
    buffer  host_destination(pool_for(devices.cpu), config.bytes, access);
    buffer  device_source(pool_for(gpu_source), config.bytes, access);
    buffer  device_destination(pool_for(gpu_destination), config.bytes, access);
    signals completion;
    auto*   expected = static_cast<uint32_t*>(host_source.data);
    for(size_t index = 0; index < config.bytes / sizeof(uint32_t); ++index)
        expected[index] = static_cast<uint32_t>(index * 13 + worker + 1);
    const auto upload   = make_route(devices.cpu, gpu_source, config.blit);
    const auto download = make_route(gpu_destination, devices.cpu, config.blit);
    route      peer{};
    if(config.peer) peer = make_route(gpu_source, gpu_destination, config.blit);
    for(size_t iteration = 0; iteration < config.iterations; ++iteration)
    {
        std::memset(host_destination.data, 0, config.bytes);
        for(auto signal : completion.values)
            hsa_signal_store_screlease(signal, 1);
        const auto start = std::chrono::steady_clock::now();
        submit(upload,
               device_source.data,
               host_source.data,
               config.bytes,
               completion.values[0],
               completion.values[1],
               config.blit);
        if(config.peer)
            submit(peer,
                   device_destination.data,
                   device_source.data,
                   config.bytes,
                   completion.values[1],
                   completion.values[2],
                   config.blit);
        submit(download,
               host_destination.data,
               config.peer ? device_destination.data : device_source.data,
               config.bytes,
               completion.values[config.peer ? 2 : 1],
               completion.values[3],
               config.blit);
        require(hsa_signal_load_scacquire(completion.values[1]) > 0 &&
                    hsa_signal_load_scacquire(completion.values[3]) > 0,
                "Copy completed before its dependency was released");
        hsa_signal_store_screlease(completion.values[0], 0);
        const bool timestamps = config.profiling && !config.benchmark;
        wait_and_check(completion.values[1], timeout, timestamps);
        if(config.peer) wait_and_check(completion.values[2], timeout, timestamps);
        wait_and_check(completion.values[3], timeout, timestamps);
        elapsed_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
        require(std::memcmp(host_source.data, host_destination.data, config.bytes) == 0,
                "Copy data verification failed");
    }
}
}  // namespace

int
main(int argc, char** argv)
{
    options config;
    try
    {
        for(int argument = 1; argument < argc; ++argument)
        {
            const std::string name = argv[argument];
            if(name == "--peer")
                config.peer = true;
            else if(name == "--blit")
                config.blit = true;
            else if(name == "--benchmark")
                config.benchmark = true;
            else
            {
                require(argument + 1 < argc, "Missing option value");
                const auto value = std::stoull(argv[++argument]);
                if(name == "--bytes")
                    config.bytes = value;
                else if(name == "--iterations")
                    config.iterations = value;
                else if(name == "--threads")
                    config.threads = value;
                else if(name == "--profiling" && value <= 1)
                    config.profiling = value != 0;
                else
                    require(false, "Unknown option or invalid profiling flag");
            }
        }
    } catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    require(config.bytes >= 4 && config.bytes % 4 == 0 && config.iterations > 0 &&
                config.threads > 0 && config.threads <= 16,
            "Require aligned nonzero bytes, nonzero iterations, and 1-16 threads");
    if(!config.blit)
        require(
            std::getenv("HSA_ENABLE_SDMA") && std::string(std::getenv("HSA_ENABLE_SDMA")) == "1",
            "Set HSA_ENABLE_SDMA=1 for the SDMA regression");
    else
        require(
            std::getenv("HSA_ENABLE_SDMA") && std::string(std::getenv("HSA_ENABLE_SDMA")) == "0",
            "Set HSA_ENABLE_SDMA=0 for the blit comparison");
    check(hsa_init());
    agents devices;
    check(hsa_iterate_agents(collect_agent, &devices));
    if(!devices.cpu.handle || devices.gpus.size() < (config.peer ? 2u : 1u))
    {
        std::cerr << "SKIP: insufficient CPU/GPU agents for requested route\n";
        check(hsa_shut_down());
        return 77;
    }
    uint64_t frequency = 0;
    check(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency));
    check(hsa_amd_profiling_async_copy_enable(config.profiling));
    std::vector<std::thread> workers;
    for(size_t worker = 0; worker < config.threads; ++worker)
        workers.emplace_back(run, std::cref(config), std::cref(devices), worker, frequency * 5);
    for(auto& worker : workers)
        worker.join();
    const auto expected = config.iterations * config.threads * (config.peer ? 3 : 2);
    require(completed == expected, "Missing completed copy records");
    require(timestamp_checks == (config.profiling && !config.benchmark ? expected : 0),
            "Missing profiling timestamp checks");
    check(hsa_amd_profiling_async_copy_enable(false));
    check(hsa_shut_down());
    std::cout << "bytes,iterations,threads,profiling,peer,blit,benchmark,copies,timestamp_checks,"
                 "elapsed_ns\n"
              << config.bytes << ',' << config.iterations << ',' << config.threads << ','
              << config.profiling << ',' << config.peer << ',' << config.blit << ','
              << config.benchmark << ',' << completed << ',' << timestamp_checks << ','
              << elapsed_ns << '\n';
    return EXIT_SUCCESS;
}
