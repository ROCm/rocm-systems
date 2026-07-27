#include <thread>
#include <iostream>
#include <cassert>
#include <variant>

#ifdef TIMESYNC_BUILD_INFLUXDB
#include <curl/curl.h>
#endif

#include <rocm-timesync/rocm_timesync.hpp>
#include <core/ipc.hpp>

#include "timesync_db.hpp"
#include "influx_client.hpp"

#define BATCH_SIZE 100

namespace rocm_timesync
{

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

static ipc::channel_t* channel = nullptr;
static bool keep_running = true;
static std::thread streamer;
static timesync_db* ts_client = nullptr;

static int _db_init(const ts_config_t& cfg)
{
    if (std::holds_alternative<ts_db_influx_t>(cfg.db_config)) {
#ifdef TIMESYNC_BUILD_INFLUXDB
        const auto& influx = std::get<ts_db_influx_t>(cfg.db_config);
        ts_client = new influx_client(
            influx.host,
            influx.port,
            influx.database
        );
        return 0;
#else
        std::cerr << "InfluxDB client not compiled" << std::endl;
#endif
    };

    return -EINVAL;
}

int timesync_init(const ts_config_t& cfg)
{
    int status;

#ifdef TIMESYNC_BUILD_INFLUXDB
    curl_global_init(CURL_GLOBAL_DEFAULT);
#endif

    // establish DB connection
    status = _db_init(cfg);
    if (status != 0)
        return status;

    // attach to ringbuffer
    channel = ipc::attach(precision_to_name(cfg.precision));
    if (channel == nullptr)
        return -ENODEV;

    // spawn thread to stream ringbuffer into database
    streamer = std::thread([]() {
        while (keep_running) {
            std::cout << "ipc streamer running...\n";

            std::vector<ipc::event_t> events;

            ipc::poll(channel, [](const ipc::event_t& event) {
                std::cout << "processed event with gpu_id:" << event.gpu_id <<
                    " and gpu_timestamp: " << event.gpu_timestamp_ns << std::endl;
                ts_client->write(event.gpu_id, event.gpu_timestamp_ns, event.system_timestamp_ns);
            });
        }
    });

    return 0;
}

int timesync_deinit()
{
    keep_running = false;
    ipc::stop(channel);
    streamer.join();

    ipc::detach(channel);

#ifdef TIMESYNC_BUILD_INFLUXDB
    curl_global_cleanup();
#endif

    delete ts_client;
    ts_client = nullptr;

    return 0;
}

int timesync_translate(uint32_t agent_kfd_gpu_id, uint64_t agent_timestamp, uint64_t *system_timestamp)
{
    std::cerr << "translate_time request for kfd gpu id: " << agent_kfd_gpu_id << std::endl << std::flush;
    return 0;
}

} // namespace rocm_timesync
