#include <thread>
#include <mutex>
#include <deque>
#include <chrono>
#include <iostream>
#include <cassert>
#include <variant>
#include <unistd.h>

#ifdef ROCM_TIMESYNC_BUILD_INFLUXDB
#include <curl/curl.h>
#endif

#include <rocm-timesync/rocm_timesync.hpp>
#include <core/ipc.hpp>

#include "config.hpp"
#include "timesync_db.hpp"
#include "influx_client.hpp"
#include "memory_client.hpp"

#define DFLT_CONFIG         "/etc/rocm/rocm-timesync/client.yml"
#define STREAM_SLEEP_MS     1

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

// TODO support mem client by itself

static std::thread streamer;
static ts_config_t cfg = {};
static ipc::channel_t* channel = nullptr;
static std::atomic<bool> stop_requested = false;
static timesync_db* db_client = nullptr;

static int _db_init()
{
    if (std::holds_alternative<ts_db_influx_t>(cfg.db)) {
#ifdef ROCM_TIMESYNC_BUILD_INFLUXDB
        const auto& influx = std::get<ts_db_influx_t>(cfg.db);
        /*
        db_client = new influx_client(
            influx.host,
            influx.port,
            influx.database
        );
        */

        db_client = new cached_timesync_db(
            std::make_unique<memory_client>(512),
            std::make_unique<influx_client>(
                influx.host,
                influx.port,
                influx.database
            )
        );
        return 0;
#else
        std::cerr << "InfluxDB client not compiled" << std::endl;
#endif
    };

    return -EINVAL;
}

int timesync_client_init(const ts_client_config_t& ccfg)
{
    int status;

#ifdef ROCM_TIMESYNC_BUILD_INFLUXDB
    curl_global_init(CURL_GLOBAL_DEFAULT);
#endif

    // load config
    std::string fname = ccfg.config_file;
    if (fname.empty())
        fname = DFLT_CONFIG;

    cfg = LoadConfig(fname);

    // create db connection
    status = _db_init();
    if (status != 0)
        return status;

    // attach to ringbuffer using client's required precision
    channel = ipc::attach(precision_to_name(ccfg.precision));
    if (channel == nullptr)
        return -ENODEV;

    // spawn thread to stream ringbuffer into database
    streamer = std::thread([]() {
        while (!stop_requested.load(std::memory_order_acquire)) {
            std::vector<timesync_db::entry_t> entries;
            std::vector<ipc::event_t> events = {};

            ipc::consume(channel, events, STREAM_SLEEP_MS);
            for (const auto& event : events) {
                entries.push_back({
                    .gpu_id = event.gpu_id,
                    .point = {
                        .gpu_timestamp = event.gpu_timestamp_ns,
                        .system_timestamp = event.system_timestamp_ns,
                    },
                });
            }

            const auto num_entries = entries.size();
            if (num_entries == 0) {
                if (STREAM_SLEEP_MS == 0)
                    std::this_thread::yield();
                continue;
            }

            db_client->write_batch(entries);
        }
    });

    return 0;
}

int timesync_client_deinit()
{
    stop_requested.store(true, std::memory_order_release);
    ipc::stop(channel);
    streamer.join();

    ipc::detach(channel);

#ifdef ROCM_TIMESYNC_BUILD_INFLUXDB
    curl_global_cleanup();
#endif

    delete db_client;
    db_client = nullptr;

    return 0;
}

int timesync_client_translate(uint32_t agent_kfd_gpu_id, uint64_t agent_timestamp, uint64_t& system_timestamp)
{
    system_timestamp = 0;

    //return db_client->lookup_or_extrapolate(agent_kfd_gpu_id, agent_timestamp, system_timestamp);
    auto start = std::chrono::steady_clock::now();
    auto ret = db_client->lookup_or_extrapolate(agent_kfd_gpu_id, agent_timestamp, system_timestamp);
    auto end = std::chrono::steady_clock::now();

    assert(ret);

    std::cout << ret << " took " << std::chrono::duration<double, std::micro>(end - start).count() << " us\n" << std::endl;
    return ret;
}

} // namespace rocm_timesync
