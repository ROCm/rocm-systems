#include <nlohmann/json.hpp>
#include <curl/curl.h>

#include "influx_client.hpp"

//#define DBG

namespace rocm_timesync
{

namespace
{
size_t write_callback(
    void* contents,
    size_t size,
    size_t nmemb,
    void* user_data)
{
    auto* response = static_cast<std::string*>(user_data);

    response->append(
        static_cast<const char*>(contents),
        size * nmemb);

    return size * nmemb;
}
}  // namespace

influx_client::influx_client(
    std::string host,
    uint16_t port,
    std::string database)
: host_(std::move(host))
, port_(port)
, database_(std::move(database))
{
    // test that host is reachable
    if (!ping())
        throw std::runtime_error("Failed to connect to the database host");

    // create database if it does not exist
    if (!ensure_database())
        throw std::runtime_error("Failed to create the database");
}

bool
influx_client::write(const entry_t& entry)
{
    std::vector<entry_t> v = {entry};
    return write_batch(v);
}

bool
influx_client::write_batch(const std::vector<entry_t>& entries)
{
    if(entries.empty())
        return true;

    std::ostringstream payload;

    for(const auto& entry : entries)
    {
        const auto& point = entry.point;

        payload << "gpu_timesync";
        payload << ",gpu_id=" << entry.gpu_id;
        payload << " ";
        payload << "system_timestamp="
                << point.system_timestamp
                << "i";
        payload << " ";
        payload << point.gpu_timestamp;
        payload << '\n';
    }

    CURL* curl = curl_easy_init();
    if(!curl)
        return false;

    const auto body = payload.str();

    curl_easy_setopt(
        curl,
        CURLOPT_URL,
        make_write_url().c_str());

    curl_easy_setopt(
        curl,
        CURLOPT_POSTFIELDS,
        body.c_str());

    curl_easy_setopt(
        curl,
        CURLOPT_POSTFIELDSIZE,
        static_cast<long>(body.size()));

    CURLcode rc = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(
        curl,
        CURLINFO_RESPONSE_CODE,
        &http_code);

    curl_easy_cleanup(curl);

    return (rc == CURLE_OK &&
            http_code >= 200 &&
            http_code < 300);
}

bool
influx_client::lookup_oldest_k(
    uint32_t gpu_id,
    uint64_t k,
    std::vector<timesync_point>& k_points)
{
    k_points.clear();

    std::ostringstream q;
    q << "SELECT system_timestamp "
      << "FROM gpu_timesync "
      << "WHERE gpu_id='"
      << gpu_id
      << "' "
      << "ORDER BY time ASC "
      << "LIMIT "
      << k;

    auto response = query(q.str());
    if(response.empty())
        return false;

    auto json = nlohmann::json::parse(response);

    auto& results = json["results"];
    if(results.empty())
        return false;

    if(results[0].contains("error"))
        return false;

    if(!results[0].contains("series"))
        return true; // no points found

    auto& series = results[0]["series"];
    if(series.empty())
        return true;

    auto& values = series[0]["values"];

    k_points.reserve(values.size());
    for(const auto& value : values)
    {
        k_points.push_back(timesync_point{
            .gpu_timestamp = value[0].get<uint64_t>(),
            .system_timestamp = value[1].get<uint64_t>(),
        });
    }

    // points already in ASC order
    return true;
}

bool
influx_client::lookup_newest_k(
    uint32_t gpu_id,
    uint64_t k,
    std::vector<timesync_point>& k_points)
{
    k_points.clear();

    std::ostringstream q;
    q << "SELECT system_timestamp "
      << "FROM gpu_timesync "
      << "WHERE gpu_id='"
      << gpu_id
      << "' "
      << "ORDER BY time DESC "
      << "LIMIT "
      << k;

    auto response = query(q.str());
    if(response.empty())
        return false;

    auto json = nlohmann::json::parse(response);

    auto& results = json["results"];
    if(results.empty())
        return false;

    if(results[0].contains("error"))
        return false;

    if(!results[0].contains("series"))
        return true; // no points found

    auto& series = results[0]["series"];
    if(series.empty())
        return true;

    auto& values = series[0]["values"];

    k_points.reserve(values.size());
    for(const auto& value : values)
    {
        k_points.push_back(timesync_point{
            .gpu_timestamp = value[0].get<uint64_t>(),
            .system_timestamp = value[1].get<uint64_t>(),
        });
    }

    // return in ASC order
    std::reverse(k_points.begin(), k_points.end());
    return true;
}

std::string
influx_client::make_write_url() const
{
    std::ostringstream ss;

    ss << "http://" << host_
       << ":" << port_
       << "/write?db=" << database_
       << "&precision=ns";

    return ss.str();
}

std::string
influx_client::make_query_url() const
{
    std::ostringstream ss;

    ss << "http://" << host_
       << ":" << port_
       << "/query?db=" << database_
       << "&precision=ns";

    return ss.str();
}

std::string
influx_client::make_ping_url() const
{
    std::ostringstream ss;

    ss << "http://" << host_
       << ":" << port_
       << "/ping";

    return ss.str();
}

std::string
influx_client::query(std::string_view influxql)
{
    CURL* curl = curl_easy_init();
    if(!curl)
        return {};

    std::string response;

    char* escaped =
        curl_easy_escape(
            curl,
            influxql.data(),
            static_cast<int>(influxql.size()));

    if(!escaped)
    {
        curl_easy_cleanup(curl);
        return {};
    }

    std::string url =
        make_query_url() +
        "&epoch=ns"
        "&q=" +
        escaped;


    curl_free(escaped);

    curl_easy_setopt(curl,
                     CURLOPT_URL,
                     url.c_str());

    curl_easy_setopt(curl,
                     CURLOPT_WRITEFUNCTION,
                     write_callback);

    curl_easy_setopt(curl,
                     CURLOPT_WRITEDATA,
                     &response);

    CURLcode rc = curl_easy_perform(curl);

    curl_easy_cleanup(curl);

    return (rc == CURLE_OK) ? response : std::string{};
}

bool
influx_client::ping() const
{
    CURL* curl = curl_easy_init();
    if(!curl)
        return false;

    curl_easy_setopt(
        curl,
        CURLOPT_URL,
        make_ping_url().c_str());

    CURLcode rc = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(
        curl,
        CURLINFO_RESPONSE_CODE,
        &http_code);

    curl_easy_cleanup(curl);

    return (rc == CURLE_OK && http_code == 204);
}

bool
influx_client::ensure_database()

{
    std::ostringstream q;

    q << "CREATE DATABASE " << database_;
    auto response = query(q.str());
    if(response.empty())
        return false;

    return true;
}

bool
influx_client::lookup_before(
    uint32_t gpu_id,
    uint64_t gpu_timestamp,
    timesync_point& point)
{
    std::ostringstream q;

    q << "SELECT system_timestamp "
      << "FROM gpu_timesync "
      << "WHERE gpu_id='"
      << gpu_id
      << "' "
      << "AND time <= "
      << gpu_timestamp
      << " "
      << "ORDER BY time DESC "
      << "LIMIT 1";

    auto response = query(q.str());

    if(response.empty())
        return false;

#ifdef DBG
    std::cerr << "\n========== INFLUX DEBUG ==========\n";
    std::cerr << "QUERY: " << q.str() << "\n";
    std::cerr << "RAW RESPONSE:\n" << response << "\n";
    std::cerr << "==================================\n";
#endif

    auto json = nlohmann::json::parse(response);

    auto& results = json["results"];

    if(results.empty())
        return false;

    if(!results[0].contains("series"))
        return false;

    auto& series = results[0]["series"];

    if(series.empty())
        return false;

    auto& values = series[0]["values"];

    if(values.empty())
        return false;

    point.gpu_timestamp =
        values[0][0].get<uint64_t>();

    point.system_timestamp =
        values[0][1].get<uint64_t>();

    return true;
}

bool
influx_client::lookup_after(
    uint32_t gpu_id,
    uint64_t gpu_timestamp,
    timesync_point& point)
{
    std::ostringstream q;

    q << "SELECT system_timestamp "
      << "FROM gpu_timesync "
      << "WHERE gpu_id='"
      << gpu_id
      << "' "
      << "AND time >= "
      << gpu_timestamp
      << " "
      << "ORDER BY time ASC "
      << "LIMIT 1";

    auto response = query(q.str());

    if(response.empty())
        return false;

#ifdef DBG
    std::cerr << "\n========== INFLUX DEBUG ==========\n";
    std::cerr << "QUERY: " << q.str() << "\n";
    std::cerr << "RAW RESPONSE:\n" << response << "\n";
    std::cerr << "==================================\n";
#endif

    auto json = nlohmann::json::parse(response);

    auto& results = json["results"];

    if(results.empty())
        return false;

    if(results[0].contains("error"))
        return false;

    if(!results[0].contains("series"))
        return false;

    auto& series = results[0]["series"];
    if(series.empty())
        return false;

    auto& values = series[0]["values"];
    if(values.empty())
        return false;

    point.gpu_timestamp = values[0][0].get<uint64_t>();
    point.system_timestamp = values[0][1].get<uint64_t>();

#ifdef DBG
    std::cerr
        << "LOOKUP_AFTER parsed gpu="
        << point.gpu_timestamp
        << " sys="
        << point.system_timestamp
        << "\n";
#endif

    return true;
}

} // namespace rocm_timesync
