#include <sstream>

#include <nlohmann/json.hpp>
#include <curl/curl.h>

#include "influx_client.hpp"

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
{}

std::string
influx_client::make_write_url() const
{
    std::ostringstream ss;

    ss << "http://" << host_
       << ":" << port_
       << "/write?db=" << database_;

    return ss.str();
}

std::string
influx_client::make_query_url() const
{
    std::ostringstream ss;

    ss << "http://" << host_
       << ":" << port_
       << "/query?db=" << database_;

    return ss.str();
}

bool
influx_client::write(uint32_t gpu_id,
                     uint64_t gpu_timestamp,
                     uint64_t system_timestamp)
{
    std::ostringstream payload;

    payload << "gpu_timesync";
    payload << ",gpu_id=" << gpu_id;
    payload << " ";
    payload << "system_timestamp=" << system_timestamp << "i";
    payload << " ";
    payload << gpu_timestamp;

    CURL* curl = curl_easy_init();
    if(!curl)
        return false;

    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    const auto body = payload.str();

    curl_easy_setopt(curl,
                     CURLOPT_URL,
                     make_write_url().c_str());

    curl_easy_setopt(curl,
                     CURLOPT_POSTFIELDS,
                     body.c_str());

    curl_easy_setopt(curl,
                     CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(body.size()));



    CURLcode rc = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl,
                      CURLINFO_RESPONSE_CODE,
                      &http_code);

    curl_easy_cleanup(curl);

    return (rc == CURLE_OK &&
            http_code >= 200 &&
            http_code < 300);
}


bool
influx_client::lookup(uint32_t gpu_id,
                      uint64_t gpu_timestamp,
                      uint64_t& system_timestamp)
{
    std::ostringstream q;

    q << "SELECT system_timestamp "
      << "FROM gpu_timesync "
      << "WHERE gpu_id='"
      << gpu_id
      << "' "
      << "AND time="
      << gpu_timestamp
      << " "
      << "LIMIT 1";

    auto response = query(q.str());

    if(response.empty())
        return false;

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

    system_timestamp =
        values[0][1].get<uint64_t>();

    return true;
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
        make_query_url() + "&q=" + escaped;

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

} // namespace rocm_timesync
