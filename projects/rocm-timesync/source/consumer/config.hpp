#pragma once

#include <variant>

struct ts_db_memory_t {
    int32_t max_entries_per_gpu{0};
};

struct ts_db_influx_t {
    std::string host{"localhost"};
    uint16_t port{8086};
    std::string database{"data"};
    // TODO: credentials, etc.
};

using ts_db_config_t = std::variant<ts_db_influx_t>;

struct ts_config_t {
    ts_db_config_t db;
    ts_db_memory_t cache;
};

ts_config_t LoadConfig(const std::string filename);
