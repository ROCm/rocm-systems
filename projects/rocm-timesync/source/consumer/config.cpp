#include <filesystem>

#include <yaml-cpp/yaml.h>
#include "config.hpp"

#include <iostream>

ts_config_t LoadConfig(const std::string filename)
{
    auto cfg = ts_config_t();
    if (!std::filesystem::exists(filename))
        return cfg;

    auto root = YAML::LoadFile(filename);
    if (auto p = root["db"])
    {
        if (auto q = p["influx"])
        {
            auto& db = cfg.db.emplace<ts_db_influx_t>();
            if (auto r = q["host"]) db.host = r.as<std::string>();
            if (auto r = q["port"]) db.port = r.as<int>();
            if (auto r = q["database"]) db.database = r.as<std::string>();
        }
    }

    // default cache size without a db is unlimited
    if (std::holds_alternative<std::monostate>(cfg.db))
        cfg.cache.max_entries_per_gpu = -1;

    if (auto q = root["cache"])
    {
        if (auto r = q["max_entries_per_gpu"]) cfg.cache.max_entries_per_gpu = r.as<int64_t>();
    }

    return cfg; 
}
