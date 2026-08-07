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
        // TODO memory
    }

    if (auto p = root["cache"])
    {
        if (auto q = p["max_entries_per_gpu"]) cfg.cache.max_entries_per_gpu = q.as<uint32_t>();
    }

    if (cfg.cache.max_entries_per_gpu < 2)
        cfg.cache.max_entries_per_gpu = 2;

    // TODO other config options

    return cfg; 
}
