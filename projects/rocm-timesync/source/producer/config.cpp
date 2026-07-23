#include <filesystem>

#include <yaml-cpp/yaml.h>
#include "config.hpp"

config_t LoadConfig(const std::string filename)
{
    auto cfg = config_t();

    if (!std::filesystem::exists(filename)) {
        return cfg;
    }

    auto root = YAML::LoadFile(filename);

    if (auto p = root["channels"])
    {
        if (auto q = p["hz_precision_high"])
        {
            if (auto r = q["hz"]) cfg.hz_high.hz = r.as<uint32_t>();
            if (auto r = q["order"]) cfg.hz_high.order = r.as<uint32_t>();
        }

        if (auto q = p["hz_precision_low"])
        {
            if (auto r = q["hz"]) cfg.hz_low.hz = r.as<uint32_t>();
            if (auto r = q["order"]) cfg.hz_low.order = r.as<uint32_t>();
        }
    }
    return cfg; 
}
