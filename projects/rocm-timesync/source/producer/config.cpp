#include <filesystem>

#include <yaml-cpp/yaml.h>
#include "config.hpp"

namespace config
{

config_t LoadConfig(const std::string filename)
{
    auto cfg = config_t();

    if (!std::filesystem::exists(filename)) {
        return cfg;
    }

    auto root = YAML::LoadFile(filename);

    if (auto p = root["hz_precision_high"])
    {
        cfg.hz_precision_high = p.as<uint32_t>();
    }

    if (auto p = root["hz_precision_low"])
    {
        cfg.hz_precision_low = p.as<uint32_t>();
    }

    if (auto p = root["ring_order"])
    {
        cfg.ring_order = p.as<uint8_t>();
    }

    return cfg; 
}

} // namespace config
