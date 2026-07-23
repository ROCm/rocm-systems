#pragma once

#include <string>

struct channel_config_t
{
    uint32_t hz;
    uint32_t order;
    bool enabled;
};

struct config_t
{
    channel_config_t hz_high = {100, 20, true};
    channel_config_t hz_low = {1, 12, true};
};

config_t LoadConfig(const std::string filename);
