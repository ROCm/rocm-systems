#pragma once

#include <string>

namespace config
{

constexpr const uint32_t DFLT_HZ_PRECISION_HIGH = 1000;
constexpr const uint32_t DFLT_HZ_PRECISION_LOW = 1;
constexpr const uint8_t DFLT_RING_ORDER = 16;

typedef struct
{
    uint32_t hz_precision_high = DFLT_HZ_PRECISION_HIGH;
    uint32_t hz_precision_low = DFLT_HZ_PRECISION_LOW;
    uint8_t ring_order = DFLT_RING_ORDER;
} config_t;

config_t LoadConfig(const std::string filename);

} // namespace config
