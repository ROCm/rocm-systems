#pragma once

#include <cstdint>

extern "C"
{
    int detach();

    int attach(uint32_t pid);
}
