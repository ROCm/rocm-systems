#pragma once

#include <cstddef>
#include <cstdint>

namespace gfx12::rare_scan
{

struct RareToken
{
    uint64_t contents;
    uint32_t type;
};

size_t scan_gfx12(const uint8_t* buf, size_t size, RareToken* out, size_t out_cap);

} // namespace gfx12::rare_scan
