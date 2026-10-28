// slice_emitter.cpp
//
// Non-template helpers for slice_emitter. The templated process /
// emit_nested bodies stay in the header (they call into the
// nested helper classes defined here).

#include "cpu_snap/processing/timeline/slice_emitter.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "cpu_snap/core/format_hex.hpp"

namespace cpu_snap::processing::timeline
{

namespace
{

std::string default_hex_name(std::uint64_t ip)
{
    return core::format_hex(ip);
}

}  // namespace

// ----- slice_emitter ------------------------------------------------------

slice_emitter::slice_emitter(slice_emitter_config cfg)
: m_cfg{ std::move(cfg) }
{
    if(!m_cfg.resolver)
    {
        m_cfg.resolver = &default_hex_name;
    }
}

std::string slice_emitter::resolve_name(std::uint64_t ip) const
{
    try
    {
        return m_cfg.resolver(ip);
    } catch(...)  // a throwing user resolver degrades to the hex name, never aborts emission
    {
        return default_hex_name(ip);
    }
}

bool slice_emitter::resolves_to_same_symbol(std::uint64_t lhs, std::uint64_t rhs) const
{
    if(!m_cfg.symbol_identity)
    {
        return false;
    }
    const auto lhs_key = m_cfg.symbol_identity(lhs);
    if(!lhs_key)
    {
        return false;
    }
    const auto rhs_key = m_cfg.symbol_identity(rhs);
    return rhs_key && *lhs_key == *rhs_key;
}

void slice_emitter::filter_library_frames(core::raw_sample& smp) const
{
    std::uint32_t out = 0;
    for(std::uint32_t i = 0; i < smp.frames.size(); ++i)
    {
        const std::uint64_t ip = smp.frames[i];
        if(!m_cfg.library_frame_filter(ip, std::string_view{}))
        {
            continue;
        }
        smp.frames[out++] = ip;
    }
    smp.frames.resize(out);
    if(out != 0)
    {
        smp.ip = smp.frames[0];
    }
}

}  // namespace cpu_snap::processing::timeline
