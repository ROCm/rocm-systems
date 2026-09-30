// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <optional>

namespace addc::mca
{

constexpr bool bit(uint64_t value, uint8_t offset) noexcept
{
    return ((value >> offset) & 1) == 1;
}

constexpr uint64_t bits(uint64_t value, uint8_t offset, uint8_t width) noexcept
{
    uint64_t mask = (width == 64) ? UINT64_MAX : ((uint64_t{1} << width) - 1);
    return (value >> offset) & mask;
}

struct McaStatus
{
    uint64_t raw;
    bool val;
    bool overflow;
    bool uc;
    bool en;
    bool misc_v;
    bool addr_v;
    bool pcc;
    bool err_core_id_val;
    bool tcc;
    bool phys_addr_val;
    bool synd_v;
    bool transparent;
    bool cecc;
    bool uecc;
    bool deferred;
    bool poison;
    bool scrub;
    uint8_t err_core_id;
    uint8_t addr_lsb;
    uint8_t error_code_ext;
    uint16_t error_code;

    explicit constexpr McaStatus(uint64_t raw_) noexcept :
        raw{raw_}, val{bit(raw_, 63)}, overflow{bit(raw_, 62)},
        uc{bit(raw_, 61)}, en{bit(raw_, 60)}, misc_v{bit(raw_, 59)},
        addr_v{bit(raw_, 58)}, pcc{bit(raw_, 57)},
        err_core_id_val{bit(raw_, 56)}, tcc{bit(raw_, 55)},
        phys_addr_val{bit(raw_, 54)}, synd_v{bit(raw_, 53)},
        transparent{bit(raw_, 52)}, cecc{bit(raw_, 46)}, uecc{bit(raw_, 45)},
        deferred{bit(raw_, 44)}, poison{bit(raw_, 43)}, scrub{bit(raw_, 40)},
        err_core_id{static_cast<uint8_t>(bits(raw_, 32, 6))},
        addr_lsb{static_cast<uint8_t>(bits(raw_, 24, 6))},
        error_code_ext{static_cast<uint8_t>(bits(raw_, 16, 6))},
        error_code{static_cast<uint16_t>(bits(raw_, 0, 16))}
    {}
};

struct McaIpid
{
    uint64_t raw;
    uint16_t mca_type;
    uint8_t instance_id_hi;
    uint16_t hardware_id;
    uint32_t instance_id;

    explicit constexpr McaIpid(uint64_t raw_) noexcept :
        raw{raw_}, mca_type{static_cast<uint16_t>(bits(raw_, 48, 16))},
        instance_id_hi{static_cast<uint8_t>(bits(raw_, 44, 4))},
        hardware_id{static_cast<uint16_t>(bits(raw_, 32, 12))},
        instance_id{static_cast<uint32_t>(bits(raw_, 0, 32))}
    {}
};

struct McaSynd
{
    uint64_t raw;
    uint32_t syndrome;
    uint8_t error_priority;
    uint8_t length;
    uint32_t error_information;

    explicit constexpr McaSynd(uint64_t raw_) noexcept :
        raw{raw_}, syndrome{static_cast<uint32_t>(bits(raw_, 32, 32))},
        error_priority{static_cast<uint8_t>(bits(raw_, 24, 3))},
        length{static_cast<uint8_t>(bits(raw_, 18, 6))},
        error_information{static_cast<uint32_t>(bits(raw_, 0, 18))}
    {}

    constexpr uint8_t effective_length() const noexcept
    {
        return length < 32 ? length : 32;
    }

    constexpr bool has_valid_syndrome() const noexcept
    {
        return effective_length() != 0;
    }

    constexpr std::optional<uint32_t> syndrome_value() const noexcept
    {
        uint8_t len = effective_length();
        if (len == 0)
            return std::nullopt;
        uint32_t mask = (len == 32) ? UINT32_MAX : ((uint32_t{1} << len) - 1);
        return syndrome & mask;
    }
};

struct McaAddr
{
    uint64_t raw;
    uint64_t error_addr;

    explicit constexpr McaAddr(uint64_t raw_) noexcept :
        raw{raw_}, error_addr{bits(raw_, 0, 64)}
    {}
};

struct McaRegisters
{
    McaStatus status;
    McaIpid ipid;
    McaSynd synd;
    McaAddr addr;
    uint64_t misc0 = 0;
    uint64_t misc1 = 0;

    constexpr McaRegisters() noexcept :
        status{McaStatus{0}}, ipid{McaIpid{0}}, synd{McaSynd{0}},
        addr{McaAddr{0}}, misc0{0}, misc1{0}
    {}

    constexpr McaRegisters(uint64_t status_, uint64_t ipid_, uint64_t synd_,
                           uint64_t addr_, uint64_t misc0_ = 0,
                           uint64_t misc1_ = 0) noexcept :
        status{McaStatus{status_}}, ipid{McaIpid{ipid_}}, synd{McaSynd{synd_}},
        addr{McaAddr{addr_}}, misc0{misc0_}, misc1{misc1_}
    {}
};

} // namespace addc::mca
