// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Included by logger/debug.hpp, so every LOG_* call site gets these
// formatters for free. WARNING: do NOT also include <fmt/chrono.h> in a TU
// that (transitively) includes logger/debug.hpp — it defines its own
// formatter for std::chrono::duration and the two will collide.
#ifdef FMT_CHRONO_H_
#    error                                                                               \
        "Do not include <fmt/chrono.h> and units/format.hpp in the same translation unit - both define fmt::formatter<std::chrono::duration>."
#endif

#include <chrono>
#include <ratio>
#include <string_view>
#include <utility>

#include <spdlog/fmt/fmt.h>

#include "common/units/chrono.hpp"
#include "common/units/data_size.hpp"
#include "common/units/frequency.hpp"
#include "common/units/power.hpp"

namespace rocprofsys::inline common::units::detail
{

/**
 * Shared parse() for the fmt::formatter specializations below: recognises a
 * leading `~` as the autoscale flag, then delegates the rest of the format
 * spec to a wrapped fmt::formatter<CountRep>. Each specialization derives from
 * this instead of repeating the identical parse()/member pair.
 */
template <typename CountRep>
struct autoscale_parser
{
    constexpr auto parse(fmt::format_parse_context& ctx) -> decltype(ctx.begin())
    {
        const auto* pos = ctx.begin();
        if(pos != ctx.end() && *pos == '~')
        {
            m_autoscale = true;
            ++pos;
            ctx.advance_to(pos);
        }
        return m_count_formatter.parse(ctx);
    }

protected:
    fmt::formatter<CountRep> m_count_formatter;
    bool                     m_autoscale = false;
};

}  // namespace rocprofsys::inline common::units::detail

// ---------------------------------------------------------------------------
// fmt::formatter<rocprofsys::common::units::frequency<Rep, Period>>
//
// Default:   fmt::format("{}", 2_mhz)         -> "2 MHz"
// Autoscale: fmt::format("{:~}", 50000000_hz) -> "50 MHz"
//            Composes with specs: "{:~.2f}"   -> "1.50 kHz"
// ---------------------------------------------------------------------------
template <typename Rep, typename Period>
struct fmt::formatter<rocprofsys::common::units::frequency<Rep, Period>>
: rocprofsys::common::units::detail::autoscale_parser<Rep>
{
    template <typename FormatContext>
    auto format(const rocprofsys::common::units::frequency<Rep, Period>& value,
                FormatContext&                                           ctx) const
    {
        if(this->m_autoscale)
        {
            using rocprofsys::common::units::frequency_cast;
            using rocprofsys::common::units::hertz;
            const auto freq_hz =
                static_cast<double>(frequency_cast<hertz>(value).count());
            const auto [scaled, suffix] = autoscale_freq(freq_hz);
            auto out = this->m_count_formatter.format(static_cast<Rep>(scaled), ctx);
            return fmt::format_to(out, " {}", suffix);
        }
        using rocprofsys::common::units::frequency_suffix;
        auto out = this->m_count_formatter.format(value.count(), ctx);
        return fmt::format_to(out, " {}", frequency_suffix<Period>::k_value);
    }

private:
    static auto autoscale_freq(double freq_hz) -> std::pair<double, std::string_view>
    {
        using rocprofsys::common::units::frequency_suffix;
        constexpr double k_ghz =
            static_cast<double>(std::giga::num) / static_cast<double>(std::giga::den);
        constexpr double k_mhz =
            static_cast<double>(std::mega::num) / static_cast<double>(std::mega::den);
        constexpr double k_khz =
            static_cast<double>(std::kilo::num) / static_cast<double>(std::kilo::den);
        const double mag = freq_hz < 0.0 ? -freq_hz : freq_hz;
        if(mag >= k_ghz)
        {
            return { freq_hz / k_ghz, frequency_suffix<std::giga>::k_value };
        }
        if(mag >= k_mhz)
        {
            return { freq_hz / k_mhz, frequency_suffix<std::mega>::k_value };
        }
        if(mag >= k_khz)
        {
            return { freq_hz / k_khz, frequency_suffix<std::kilo>::k_value };
        }
        return { freq_hz, frequency_suffix<std::ratio<1>>::k_value };
    }
};

// ---------------------------------------------------------------------------
// fmt::formatter<rocprofsys::common::units::data_size<Rep, Scale>>
//
// Default:   fmt::format("{}", 2_mb)         -> "2 MB"
//            fmt::format("{}", 2_mib)        -> "2 MiB"
// Autoscale: fmt::format("{:~}", 5000000_b)  -> "5 MB"
//            Autoscaling always reports decimal units, whatever family the
//            input used, since the value is normalised to bytes first.
// ---------------------------------------------------------------------------
template <typename Rep, typename Scale>
struct fmt::formatter<rocprofsys::common::units::data_size<Rep, Scale>>
: rocprofsys::common::units::detail::autoscale_parser<Rep>
{
    template <typename FormatContext>
    auto format(const rocprofsys::common::units::data_size<Rep, Scale>& value,
                FormatContext&                                          ctx) const
    {
        if(this->m_autoscale)
        {
            using rocprofsys::common::units::bytes;
            using rocprofsys::common::units::data_size_cast;
            const auto bytes_val =
                static_cast<double>(data_size_cast<bytes>(value).count());
            const auto [scaled, suffix] = autoscale_size(bytes_val);
            auto out = this->m_count_formatter.format(static_cast<Rep>(scaled), ctx);
            return fmt::format_to(out, " {}", suffix);
        }
        using rocprofsys::common::units::data_size_suffix;
        auto out = this->m_count_formatter.format(value.count(), ctx);
        return fmt::format_to(out, " {}", data_size_suffix<Scale>::k_value);
    }

private:
    static auto autoscale_size(double bytes_val) -> std::pair<double, std::string_view>
    {
        using rocprofsys::common::units::data_size_suffix;
        using rocprofsys::common::units::gigabytes;
        using rocprofsys::common::units::kilobytes;
        using rocprofsys::common::units::megabytes;
        using rocprofsys::common::units::terabytes;
        constexpr double k_kb = static_cast<double>(kilobytes::scale::num) /
                                static_cast<double>(kilobytes::scale::den);
        constexpr double k_mb = static_cast<double>(megabytes::scale::num) /
                                static_cast<double>(megabytes::scale::den);
        constexpr double k_gb = static_cast<double>(gigabytes::scale::num) /
                                static_cast<double>(gigabytes::scale::den);
        constexpr double k_tb = static_cast<double>(terabytes::scale::num) /
                                static_cast<double>(terabytes::scale::den);
        const double mag = bytes_val < 0.0 ? -bytes_val : bytes_val;
        if(mag >= k_tb)
        {
            return { bytes_val / k_tb, data_size_suffix<terabytes::scale>::k_value };
        }
        if(mag >= k_gb)
        {
            return { bytes_val / k_gb, data_size_suffix<gigabytes::scale>::k_value };
        }
        if(mag >= k_mb)
        {
            return { bytes_val / k_mb, data_size_suffix<megabytes::scale>::k_value };
        }
        if(mag >= k_kb)
        {
            return { bytes_val / k_kb, data_size_suffix<kilobytes::scale>::k_value };
        }
        return { bytes_val, data_size_suffix<std::ratio<1>>::k_value };
    }
};

// ---------------------------------------------------------------------------
// fmt::formatter<rocprofsys::common::units::power<Rep, Period>>
//
// Default:   fmt::format("{}", 250_mw)  -> "250 mW"
// Autoscale: fmt::format("{:~}", 0.5_w) -> "500 mW"
// ---------------------------------------------------------------------------
template <typename Rep, typename Period>
struct fmt::formatter<rocprofsys::common::units::power<Rep, Period>>
: rocprofsys::common::units::detail::autoscale_parser<Rep>
{
    template <typename FormatContext>
    auto format(const rocprofsys::common::units::power<Rep, Period>& value,
                FormatContext&                                       ctx) const
    {
        if(this->m_autoscale)
        {
            using rocprofsys::common::units::power_cast;
            using rocprofsys::common::units::watt;
            const auto watts_val = static_cast<double>(power_cast<watt>(value).count());
            const auto [scaled, suffix] = autoscale_power(watts_val);
            auto out = this->m_count_formatter.format(static_cast<Rep>(scaled), ctx);
            return fmt::format_to(out, " {}", suffix);
        }
        using rocprofsys::common::units::power_suffix;
        auto out = this->m_count_formatter.format(value.count(), ctx);
        return fmt::format_to(out, " {}", power_suffix<Period>::k_value);
    }

private:
    static auto autoscale_power(double watts_val) -> std::pair<double, std::string_view>
    {
        using rocprofsys::common::units::power_suffix;
        constexpr double k_kw =
            static_cast<double>(std::kilo::num) / static_cast<double>(std::kilo::den);
        constexpr double k_mw =
            static_cast<double>(std::milli::num) / static_cast<double>(std::milli::den);
        constexpr double k_uw =
            static_cast<double>(std::micro::num) / static_cast<double>(std::micro::den);
        constexpr double k_nw =
            static_cast<double>(std::nano::num) / static_cast<double>(std::nano::den);
        const double mag = watts_val < 0.0 ? -watts_val : watts_val;
        if(mag == 0.0)
        {
            return { watts_val, power_suffix<std::ratio<1>>::k_value };
        }
        if(mag >= k_kw)
        {
            return { watts_val / k_kw, power_suffix<std::kilo>::k_value };
        }
        if(mag >= 1.0)
        {
            return { watts_val, power_suffix<std::ratio<1>>::k_value };
        }
        if(mag >= k_mw)
        {
            return { watts_val / k_mw, power_suffix<std::milli>::k_value };
        }
        if(mag >= k_uw)
        {
            return { watts_val / k_uw, power_suffix<std::micro>::k_value };
        }
        return { watts_val / k_nw, power_suffix<std::nano>::k_value };
    }
};

// ---------------------------------------------------------------------------
// fmt::formatter<std::chrono::duration<Rep, Period>>
//
// Default:   fmt::format("{}", 1500ms)   -> "1500 ms"
// Autoscale: fmt::format("{:~}", 1500ms) -> "1.5 s"
// ---------------------------------------------------------------------------
template <typename Rep, typename Period>
struct fmt::formatter<std::chrono::duration<Rep, Period>>
: rocprofsys::common::units::detail::autoscale_parser<double>
{
    template <typename FormatContext>
    auto format(const std::chrono::duration<Rep, Period>& value, FormatContext& ctx) const
    {
        if(this->m_autoscale)
        {
            const auto [scaled, suffix] = autoscale_duration(value);
            auto out                    = this->m_count_formatter.format(scaled, ctx);
            return fmt::format_to(out, " {}", suffix);
        }
        auto out =
            this->m_count_formatter.format(static_cast<double>(value.count()), ctx);
        using rocprofsys::common::units::duration_suffix;
        return fmt::format_to(out, " {}", duration_suffix<Period>::k_value);
    }

private:
    /**
     * Uses duration<double, TargetPeriod> cast rather than dividing by a tiny
     * constant to avoid FP rounding errors (e.g. 250000ns -> "250 us" not
     * "250.000...03 us").
     */
    static auto autoscale_duration(const std::chrono::duration<Rep, Period>& dur)
        -> std::pair<double, std::string_view>
    {
        using rocprofsys::common::units::duration_suffix;
        constexpr double k_milli =
            static_cast<double>(std::milli::num) / static_cast<double>(std::milli::den);
        constexpr double k_micro =
            static_cast<double>(std::micro::num) / static_cast<double>(std::micro::den);
        const double secs = std::chrono::duration<double>(dur).count();
        const double mag  = secs < 0.0 ? -secs : secs;
        if(mag >= 1.0)
        {
            return { secs, duration_suffix<std::ratio<1>>::k_value };
        }
        if(mag >= k_milli)
        {
            return { std::chrono::duration<double, std::milli>(dur).count(),
                     duration_suffix<std::milli>::k_value };
        }
        if(mag >= k_micro)
        {
            return { std::chrono::duration<double, std::micro>(dur).count(),
                     duration_suffix<std::micro>::k_value };
        }
        return { std::chrono::duration<double, std::nano>(dur).count(),
                 duration_suffix<std::nano>::k_value };
    }
};
