// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <fstream>
#include <memory>
#include <set>
#include <ranges>
#include <string>
#include <vector>

#include "core/utility.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

namespace rocprofsys::pmc::collectors::nic
{

// ---------------------------------------------------------------------------
// /proc/net/dev enumeration
// ---------------------------------------------------------------------------

/// Returns the names of all network interfaces found in /proc/net/dev,
/// excluding the loopback interface.
inline std::vector<std::string>
enumerate_proc_net_dev()
{
    std::vector<std::string> result;
    std::ifstream            ifs{ "/proc/net/dev" };
    if(!ifs) return result;

    std::string line;
    std::getline(ifs, line);  // skip header line 1
    std::getline(ifs, line);  // skip header line 2

    while(std::getline(ifs, line))
    {
        const auto colon = line.find(':');
        if(colon == std::string::npos) continue;
        auto name = line.substr(0, colon);
        const auto start = name.find_first_not_of(" ");
        if(start == std::string::npos) continue;
        name = name.substr(start);
        if(name == "lo") continue;
        result.push_back(std::move(name));
    }
    return result;
}

// ---------------------------------------------------------------------------
// NIC classification
// ---------------------------------------------------------------------------

struct nic_classification
{
    std::vector<std::string> conventional;  ///< Route to PAPI net component
    std::vector<std::string> ai;            ///< Route to AMD SMI
};

/// Splits the interfaces named in @p requested into conventional and AI
/// categories.
///
/// @param requested   "all", "none", or a comma-separated list of interface
///                    names (as produced by ROCPROFSYS_SAMPLING_NICS).
/// @param ai_names    Set of interface names identified as AI NICs by AMD SMI.
///                    May be empty when ROCPROFSYS_BUILD_AINIC is not defined.
/// @param available   Interfaces found in /proc/net/dev (loopback excluded).
inline nic_classification
classify_nics(const std::string&              requested,
              const std::set<std::string>&    ai_names,
              const std::vector<std::string>& available)
{
    nic_classification result;
    if(requested.empty() || requested == "none") return result;

    auto is_ai = [&](const std::string& name) { return ai_names.count(name) > 0; };

    if(requested == "all")
    {
        for(const auto& iface : available)
        {
            if(is_ai(iface))
                result.ai.push_back(iface);
            else
                result.conventional.push_back(iface);
        }
        // Include AI NICs that may not appear in /proc/net/dev
        for(const auto& ai : ai_names)
        {
            if(std::find(available.begin(), available.end(), ai) == available.end())
                result.ai.push_back(ai);
        }
    }
    else
    {
        // Parse comma-separated list and trim whitespace from each token
        for(auto part : std::views::split(requested, ','))
        {
            std::string name;
            std::ranges::copy(part, std::back_inserter(name));
            rocprofsys::utility::trim_str(name);
            if(name.empty()) continue;
            if(is_ai(name))
                result.ai.push_back(name);
            else
                result.conventional.push_back(name);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// PAPI event string construction
// ---------------------------------------------------------------------------

/// Builds a space-separated string of PAPI net::: events for the given
/// interfaces.  The minimal default set is rx/tx byte and rx/tx packet per
/// interface.
inline std::string
build_papi_net_events(const std::vector<std::string>& interfaces)
{
    if(interfaces.empty()) return {};
    std::string events;
    for(const auto& iface : interfaces)
    {
        if(!events.empty()) events += ' ';
        events += fmt::format("net:::{}:rx:byte net:::{}:tx:byte "
                              "net:::{}:rx:packet net:::{}:tx:packet",
                              iface, iface, iface, iface);
    }
    return events;
}

// ---------------------------------------------------------------------------
// AI NIC name discovery (gated on ROCPROFSYS_BUILD_AINIC)
// ---------------------------------------------------------------------------

/// Returns the set of network interface names reported by the AMD SMI
/// provider.  Returns an empty set when the build does not include AI NIC
/// support (@c ROCPROFSYS_BUILD_AINIC) or when @p provider is null.
///
/// @tparam DeviceType  Concrete NIC device type (e.g. nic::device<backend>).
/// @tparam Provider    AMD SMI provider type; must expose get_nic_devices<T>().
template <typename DeviceType, typename Provider>
inline std::set<std::string>
get_ai_nic_names([[maybe_unused]] std::shared_ptr<Provider> provider)
{
#if defined(ROCPROFSYS_BUILD_AINIC)
    std::set<std::string> names;
    if(!provider) return names;
    const auto devices = provider->template get_nic_devices<DeviceType>();
    for(const auto& dev : devices)
        names.insert(dev->get_name());
    return names;
#else
    return {};
#endif
}

}  // namespace rocprofsys::pmc::collectors::nic
