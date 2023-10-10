/******************************************************************************
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#include "metrics.hpp"

#include <dlfcn.h>  // for dladdr
#include <cstdlib>
#include <filesystem>
#include <optional>

#include "glog/logging.h"
#include "lib/common/xml.hpp"
#include "rocprofiler/rocprofiler.h"

namespace counters
{
namespace
{
MetricMap
loadXml(const std::string& filename)
{
    MetricMap ret;
    DLOG(INFO) << "Loading Counter Config: " << filename;
    // todo: return unique_ptr....
    auto xml = xml::Xml::Create(filename);
    LOG_IF(FATAL, !xml)
        << "Could not open XML Counter Config File (set env ROCPROFILER_METRICS_PATH)";

    for(const auto& [gfx_name, nodes] : xml->GetAllNodes())
    {
        /**
         * "top." is used to designate the root encapsulation of all contained XML subroots (in our
         * case "gfxX"). This is inserted by the parser so it will always be present. .metric
         * denotes XML tags that are contained in the subroots. This will not change unless we
         * respec the XML (which we should...).
         */
        if(gfx_name.find("metric") == std::string::npos ||
           gfx_name.find("top.") == std::string::npos)
            continue;

        auto& metricVec =
            ret.emplace(gfx_name.substr(strlen("top."),
                                        gfx_name.size() - strlen("top.") - strlen(".metric")),
                        std::vector<Metric>())
                .first->second;
        for(const auto& node : nodes)
        {
            metricVec.emplace_back(node->opts["name"],
                                   node->opts["block"],
                                   node->opts["event"],
                                   node->opts["descr"],
                                   node->opts["expr"]);
        }
    }

    DLOG(INFO) << fmt::format("{}", ret);
    return ret;
}

std::string
findViaInstallPath(const std::string& filename)
{
    Dl_info dl_info;
    DLOG(INFO) << filename << " is being looked up via install path";
    if(dladdr(reinterpret_cast<const void*>(rocprofiler_query_available_agents), &dl_info) != 0)
    {
        return std::filesystem::path{dl_info.dli_fname}.remove_filename() /
               fmt::format("../lib/{}", filename);
    }
    return filename;
}

std::string
findViaEnvironment(const std::string& filename)
{
    if(getenv("ROCPROFILER_METRICS_PATH"))
    {
        DLOG(INFO) << filename << " is being looked up via env variable ROCPROFILER_METRICS_PATH";
        return std::filesystem::path{std::string(getenv("ROCPROFILER_METRICS_PATH"))} / filename;
    }
    // No environment variable, lookup via install path
    return findViaInstallPath(filename);
}

}  // namespace

MetricMap
getDerivedHardwareMetrics()
{
    return loadXml(findViaEnvironment("derived_counters.xml"));
}

MetricMap
getBaseHardwareMetrics()
{
    return loadXml(findViaEnvironment("basic_counters.xml"));
}

};  // namespace counters
