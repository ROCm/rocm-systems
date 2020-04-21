/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
*/
#include "RdciStatsSubSystem.h"
#include <getopt.h>
#include <unistd.h>
#include <signal.h>
#include <limits>
#include <iomanip>
#include "rdc_lib/rdc_common.h"
#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"

namespace amd {
namespace rdc {

RdciStatsSubSystem::RdciStatsSubSystem() {
}

RdciStatsSubSystem::~RdciStatsSubSystem() {
}


void RdciStatsSubSystem::parse_cmd_opts(int argc, char ** argv) {
    const int HOST_OPTIONS = 1000;
    const struct option long_options[] = {
        {"host", required_argument, nullptr, HOST_OPTIONS },
        {"help", optional_argument, nullptr, 'h' },
        {"unauth", optional_argument, nullptr, 'u' },
        {"jstart", required_argument, nullptr, 's' },
        {"jstop", required_argument, nullptr, 'x' },
        {"job", required_argument, nullptr, 'j' },
        {"jremove", required_argument, nullptr, 'r'},
        {"jremoveall", optional_argument, nullptr, 'a' },
        {"verbose", optional_argument, nullptr, 'v'},
        {"group", required_argument, nullptr, 'g'},
        { nullptr,  0 , nullptr, 0 }
    };

    bool is_group_id_set = false;
    int option_index = 0;
    int opt = 0;

    while ((opt = getopt_long(argc, argv, "huvas:x:j:r:g:",
                long_options, &option_index)) != -1) {
        switch (opt) {
            case HOST_OPTIONS:
                ip_port_ = optarg;
                break;
            case 'h':
                stats_ops_ = STATS_HELP;
                return;
            case 'u':
                 use_auth_ = false;
                 break;
            case 's':
                stats_ops_ = STATS_START_RECORDING;
                job_id_ = optarg;
                break;
            case 'g':
                if (!IsNumber(optarg)) {
                    show_help();
                    throw RdcException(RDC_ST_BAD_PARAMETER,
                        "The group id needs to be a number");
                }
                group_id_ = std::stoi(optarg);
                is_group_id_set = true;
                break;
            case 'x':
                stats_ops_ = STATS_STOP_RECORDING;
                job_id_ = optarg;
                break;
            case 'j':
                stats_ops_ = STATS_DISPLAY;
                job_id_ = optarg;
                break;
            case 'v':
                is_verbose_ = true;
                break;
            case 'r':
                stats_ops_ = STATS_REMOVE;
                job_id_ = optarg;
                break;
            case 'a':
                stats_ops_ = STATS_REMOVE_ALL;
                break;
            default:
                show_help();
                throw RdcException(RDC_ST_BAD_PARAMETER,
                        "Unknown command line options");
        }
    }

    if (stats_ops_ == STATS_START_RECORDING
        && is_group_id_set == false) {
            show_help();
            throw RdcException(RDC_ST_BAD_PARAMETER,
                    "Need to specify the group id to start recording");
    }
}

void RdciStatsSubSystem::show_help() const {
    std::cout << " stats -- Used to view job statistics.\n\n";
    std::cout << "Usage\n";
    std::cout << "    rdci stats [--host <IP/FQDN>:port] [-u] -s <jobId>"
            << " -g <groupId>\n";
    std::cout << "    rdci stats [--host <IP/FQDN>:port] [-u] -x <jobId>\n";
    std::cout << "    rdci stats [--host <IP/FQDN>:port] [-u] [-v] "
              << "-j <jobId>\n";
    std::cout << "    rdci stats [--host <IP/FQDN>:port] [-u] -r <jobId>\n";
    std::cout << "    rdci stats [--host <IP/FQDN>:port] [-u] -a\n";
    std::cout << "\nFlags:\n";
    show_common_usage();
    std::cout << "  -s  --jstart                   Start recording "
              << "job statistics.\n";
    std::cout << "  -g  --group-id                 The GPU group to query "
              << "on the specified host.\n";
    std::cout << "  -x  --jstop                    Stop recording "
              << "job statistics.\n";
    std::cout << "  -j  --job                      Display "
              << "job statistics.\n";
    std::cout << "  -v  --verbose                  Show job information "
              << "for each GPU.\n";
    std::cout << "  -r  --jremove                  Remove "
              << "job statistics.\n";
    std::cout << "  -a  --jremoveall               Remove "
              << "all job statistics.\n";
}

void RdciStatsSubSystem::show_job_stats(
        const rdc_gpu_usage_info_t& gpu_info) const {
    std::cout << "|------- Execution Stats ----------"
        << "+------------------------------------\n";
    std::cout << "| Start Time *                     | "
        << gpu_info.start_time << "\n";
    std::cout << "| End Time *                       | "
        << gpu_info.end_time << "\n";
    std::cout << "| Total Execution Time (sec) *     | "
        << (gpu_info.end_time-gpu_info.start_time) << "\n";
    std::cout << "+------- Performance Stats --------"
        << "+------------------------------------\n";
    std::cout << "| Energy Consumed (Joules)         | "
        << gpu_info.energy_consumed << "\n";
    std::cout << "| Power Usage (Watts)              | "  << "Max: "
        << gpu_info.power_usage.max_value<< " Min: "<<
        gpu_info.power_usage.min_value << " Avg: "
        << gpu_info.power_usage.average << "\n";
    std::cout << "| SM Clock (MHz)                   | "  << "Max: "
        << gpu_info.gpu_clock.max_value << " Min: " <<
        gpu_info.gpu_clock.min_value << " Avg: "
        << gpu_info.gpu_clock.average << "\n";
    std::cout << "| SM Utilization (%)               | "  << "Max: "
        << gpu_info.gpu_utilization.max_value <<" Min: " <<
        gpu_info.gpu_utilization.min_value << " Avg: " <<
        gpu_info.gpu_utilization.average << "\n";
    std::cout << "| Max GPU Memory Used (bytes) *    | "  <<
        gpu_info.max_gpu_memory_used << "\n";
    std::cout << "| Memory Utilization (%)           | "
        << "Max: " << gpu_info.memory_utilization.max_value
            <<" Min: "<< gpu_info.memory_utilization.min_value
            << " Avg: " << gpu_info.memory_utilization.average << "\n";
    std::cout << "+----------------------------------"
        << "+------------------------------------\n";
}

void RdciStatsSubSystem::process() {
    if (stats_ops_ == STATS_HELP ||
            stats_ops_ == STATS_UNKNOWN) {
        show_help();
        return;
    }

    rdc_status_t result;
    if (stats_ops_ == STATS_START_RECORDING) {
        // Record job every 1 second
        result = rdc_job_start_stats(rdc_handle_, group_id_,
                    const_cast<char*>(job_id_.c_str()), 1000000);
        if (result != RDC_ST_OK) {
            throw RdcException(result, rdc_status_string(result));
        }
        std::cout << "Successfully started recording job "
                << job_id_ << " with a group ID " << group_id_ << std::endl;
        return;
    }

    if (stats_ops_ == STATS_STOP_RECORDING) {
        result = rdc_job_stop_stats(rdc_handle_,
                    const_cast<char*>(job_id_.c_str()));
        if (result != RDC_ST_OK) {
            throw RdcException(result, rdc_status_string(result));
        }
        std::cout << "Successfully stopped recording job "
                << job_id_ << std::endl;
        return;
    }

    if (stats_ops_ == STATS_DISPLAY) {
        rdc_job_info_t job_info;
        result = rdc_job_get_stats(rdc_handle_,
                    const_cast<char*>(job_id_.c_str()), &job_info);
        if (result != RDC_ST_OK) {
            throw RdcException(result, rdc_status_string(result));
        }

        std::cout << "|         Summary \n";
        show_job_stats(job_info.summary);
        if (is_verbose_ == false) {
            return;
        }
        for (uint32_t i = 0; i < job_info.num_gpus; i++) {
            std::cout << "|         GPU " << i << "\n";
            show_job_stats(job_info.gpus[i]);
        }
        return;
    }

    if (stats_ops_ == STATS_REMOVE) {
        result = rdc_job_remove(rdc_handle_,
                    const_cast<char*>(job_id_.c_str()));
        if (result != RDC_ST_OK) {
            throw RdcException(result, rdc_status_string(result));
        }
        std::cout << "Successfully removed job "
                << job_id_ << std::endl;
        return;
    }

    if (stats_ops_ == STATS_REMOVE_ALL) {
        result = rdc_job_remove_all(rdc_handle_);
        if (result != RDC_ST_OK) {
            throw RdcException(result, rdc_status_string(result));
        }
        std::cout << "Successfully removed all jobs\n";
        return;
    }
}

}  // namespace rdc
}  // namespace amd


