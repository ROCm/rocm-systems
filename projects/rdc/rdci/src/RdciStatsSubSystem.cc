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
#include <signal.h>
#include <unistd.h>

#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdciStatsSubSystem::RdciStatsSubSystem() {}

RdciStatsSubSystem::~RdciStatsSubSystem() {}

void RdciStatsSubSystem::parse_cmd_opts(int argc, char** argv) {
  const int HOST_OPTIONS = 1000;
  const int JSON_OPTIONS = 1001;
  const struct option long_options[] = {{"host", required_argument, nullptr, HOST_OPTIONS},
                                        {"help", optional_argument, nullptr, 'h'},
                                        {"unauth", optional_argument, nullptr, 'u'},
                                        {"jstart", required_argument, nullptr, 's'},
                                        {"jstop", required_argument, nullptr, 'x'},
                                        {"job", required_argument, nullptr, 'j'},
                                        {"jremove", required_argument, nullptr, 'r'},
                                        {"jremoveall", optional_argument, nullptr, 'a'},
                                        {"verbose", optional_argument, nullptr, 'v'},
                                        {"group", required_argument, nullptr, 'g'},
                                        {"json", optional_argument, nullptr, JSON_OPTIONS},
                                        {nullptr, 0, nullptr, 0}};

  bool is_group_id_set = false;
  int option_index = 0;
  int opt = 0;

  while ((opt = getopt_long(argc, argv, "huvas:x:j:r:g:", long_options, &option_index)) != -1) {
    switch (opt) {
      case HOST_OPTIONS:
        ip_port_ = optarg;
        break;
      case JSON_OPTIONS:
        set_json_output(true);
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
          throw RdcException(RDC_ST_BAD_PARAMETER, "The group id needs to be a number");
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
        throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command line options");
    }
  }

  if (stats_ops_ == STATS_START_RECORDING && is_group_id_set == false) {
    show_help();
    throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the group id to start recording");
  }
}

void RdciStatsSubSystem::show_help() const {
  if (is_json_output()) return;
  std::cout << " stats -- Used to view job statistics.\n\n";
  std::cout << "Usage\n";
  std::cout << "    rdci stats [--host <IP/FQDN>:port] [-u] [--json] "
            << "-s <jobId> -g <groupId>\n";
  std::cout << "    rdci stats [--host <IP/FQDN>:port] [-u] [--json] "
            << "-x <jobId>\n";
  std::cout << "    rdci stats [--host <IP/FQDN>:port] [-u] [--json] [-v] "
            << "-j <jobId>\n";
  std::cout << "    rdci stats [--host <IP/FQDN>:port] [-u] [--json] "
            << "-r <jobId>\n";
  std::cout << "    rdci stats [--host <IP/FQDN>:port] [-u] [--json] -a\n";
  std::cout << "\nFlags:\n";
  show_common_usage();
  std::cout << "  --json                         "
            << "Output using json.\n";
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

// See rdc_stats_summary_t: min_value is UINT64_MAX for a statistic without samples.
static bool has_samples(const rdc_stats_summary_t& stats) {
  return stats.min_value != std::numeric_limits<uint64_t>::max();
}

// The four JSON values of a statistic, each null without samples. The deviation
// keys keep their published spellings.
static void show_stats_json(const std::string& name, const rdc_stats_summary_t& stats,
                            const std::string& deviation_key) {
  auto value = [&stats](auto v) {
    std::ostringstream out;
    if (has_samples(stats)) {
      out << v;
    } else {
      out << "null";
    }
    return out.str();
  };
  std::cout << "\"" << name << "_max\": " << value(stats.max_value) << ",";
  std::cout << "\"" << name << "_min\": " << value(stats.min_value) << ",";
  std::cout << "\"" << name << "_avg\": " << value(stats.average) << ",";
  std::cout << "\"" << deviation_key << "\": " << value(stats.standard_deviation) << ",";
}

void RdciStatsSubSystem::show_job_stats_json(const rdc_gpu_usage_info_t& gpu_info) const {
  std::cout << "\"start_time\": " << gpu_info.start_time << ",";
  std::cout << "\"end_time\": " << gpu_info.end_time << ",";
  std::cout << "\"execution_time\": " << (gpu_info.end_time - gpu_info.start_time) << ",";
  std::cout << "\"energy_consumed\": " << gpu_info.energy_consumed << ",";

  show_stats_json("power_usage", gpu_info.power_usage, "power_usage_stanard_deviation");
  show_stats_json("gpu_clock", gpu_info.gpu_clock, "gpu_clock_stanard_deviation");
  show_stats_json("memory_clock", gpu_info.memory_clock, "memory_clock_stanard_deviation");
  show_stats_json("gpu_utilization", gpu_info.gpu_utilization, "gpu_utilization_deviation");
  std::cout << "\"max_gpu_memory_used\": " << gpu_info.max_gpu_memory_used << ",";
  show_stats_json("memory_utilization", gpu_info.memory_utilization,
                  "memory_utilization_stanard_deviation");
  show_stats_json("gpu_temperature", gpu_info.gpu_temperature, "gpu_temperature_stanard_deviation");
  show_stats_json("pcie_rx", gpu_info.pcie_rx, "pcie_rx_stanard_deviation");
  show_stats_json("pcie_tx", gpu_info.pcie_tx, "pcie_tx_stanard_deviation");
  show_stats_json("pcie_total", gpu_info.pcie_total, "pcie_total_stanard_deviation");

  std::cout << "\"ecc_correct\": " << gpu_info.ecc_correct << ",";
  std::cout << "\"ecc_uncorrect\": " << gpu_info.ecc_uncorrect;
}

// "Max: 3 Min: 1 Avg: 2 SD: 0.82", or N/A for a statistic without samples.
static std::string stats_text(const rdc_stats_summary_t& stats) {
  if (!has_samples(stats)) {
    return "N/A";
  }
  std::ostringstream out;
  out << "Max: " << stats.max_value << " Min: " << stats.min_value << " Avg: " << stats.average
      << " SD: " << std::fixed << std::setprecision(2) << stats.standard_deviation;
  return out.str();
}

void RdciStatsSubSystem::show_job_stats(const rdc_gpu_usage_info_t& gpu_info) const {
  std::cout << "|------- Execution Stats ----------"
            << "+------------------------------------\n";
  std::cout << "| Start Time                       | "
            << std::put_time(std::gmtime(reinterpret_cast<const time_t*>(&gpu_info.start_time)),
                             "%c %Z")
            << "\n";
  std::cout << "| End Time                         | "
            << std::put_time(std::gmtime(reinterpret_cast<const time_t*>(&gpu_info.end_time)),
                             "%c %Z")
            << "\n";
  std::cout << "| Total Execution Time (sec)       | " << (gpu_info.end_time - gpu_info.start_time)
            << "\n";
  std::cout << "+------- Performance Stats --------"
            << "+------------------------------------\n";
  std::cout << "| Energy Consumed (Joules)         | " << gpu_info.energy_consumed << "\n";
  std::cout << "| Power Usage (Watts)              | " << stats_text(gpu_info.power_usage) << "\n";
  std::cout << "| GPU Clock (MHz)                  | " << stats_text(gpu_info.gpu_clock) << "\n";
  std::cout << "| Memory Clock (MHz)               | " << stats_text(gpu_info.memory_clock) << "\n";
  std::cout << "| GPU Utilization (%)              | " << stats_text(gpu_info.gpu_utilization)
            << "\n";
  std::cout << "| Max GPU Memory Used (bytes)      | " << gpu_info.max_gpu_memory_used << "\n";
  std::cout << "| Memory Utilization (%)           | " << stats_text(gpu_info.memory_utilization)
            << "\n";
  std::cout << "| GPU Temperature (Celsius)        | " << stats_text(gpu_info.gpu_temperature)
            << "\n";
  std::cout << "| PCIe Rx Bandwidth (megabytes)    | " << stats_text(gpu_info.pcie_rx) << "\n";
  std::cout << "| PCIe Tx Bandwidth (megabytes)    | " << stats_text(gpu_info.pcie_tx) << "\n";
  std::cout << "| PCIe Total Bandwidth (megabytes) | " << stats_text(gpu_info.pcie_total) << "\n";
  std::cout << "| Correctable ECC Errors           | " << gpu_info.ecc_correct << "\n";
  std::cout << "| Uncorrectable ECC Errors         | " << gpu_info.ecc_uncorrect << "\n";
  std::cout << "+----------------------------------"
            << "+------------------------------------\n";
}

void RdciStatsSubSystem::process() {
  if (stats_ops_ == STATS_HELP || stats_ops_ == STATS_UNKNOWN) {
    show_help();
    return;
  }

  rdc_status_t result;
  if (stats_ops_ == STATS_START_RECORDING) {
    // Record job every 1 second
    result =
        rdc_job_start_stats(rdc_handle_, group_id_, const_cast<char*>(job_id_.c_str()), 1000000);
    if (result != RDC_ST_OK) {
      throw RdcException(result, rdc_status_string(result));
    }
    if (is_json_output()) {
      std::cout << "\"job_id\": \"" << job_id_ << "\", \"group_id\": \"" << group_id_
                << "\", \"status\": \"ok\"";
    } else {
      std::cout << "Successfully started recording job " << job_id_ << " with a group ID "
                << group_id_ << std::endl;
    }
    return;
  }

  if (stats_ops_ == STATS_STOP_RECORDING) {
    result = rdc_job_stop_stats(rdc_handle_, const_cast<char*>(job_id_.c_str()));
    if (result != RDC_ST_OK) {
      throw RdcException(result, rdc_status_string(result));
    }
    if (is_json_output()) {
      std::cout << "\"job_id\": \"" << job_id_ << "\", \"status\": \"ok\"";
    } else {
      std::cout << "Successfully stopped recording job " << job_id_ << std::endl;
    }
    return;
  }

  if (stats_ops_ == STATS_DISPLAY) {
    rdc_job_info_t job_info{};
    result = rdc_job_get_stats(rdc_handle_, const_cast<char*>(job_id_.c_str()), &job_info);
    if (result != RDC_ST_OK) {
      throw RdcException(result, rdc_status_string(result));
    }

    if (!is_json_output()) {
      std::cout << "|         Summary \n";
      show_job_stats(job_info.summary);
    } else {
      std::cout << "\"job_summary\" : {";
      show_job_stats_json(job_info.summary);
      std::cout << "}";
    }
    if (is_verbose_ == false) {
      return;
    }
    for (uint32_t i = 0; i < job_info.num_gpus; i++) {
      const std::string gpu = entity_to_string(job_info.gpus[i].gpu_id);
      if (!is_json_output()) {
        std::cout << "|         GPU " << gpu << "\n";
        show_job_stats(job_info.gpus[i]);
      } else {
        std::cout << ", \"gpu_" << gpu << "\": {";
        show_job_stats_json(job_info.gpus[i]);
        std::cout << "}";
      }
    }
    if (!is_json_output()) {
      std::cout << "| Processes Tracked                | " << job_info.num_processes << "\n";
      for (uint32_t p = 0; p < job_info.num_processes; ++p) {
        auto& pr = job_info.processes[p];
        std::ostringstream oss;
        oss << "PID: " << pr.pid << ", Start: " << pr.start_time / 1000
            << ", End: " << pr.stop_time / 1000;
        std::cout << "| " << std::setw(33) << std::left << pr.process_name << "| " << oss.str()
                  << "\n";
      }
      std::cout << "+----------------------------------+------------------------------------\n";
    } else {
      std::cout << ", \"processes_tracked\": " << job_info.num_processes;
      std::cout << ", \"processes\": [";
      for (uint32_t p = 0; p < job_info.num_processes; ++p) {
        auto& pr = job_info.processes[p];
        std::cout << (p ? "," : "") << "{"
                  << "\"pid\":" << pr.pid << ","
                  << "\"process_name\":\"" << pr.process_name << "\","
                  << "\"start_time\":" << pr.start_time / 1000 << ","
                  << "\"stop_time\":" << pr.stop_time / 1000 << "}";
      }
      std::cout << "]";
    }
    return;
  }

  if (stats_ops_ == STATS_REMOVE) {
    result = rdc_job_remove(rdc_handle_, const_cast<char*>(job_id_.c_str()));
    if (result != RDC_ST_OK) {
      throw RdcException(result, rdc_status_string(result));
    }
    if (is_json_output()) {
      std::cout << "\"job_id\": \"" << job_id_ << "\", \"status\": \"ok\"";
    } else {
      std::cout << "Successfully removed job " << job_id_ << std::endl;
    }
    return;
  }

  if (stats_ops_ == STATS_REMOVE_ALL) {
    result = rdc_job_remove_all(rdc_handle_);
    if (result != RDC_ST_OK) {
      throw RdcException(result, rdc_status_string(result));
    }
    if (is_json_output()) {
      std::cout << "\"status\": \"ok\"";
    } else {
      std::cout << "Successfully removed all jobs\n";
    }
    return;
  }
}

}  // namespace rdc
}  // namespace amd
