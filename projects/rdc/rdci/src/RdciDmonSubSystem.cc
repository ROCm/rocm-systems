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
#include "RdciDmonSubSystem.h"
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

// When ctrl-C the program, the SIGINT handler will set the is_terminating
// to notify the program to clean up the resources created by the subsystem.
volatile sig_atomic_t RdciDmonSubSystem::is_terminating_ = 0;

RdciDmonSubSystem::RdciDmonSubSystem():
    dmon_ops_(DMON_MONITOR)
    , need_cleanup_(false) {
        signal(SIGINT, set_terminating);
}

RdciDmonSubSystem::~RdciDmonSubSystem() {
    clean_up();
}

void RdciDmonSubSystem::set_terminating(int sig) {
  if (sig == SIGINT) {
       is_terminating_ = 1;
  }
}

void RdciDmonSubSystem::parse_cmd_opts(int argc, char ** argv) {
    const int HOST_OPTIONS = 1000;
    const struct option long_options[] = {
        {"host", required_argument, nullptr, HOST_OPTIONS },
        {"help", optional_argument, nullptr, 'h' },
        {"list", optional_argument, nullptr, 'l' },
        {"field-group-id", required_argument, nullptr, 'f' },
        {"field-id", required_argument, nullptr, 'e' },
        {"gpu_index", required_argument, nullptr, 'i'},
        {"group-id", required_argument, nullptr, 'g' },
        {"count", required_argument, nullptr, 'c'},
        {"delay", required_argument, nullptr, 'd'},
        { nullptr,  0 , nullptr, 0 }
    };

    int option_index = 0;
    int opt = 0;
    std::string gpu_indexes;
    std::string field_ids;

    while ((opt = getopt_long(argc, argv, "hlf:g:c:d:e:i:",
                long_options, &option_index)) != -1) {
        switch (opt) {
            case HOST_OPTIONS:
                ip_port_ = optarg;
                break;
            case 'h':
                dmon_ops_ = DMON_HELP;
                return;
            case 'l':
                dmon_ops_ = DMON_LIST_FIELDS;
                return;
            case 'f':
                if (!IsNumber(optarg)) {
                    show_help();
                    throw RdcException(RDC_ST_BAD_PARAMETER,
                        "The field group id needs to be a number");
                }
                options_.insert({OPTIONS_FIELD_GROUP_ID, std::stoi(optarg)});
                break;
            case 'e':
                field_ids = optarg;
                break;
            case 'i':
                gpu_indexes = optarg;
                break;
            case 'g':
                if (!IsNumber(optarg)) {
                    show_help();
                    throw RdcException(RDC_ST_BAD_PARAMETER,
                        "The group id needs to be a number");
                }
                options_.insert({OPTIONS_GROUP_ID, std::stoi(optarg)});
                break;
            case 'c':
                if (!IsNumber(optarg)) {
                    show_help();
                    throw RdcException(RDC_ST_BAD_PARAMETER,
                            "The count needs to be a number");
                }
                options_.insert({OPTIONS_COUNT, std::stoi(optarg)});
                break;
            case 'd':
                if (!IsNumber(optarg)) {
                    show_help();
                    throw RdcException(RDC_ST_BAD_PARAMETER,
                            "The delay needs to be a number");
                }
                options_.insert({OPTIONS_DELAY, std::stoi(optarg)});
                break;
            default:
                show_help();
                throw RdcException(RDC_ST_BAD_PARAMETER,
                        "Unknown command line options");
        }
    }

    if (options_.find(OPTIONS_FIELD_GROUP_ID) == options_.end()) {
        if (field_ids == "") {
            show_help();
            throw RdcException(RDC_ST_BAD_PARAMETER,
                    "Need to specify the fields or field group id");
        } else {
            std::vector<std::string> vec_ids = split_string(field_ids, ',');
            for (uint32_t i = 0; i < vec_ids.size(); i++) {
                if (!IsNumber(vec_ids[i])) {
                    throw RdcException(RDC_ST_BAD_PARAMETER, "The field Id "
                            +vec_ids[i]+" needs to be a number");
                }
                field_ids_.push_back(std::stoi(vec_ids[i]));
            }
        }
    }

    if (options_.find(OPTIONS_GROUP_ID) == options_.end()) {
        if (gpu_indexes == "") {
            show_help();
            throw RdcException(RDC_ST_BAD_PARAMETER,
                    "Need to specify the GPUs or group id");
        } else {
            std::vector<std::string> vec_ids = split_string(gpu_indexes, ',');
            for (uint32_t i = 0; i < vec_ids.size(); i++) {
                if (!IsNumber(vec_ids[i])) {
                    throw RdcException(RDC_ST_BAD_PARAMETER,
                        "The GPU index "+vec_ids[i]+" needs to be a number");
                }
                gpu_indexes_.push_back(std::stoi(vec_ids[i]));
            }
        }
    }

    // Group and GPU index cannot co-exist
    if (gpu_indexes != "" &&
            options_.find(OPTIONS_GROUP_ID) != options_.end()) {
                show_help();
                throw RdcException(RDC_ST_BAD_PARAMETER,
                    "Use either the group or GPU indexes");
    }

    // Field group and field Ids cannot co-exist
    if (field_ids != "" &&
            options_.find(OPTIONS_FIELD_GROUP_ID) != options_.end()) {
                show_help();
                throw RdcException(RDC_ST_BAD_PARAMETER,
                    "Use either the field group or field IDs");
    }

    // Set default delay to 1 second
    if (options_.find(OPTIONS_DELAY) == options_.end()) {
        options_.insert({OPTIONS_DELAY, 1000});
    }

    // Set default count to max integer
    if (options_.find(OPTIONS_COUNT) == options_.end()) {
        options_.insert({OPTIONS_COUNT, std::numeric_limits<uint32_t>::max()});
    }
}

void RdciDmonSubSystem::show_help() const {
    std::cout << " dmon -- Used to monitor GPUs and their stats.\n\n";
    std::cout << "Usage\n";
    std::cout << "    rdci dmon [--host <IP/FQDN>:port] [-u] -f <fieldGroupId>"
            << " -g <groupId>\n";
    std::cout << "         [-d <delay>] [-c <count>]\n";
    std::cout << "    rdci dmon [--host <IP/FQDN>:port] [-u] -e <fieldIds>"
              << " -i <gpuIndexes>\n";
    std::cout << "         [-d <delay>] [-c <count>]\n";
    std::cout << "    rdci dmon [--host <IP/FQDN>:port] [-u] -l \n";
    std::cout << "\nFlags:\n";
    show_common_usage();
    std::cout << "  -f  --field-group-id           The field group "
              << "to query on the specified host.\n";
    std::cout << "  -g  --group-id                 The GPU group to query "
              << "on the specified host.\n";
    std::cout << "  -c  --count       count        Integer representing How"
        << " many times to loop before exiting. [default = runs forever.]\n";
    std::cout << "  -e  --field-id    fieldIds     Comma-separated list "
              << "of the field ids to monitor.\n";
    std::cout << "  -i  --gpu_index   gpuIndexes   Comma-separated list "
              << "of the GPU index to monitor.\n";
    std::cout << "  -d  --delay       delay        How often to query RDC "
              << "in milli seconds. [default = 1000 msec, "
              << "Minimum value = 100 msec.]\n";
    std::cout << "  -l  --list                     List to look up the long "
              << "names and descriptions of the field ids\n";
}

void RdciDmonSubSystem::create_temp_group() {
    if (gpu_indexes_.size() == 0) {
        return;
    }

    const std::string group_name("rdci-dmon-group");
    rdc_gpu_group_t group_id;
    rdc_status_t result = rdc_group_gpu_create(rdc_handle_,
                RDC_GROUP_EMPTY, group_name.c_str(), &group_id);
    if (result != RDC_ST_OK) {
        throw RdcException(result, "Fail to create the dmon group");
    }
    need_cleanup_ = true;

    for (uint32_t i = 0; i < gpu_indexes_.size() ; i++) {
        result = rdc_group_gpu_add(rdc_handle_, group_id, gpu_indexes_[i]);
        if (result != RDC_ST_OK) {
            throw RdcException(result, "Fail to add " +
                std::to_string(gpu_indexes_[i])+" to the dmon group.");
        }
    }
    options_.insert({OPTIONS_GROUP_ID, group_id});
}


void RdciDmonSubSystem::create_temp_field_group() {
    if (field_ids_.size() == 0) {
        return;
    }

    const std::string field_group_name("rdci-dmon-field-group");
    rdc_field_grp_t group_id;
    uint32_t field_ids[RDC_MAX_FIELD_IDS_PER_FIELD_GROUP];
    for (uint32_t i = 0; i < field_ids_.size(); i++) {
        field_ids[i] = field_ids_[i];
    }

    rdc_status_t result = rdc_group_field_create(rdc_handle_,
         field_ids_.size(), &field_ids[0], field_group_name.c_str(), &group_id);
    if (result != RDC_ST_OK) {
        throw RdcException(result, "Fail to create the dmon field group.");
    }

    need_cleanup_ = true;
    options_.insert({OPTIONS_FIELD_GROUP_ID, group_id});
}

void RdciDmonSubSystem::show_field_usage() const {
    std::cout << "Supported fields Ids:\n";
    std::cout << "100 RDC_FI_GPU_SM_CLOCK: Current GPU clock frequencies.\n";
    std::cout << "150 RDC_FI_GPU_TEMP: GPU "
              << "temperature in millidegrees Celcius.\n";
    std::cout << "155 RDC_FI_POWER_USAGE: Power usage in microwatts.\n";
    std::cout << "203 RDC_FI_GPU_UTIL: GPU busy percentage.\n";
    std::cout << "525 RDC_FI_GPU_MEMORY_USAGE: Memory usage of the GPU "
              << "instance in bytes.\n";
}

void RdciDmonSubSystem::process() {
    if (dmon_ops_ == DMON_HELP ||
            dmon_ops_ == DMON_UNKNOWN) {
        show_help();
        return;
    }

    if (dmon_ops_ == DMON_LIST_FIELDS) {
        show_field_usage();
        return;
    }

    rdc_status_t result;
    rdc_group_info_t group_info;
    rdc_field_group_info_t field_info;

    // Create a temporary group/field if pass as GPU indexes or field ids
    create_temp_group();
    create_temp_field_group();

    result = rdc_group_gpu_get_info(rdc_handle_,
                options_[OPTIONS_GROUP_ID], &group_info);
    if (result != RDC_ST_OK) {
        throw RdcException(result, rdc_status_string(result));
    }
    result = rdc_group_field_get_info(rdc_handle_,
        options_[OPTIONS_FIELD_GROUP_ID], &field_info);
    if (result != RDC_ST_OK) {
        throw RdcException(result, rdc_status_string(result));
    }

    // keep extra 1 minute data
    double max_keep_age = options_[OPTIONS_DELAY]/1000.0 + 60;
    const int max_keep_samples = 10;  // keep only 10 samples
    result = rdc_field_watch(rdc_handle_,
        options_[OPTIONS_GROUP_ID], options_[OPTIONS_FIELD_GROUP_ID],
        options_[OPTIONS_DELAY]*1000, max_keep_age, max_keep_samples);
    need_cleanup_ = true;
    std::cout << "GPU\t";
    for (uint32_t findex = 0; findex < field_info.count; findex++) {
       std::cout << std::left << std::setw(20)
            << field_id_string(field_info.field_ids[findex]);
    }
    std::cout << std::endl;

    for (uint32_t i = 0; i < options_[OPTIONS_COUNT]; i++) {
        usleep(options_[OPTIONS_DELAY]*1000);
        for (uint32_t gindex = 0; gindex < group_info.count; gindex++) {
            std::cout << group_info.entity_ids[gindex] << "\t";
            for (uint32_t findex = 0; findex < field_info.count; findex++) {
                 rdc_field_value value;
                 result = rdc_field_get_latest_value(rdc_handle_,
                    group_info.entity_ids[gindex],
                    field_info.field_ids[findex], &value);
                 if (result != RDC_ST_OK) {
                     std::cout << std::left << std::setw(20) << "error";
                 } else {
                     if (value.type == INTEGER) {
                        std::cout << std::left << std::setw(20)
                            << value.value.l_int;
                     } else if (value.type == DOUBLE) {
                         std::cout << std::left << std::setw(20)
                            << value.value.dbl;
                     } else {
                         std::cout << std::left << std::setw(20)
                            << value.value.str;
                     }
                 }

                 if (is_terminating_) {
                     clean_up();
                     return;
                 }
            }
            std::cout << std::endl;
        }
    }

    clean_up();
}


void RdciDmonSubSystem::clean_up() {
    if (!need_cleanup_) {
        return;
    }

    // Not throw the errors in order to clean up all resources created
    if (options_.find(OPTIONS_GROUP_ID) != options_.end() &&
            options_.find(OPTIONS_FIELD_GROUP_ID) != options_.end()) {
        rdc_field_unwatch(rdc_handle_, options_[OPTIONS_GROUP_ID],
                options_[OPTIONS_FIELD_GROUP_ID]);
    }

    if (gpu_indexes_.size() != 0) {
        auto group = options_.find(OPTIONS_GROUP_ID);
        if (group != options_.end()) {
            rdc_group_gpu_destroy(rdc_handle_, group->second);
        }
    }

    if (field_ids_.size() != 0) {
        auto fgroup = options_.find(OPTIONS_FIELD_GROUP_ID);
        if (fgroup != options_.end()) {
            rdc_group_field_destroy(rdc_handle_, fgroup->second);
        }
    }

    need_cleanup_ = false;
}

}  // namespace rdc
}  // namespace amd


