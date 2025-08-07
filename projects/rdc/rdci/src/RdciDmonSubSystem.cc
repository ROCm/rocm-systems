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

#include <assert.h>
#include <getopt.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/rdc_fields_supported.h"
#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

// When ctrl-C the program, the SIGINT handler will set the is_terminating
// to notify the program to clean up the resources created by the subsystem.
volatile sig_atomic_t RdciDmonSubSystem::is_terminating_ = 0;

RdciDmonSubSystem::RdciDmonSubSystem()
    : dmon_ops_(DMON_MONITOR), need_cleanup_(false), show_timpstamps_(false) {
  signal(SIGINT, set_terminating);
}

RdciDmonSubSystem::~RdciDmonSubSystem() { clean_up(); }

void RdciDmonSubSystem::set_terminating(int sig) {
  if (sig == SIGINT) {
    is_terminating_ = 1;
  }
}

std::string entity_to_string(uint32_t entity_index) {
  rdc_entity_info_t info = rdc_get_info_from_entity_index(entity_index);

  if (info.entity_role == RDC_DEVICE_ROLE_PARTITION_INSTANCE) {
    return "g" + std::to_string(info.device_index) + "." + std::to_string(info.instance_index);
  }
  return std::to_string(info.device_index);
}

void RdciDmonSubSystem::parse_cmd_opts(int argc, char** argv) {
  const int HOST_OPTIONS = 1000;
  const int LIST_ALL_FIELDS_OPT = 1001;
  const struct option long_options[] = {
      {"host", required_argument, nullptr, HOST_OPTIONS},
      {"help", optional_argument, nullptr, 'h'},
      {"unauth", optional_argument, nullptr, 'u'},
      {"list", optional_argument, nullptr, 'l'},
      {"time-stamp", optional_argument, nullptr, 't'},
      {"list-all", optional_argument, nullptr, LIST_ALL_FIELDS_OPT},
      {"field-group-id", required_argument, nullptr, 'f'},
      {"field-id", required_argument, nullptr, 'e'},
      {"gpu_index", required_argument, nullptr, 'i'},
      {"group-id", required_argument, nullptr, 'g'},
      {"count", required_argument, nullptr, 'c'},
      {"delay", required_argument, nullptr, 'd'},
      {nullptr, 0, nullptr, 0}};

  int option_index = 0;
  int opt = 0;
  std::string gpu_indexes;
  std::string field_ids;

  while ((opt = getopt_long(argc, argv, "hltuf:g:c:d:e:i:", long_options, &option_index)) != -1) {
    switch (opt) {
      case HOST_OPTIONS:
        ip_port_ = optarg;
        break;
      case 'h':
        dmon_ops_ = DMON_HELP;
        return;
      case 'u':
        use_auth_ = false;
        break;
      case 't':
        show_timpstamps_ = true;
        break;
      case 'l':
        dmon_ops_ = DMON_LIST_FIELDS;
        break;
      case 'f':
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The field group id needs to be a number");
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
          throw RdcException(RDC_ST_BAD_PARAMETER, "The group id needs to be a number");
        }
        options_.insert({OPTIONS_GROUP_ID, std::stoi(optarg)});
        break;
      case 'c':
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The count needs to be a number");
        }
        options_.insert({OPTIONS_COUNT, std::stoi(optarg)});
        break;
      case 'd':
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The delay needs to be a number");
        }
        options_.insert({OPTIONS_DELAY, std::stoi(optarg)});
        break;
      case LIST_ALL_FIELDS_OPT:
        dmon_ops_ = DMON_LIST_ALL_FIELDS;
        break;
      default:
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command line options");
    }
  }

  if (dmon_ops_ == DMON_LIST_FIELDS || dmon_ops_ == DMON_LIST_ALL_FIELDS) {
    return;
  }

  if (options_.find(OPTIONS_FIELD_GROUP_ID) == options_.end()) {
    if (field_ids == "") {
      show_help();
      throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the fields or field group id");
    } else {
      std::vector<std::string> vec_ids = split_string(field_ids, ',');
      for (uint32_t i = 0; i < vec_ids.size(); i++) {
        if (!IsNumber(vec_ids[i])) {
          rdc_field_t field_id = RDC_FI_INVALID;
          if (!amd::rdc::get_field_id_from_name(vec_ids[i], &field_id)) {
            throw RdcException(RDC_ST_BAD_PARAMETER,
                               "The field name " + vec_ids[i] + " is not valid");
          }
          field_ids_.push_back(field_id);
        } else {
          field_ids_.push_back(static_cast<rdc_field_t>(std::stoi(vec_ids[i])));
        }
      }
    }
  }

  if (options_.find(OPTIONS_GROUP_ID) == options_.end()) {
    if (gpu_indexes == "") {
      show_help();
      throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the GPUs or group id");
    }
  }

  // Group and GPU index cannot co-exist
  if (gpu_indexes != "" && options_.find(OPTIONS_GROUP_ID) != options_.end()) {
    show_help();
    throw RdcException(RDC_ST_BAD_PARAMETER, "Use either the group or GPU indexes");
  }

  // Field group and field Ids cannot co-exist
  if (field_ids != "" && options_.find(OPTIONS_FIELD_GROUP_ID) != options_.end()) {
    show_help();
    throw RdcException(RDC_ST_BAD_PARAMETER, "Use either the field group or field IDs");
  }

  // Set default delay to 1 second
  if (options_.find(OPTIONS_DELAY) == options_.end()) {
    options_.insert({OPTIONS_DELAY, 1000});
  }

  // Set default count to max integer
  if (options_.find(OPTIONS_COUNT) == options_.end()) {
    options_.insert({OPTIONS_COUNT, std::numeric_limits<uint32_t>::max()});
  }

  // Store gpu indexes to parse later
  raw_gpu_indexes_ = gpu_indexes;
}

void RdciDmonSubSystem::show_help() const {
  // Try to keep total output line length to  <= 80 chars for better
  // readability. For reference:
  //            *********************** 60 Chars **************************
  //            ************** 40 Chars ***************
  //            ***** 20 Chars ****
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
            << " many times to loop\n"
            << "                                 before exiting. [default "
            << "= runs forever.]\n";
  std::cout << "  -e  --field-id    fieldIds     Comma-separated list "
            << "of field ids to monitor.\n";
  std::cout << "  -i  --gpu_index   gpuIndexes   Comma-separated list "
            << "of GPU indexes to monitor.\n";
  std::cout << "  -d  --delay       delay        How often to query RDC "
            << "in milli seconds. \n"
            << "                                 [default = 1000 msec, "
            << "Minimum value = 100 msec.]\n";
  std::cout << "  -l  --list                     List to look up the long "
            << "names and \n"
            << "                                 descriptions of the field "
            << "ids\n";
  std::cout << "  -t  --time-stamp               Include timestamps in "
            << "display\n";
  std::cout << "  --list-all                     Same as -l, except this "
            << "lists all possible\n"
            << "                                 fields, including "
            << "those that are less \n"
            << "                                 commonly used.\n";
}

void RdciDmonSubSystem::create_temp_group() {
  if (gpu_indexes_.size() == 0) {
    return;
  }

  const std::string group_name("rdci-dmon-group");
  rdc_gpu_group_t group_id;
  rdc_status_t result =
      rdc_group_gpu_create(rdc_handle_, RDC_GROUP_EMPTY, group_name.c_str(), &group_id);
  if (result != RDC_ST_OK) {
    throw RdcException(result, "Fail to create the dmon group");
  }
  need_cleanup_ = true;

  for (uint32_t i = 0; i < gpu_indexes_.size(); i++) {
    result = rdc_group_gpu_add(rdc_handle_, group_id, gpu_indexes_[i]);
    if (result != RDC_ST_OK) {
      rdc_entity_info_t info = rdc_get_info_from_entity_index(gpu_indexes_[i]);
      std::string info_str;
      if (info.entity_role == RDC_DEVICE_ROLE_PARTITION_INSTANCE) {
        info_str =
            "g" + std::to_string(info.device_index) + "." + std::to_string(info.instance_index);
      } else {
        info_str = std::to_string(info.device_index);
      }
      throw RdcException(result, "Fail to add " + info_str + " to the dmon group.");
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
  rdc_field_t field_ids[RDC_MAX_FIELD_IDS_PER_FIELD_GROUP];
  for (uint32_t i = 0; i < field_ids_.size(); i++) {
    field_ids[i] = field_ids_[i];
  }

  rdc_status_t result = rdc_group_field_create(rdc_handle_, field_ids_.size(), &field_ids[0],
                                               field_group_name.c_str(), &group_id);
  if (result != RDC_ST_OK) {
    throw RdcException(result, "Fail to create the dmon field group.");
  }

  need_cleanup_ = true;
  options_.insert({OPTIONS_FIELD_GROUP_ID, group_id});
}

void RdciDmonSubSystem::resolve_gpu_indexes() {
  uint32_t device_list[RDC_MAX_NUM_DEVICES];
  uint32_t count = 0;
  rdc_status_t res = rdc_device_get_all(rdc_handle_, device_list, &count);
  if (res != RDC_ST_OK) {
    throw RdcException(res, "Failed to get all devices");
  }

  std::vector<std::string> vec_ids = split_string(raw_gpu_indexes_, ',');
  for (uint32_t i = 0; i < vec_ids.size(); i++) {
    if (rdc_is_partition_string(vec_ids[i].c_str())) {
      uint32_t logicalPhysicalGpu;
      uint32_t partition;
      if (!rdc_parse_partition_string(vec_ids[i].c_str(), &logicalPhysicalGpu, &partition)) {
        throw RdcException(RDC_ST_BAD_PARAMETER, "Invalid partition format: " + vec_ids[i]);
      }

      if (logicalPhysicalGpu >= count) {
        throw RdcException(RDC_ST_BAD_PARAMETER,
                           "GPU " + std::to_string(logicalPhysicalGpu) + " is out of range");
      }

      uint32_t physicalGpu = device_list[logicalPhysicalGpu];

      uint16_t num_partitions = 0;
      rdc_status_t st = rdc_get_num_partition(rdc_handle_, physicalGpu, &num_partitions);
      if (st != RDC_ST_OK) {
        throw RdcException(st,
                           "Failed to get partition info for GPU " + std::to_string(physicalGpu));
      }

      if (num_partitions == UINT16_MAX || num_partitions <= 1) {
        if (partition != 0) {
          throw RdcException(RDC_ST_BAD_PARAMETER, "GPU " + std::to_string(physicalGpu) +
                                                       " is not partitioned, so partition " +
                                                       std::to_string(partition) + " is invalid");
        }
      } else {
        if (partition >= num_partitions) {
          throw RdcException(RDC_ST_BAD_PARAMETER,
                             "GPU " + std::to_string(physicalGpu) + " supports only " +
                                 std::to_string(num_partitions) + " partitions, partition " +
                                 std::to_string(partition) + " is invalid");
        }
      }

      rdc_entity_info_t phys_info;
      phys_info.device_index = physicalGpu;
      phys_info.instance_index = partition;
      phys_info.entity_role = RDC_DEVICE_ROLE_PARTITION_INSTANCE;
      phys_info.device_type = RDC_DEVICE_TYPE_GPU;
      uint32_t phys_entity_index = rdc_get_entity_index_from_info(phys_info);
      gpu_indexes_.push_back(phys_entity_index);
    } else if (IsNumber(vec_ids[i])) {
      uint32_t logicalIndex = std::stoi(vec_ids[i]);
      if (logicalIndex >= count) {
        throw RdcException(RDC_ST_BAD_PARAMETER,
                           "GPU " + std::to_string(logicalIndex) + " is out of range");
      }
      gpu_indexes_.push_back(std::stoi(vec_ids[i]));
    } else {
      throw RdcException(RDC_ST_BAD_PARAMETER, "The GPU index " + vec_ids[i] +
                                                   " needs to be a number or a valid partition");
    }
  }
}

void RdciDmonSubSystem::show_field_usage() const {
  std::cout << "Supported fields Ids:" << std::endl;

  amd::rdc::fld_id2name_map_t& field_id_to_descript = amd::rdc::get_field_id_description_from_id();
  for (auto i = field_id_to_descript.begin(); i != field_id_to_descript.end(); i++) {
    if (i->second.do_display || dmon_ops_ == DMON_LIST_ALL_FIELDS) {
      std::cout << i->first << " " << i->second.enum_name << " : " << i->second.description << "."
                << std::endl;
    }
  }
  std::cout << std::endl;
  std::cout << "* Note: The field ID number associated with a field ID can "
               "change"
            << std::endl;
  std::cout << "  from release to release. Field name strings should be "
               "used in scripts."
            << std::endl;
}

static void separate_notf_events(const rdc_field_group_info_t* f_info,
                                 std::vector<rdc_field_t>* notif,
                                 std::vector<rdc_field_t>* reg_ev) {
  assert(f_info != nullptr && notif != nullptr && reg_ev != nullptr);

  for (uint32_t i = 0; i < f_info->count; ++i) {
    if (RDC_EVNT_IS_NOTIF_FIELD(f_info->field_ids[i])) {
      notif->push_back(f_info->field_ids[i]);
    } else {
      reg_ev->push_back(f_info->field_ids[i]);
    }
  }
}

typedef struct {
  uint32_t dev_ind;
  rdc_field_value val;
} notif_dev_value;

struct Compare_ts {
  bool operator()(const notif_dev_value& r1, const notif_dev_value& r2) {
    return r1.val.ts > r2.val.ts;
  }
};

typedef std::priority_queue<notif_dev_value, std::vector<notif_dev_value>, Compare_ts> field_pq_t;

static void collect_new_notifs(rdc_handle_t h, const rdc_group_info_t& group_info,
                               const std::vector<rdc_field_t>& notif_fields,
                               std::vector<uint64_t>* notif_ts, field_pq_t* notif_pq) {
  rdc_status_t ret;
  notif_dev_value value;
  std::string error_msg;
  uint64_t next_ts;

  assert(notif_ts != nullptr);

  for (uint32_t gindex = 0; gindex < group_info.count; gindex++) {
    for (uint32_t findex = 0; findex < notif_fields.size(); findex++) {
      // There may be multiple, repeated events; get all of them
      while (true) {
        ret = rdc_field_get_value_since(h, group_info.entity_ids[gindex], notif_fields[findex],
                                        (*notif_ts)[findex], &next_ts, &value.val);

        if (ret == RDC_ST_NOT_FOUND) {
          break;
        } else if (ret == RDC_ST_OK) {
          (*notif_ts)[findex] = next_ts;
          value.dev_ind = group_info.entity_ids[gindex];
          if (notif_pq != nullptr) {
            notif_pq->push(value);
          }
        } else {
          error_msg = "rdc_field_get_value_since() failed";
          throw RdcException(ret, error_msg.c_str());
        }
      }
    }
  }
}

// ts is milliseconds
static std::string ts_string(const time_t ts) {
  struct tm* timeinfo;
  time_t tmp_ts = ts / 1000;
  std::string ret;

  timeinfo = localtime(&tmp_ts);  // NOLINT

  ret = asctime(timeinfo);  // NOLINT
  ret.pop_back();
  return ret;
}

static void print_and_clr_notif_pq(field_pq_t* notif_pq, bool ts) {
  assert(notif_pq != nullptr);
  notif_dev_value v;
  amd::rdc::fld_id2name_map_t& field_id_to_descript = amd::rdc::get_field_id_description_from_id();
  while (!notif_pq->empty()) {
    v = notif_pq->top();
    notif_pq->pop();

    std::cout << v.dev_ind << "\t";

    if (ts) {
      std::cout << std::left << std::setw(25) << ts_string(v.val.ts);
    }

    std::cout << std::left << "   **Event: " << field_id_to_descript.at(v.val.field_id).label;
    std::cout << std::left << "\t\"" << v.val.value.str << "\"";

    std::cout << std::endl;
  }
}

void RdciDmonSubSystem::process() {
  if (dmon_ops_ == DMON_HELP || dmon_ops_ == DMON_UNKNOWN) {
    show_help();
    return;
  }

  if (dmon_ops_ == DMON_LIST_FIELDS || dmon_ops_ == DMON_LIST_ALL_FIELDS) {
    show_field_usage();
    return;
  }

  rdc_status_t result;
  rdc_group_info_t group_info;
  rdc_field_group_info_t field_info;

  resolve_gpu_indexes();

  // Create a temporary group/field if pass as GPU indexes or field ids
  create_temp_group();
  create_temp_field_group();

  result = rdc_group_gpu_get_info(rdc_handle_, options_[OPTIONS_GROUP_ID], &group_info);
  if (result != RDC_ST_OK) {
    std::string error_msg = rdc_status_string(result);
    if (result == RDC_ST_NOT_FOUND) {
      error_msg = "Cannot find the group " + std::to_string(options_[OPTIONS_GROUP_ID]);
    }
    throw RdcException(result, error_msg.c_str());
  }
  if (group_info.count == 0) {
    throw RdcException(RDC_ST_NOT_FOUND, "The gpu group " +
                                             std::to_string(options_[OPTIONS_GROUP_ID]) +
                                             " must contain at least 1 GPU.");
  }
  result = rdc_group_field_get_info(rdc_handle_, options_[OPTIONS_FIELD_GROUP_ID], &field_info);
  if (result != RDC_ST_OK) {
    std::string error_msg = rdc_status_string(result);
    if (result == RDC_ST_NOT_FOUND) {
      error_msg = "Cannot find the field group " + std::to_string(options_[OPTIONS_FIELD_GROUP_ID]);
    }
    throw RdcException(result, error_msg.c_str());
  }
  if (field_info.count == 0) {
    throw RdcException(RDC_ST_NOT_FOUND, "The field group " +
                                             std::to_string(options_[OPTIONS_FIELD_GROUP_ID]) +
                                             " must contain at least 1 field.");
  }
  // Divide field_info fields into 2 vectors, 1 for notifications
  // and one for non-notifications. Handle these separately below.
  std::vector<rdc_field_t> notif_fields;
  std::vector<rdc_field_t> reg_fields;
  separate_notf_events(&field_info, &notif_fields, &reg_fields);

  // keep extra 1 minute data
  double max_keep_age = options_[OPTIONS_DELAY] / 1000.0 + 60;
  const int max_keep_samples = 10;  // keep only 10 samples
  result =
      rdc_field_watch(rdc_handle_, options_[OPTIONS_GROUP_ID], options_[OPTIONS_FIELD_GROUP_ID],
                      options_[OPTIONS_DELAY] * 1000, max_keep_age, max_keep_samples);
  need_cleanup_ = true;

  std::stringstream ss;
  amd::rdc::fld_id2name_map_t& field_id_to_descript = amd::rdc::get_field_id_description_from_id();

  if (notif_fields.size() > 0) {
    ss << "Listening for events: ";
    uint32_t i = 0;
    for (i = 0; i < notif_fields.size() - 1; ++i) {
      ss << field_id_to_descript.at(notif_fields[i]).label << ", ";
    }
    ss << field_id_to_descript.at(notif_fields[i]).label << std::endl;
  }
  ss << std::left << std::setw(10) << "GPU";
  if (show_timpstamps_) {
    ss << std::left << std::setw(25) << "TIMESTAMP";
    ss << "  ";
  }
  for (uint32_t findex = 0; findex < reg_fields.size(); findex++) {
    ss << std::left << std::setw(20) << field_id_string(reg_fields[findex]);
  }
  ss << std::endl;

  std::string header_line((std::istreambuf_iterator<char>(ss)), (std::istreambuf_iterator<char>()));

  std::vector<uint64_t> notif_ts(notif_fields.size());
  field_pq_t notif_pq;

  // Call this once without printing out notfications to initialize
  // timestamps. There may be very stale timestamps in cache.
  collect_new_notifs(rdc_handle_, group_info, notif_fields, &notif_ts, nullptr);

  for (uint32_t i = 0; i < options_[OPTIONS_COUNT]; i++) {
    if (i % 50 == 0) {
      std::cout << header_line;
    }

    usleep(options_[OPTIONS_DELAY] * 1000);

    collect_new_notifs(rdc_handle_, group_info, notif_fields, &notif_ts, &notif_pq);

    print_and_clr_notif_pq(&notif_pq, show_timpstamps_);

    for (uint32_t gindex = 0; gindex < group_info.count; gindex++) {
      std::cout << std::left << std::setw(10) << entity_to_string(group_info.entity_ids[gindex]);
      for (uint32_t findex = 0; findex < reg_fields.size(); findex++) {
        rdc_field_value value;

        result = rdc_field_get_latest_value(rdc_handle_, group_info.entity_ids[gindex],
                                            reg_fields[findex], &value);
        if (result != RDC_ST_OK) {
          std::cout << std::left << std::setw(20) << "N/A";
        } else {
          if (show_timpstamps_ && findex == 0) {
            std::cout << std::left << std::setw(25) << ts_string(value.ts) << "  ";
          }

          if (value.type == INTEGER) {
            std::cout << std::left << std::setw(20) << value.value.l_int;
          } else if (value.type == DOUBLE) {
            std::cout << std::left << std::setw(20) << std::fixed << std::setprecision(3)
                      << value.value.dbl;
          } else {
            std::cout << std::left << std::setw(20) << value.value.str;
          }
        }

        if (is_terminating_) {
          clean_up();
          return;
        }
      }
      if (reg_fields.size()) {
        std::cout << std::endl;
      }
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
    rdc_field_unwatch(rdc_handle_, options_[OPTIONS_GROUP_ID], options_[OPTIONS_FIELD_GROUP_ID]);
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
