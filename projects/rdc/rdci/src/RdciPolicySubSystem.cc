/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "RdciPolicySubSystem.h"

#include <getopt.h>
#include <signal.h>
#include <unistd.h>

#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {
RdciPolicySubSystem::RdciPolicySubSystem() : policy_ops_(POLICY_UNKNOWN), is_group_id_set(false) {}

void RdciPolicySubSystem::parse_cmd_opts(int argc, char** argv) {
  const int HOST_OPTIONS = 1000;
  const int POLICY_OPTION_GET = 1001;
  const int POLICY_OPTION_CLEAR = 1002;
  const int POLICY_OPTION_REG = 1003;
  const struct option long_options[] = {{"host", required_argument, nullptr, HOST_OPTIONS},
                                        {"help", optional_argument, nullptr, 'h'},
                                        {"unauth", optional_argument, nullptr, 'u'},
                                        {"max", required_argument, nullptr, 'M'},
                                        {"temp", required_argument, nullptr, 'T'},
                                        {"power", required_argument, nullptr, 'P'},
                                        {"action", required_argument, nullptr, 'A'},
                                        {"group", required_argument, nullptr, 'g'},
                                        {"get", optional_argument, nullptr, POLICY_OPTION_GET},
                                        {"clear", optional_argument, nullptr, POLICY_OPTION_CLEAR},
                                        {"reg", optional_argument, nullptr, POLICY_OPTION_REG},
                                        {nullptr, 0, nullptr, 0}};
  int option_index = 0;
  int opt = 0;

  while ((opt = getopt_long(argc, argv, "hucr:M:T:P:A:g:", long_options, &option_index)) != -1) {
    switch (opt) {
      case HOST_OPTIONS:
        ip_port_ = optarg;
        break;
      case 'h':
        policy_ops_ = POLICY_HELP;
        break;
      case 'u':
        use_auth_ = false;
        break;
      case 'M':
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The field max page needs to be a number");
        }

        options_.insert({POLICY_OPT_MAX_PAGE, std::stoi(optarg)});
        policy_ops_ = POLICY_SET;
        break;
      case 'T':
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The field temp needs to be a number");
        }

        options_.insert({POLICY_OPT_TEMP, std::stoi(optarg)});
        policy_ops_ = POLICY_SET;
        break;
      case 'P':
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The field power needs to be a number");
        }

        options_.insert({POLICY_OPT_POWER, std::stoi(optarg)});
        policy_ops_ = POLICY_SET;
        break;
      case 'A':
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The field action needs to be a number");
        }

        options_.insert({POLICY_OPT_ACTION, std::stoi(optarg)});
        policy_ops_ = POLICY_SET;
        break;
      case 'g':
        if (!IsNumber(optarg)) {
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "The field group id needs to be a number");
        }
        group_id_ = std::stoi(optarg);
        is_group_id_set = true;
        break;
      case POLICY_OPTION_GET:
        policy_ops_ = POLICY_GET;
        break;
      case POLICY_OPTION_CLEAR:
        policy_ops_ = POLICY_CLEAR;
        break;
      case POLICY_OPTION_REG:
        policy_ops_ = POLICY_REGISTER;
        break;
      default:
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command line options");
    }
  }
  if (policy_ops_ == POLICY_HELP) {
    show_help();
    return;
  }
  if (is_group_id_set == false) {
    show_help();
    throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the group id to notify");
  }

  // Set default action to 0
  if (policy_ops_ == POLICY_SET && options_.find(POLICY_OPT_ACTION) == options_.end()) {
    options_.insert({POLICY_OPT_ACTION, RDC_POLICY_ACTION_NONE});
  }
}

void RdciPolicySubSystem::show_help() const {
  if (is_json_output()) return;
  std::cout << " policy -- Used to  policy  notify the GPU "
            << "whether an event is occurring.\n\n";
  std::cout << "Usage\n";
  std::cout << "    rdci policy [--host <IP/FQDN>:port] [--reg] -g <groupId>\n";
  std::cout << "\nFlags:\n";
  show_common_usage();
  std::cout << "  --get                          can also list the "
            << "policy defined on GPU group id \n";
  std::cout << "  --clear                        can also clear "
            << "the policy defined on a GPU group.\n";
  std::cout << "  --reg                          can register "
            << "the rdci to listen for those events.\n";
  std::cout << "  -h  --help                     Displays usage "
            << "information and exits.\n";
  std::cout << "  -M  --max                      Specify the maximum number "
            << "of retired pages for the Page Retirement event.\n";
  std::cout << "  -T  --temp                     Specify the maximum "
            << "temperature for the Temperature Retirement event.\n";
  std::cout << "  -P  --power                    Specify the maximum "
            << "power for the Power Limit event.\n";
  std::cout << "  -A  --action                   Action to take when "
            << "event happens: 0 is print out and 1 is for GPU reset.\n";
  std::cout << "  -g  --group                    GPU group ID type \n";
}

static const char* condition_type_to_str(rdc_policy_condition_type_t type) {
  if (type == RDC_POLICY_COND_MAX_PAGE_RETRIED) return "Retried Page Limit";
  if (type == RDC_POLICY_COND_THERMAL) return "Temperature Limit";
  if (type == RDC_POLICY_COND_POWER) return "Power Limit";
  return "Unknown_Type";
}

static time_t last_time = 0;  // last time to print message
int rdc_policy_callback(rdc_policy_callback_response_t* userData) {
  if (userData == nullptr) {
    throw RdcException(RDC_ST_BAD_PARAMETER, "The rdc_policy_callback returns null data");
    return 1;
  }

  // To avoid flooding too many messages, only print message every 5 seconds
  time_t now = time(NULL);
  if (difftime(now, last_time) < 5) {
    return 0;
  }

  int64_t value = userData->value;
  int64_t threshold = userData->condition.value;
  if (userData->condition.type == RDC_POLICY_COND_THERMAL) {
    value /= 1000;
    threshold /= 1000;
  } else if (userData->condition.type == RDC_POLICY_COND_POWER) {
    value /= 1000000;
    threshold /= 1000000;
  }

  std::cout << "A " << condition_type_to_str(userData->condition.type) << " exceeds the threshold "
            << threshold << " with the value " << value << std::endl;
  last_time = now;  // update the last time
  return 0;
}

volatile sig_atomic_t RdciPolicySubSystem::keep_running_ = 1;

void RdciPolicySubSystem::set_terminating(int sig) {
  if (sig == SIGINT) {
    keep_running_ = 0;
  }
}

void RdciPolicySubSystem::process() {
  rdc_status_t result = RDC_ST_OK;
  rdc_policy_t policy;
  rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS];

  switch (policy_ops_) {
    case POLICY_SET: {
      if (!is_group_id_set) {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the group id to set a group");
      }
      for (const auto& option : options_) {
        switch (option.first) {
          case POLICY_OPT_MAX_PAGE:
            policy.condition = {RDC_POLICY_COND_MAX_PAGE_RETRIED, option.second};
            break;
          case POLICY_OPT_TEMP:
            policy.condition = {RDC_POLICY_COND_THERMAL, option.second * 1000};
            break;
          case POLICY_OPT_POWER:
            policy.condition = {RDC_POLICY_COND_POWER, option.second * 1000000};
            break;
          case POLICY_OPT_ACTION:
            if (option.second == 0) {
              policy.action = RDC_POLICY_ACTION_NONE;
            } else if (option.second == 1) {
              policy.action = RDC_POLICY_ACTION_GPU_RESET;
            } else {
              throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command");
            }
            break;
          default:
            throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command");
        }
      }
      result = rdc_policy_set(rdc_handle_, group_id_, policy);
      if (result == RDC_ST_NOT_SUPPORTED) {
        std::cout << "GPU NO SUPPORT SET " << condition_type_to_str(policy.condition.type) << "\n";
      }
      break;
    }

    case POLICY_GET: {
      if (!is_group_id_set) {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER, "Need to specify the group id to get a group");
      }
      uint32_t count = 0;
      result = rdc_policy_get(rdc_handle_, group_id_, &count, policies);
      if (result == RDC_ST_OK) {
        std::cout << "Policy information \n";
        std::cout << "+------------------------+"
                  << "-----------------------------"
                  << "-----------------+\n";
        std::cout << "| Policy Name     \t "
                  << "| Threshold\t\t"
                  << "| Action\t\t|\n";
        std::cout << "+========================+"
                  << "============================="
                  << "=================+\n";

        for (unsigned int i = 0; i < count; i++) {
          if (policies[i].condition.type == RDC_POLICY_COND_MAX_PAGE_RETRIED) {
            std::cout << "| Page Retirement\t "
                      << "|   " << policies[i].condition.value;
            if (policies[i].condition.value < 100) {
              std::cout << "\t";
            }
          } else if (policies[i].condition.type == RDC_POLICY_COND_THERMAL) {
            std::cout << "| Temperature Limit\t "
                      << "|   " << policies[i].condition.value / 1000;
            if (policies[i].condition.value / 1000 < 100) {
              std::cout << "\t";
            }
          } else if (policies[i].condition.type == RDC_POLICY_COND_POWER) {
            std::cout << "| Power Limit     \t "
                      << "|   " << policies[i].condition.value / 1000000;
            if (policies[i].condition.value / 1000000 < 100) {
              std::cout << "\t";
            }
          }
          if (policies[i].action == 0) {
            std::cout << "\t\t| Notify\t\t|\n";
          } else if (policies[i].action == 1) {
            std::cout << "\t\t| GPU Reset\t\t|\n";
          } else {
            throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command");
          }
        }
        std::cout << "+========================+"
                  << "============================="
                  << "=================+\n";
      }
      break;
    }
    case POLICY_REGISTER: {
      result = rdc_policy_register(rdc_handle_, group_id_, rdc_policy_callback);
      if (result == RDC_ST_OK) {
        std::cout << "Waiting for policy events ... ...\n" << std::endl;
        keep_running_ = 1;
        signal(SIGINT, set_terminating);
        while (keep_running_) {
          sleep(1);
        }
        result = rdc_policy_unregister(rdc_handle_, group_id_);
        if (result != RDC_ST_OK) {
          throw RdcException(RDC_ST_BAD_PARAMETER, "Error unregister policy");
        }
      } else {
        throw RdcException(RDC_ST_BAD_PARAMETER, "Error register policy");
      }
      break;
    }
    case POLICY_CLEAR: {
      for (uint32_t cond = RDC_POLICY_COND_FIRST; cond <= RDC_POLICY_COND_LAST; cond++) {
        rdc_policy_condition_type_t condition_type = static_cast<rdc_policy_condition_type_t>(cond);
        result = rdc_policy_delete(rdc_handle_, group_id_, condition_type);
      }
      std::cout << "Policy is cleared on GPU group " << group_id_ << std::endl;
      break;
    }
    default:
      throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command");
      break;
  }
}
}  // namespace rdc
}  // namespace amd
