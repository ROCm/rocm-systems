/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "RdciDiagSubSystem.h"
#include <getopt.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <assert.h>

#include <limits>
#include <iomanip>
#include <unordered_map>
#include <vector>
#include <string>
#include <queue>
#include <ctime>
#include <sstream>

#include "rdc_lib/rdc_common.h"
#include "common/rdc_utils.h"
#include "common/rdc_fields_supported.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"

namespace amd {
namespace rdc {


RdciDiagSubSystem::RdciDiagSubSystem(): diag_ops_(DIAG_RUN)
            , run_level_(RDC_DIAG_LVL_SHORT) {
}

RdciDiagSubSystem::~RdciDiagSubSystem() {
}

void RdciDiagSubSystem::parse_cmd_opts(int argc, char ** argv) {
    const int HOST_OPTIONS = 1000;
    const struct option long_options[] = {
        {"host", required_argument, nullptr, HOST_OPTIONS},
        {"help", optional_argument, nullptr, 'h'},
        {"unauth", optional_argument, nullptr, 'u'},
        {"run-level", required_argument, nullptr, 'r'},
        {"group-id", required_argument, nullptr, 'g'},
        { nullptr,  0 , nullptr, 0 }
    };

    bool group_id_set = false;
    int option_index = 0;
    int opt = 0;

    while ((opt = getopt_long(argc, argv, "hug:r:",
                long_options, &option_index)) != -1) {
        switch (opt) {
            case HOST_OPTIONS:
                ip_port_ = optarg;
                break;
            case 'h':
                diag_ops_ = DIAG_HELP;
                return;
            case 'u':
                 use_auth_ = false;
                 break;
            case 'g':
                if (!IsNumber(optarg)) {
                    show_help();
                    throw RdcException(RDC_ST_BAD_PARAMETER,
                        "The group id needs to be a number");
                }
                group_id_ = std::stoi(optarg);
                group_id_set = true;
                break;
            case 'r':
                if (!IsNumber(optarg)) {
                    show_help();
                    throw RdcException(RDC_ST_BAD_PARAMETER,
                            "The run level needs to be a number");
                }
                run_level_ = static_cast<rdc_diag_level_t>(std::stoi(optarg));
                break;
            default:
                show_help();
                throw RdcException(RDC_ST_BAD_PARAMETER,
                        "Unknown command line options");
        }
    }

    if (!group_id_set) {
        show_help();
        throw RdcException(RDC_ST_BAD_PARAMETER,
               "Need to specify the GPU group id");
    }
}

void RdciDiagSubSystem::show_help() const {
    // Try to keep total output line length to  <= 80 chars for better
    // readability. For reference:
    //            *********************** 60 Chars **************************
    //            ************** 40 Chars ***************
    //            ***** 20 Chars ****
    std::cout << " diag -- Used to run diagnostic for GPUs.\n\n";
    std::cout << "Usage\n";
    std::cout << "    rdci diag [--host <IP/FQDN>:port] [-u] -g <groupId>"
            << " -r <runLevel>\n";
    std::cout << "\nFlags:\n";
    show_common_usage();
    std::cout << "  -g  --group-id                 The GPU group to diagnose"
              << " on the specified host.\n";
    std::cout << "  -r  --run-level   level        Integer representing test"
              << " run levels [default = 1].\n"
              << "                                 level 1: Tests take a "
              << "few seconds to run.\n"
              << "                                 level 2: Tests take a "
              << "few minutes to run (To be implemented).\n"
              << "                                 level 3: Tests take "
              << "half an hour to run (To be implemented).\n";
}

std::string RdciDiagSubSystem::get_test_name
                    (rdc_diag_test_cases_t test_case) const {
        const std::map<rdc_diag_test_cases_t, std::string> test_desc = {
            {RDC_DIAG_COMPUTE_PROCESS, "No compute process"},
            {RDC_DIAG_COMPUTE_QUEUE, "Compute Queue ready"},
            {RDC_DIAG_SYS_MEM_CHECK, "System memory check"},
            {RDC_DIAG_NODE_TOPOLOGY, "Node topology check"},
            {RDC_DIAG_GPU_PARAMETERS, "GPU parameters check"},
            {RDC_DIAG_TEST_LAST, "Unknown"}
    };

    auto test_name = test_desc.find(test_case);
    if (test_name == test_desc.end()) {
        return "Unknown Test";
    }
    return test_name->second;
}

void RdciDiagSubSystem::process() {
    if (diag_ops_ == DIAG_HELP ||
            diag_ops_ == DIAG_UNKNOWN) {
        show_help();
        return;
    }

    rdc_status_t result;
    rdc_diag_response_t response;
    result = rdc_diagnostic_run(rdc_handle_, group_id_,
                run_level_, &response);

    if (result != RDC_ST_OK) {
        std::string error_msg = rdc_status_string(result);
        throw RdcException(result, error_msg.c_str());
    }

    // (3) Check diagnostic results
    for (uint32_t i=0 ; i < response.results_count; i++) {
        const rdc_diag_test_result_t& test_result =
                            response.diag_info[i];
        std::cout << std::setw(26) << std::left
                  << get_test_name(test_result.test_case) + ":"
                  << rdc_diagnostic_result_string(test_result.status) << "\n";
    }

    // (4) diagnostic detail information
    std::cout <<" =============== Diagnostic Details ==================\n";
    for (uint32_t i=0 ; i < response.results_count; i++) {
        const rdc_diag_test_result_t& test_result =
                            response.diag_info[i];
        if (test_result.info[0] != '\0') {
            std::cout << std::setw(26) << std::left
                << get_test_name(test_result.test_case) + ":"
                << test_result.info << "\n";
        }
        for (uint32_t j=0; j < test_result.per_gpu_result_count; j++) {
            const rdc_diag_per_gpu_result_t& gpu_result
                                    = test_result.gpu_results[j];
            if (strlen(gpu_result.gpu_result.msg) > 0) {
                std::cout << " GPU " << gpu_result.gpu_index << " " <<
                        gpu_result.gpu_result.msg << "\n";
            }
        }
    }
}


}  // namespace rdc
}  // namespace amd


