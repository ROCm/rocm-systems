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
#include "rdc_lib/impl/RdcDiagnosticModule.h"
#include <map>
#include <vector>
#include <functional>
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/RdcSmiLib.h"
#include "rdc_lib/impl/RdcRasLib.h"

namespace amd {
namespace rdc {

rdc_status_t RdcDiagnosticModule::rdc_diag_test_cases_query(
        rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
        uint32_t* test_case_count) {
    if (test_case_count == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    auto ite = diagnostic_modules_.begin();
    *test_case_count = 0;
    for (; ite != diagnostic_modules_.end(); ite++) {
        uint32_t count = 0;
        rdc_status_t status = (*ite)->rdc_diag_test_cases_query(
                &(test_cases[*test_case_count]), &count);
        if (status == RDC_ST_OK) {
            *test_case_count += count;
        }
    }
    return RDC_ST_OK;
}

rdc_status_t RdcDiagnosticModule::rdc_test_case_run(
        rdc_diag_test_cases_t test_case,
        uint32_t gpu_index[RDC_MAX_NUM_DEVICES],
        uint32_t gpu_count,
        rdc_diag_test_result_t* result) {
    if (result == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    // Init test status
    auto ite = testcases_to_module_.find(test_case);
    if (ite == testcases_to_module_.end()) {
        result->status = RDC_DIAG_RESULT_SKIP;
        strncpy_with_null(result->info, "Not implemented", MAX_DIAG_MSG_LENGTH);
        return RDC_ST_NOT_SUPPORTED;
    }
    return ite->second->rdc_test_case_run(test_case,
            gpu_index, gpu_count, result);
}

rdc_status_t RdcDiagnosticModule::rdc_diagnostic_run(
        const rdc_group_info_t& gpus,
        rdc_diag_level_t level,
        rdc_diag_response_t* response) {
    if (response == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    std::vector<rdc_diag_test_cases_t> rdc_runs;
    if (level >= RDC_DIAG_LVL_SHORT) {  // Short run and above
        rdc_runs.push_back(RDC_DIAG_COMPUTE_PROCESS);
        rdc_runs.push_back(RDC_DIAG_NODE_TOPOLOGY);
        rdc_runs.push_back(RDC_DIAG_GPU_PARAMETERS);
        rdc_runs.push_back(RDC_DIAG_COMPUTE_QUEUE);
        rdc_runs.push_back(RDC_DIAG_SYS_MEM_CHECK);
    }

    response->results_count = 0;
    for (unsigned int i=0; i < rdc_runs.size(); i++) {
        response->diag_info[i].test_case = rdc_runs[i];
        rdc_test_case_run(rdc_runs[i],
            const_cast<uint32_t*>(gpus.entity_ids),
            gpus.count, &(response->diag_info[i]));
        response->results_count++;
    }

    return RDC_ST_OK;
}

rdc_status_t RdcDiagnosticModule::rdc_diag_init(uint64_t flag) {
    auto ite = diagnostic_modules_.begin();
    for (; ite != diagnostic_modules_.end(); ite++) {
        (*ite)->rdc_diag_init(flag);
    }
    return RDC_ST_OK;
}

rdc_status_t RdcDiagnosticModule::RdcDiagnosticModule::rdc_diag_destroy() {
    auto ite = diagnostic_modules_.begin();
    for (; ite != diagnostic_modules_.end(); ite++) {
        (*ite)->rdc_diag_destroy();
    }
    return RDC_ST_OK;
}

RdcDiagnosticModule::RdcDiagnosticModule(const RdcSmiLibPtr& smi_lib,
    const RdcRasLibPtr& ras_module, const RdcRocrLibPtr& rocr_module) {
    if (smi_lib) {
       diagnostic_modules_.push_back(smi_lib);
    }
    if (rocr_module) {
       diagnostic_modules_.push_back(rocr_module);
    }
    if (ras_module) {
       diagnostic_modules_.push_back(ras_module);
    }

    auto ite = diagnostic_modules_.begin();
    for (; ite != diagnostic_modules_.end(); ite++) {
       rdc_diag_test_cases_t test_cases[MAX_TEST_CASES];
       uint32_t test_count = 0;
       rdc_status_t status = (*ite)->
            rdc_diag_test_cases_query(test_cases, &test_count);
       if (status == RDC_ST_OK) {
            for (uint32_t index = 0; index < test_count; index++) {
                testcases_to_module_.insert({test_cases[index], (*ite)});
            }
       }
    }
}


}  // namespace rdc
}  // namespace amd

