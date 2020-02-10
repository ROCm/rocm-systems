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
#include <dlfcn.h>
#include "rdc/rdc.h"
#include "rdc_lib/RdcHandler.h"
#include "rdc_lib/rdc_common.h"

static void* libHandler = nullptr;

rdc_status_t rdc_init(uint64_t) {
       return RDC_ST_OK;
}

rdc_status_t rdc_shutdown() {
        if (libHandler) {
            dlclose(libHandler);
            libHandler = nullptr;
        }
        return RDC_ST_OK;
}

rdc_status_t rdc_connect(const char* ipAddress,
                rdc_handle_t* p_rdc_handle ) {
        amd::rdc::RdcHandler* (*func_make_handler)(const char*);

        if (!ipAddress || !p_rdc_handle) {
                return RDC_ST_FAIL_LOAD_MODULE;
        }

        if (!libHandler) {
             libHandler = dlopen("librdc_client.so", RTLD_LAZY);
        }

        if (!libHandler) {
                return RDC_ST_FAIL_LOAD_MODULE;
        }

        *reinterpret_cast<void**>(&func_make_handler) =
                dlsym(libHandler, "make_handler");
        if (!func_make_handler) {
                return RDC_ST_FAIL_LOAD_MODULE;
        }

        *p_rdc_handle = static_cast<rdc_handle_t>
                (func_make_handler(ipAddress));
        return RDC_ST_OK;
}

rdc_status_t rdc_disconnect(rdc_handle_t p_rdc_handle) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }
        delete static_cast<amd::rdc::RdcHandler*>(p_rdc_handle);
        p_rdc_handle = nullptr;
        return RDC_ST_OK;
}

rdc_status_t rdc_start_embedded(rdc_operation_mode_t op_mode,
                rdc_handle_t* p_rdc_handle ) {
        amd::rdc::RdcHandler* (*func_make_handler)(rdc_operation_mode_t);
        char *error;
        if (!p_rdc_handle) {
                return RDC_ST_FAIL_LOAD_MODULE;
        }

        dlerror();

        if (!libHandler) {
             libHandler = dlopen("librdc.so", RTLD_LAZY);
        }

        if (!libHandler) {
                error = dlerror();
                LOG_DEBUG("Fail to open librdc.so: " << error);
                return RDC_ST_FAIL_LOAD_MODULE;
        }

        *reinterpret_cast<void**>(&func_make_handler) =
                dlsym(libHandler, "make_handler");
        if (!func_make_handler) {
                error = dlerror();
                LOG_DEBUG("Fail to find function make_handler:" << error);
                return RDC_ST_FAIL_LOAD_MODULE;
        }

        *p_rdc_handle = static_cast<rdc_handle_t>
                        (func_make_handler(op_mode));

        return RDC_ST_OK;
}

rdc_status_t rdc_stop_embedded(rdc_handle_t p_rdc_handle) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }
        delete static_cast<amd::rdc::RdcHandler*>(p_rdc_handle);
        p_rdc_handle = nullptr;
        return RDC_ST_OK;
}


rdc_status_t rdc_watch_job_fields(rdc_handle_t p_rdc_handle,
        rdc_gpu_group_t group_id,  uint64_t update_freq, double  max_keep_age,
        uint32_t max_keep_samples) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
              rdc_watch_job_fields(group_id, update_freq,
                                max_keep_age, max_keep_samples);
}

rdc_status_t rdc_update_all_fields(rdc_handle_t p_rdc_handle,
        uint32_t wait_for_update) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_update_all_fields(wait_for_update);
}

rdc_status_t rdc_job_get_stats(rdc_handle_t p_rdc_handle, char  job_id[64] ,
                rdc_job_info_t* p_job_info) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_job_get_stats(job_id, p_job_info);
}

rdc_status_t rdc_job_start_stats(rdc_handle_t p_rdc_handle,
                rdc_gpu_group_t groupId, char  job_id[64] ) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_job_start_stats(groupId, job_id);
}


rdc_status_t rdc_job_stop_stats(rdc_handle_t p_rdc_handle, char  job_id[64] ) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_job_stop_stats(job_id);
}

rdc_status_t rdc_group_gpu_create(rdc_handle_t p_rdc_handle,
                rdc_group_type_t type, const char* group_name,
                rdc_gpu_group_t* p_rdc_group_id) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
              rdc_group_gpu_create(type, group_name, p_rdc_group_id);
}

rdc_status_t rdc_group_gpu_add(rdc_handle_t p_rdc_handle,
        rdc_gpu_group_t groupId, uint32_t gpuIndex ) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_group_gpu_add(groupId, gpuIndex);
}

const char* rdc_status_string(rdc_status_t result) {
    switch (result) {
        case RDC_ST_OK:
                return "Success";
        case RDC_ST_NOT_SUPPORTED:
                return "Not supported";
        case RDC_ST_FAIL_LOAD_MODULE:
                return "Fail to load module";
        case RDC_ST_INVALID_HANDLER:
                return "Invalid handler";
        default:
                return "Unknown";
    }
}

