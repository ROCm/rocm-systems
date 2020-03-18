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
#include <string.h>
#include <map>
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
                rdc_handle_t* p_rdc_handle,
                const char* root_ca, const char* client_cert,
                const char* client_key ) {
        amd::rdc::RdcHandler* (*func_make_handler)(const char*,
        const char*, const char*, const char*);

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
                (func_make_handler(ipAddress,
                                root_ca, client_cert, client_key));
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

rdc_status_t rdc_field_update_all(rdc_handle_t p_rdc_handle,
        uint32_t wait_for_update) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_field_update_all(wait_for_update);
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
                rdc_gpu_group_t groupId, char  job_id[64], uint64_t update_freq,
                double  max_keep_age, uint32_t  max_keep_samples ) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_job_start_stats(groupId, job_id, update_freq,
                                max_keep_age, max_keep_samples);
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

rdc_status_t rdc_device_get_all(rdc_handle_t p_rdc_handle,
            uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES], uint32_t* count) {
        if (!p_rdc_handle || !count) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_device_get_all(gpu_index_list, count);
}

rdc_status_t rdc_device_get_attributes(rdc_handle_t p_rdc_handle,
            uint32_t gpu_index, rdc_device_attributes_t* p_rdc_attr) {
        if (!p_rdc_handle || !p_rdc_attr) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_device_get_attributes(gpu_index, p_rdc_attr);
}

rdc_status_t rdc_group_field_create(rdc_handle_t p_rdc_handle,
            uint32_t num_field_ids, uint32_t* field_ids,
            const char* field_group_name, rdc_field_grp_t* rdc_field_group_id) {
        if (!p_rdc_handle || !field_ids ||
                !field_group_name || !rdc_field_group_id) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_group_field_create(num_field_ids, field_ids,
                field_group_name, rdc_field_group_id);
}

rdc_status_t rdc_group_field_get_info(rdc_handle_t p_rdc_handle,
            rdc_field_grp_t rdc_field_group_id,
            rdc_field_group_info_t* field_group_info) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_group_field_get_info(rdc_field_group_id, field_group_info);
}

rdc_status_t rdc_group_gpu_get_info(rdc_handle_t p_rdc_handle,
        rdc_gpu_group_t p_rdc_group_id, rdc_group_info_t* p_rdc_group_info) {
        if (!p_rdc_handle || !p_rdc_group_info) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_group_gpu_get_info(p_rdc_group_id, p_rdc_group_info);
}

rdc_status_t rdc_group_get_all_ids(rdc_handle_t p_rdc_handle,
            rdc_gpu_group_t group_id_list[], uint32_t* count) {
        if (!p_rdc_handle || !count) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
              rdc_group_get_all_ids(group_id_list, count);
}

rdc_status_t rdc_group_field_get_all_ids(rdc_handle_t p_rdc_handle,
        rdc_field_grp_t field_group_id_list[], uint32_t* count) {
        if (!p_rdc_handle || !count) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
              rdc_group_field_get_all_ids(field_group_id_list, count);
}

rdc_status_t rdc_field_watch(rdc_handle_t p_rdc_handle,
        rdc_gpu_group_t group_id, rdc_field_grp_t field_group_id,
        uint64_t update_freq, double max_keep_age, uint32_t max_keep_samples) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
              rdc_field_watch(group_id, field_group_id, update_freq,
              max_keep_age, max_keep_samples);
}

rdc_status_t rdc_field_get_latest_value(rdc_handle_t p_rdc_handle,
        uint32_t gpu_index, uint32_t field, rdc_field_value* value) {
        if (!p_rdc_handle || !value) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_field_get_latest_value(gpu_index, field, value);
}

rdc_status_t rdc_field_get_value_since(rdc_handle_t p_rdc_handle,
        uint32_t gpu_index, uint32_t field, uint64_t since_time_stamp,
        uint64_t *next_since_time_stamp, rdc_field_value* value) {
        if (!p_rdc_handle || !next_since_time_stamp || !value) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
                rdc_field_get_value_since(gpu_index, field, since_time_stamp,
                next_since_time_stamp, value);
}

rdc_status_t rdc_field_unwatch(rdc_handle_t p_rdc_handle,
        rdc_gpu_group_t group_id, rdc_field_grp_t field_group_id) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
              rdc_field_unwatch(group_id, field_group_id);
}

rdc_status_t rdc_group_gpu_destroy(rdc_handle_t p_rdc_handle,
        rdc_gpu_group_t p_rdc_group_id) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
              rdc_group_gpu_destroy(p_rdc_group_id);
}

rdc_status_t rdc_group_field_destroy(rdc_handle_t p_rdc_handle,
        rdc_field_grp_t rdc_field_group_id) {
        if (!p_rdc_handle) {
                return RDC_ST_INVALID_HANDLER;
        }

        return static_cast<amd::rdc::RdcHandler*>(p_rdc_handle)->
              rdc_group_field_destroy(rdc_field_group_id);
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
        case RDC_ST_NOT_FOUND:
                return "Cannot find the value";
        case RDC_ST_BAD_PARAMETER:
                return "Invalid parameters";
        case RDC_ST_MSI_ERROR:
                return "SMI error";
        case RDC_ST_MAX_LIMIT:
                return "The max limit reached";
        case RDC_ST_CONFLICT:
                return "Conflict with current state";
        case RDC_ST_CLIENT_ERROR:
                return "RDC Client error";
        default:
                return "Unknown";
    }
}

const char* field_id_string(uint32_t field_id) {
       const std::map<uint16_t, const char*> id_name = {
        {RDC_FI_GPU_MEMORY_USAGE, "GPU_MEMORY_USAGE"},
        {RDC_FI_GPU_MEMORY_TOTAL, "GPU_MEMORY_TOTAL"},
        {RDC_FI_POWER_USAGE, "POWER_USAGE"},
        {RDC_FI_GPU_SM_CLOCK, "GPU_SM_CLOCK"},
        {RDC_FI_GPU_UTIL, "GPU_UTIL"},
        {RDC_FI_GPU_TEMP, "GPU_TEMP"},
        {RDC_FI_GPU_COUNT, "GPU_COUNT"},
        {RDC_FI_DEV_NAME, "DEV_NAME"}
      };

      auto search = id_name.find(field_id);
      if (search == id_name.end()) {
          return "UNKNOWN_FIELD";
      }

      return search->second;
}

char *strncpy_with_null(char *dest, const char *src, size_t n) {
     if (n == 0) {
        return dest;
     }
     strncpy(dest, src, n - 1);
     dest[n - 1]= '\0';
     return dest;
}
