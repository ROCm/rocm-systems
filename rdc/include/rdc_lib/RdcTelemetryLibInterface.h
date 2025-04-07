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
#ifndef INCLUDE_RDC_LIB_RDCTELEMETRYLIBINTERFACE_H_
#define INCLUDE_RDC_LIB_RDCTELEMETRYLIBINTERFACE_H_

// The telemetry interface for libraries, for example, AMD-SMI.
#include <rdc/rdc.h>

#include <cstdint>

extern "C" {

// Structure to keep both gup index and field value
typedef struct {
  uint32_t gpu_index;
  rdc_field_value field_value;
} rdc_gpu_field_value_t;

typedef struct {
  uint32_t gpu_index;
  rdc_field_t field_id;
} rdc_gpu_field_t;

#define MAX_NUM_FIELDS 8192
typedef rdc_status_t (*rdc_field_value_f)(rdc_gpu_field_value_t* values, uint32_t num_values,
                                          void* user_data);

// The library will implement below function

rdc_status_t rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS], uint32_t* field_count);

rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields, uint32_t fields_count,
                                            rdc_field_value_f callback, void* user_data);

rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t* fields, uint32_t fields_count);

rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields, uint32_t fields_count);

rdc_status_t rdc_module_init(uint64_t flags);

rdc_status_t rdc_module_destroy();
}

#endif  // INCLUDE_RDC_LIB_RDCTELEMETRYLIBINTERFACE_H_
