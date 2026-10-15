// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_GO_MOCK_H
#define AMDSMI_GO_MOCK_H

#include <amd_smi/amdsmi.h>
#include <stddef.h>
#include <stdint.h>

void mock_reset(void);
void mock_configure(const char* op, uint32_t status, uint32_t mode);
void mock_topology(uint32_t sockets, uint32_t processors_per_socket, uint32_t non_gpu_stride);
uint32_t mock_mode(const char* op);
uint64_t mock_calls(const char* op);
uint32_t mock_max_active(void);
uint32_t mock_native_refs(void);
amdsmi_status_t mock_begin(const char* op, amdsmi_processor_handle processor);
amdsmi_status_t mock_finish(amdsmi_status_t status);
int mock_is_zero(const void* value, size_t size);

#endif
