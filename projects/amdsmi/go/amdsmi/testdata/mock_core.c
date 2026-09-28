// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "mock.h"

struct operation {
  const char* name;
  _Atomic uint32_t status;
  _Atomic uint32_t mode;
  _Atomic uint64_t calls;
};

#define OP(name) {#name, AMDSMI_STATUS_SUCCESS, 0, 0}
static struct operation operations[] = {
    OP(amdsmi_init),
    OP(amdsmi_shut_down),
    OP(amdsmi_get_lib_version),
    OP(amdsmi_status_code_to_string),
    OP(amdsmi_get_socket_handles),
    OP(amdsmi_get_processor_handles),
    OP(amdsmi_get_processor_type),
    OP(amdsmi_get_processor_handle_from_bdf),
    OP(amdsmi_get_gpu_device_bdf),
    OP(amdsmi_get_gpu_device_uuid),
    OP(amdsmi_get_gpu_asic_info),
    OP(amdsmi_get_gpu_driver_info),
    OP(amdsmi_get_gpu_board_info),
    OP(amdsmi_get_fw_info),
    OP(amdsmi_get_gpu_vbios_info),
    OP(amdsmi_get_temp_metric),
    OP(amdsmi_get_power_info),
    OP(amdsmi_get_power_cap_info),
    OP(amdsmi_get_clock_info),
    OP(amdsmi_get_clk_freq),
    OP(amdsmi_get_gpu_activity),
    OP(amdsmi_get_gpu_memory_total),
    OP(amdsmi_get_gpu_memory_usage),
    OP(amdsmi_get_gpu_vram_info),
    OP(amdsmi_get_gpu_kfd_info),
    OP(amdsmi_get_gpu_memory_partition_config),
    OP(amdsmi_get_gpu_accelerator_partition_profile),
    OP(amdsmi_get_gpu_ecc_enabled),
    OP(amdsmi_get_gpu_ecc_count),
    OP(amdsmi_get_gpu_total_ecc_count),
    OP(amdsmi_get_gpu_ras_block_features_enabled),
    OP(amdsmi_get_gpu_ras_feature_info),
};
#undef OP

static _Atomic uint32_t native_refs;
static _Atomic uint32_t active;
static _Atomic uint32_t max_active;

#define MOCK_MAX_SOCKETS 3
#define MOCK_MAX_PROCESSORS 64
static uint32_t socket_count = 1;
static uint32_t processors_per_socket = 1;
static uint32_t non_gpu_stride;

struct mock_socket {
  uint32_t index;
};

struct mock_processor {
  uint32_t index;
};

static struct mock_socket sockets[MOCK_MAX_SOCKETS];
static struct mock_processor processors[MOCK_MAX_SOCKETS][MOCK_MAX_PROCESSORS];

static struct mock_socket* find_socket(amdsmi_socket_handle handle) {
  for (uint32_t i = 0; i < socket_count; ++i) {
    if (handle == &sockets[i]) return &sockets[i];
  }
  return NULL;
}

static struct mock_processor* find_processor(amdsmi_processor_handle handle) {
  for (uint32_t i = 0; i < socket_count; ++i) {
    for (uint32_t j = 0; j < processors_per_socket; ++j) {
      if (handle == &processors[i][j]) return &processors[i][j];
    }
  }
  return NULL;
}

static struct operation* find_operation(const char* name) {
  if (name != NULL) {
    for (size_t i = 0; i < sizeof(operations) / sizeof(operations[0]); ++i) {
      if (strcmp(name, operations[i].name) == 0) return &operations[i];
    }
  }
  abort();
}

void mock_reset(void) {
  if (atomic_load(&native_refs) != 0 || atomic_load(&active) != 0) abort();
  for (size_t i = 0; i < sizeof(operations) / sizeof(operations[0]); ++i) {
    atomic_store(&operations[i].status, AMDSMI_STATUS_SUCCESS);
    atomic_store(&operations[i].mode, 0);
    atomic_store(&operations[i].calls, 0);
  }
  atomic_store(&max_active, 0);
  mock_topology(1, 1, 0);
}

void mock_topology(uint32_t socket_total, uint32_t processor_total, uint32_t stride) {
  if (atomic_load(&native_refs) != 0 || atomic_load(&active) != 0 ||
      socket_total > MOCK_MAX_SOCKETS || processor_total > MOCK_MAX_PROCESSORS)
    abort();
  socket_count = socket_total;
  processors_per_socket = processor_total;
  non_gpu_stride = stride;
  for (uint32_t i = 0; i < socket_count; ++i) {
    sockets[i].index = i;
    for (uint32_t j = 0; j < processors_per_socket; ++j) {
      processors[i][j].index = i * processors_per_socket + j;
    }
  }
}

void mock_configure(const char* op, uint32_t status, uint32_t mode) {
  struct operation* entry = find_operation(op);
  atomic_store(&entry->status, status);
  atomic_store(&entry->mode, mode);
}

uint32_t mock_mode(const char* op) { return atomic_load(&find_operation(op)->mode); }
uint64_t mock_calls(const char* op) { return atomic_load(&find_operation(op)->calls); }
uint32_t mock_max_active(void) { return atomic_load(&max_active); }
uint32_t mock_native_refs(void) { return atomic_load(&native_refs); }

amdsmi_status_t mock_begin(const char* op, amdsmi_processor_handle processor) {
  struct operation* entry = find_operation(op);
  atomic_fetch_add(&entry->calls, 1);
  uint32_t current = atomic_fetch_add(&active, 1) + 1;
  uint32_t maximum = atomic_load(&max_active);
  while (current > maximum && !atomic_compare_exchange_weak(&max_active, &maximum, current)) {
  }
  sched_yield();
  if (processor != NULL) {
    if (atomic_load(&native_refs) == 0) return AMDSMI_STATUS_NOT_INIT;
    if (!find_processor(processor)) return AMDSMI_STATUS_INVAL;
  }
  return (amdsmi_status_t)atomic_load(&entry->status);
}

amdsmi_status_t mock_finish(amdsmi_status_t status) {
  if (atomic_fetch_sub(&active, 1) == 0) abort();
  return status;
}

int mock_is_zero(const void* value, size_t size) {
  if (value == NULL) return size == 0;
  const unsigned char* bytes = value;
  for (size_t i = 0; i < size; ++i) {
    if (bytes[i] != 0) return 0;
  }
  return 1;
}

amdsmi_status_t amdsmi_init(uint64_t flags) {
  amdsmi_status_t status = mock_begin(__func__, NULL);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (flags != AMDSMI_INIT_AMD_GPUS) return mock_finish(AMDSMI_STATUS_INVAL);
  atomic_fetch_add(&native_refs, 1);
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_shut_down(void) {
  amdsmi_status_t status = mock_begin(__func__, NULL);
  if (atomic_load(&native_refs) == 0) return mock_finish(AMDSMI_STATUS_NOT_INIT);
  atomic_fetch_sub(&native_refs, 1);
  return mock_finish(status);
}

amdsmi_status_t amdsmi_get_socket_handles(uint32_t* count, amdsmi_socket_handle* out) {
  amdsmi_status_t status = mock_begin(__func__, NULL);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (atomic_load(&native_refs) == 0) return mock_finish(AMDSMI_STATUS_NOT_INIT);
  if (!count) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (!out) {
    if (*count != 0) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
    *count = socket_count;
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  if (*count != socket_count || !mock_is_zero(out, *count * sizeof(*out)))
    return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  uint32_t mode = mock_mode(__func__);
  if (mode == 5) return mock_finish(AMDSMI_STATUS_IO);
  uint32_t copied = socket_count - (mode == 3 && socket_count != 0);
  for (uint32_t i = 0; i < copied; ++i) out[i] = &sockets[i];
  if (mode == 2 && copied != 0) out[0] = NULL;
  *count = copied + (mode == 1);
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle socket, uint32_t* count,
                                             amdsmi_processor_handle* out) {
  amdsmi_status_t status = mock_begin(__func__, NULL);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (atomic_load(&native_refs) == 0) return mock_finish(AMDSMI_STATUS_NOT_INIT);
  struct mock_socket* entry = find_socket(socket);
  if (!entry) return mock_finish(AMDSMI_STATUS_INVAL);
  if (!count) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (!out) {
    if (*count != 0) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
    *count = mock_mode(__func__) == 4 && entry->index == 0 ? 0 : processors_per_socket;
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  if (*count != processors_per_socket || !mock_is_zero(out, *count * sizeof(*out)))
    return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  uint32_t mode = mock_mode(__func__);
  if (mode == 5) return mock_finish(AMDSMI_STATUS_IO);
  uint32_t copied = processors_per_socket - (mode == 3 && processors_per_socket != 0);
  for (uint32_t i = 0; i < copied; ++i) out[i] = &processors[entry->index][i];
  if (mode == 2 && copied != 0) out[0] = NULL;
  *count = copied + (mode == 1);
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle processor,
                                          amdsmi_processor_type_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  struct mock_processor* entry = find_processor(processor);
  if (!entry) return mock_finish(AMDSMI_STATUS_INVAL);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  *out = non_gpu_stride != 0 && (entry->index + 1) % non_gpu_stride == 0
             ? AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU
             : AMDSMI_PROCESSOR_TYPE_AMD_GPU;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

static amdsmi_bdf_t processor_bdf(const struct mock_processor* processor) {
  amdsmi_bdf_t bdf = {0};
  bdf.bdf.domain_number = UINT64_C(0xabcde1234567);
  bdf.bdf.bus_number = processor->index / 32;
  bdf.bdf.device_number = (processor->index % 32) / 8;
  bdf.bdf.function_number = processor->index % 8;
  return bdf;
}

amdsmi_status_t amdsmi_get_gpu_device_bdf(amdsmi_processor_handle processor, amdsmi_bdf_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  struct mock_processor* entry = find_processor(processor);
  if (!entry) return mock_finish(AMDSMI_STATUS_INVAL);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  *out = processor_bdf(entry);
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_processor_handle_from_bdf(amdsmi_bdf_t bdf,
                                                     amdsmi_processor_handle* out) {
  amdsmi_status_t status = mock_begin(__func__, NULL);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (atomic_load(&native_refs) == 0) return mock_finish(AMDSMI_STATUS_NOT_INIT);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (mock_mode(__func__) == 1) return mock_finish(AMDSMI_STATUS_SUCCESS);
  for (uint32_t i = 0; i < socket_count; ++i) {
    for (uint32_t j = 0; j < processors_per_socket; ++j) {
      amdsmi_bdf_t candidate = processor_bdf(&processors[i][j]);
      if (bdf.bdf.domain_number == candidate.bdf.domain_number &&
          bdf.bdf.bus_number == candidate.bdf.bus_number &&
          bdf.bdf.device_number == candidate.bdf.device_number &&
          bdf.bdf.function_number == candidate.bdf.function_number) {
        *out = &processors[i][j];
        return mock_finish(AMDSMI_STATUS_SUCCESS);
      }
    }
  }
  return mock_finish(AMDSMI_STATUS_NOT_FOUND);
}

amdsmi_status_t amdsmi_get_lib_version(amdsmi_version_t* out) {
  amdsmi_status_t status = mock_begin(__func__, NULL);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  out->major = AMDSMI_LIB_VERSION_MAJOR;
  out->minor = AMDSMI_LIB_VERSION_MINOR;
  out->release = AMDSMI_LIB_VERSION_RELEASE;
  out->build = mock_mode(__func__) == 1 ? NULL : AMDSMI_LIB_VERSION_STRING;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_status_code_to_string(amdsmi_status_t code, const char** out) {
  amdsmi_status_t status = mock_begin(__func__, NULL);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (mock_mode(__func__) == 1) return mock_finish(AMDSMI_STATUS_SUCCESS);
#define STATUS(name) \
  case name:         \
    *out = #name;    \
    break
  switch (code) {
    STATUS(AMDSMI_STATUS_SUCCESS);
    STATUS(AMDSMI_STATUS_INVAL);
    STATUS(AMDSMI_STATUS_NOT_SUPPORTED);
    STATUS(AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    STATUS(AMDSMI_STATUS_FAIL_LOAD_MODULE);
    STATUS(AMDSMI_STATUS_FAIL_LOAD_SYMBOL);
    STATUS(AMDSMI_STATUS_DRM_ERROR);
    STATUS(AMDSMI_STATUS_API_FAILED);
    STATUS(AMDSMI_STATUS_RETRY);
    STATUS(AMDSMI_STATUS_NO_PERM);
    STATUS(AMDSMI_STATUS_INTERRUPT);
    STATUS(AMDSMI_STATUS_IO);
    STATUS(AMDSMI_STATUS_ADDRESS_FAULT);
    STATUS(AMDSMI_STATUS_FILE_ERROR);
    STATUS(AMDSMI_STATUS_OUT_OF_RESOURCES);
    STATUS(AMDSMI_STATUS_INTERNAL_EXCEPTION);
    STATUS(AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS);
    STATUS(AMDSMI_STATUS_INIT_ERROR);
    STATUS(AMDSMI_STATUS_REFCOUNT_OVERFLOW);
    STATUS(AMDSMI_STATUS_DIRECTORY_NOT_FOUND);
    STATUS(AMDSMI_STATUS_IPC_ERROR);
    STATUS(AMDSMI_STATUS_BUSY);
    STATUS(AMDSMI_STATUS_NOT_FOUND);
    STATUS(AMDSMI_STATUS_NOT_INIT);
    STATUS(AMDSMI_STATUS_NO_SLOT);
    STATUS(AMDSMI_STATUS_DRIVER_NOT_LOADED);
    STATUS(AMDSMI_STATUS_NO_DATA);
    STATUS(AMDSMI_STATUS_INSUFFICIENT_SIZE);
    STATUS(AMDSMI_STATUS_UNEXPECTED_SIZE);
    STATUS(AMDSMI_STATUS_UNEXPECTED_DATA);
    STATUS(AMDSMI_STATUS_NON_AMD_CPU);
    STATUS(AMDSMI_STATUS_NO_ENERGY_DRV);
    STATUS(AMDSMI_STATUS_NO_MSR_DRV);
    STATUS(AMDSMI_STATUS_NO_HSMP_DRV);
    STATUS(AMDSMI_STATUS_NO_HSMP_SUP);
    STATUS(AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    STATUS(AMDSMI_STATUS_HSMP_TIMEOUT);
    STATUS(AMDSMI_STATUS_NO_DRV);
    STATUS(AMDSMI_STATUS_FILE_NOT_FOUND);
    STATUS(AMDSMI_STATUS_ARG_PTR_NULL);
    STATUS(AMDSMI_STATUS_AMDGPU_RESTART_ERR);
    STATUS(AMDSMI_STATUS_SETTING_UNAVAILABLE);
    STATUS(AMDSMI_STATUS_CORRUPTED_EEPROM);
    STATUS(AMDSMI_STATUS_MAP_ERROR);
    STATUS(AMDSMI_STATUS_UNKNOWN_ERROR);
    default:
      return mock_finish(AMDSMI_STATUS_UNKNOWN_ERROR);
  }
#undef STATUS
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}
