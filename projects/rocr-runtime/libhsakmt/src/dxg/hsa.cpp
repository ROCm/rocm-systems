/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#if defined(__linux__)
#include <dlfcn.h>
#endif
#include "hsa-runtime/inc/hsa.h"
#include "hsa-runtime/inc/hsa_ven_amd_loader.h"

static hsa_status_t (*fn_hsa_ven_amd_loader_query_host_address)(
    const void *device_address, const void **host_address);

static std::mutex* lock_ = new std::mutex();

#if defined(__linux__)
#define _HSAKMT_LOOKUP_SYMS(_sym)                                              \
if (fn_##_sym == nullptr) {                                                    \
    std::lock_guard<std::mutex> gard(*lock_);                                  \
    if (fn_##_sym == nullptr) {                                                \
      fn_##_sym =                                                              \
        reinterpret_cast<decltype(fn_##_sym)>(dlsym(RTLD_DEFAULT, #_sym));     \
      if (!fn_##_sym) {                                                        \
        pr_err("%s not found - %s\n", #_sym, dlerror());                       \
      }                                                                        \
    }                                                                          \
}

#define _HSAKMT_EXEC_API(_sym, ...) \
do { \
    if (fn_##_sym != nullptr) {    \
        return fn_##_sym(__VA_ARGS__);   \
    } \
} while(0);

bool hsakmt_hsa_loader_init() {
  void *hsa_loader_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
  if (hsa_loader_handle == nullptr) {
    pr_err("dlopen libhsa-runtime64.so failed - %s\n", dlerror());
    return false;
  }
  dlclose(hsa_loader_handle);
  return true;
}

hsa_signal_value_t hsakmt_hsa_signal_load_relaxed(hsa_signal_t signal) {
  static hsa_signal_value_t (*fn_hsa_signal_load_relaxed)(hsa_signal_t signal) = nullptr;

  _HSAKMT_LOOKUP_SYMS(hsa_signal_load_relaxed);
  _HSAKMT_EXEC_API(hsa_signal_load_relaxed, signal);

  return 0;
}

hsa_signal_value_t hsakmt_hsa_signal_wait_relaxed(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint) {
static hsa_signal_value_t (*fn_hsa_signal_wait_relaxed)(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint) = nullptr;

  _HSAKMT_LOOKUP_SYMS(hsa_signal_wait_relaxed);
  _HSAKMT_EXEC_API(hsa_signal_wait_relaxed, signal, condition, compare_value,
                   timeout_hint, wait_state_hint);

  return 0;
}

void hsakmt_hsa_signal_store_screlease(hsa_signal_t hsa_signal,
                                      hsa_signal_value_t value){
static void (*fn_hsa_signal_store_screlease)(hsa_signal_t hsa_signal,
                                      hsa_signal_value_t value) = nullptr;

  _HSAKMT_LOOKUP_SYMS(hsa_signal_store_screlease);
  _HSAKMT_EXEC_API(hsa_signal_store_screlease, hsa_signal, value);
}

hsa_status_t hsakmt_hsa_ven_amd_loader_query_host_address(
    const void *device_address, const void **host_address) {
  static hsa_status_t (*fn_hsa_ven_amd_loader_query_host_address)(
    const void *device_address, const void **host_address) = nullptr;

  if (fn_hsa_ven_amd_loader_query_host_address == nullptr) {
    std::lock_guard<std::mutex> gard(*lock_);
    if (fn_hsa_ven_amd_loader_query_host_address == nullptr) {
      hsa_status_t (*fn_hsa_system_get_extension_table)(
      uint16_t extension, uint16_t version_major, uint16_t version_minor, void *table);
      fn_hsa_system_get_extension_table =
        reinterpret_cast<decltype(fn_hsa_system_get_extension_table)>(dlsym(RTLD_DEFAULT, "hsa_system_get_extension_table"));
      if (fn_hsa_system_get_extension_table == nullptr) {
        pr_err("%s not found - %s\n", "hsa_system_get_extension_table", dlerror());
        return HSA_STATUS_ERROR;
      }

      hsa_ven_amd_loader_1_03_pfn_t table;
      fn_hsa_system_get_extension_table(HSA_EXTENSION_AMD_LOADER, 1, 3, &table);
      fn_hsa_ven_amd_loader_query_host_address =
          table.hsa_ven_amd_loader_query_host_address;
    }
  }

  _HSAKMT_EXEC_API(hsa_ven_amd_loader_query_host_address, device_address, host_address);
  return HSA_STATUS_ERROR;
}

#else
hsa_signal_value_t hsakmt_hsa_signal_load_relaxed(hsa_signal_t signal) {
  return hsa_signal_load_relaxed(signal);
}

hsa_signal_value_t hsakmt_hsa_signal_wait_relaxed(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint) {
  return hsa_signal_wait_relaxed(signal, condition, compare_value, timeout_hint,
                                 wait_state_hint);
}

void hsakmt_hsa_signal_store_screlease(hsa_signal_t hsa_signal,
                                      hsa_signal_value_t value) {
  hsa_signal_store_screlease(hsa_signal, value);
}

hsa_status_t hsakmt_hsa_ven_amd_loader_query_host_address(
    const void *device_address, const void **host_address) {
  static hsa_status_t (*fn_hsa_ven_amd_loader_query_host_address)(
    const void *device_address, const void **host_address) = nullptr;

  if (fn_hsa_ven_amd_loader_query_host_address == nullptr) {
    std::lock_guard<std::mutex> gard(*lock_);
    if (fn_hsa_ven_amd_loader_query_host_address == nullptr) {
      hsa_ven_amd_loader_1_03_pfn_t table;
      hsa_system_get_extension_table(HSA_EXTENSION_AMD_LOADER, 1, 3, &table);
      fn_hsa_ven_amd_loader_query_host_address =
          table.hsa_ven_amd_loader_query_host_address;
    }
  }

  if (fn_hsa_ven_amd_loader_query_host_address)
    return fn_hsa_ven_amd_loader_query_host_address(device_address, host_address);

  return HSA_STATUS_ERROR;
}
#endif
