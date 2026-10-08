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
static hsa_signal_value_t (*fn_hsa_signal_load_relaxed)(hsa_signal_t signal);
static hsa_signal_value_t (*fn_hsa_signal_wait_relaxed)(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint);
static void (*fn_hsa_signal_store_screlease)(hsa_signal_t hsa_signal,
                                             hsa_signal_value_t value);

// Resolves libhsa-runtime64 entry points. Called once via std::call_once on
// the first queue create, before any queue's PM4 thread reads the pointers.
bool hsakmt_hsa_loader_init() {
  // At queue-create time librocdxg is already resident as a dependency of
  // libhsa-runtime64, so only look it up in the current address space
  // (RTLD_NOLOAD) — never pull a fresh copy from disk. RTLD_GLOBAL promotes its
  // symbols in case the app dlopen'd it lazily/locally, which would otherwise
  // leave these symbols unresolvable.
  void *handle = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  if (!handle) {
    pr_err("libhsa-runtime64.so.1 not resident - %s\n", dlerror());
    return false;
  }

  fn_hsa_signal_load_relaxed = reinterpret_cast<decltype(fn_hsa_signal_load_relaxed)>(
      dlsym(handle, "hsa_signal_load_relaxed"));
  fn_hsa_signal_wait_relaxed = reinterpret_cast<decltype(fn_hsa_signal_wait_relaxed)>(
      dlsym(handle, "hsa_signal_wait_relaxed"));
  fn_hsa_signal_store_screlease = reinterpret_cast<decltype(fn_hsa_signal_store_screlease)>(
      dlsym(handle, "hsa_signal_store_screlease"));

  auto fn_hsa_system_get_extension_table =
      reinterpret_cast<hsa_status_t (*)(uint16_t, uint16_t, uint16_t, void *)>(
          dlsym(handle, "hsa_system_get_extension_table"));
  if (fn_hsa_system_get_extension_table) {
    hsa_ven_amd_loader_1_03_pfn_t table;
    fn_hsa_system_get_extension_table(HSA_EXTENSION_AMD_LOADER, 1, 3, &table);
    fn_hsa_ven_amd_loader_query_host_address = table.hsa_ven_amd_loader_query_host_address;
  }

  // Function pointers captured; drop our reference. The library stays resident
  // via libhsa-runtime64's own reference.
  dlclose(handle);

  bool ok = fn_hsa_signal_load_relaxed && fn_hsa_signal_wait_relaxed &&
            fn_hsa_signal_store_screlease && fn_hsa_ven_amd_loader_query_host_address;
  if (!ok)
    pr_err("failed to resolve libhsa-runtime64 symbols\n");
  return ok;
}

hsa_signal_value_t hsakmt_hsa_signal_load_relaxed(hsa_signal_t signal) {
  if (fn_hsa_signal_load_relaxed)
    return fn_hsa_signal_load_relaxed(signal);
  return 0;
}

hsa_signal_value_t hsakmt_hsa_signal_wait_relaxed(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint) {
  if (fn_hsa_signal_wait_relaxed)
    return fn_hsa_signal_wait_relaxed(signal, condition, compare_value,
                                      timeout_hint, wait_state_hint);
  return 0;
}

void hsakmt_hsa_signal_store_screlease(hsa_signal_t hsa_signal,
                                       hsa_signal_value_t value) {
  if (fn_hsa_signal_store_screlease)
    fn_hsa_signal_store_screlease(hsa_signal, value);
}

hsa_status_t hsakmt_hsa_ven_amd_loader_query_host_address(
    const void *device_address, const void **host_address) {
  if (fn_hsa_ven_amd_loader_query_host_address)
    return fn_hsa_ven_amd_loader_query_host_address(device_address, host_address);
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
