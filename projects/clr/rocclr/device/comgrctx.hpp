/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mutex>
#include "top.hpp"
#include "amd_comgr/amd_comgr.h"

namespace amd {
// The entry points of the compile path: OpenCL builds, hipRTC, hipLink, SPIR-V, and compressed
// bundles in a format the runtime doesn't read. Loading code objects doesn't use comgr.
typedef void (*t_amd_comgr_get_version)(size_t* major, size_t* minor);
typedef amd_comgr_status_t (*t_amd_comgr_create_data)(amd_comgr_data_kind_t kind,
                                                      amd_comgr_data_t* data);
typedef amd_comgr_status_t (*t_amd_comgr_release_data)(amd_comgr_data_t data);
typedef amd_comgr_status_t (*t_amd_comgr_set_data)(amd_comgr_data_t data, size_t size,
                                                   const char* bytes);
typedef amd_comgr_status_t (*t_amd_comgr_set_data_name)(amd_comgr_data_t data, const char* name);
typedef amd_comgr_status_t (*t_amd_comgr_get_data)(amd_comgr_data_t data, size_t* size,
                                                   char* bytes);
typedef amd_comgr_status_t (*t_amd_comgr_get_data_name)(amd_comgr_data_t data, size_t* size,
                                                        char* name);
typedef amd_comgr_status_t (*t_amd_comgr_create_data_set)(amd_comgr_data_set_t* data_set);
typedef amd_comgr_status_t (*t_amd_comgr_destroy_data_set)(amd_comgr_data_set_t data_set);
typedef amd_comgr_status_t (*t_amd_comgr_data_set_add)(amd_comgr_data_set_t data_set,
                                                       amd_comgr_data_t data);
typedef amd_comgr_status_t (*t_amd_comgr_action_data_count)(amd_comgr_data_set_t data_set,
                                                            amd_comgr_data_kind_t data_kind,
                                                            size_t* count);
typedef amd_comgr_status_t (*t_amd_comgr_action_data_get_data)(amd_comgr_data_set_t data_set,
                                                               amd_comgr_data_kind_t data_kind,
                                                               size_t index,
                                                               amd_comgr_data_t* data);
typedef amd_comgr_status_t (*t_amd_comgr_create_action_info)(amd_comgr_action_info_t* action_info);
typedef amd_comgr_status_t (*t_amd_comgr_destroy_action_info)(amd_comgr_action_info_t action_info);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_isa_name)(
    amd_comgr_action_info_t action_info, const char* isa_name);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_language)(
    amd_comgr_action_info_t action_info, amd_comgr_language_t language);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_option_list)(
    amd_comgr_action_info_t action_info, const char* options[], size_t count);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_logging)(
    amd_comgr_action_info_t action_info, bool logging);
typedef amd_comgr_status_t (*t_amd_comgr_do_action)(amd_comgr_action_kind_t kind,
                                                    amd_comgr_action_info_t info,
                                                    amd_comgr_data_set_t input,
                                                    amd_comgr_data_set_t result);
typedef amd_comgr_status_t (*t_amd_comgr_populate_name_expression_map)(amd_comgr_data_t data,
                                                                       size_t* count);
typedef amd_comgr_status_t (*t_amd_comgr_map_name_expression_to_symbol_name)(amd_comgr_data_t data,
                                                                             size_t* size,
                                                                             char* name_expression,
                                                                             char* symbol_name);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_device_lib_linking)(
    amd_comgr_action_info_t action_info, bool link_device_lib);
typedef amd_comgr_status_t (*t_amd_comgr_action_info_set_bundle_entry_ids)(
    amd_comgr_action_info_t action_info, const char* bundle_entry_ids[], size_t count);

struct ComgrEntryPoints {
  void* handle;
  t_amd_comgr_get_version amd_comgr_get_version;
  t_amd_comgr_create_data amd_comgr_create_data;
  t_amd_comgr_release_data amd_comgr_release_data;
  t_amd_comgr_set_data amd_comgr_set_data;
  t_amd_comgr_set_data_name amd_comgr_set_data_name;
  t_amd_comgr_get_data amd_comgr_get_data;
  t_amd_comgr_get_data_name amd_comgr_get_data_name;
  t_amd_comgr_create_data_set amd_comgr_create_data_set;
  t_amd_comgr_destroy_data_set amd_comgr_destroy_data_set;
  t_amd_comgr_data_set_add amd_comgr_data_set_add;
  t_amd_comgr_action_data_count amd_comgr_action_data_count;
  t_amd_comgr_action_data_get_data amd_comgr_action_data_get_data;
  t_amd_comgr_create_action_info amd_comgr_create_action_info;
  t_amd_comgr_destroy_action_info amd_comgr_destroy_action_info;
  t_amd_comgr_action_info_set_isa_name amd_comgr_action_info_set_isa_name;
  t_amd_comgr_action_info_set_language amd_comgr_action_info_set_language;
  t_amd_comgr_action_info_set_option_list amd_comgr_action_info_set_option_list;
  t_amd_comgr_action_info_set_logging amd_comgr_action_info_set_logging;
  t_amd_comgr_do_action amd_comgr_do_action;
  t_amd_comgr_populate_name_expression_map amd_comgr_populate_name_expression_map;
  t_amd_comgr_map_name_expression_to_symbol_name amd_comgr_map_name_expression_to_symbol_name;
  t_amd_comgr_action_info_set_device_lib_linking amd_comgr_action_info_set_device_lib_linking;
  t_amd_comgr_action_info_set_bundle_entry_ids amd_comgr_action_info_set_bundle_entry_ids;
};

#ifdef COMGR_DYN_DLL
// comgr maps the LLVM/clang libraries, so it is loaded on first use, not at device init.
// If it can't be loaded, entry points resolve to a stub that returns AMD_COMGR_STATUS_ERROR.
#define COMGR_DYN(NAME) Entry(&ComgrEntryPoints::NAME)
#define GET_COMGR_SYMBOL(NAME)                                                                     \
  cep_.NAME = reinterpret_cast<t_##NAME>(Os::getSymbol(cep_.handle, #NAME));                       \
  if (nullptr == cep_.NAME) {                                                                      \
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "Failed to load COMGR function %s", #NAME);             \
    return false;                                                                                  \
  }
#define GET_COMGR_OPTIONAL_SYMBOL(NAME)                                                            \
  cep_.NAME = reinterpret_cast<t_##NAME>(Os::getSymbol(cep_.handle, #NAME));
#else
#define COMGR_DYN(NAME) NAME
#define GET_COMGR_SYMBOL(NAME)
#define GET_COMGR_OPTIONAL_SYMBOL(NAME)
#endif

#ifdef COMGR_DYN_DLL
//! Fallback for a comgr entry point that isn't available: comgr failed to load or is too old
template <typename F> struct ComgrStub;
template <typename... Args> struct ComgrStub<amd_comgr_status_t (*)(Args...)> {
  static amd_comgr_status_t call(Args...) { return AMD_COMGR_STATUS_ERROR; }
};
template <typename... Args> struct ComgrStub<void (*)(Args...)> {
  static void call(Args...) {}
};
#endif

class Comgr : public amd::AllStatic {
 public:
  static std::once_flag initialized;

  static bool LoadLib();

  //! Loads comgr on first use. Returns true if it is loaded
  static bool Load() {
    std::call_once(initialized, LoadLib);
    return is_ready_;
  }

  static bool IsReady() { return is_ready_; }

  static void get_version(size_t* major, size_t* minor) {
    COMGR_DYN(amd_comgr_get_version)(major, minor);
  }
  static amd_comgr_status_t create_data(amd_comgr_data_kind_t kind, amd_comgr_data_t* data) {
    return COMGR_DYN(amd_comgr_create_data)(kind, data);
  }
  static amd_comgr_status_t release_data(amd_comgr_data_t data) {
    return COMGR_DYN(amd_comgr_release_data)(data);
  }
  static amd_comgr_status_t set_data(amd_comgr_data_t data, size_t size, const char* bytes) {
    return COMGR_DYN(amd_comgr_set_data)(data, size, bytes);
  }
  static amd_comgr_status_t set_data_name(amd_comgr_data_t data, const char* name) {
    return COMGR_DYN(amd_comgr_set_data_name)(data, name);
  }
  static amd_comgr_status_t get_data(amd_comgr_data_t data, size_t* size, char* bytes) {
    return COMGR_DYN(amd_comgr_get_data)(data, size, bytes);
  }
  static amd_comgr_status_t get_data_name(amd_comgr_data_t data, size_t* size, char* name) {
    return COMGR_DYN(amd_comgr_get_data_name)(data, size, name);
  }
  static amd_comgr_status_t create_data_set(amd_comgr_data_set_t* data_set) {
    return COMGR_DYN(amd_comgr_create_data_set)(data_set);
  }
  static amd_comgr_status_t destroy_data_set(amd_comgr_data_set_t data_set) {
    return COMGR_DYN(amd_comgr_destroy_data_set)(data_set);
  }
  static amd_comgr_status_t data_set_add(amd_comgr_data_set_t data_set, amd_comgr_data_t data) {
    return COMGR_DYN(amd_comgr_data_set_add)(data_set, data);
  }
  static amd_comgr_status_t action_data_count(amd_comgr_data_set_t data_set,
                                              amd_comgr_data_kind_t data_kind, size_t* count) {
    return COMGR_DYN(amd_comgr_action_data_count)(data_set, data_kind, count);
  }
  static amd_comgr_status_t action_data_get_data(amd_comgr_data_set_t data_set,
                                                 amd_comgr_data_kind_t data_kind, size_t index,
                                                 amd_comgr_data_t* data) {
    return COMGR_DYN(amd_comgr_action_data_get_data)(data_set, data_kind, index, data);
  }
  static amd_comgr_status_t create_action_info(amd_comgr_action_info_t* action_info) {
    return COMGR_DYN(amd_comgr_create_action_info)(action_info);
  }
  static amd_comgr_status_t destroy_action_info(amd_comgr_action_info_t action_info) {
    return COMGR_DYN(amd_comgr_destroy_action_info)(action_info);
  }
  static amd_comgr_status_t action_info_set_isa_name(amd_comgr_action_info_t action_info,
                                                     const char* isa_name) {
    return COMGR_DYN(amd_comgr_action_info_set_isa_name)(action_info, isa_name);
  }
  static amd_comgr_status_t action_info_set_language(amd_comgr_action_info_t action_info,
                                                     amd_comgr_language_t language) {
    return COMGR_DYN(amd_comgr_action_info_set_language)(action_info, language);
  }
  static amd_comgr_status_t action_info_set_option_list(amd_comgr_action_info_t action_info,
                                                        const char* options[], size_t count) {
    return COMGR_DYN(amd_comgr_action_info_set_option_list)(action_info, options, count);
  }
  static amd_comgr_status_t action_info_set_logging(amd_comgr_action_info_t action_info,
                                                    bool logging) {
    return COMGR_DYN(amd_comgr_action_info_set_logging)(action_info, logging);
  }
  static amd_comgr_status_t do_action(amd_comgr_action_kind_t kind, amd_comgr_action_info_t info,
                                      amd_comgr_data_set_t input, amd_comgr_data_set_t result) {
    return COMGR_DYN(amd_comgr_do_action)(kind, info, input, result);
  }
  static amd_comgr_status_t populate_name_expression_map(amd_comgr_data_t data, size_t* count) {
    return COMGR_DYN(amd_comgr_populate_name_expression_map)(data, count);
  }
  static amd_comgr_status_t map_name_expression_to_symbol_name(amd_comgr_data_t data, size_t* size,
                                                               char* name_expression,
                                                               char* symbol_name) {
    return COMGR_DYN(amd_comgr_map_name_expression_to_symbol_name)(data, size, name_expression,
                                                                   symbol_name);
  }
  static amd_comgr_status_t action_info_set_device_lib_linking(amd_comgr_action_info_t action_info,
                                                               bool link_device_lib) {
    return COMGR_DYN(amd_comgr_action_info_set_device_lib_linking)(action_info, link_device_lib);
  }
  static amd_comgr_status_t action_info_set_bundle_entry_ids(amd_comgr_action_info_t action_info,
                                                             const char* bundle_entry_ids[],
                                                             size_t count) {
#if defined(COMGR_DYN_DLL)
    if (!Load() || cep_.amd_comgr_action_info_set_bundle_entry_ids == nullptr) {
      // comgr version 2.7 or less is loaded
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
              "Failed to load COMGR function amd_comgr_action_info_set_bundle_entry_ids");
      return AMD_COMGR_STATUS_ERROR;
    }
#endif
    return COMGR_DYN(amd_comgr_action_info_set_bundle_entry_ids)(action_info, bundle_entry_ids,
                                                                 count);
  }

 private:
#ifdef COMGR_DYN_DLL
  //! Returns the entry point @p fn, loading comgr on first use, or a stub if it's unavailable
  template <typename F> static F Entry(F ComgrEntryPoints::*fn) {
    F f = Load() ? cep_.*fn : nullptr;
    return (f != nullptr) ? f : &ComgrStub<F>::call;
  }
#endif

  static ComgrEntryPoints cep_;
  static bool is_ready_;
};

}  // namespace amd
