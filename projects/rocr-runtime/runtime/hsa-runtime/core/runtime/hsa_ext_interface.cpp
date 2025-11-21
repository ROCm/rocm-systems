////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
// 
// Developed by:
// 
//                 AMD Research and AMD HSA Software Development
// 
//                 Advanced Micro Devices, Inc.
// 
//                 www.amd.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "image/inc/hsa_ext_image_impl.h"
#include "pcs/inc/hsa_ven_amd_pc_sampling_impl.h"
#include "core/inc/hsa_ext_interface.h"
#include "core/inc/runtime.h"

#include <string>

namespace rocr {
// Implementations for missing / unsupported extensions
template <class R, class... ARGS> static R hsa_ext_null(ARGS...) {
  return HSA_STATUS_ERROR_NOT_INITIALIZED;
}

namespace core {
ExtensionEntryPoints::ExtensionEntryPoints() {
  InitFinalizerExtTable();
  InitImageExtTable();
  InitPcSamplingExtTable();
  InitAmdExtTable();
}

// Initialize Finalizer function table to be NULLs
void ExtensionEntryPoints::InitFinalizerExtTable() {
  
  // Initialize Version of Api Table
  finalizer_api.version.major_id = 0x00;
  finalizer_api.version.minor_id = 0x00;
  finalizer_api.version.step_id = 0x00;

  finalizer_api.hsa_ext_program_create_fn = hsa_ext_null;
  finalizer_api.hsa_ext_program_destroy_fn = hsa_ext_null;
  finalizer_api.hsa_ext_program_add_module_fn = hsa_ext_null;
  finalizer_api.hsa_ext_program_iterate_modules_fn = hsa_ext_null;
  finalizer_api.hsa_ext_program_get_info_fn = hsa_ext_null;
  finalizer_api.hsa_ext_program_finalize_fn = hsa_ext_null;
}

// Initialize Image function table to be NULLs
void ExtensionEntryPoints::InitImageExtTable() {
 
  // Initialize Version of Api Table
  image_api.version.major_id = 0x00;
  image_api.version.minor_id = 0x00;
  image_api.version.step_id = 0x00;

  image_api.hsa_ext_image_get_capability_fn = hsa_ext_null;
  image_api.hsa_ext_image_data_get_info_fn = hsa_ext_null;
  image_api.hsa_ext_image_create_fn = hsa_ext_null;
  image_api.hsa_ext_image_import_fn = hsa_ext_null;
  image_api.hsa_ext_image_export_fn = hsa_ext_null;
  image_api.hsa_ext_image_copy_fn = hsa_ext_null;
  image_api.hsa_ext_image_clear_fn = hsa_ext_null;
  image_api.hsa_ext_image_destroy_fn = hsa_ext_null;
  image_api.hsa_ext_sampler_create_fn = hsa_ext_null;
  image_api.hsa_ext_sampler_destroy_fn = hsa_ext_null;
  image_api.hsa_amd_image_get_info_max_dim_fn = hsa_ext_null;
  image_api.hsa_ext_image_get_capability_with_layout_fn = hsa_ext_null;
  image_api.hsa_ext_image_data_get_info_with_layout_fn = hsa_ext_null;
  image_api.hsa_ext_image_create_with_layout_fn = hsa_ext_null;
}

// Initialize PC Sampling function table to be NULLs
void ExtensionEntryPoints::InitPcSamplingExtTable() {
  // Initialize Version of Api Table
  pcs_api.version.major_id = 0x00;
  pcs_api.version.minor_id = 0x00;
  pcs_api.version.step_id = 0x00;

  pcs_api.hsa_ven_amd_pcs_iterate_configuration_fn = hsa_ext_null;
  pcs_api.hsa_ven_amd_pcs_create_fn = hsa_ext_null;
  pcs_api.hsa_ven_amd_pcs_create_from_id_fn = hsa_ext_null;
  pcs_api.hsa_ven_amd_pcs_destroy_fn = hsa_ext_null;
  pcs_api.hsa_ven_amd_pcs_start_fn = hsa_ext_null;
  pcs_api.hsa_ven_amd_pcs_stop_fn = hsa_ext_null;
  pcs_api.hsa_ven_amd_pcs_flush_fn = hsa_ext_null;
}

// Initialize Amd Ext table for Api related to Images
void ExtensionEntryPoints::InitAmdExtTable() {
  hsa_api_table().amd_ext_api.hsa_amd_image_create_fn = hsa_ext_null;
  hsa_internal_api_table().amd_ext_api.hsa_amd_image_create_fn = hsa_ext_null;
}

// Update Amd Ext table for Api related to Images.
// @note: Interface should be updated when Amd Ext table
// begins hosting Api's from other extension libraries
void ExtensionEntryPoints::UpdateAmdExtTable(decltype(::hsa_amd_image_create)* func_ptr) {
  assert(hsa_api_table().amd_ext_api.hsa_amd_image_create_fn ==
             (decltype(hsa_amd_image_create)*)hsa_ext_null && 
             "Duplicate load of extension import.");
  assert(hsa_internal_api_table().amd_ext_api.hsa_amd_image_create_fn ==
             (decltype(hsa_amd_image_create)*)hsa_ext_null && 
             "Duplicate load of extension import.");
  hsa_api_table().amd_ext_api.hsa_amd_image_create_fn = func_ptr;
  hsa_internal_api_table().amd_ext_api.hsa_amd_image_create_fn = func_ptr;
}

void ExtensionEntryPoints::UnloadImage() {
  InitAmdExtTable();
  InitImageExtTable();
  core::hsa_internal_api_table().Reset();
#ifdef HSA_IMAGE_SUPPORT
  rocr::image::ReleaseImageRsrcs();
#endif
}

void ExtensionEntryPoints::Unload() {
  // Reset Image apis to hsa_ext_null function
  UnloadImage();
#ifdef HSA_PC_SAMPLING_SUPPORT
  rocr::pcs::ReleasePcSamplingRsrcs();
#endif

  for (auto lib : libs_) {
    void* ptr = os::GetExportAddress(lib, "Unload");
    if (ptr) {
      ((Unload_t)ptr)();
    }
  }
  // Due to valgrind bug, runtime cannot dlclose extensions see:
  // http://valgrind.org/docs/manual/faq.html#faq.unhelpful
  if (!core::Runtime::runtime_singleton_->flag().running_valgrind()) {
    for (auto lib : libs_) {
      os::CloseLib(lib);
    }
  }
  libs_.clear();

  InitFinalizerExtTable();
  InitPcSamplingExtTable();
  InitImageExtTable();
  InitAmdExtTable();
  core::hsa_internal_api_table().Reset();
}

bool ExtensionEntryPoints::LoadImage() {
#ifdef HSA_IMAGE_SUPPORT
  // Consult user input on linking to Image implementation
  bool disable_image = core::Runtime::runtime_singleton_->flag().disable_image();
  if (disable_image) {
    return true;
  }

  // Bind to Image implementation api's
  decltype(::hsa_amd_image_create)* func;
  rocr::image::LoadImage(&image_api, &func);

  // Initialize Version of Api Table
  image_api.version.major_id = HSA_IMAGE_API_TABLE_MAJOR_VERSION;
  image_api.version.minor_id = sizeof(ImageExtTable);
  image_api.version.step_id = HSA_IMAGE_API_TABLE_STEP_VERSION;

  // Update private copy of Api table with handle for Image extensions
  hsa_internal_api_table().CloneExts(&image_api,
                                    core::HsaApiTable::HSA_EXT_IMAGE_API_TABLE_ID);

  // Update Amd Ext Api table Api that deals with Images
  UpdateAmdExtTable(func);
#endif
  return true;
}

void ExtensionEntryPoints::LoadPcSampling() {
#ifdef HSA_PC_SAMPLING_SUPPORT
  if (core::Runtime::runtime_singleton_->flag().disable_pc_sampling()) return;

  // Bind to Image implementation api's
  rocr::pcs::LoadPcSampling(&pcs_api);

  // Initialize Version of Api Table
  pcs_api.version.major_id = HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION;
  pcs_api.version.minor_id = sizeof(PcSamplingExtTable);
  pcs_api.version.step_id = HSA_PC_SAMPLING_API_TABLE_STEP_VERSION;

  // Update private copy of Api table with handle for Image extensions
  hsa_internal_api_table().CloneExts(&pcs_api,
                        core::HsaApiTable::HSA_EXT_PC_SAMPLING_API_TABLE_ID);
#endif
}

bool ExtensionEntryPoints::LoadFinalizer(std::string library_name) {
  os::LibHandle lib = os::LoadLib(library_name);
  if (lib == NULL) {
    return false;
  }
  libs_.push_back(lib);
  
  void* ptr;

  ptr = os::GetExportAddress(lib, "hsa_ext_program_create_impl");
  if (ptr != NULL) {
    assert(finalizer_api.hsa_ext_program_create_fn ==
               (decltype(::hsa_ext_program_create)*)hsa_ext_null &&
           "Duplicate load of extension import.");
    finalizer_api.hsa_ext_program_create_fn = (decltype(::hsa_ext_program_create)*)ptr;
  }

  ptr = os::GetExportAddress(lib, "hsa_ext_program_destroy_impl");
  if (ptr != NULL) {
    assert(finalizer_api.hsa_ext_program_destroy_fn ==
               (decltype(::hsa_ext_program_destroy)*)hsa_ext_null &&
           "Duplicate load of extension import.");
    finalizer_api.hsa_ext_program_destroy_fn =
        (decltype(::hsa_ext_program_destroy)*)ptr;
  }

  ptr = os::GetExportAddress(lib, "hsa_ext_program_add_module_impl");
  if (ptr != NULL) {
    assert(finalizer_api.hsa_ext_program_add_module_fn ==
               (decltype(::hsa_ext_program_add_module)*)hsa_ext_null &&
           "Duplicate load of extension import.");
    finalizer_api.hsa_ext_program_add_module_fn =
        (decltype(::hsa_ext_program_add_module)*)ptr;
  }

  ptr = os::GetExportAddress(lib, "hsa_ext_program_iterate_modules_impl");
  if (ptr != NULL) {
    assert(finalizer_api.hsa_ext_program_iterate_modules_fn ==
               (decltype(::hsa_ext_program_iterate_modules)*)hsa_ext_null &&
           "Duplicate load of extension import.");
    finalizer_api.hsa_ext_program_iterate_modules_fn =
        (decltype(::hsa_ext_program_iterate_modules)*)ptr;
  }

  ptr = os::GetExportAddress(lib, "hsa_ext_program_get_info_impl");
  if (ptr != NULL) {
    assert(finalizer_api.hsa_ext_program_get_info_fn ==
               (decltype(::hsa_ext_program_get_info)*)hsa_ext_null &&
           "Duplicate load of extension import.");
    finalizer_api.hsa_ext_program_get_info_fn =
        (decltype(::hsa_ext_program_get_info)*)ptr;
  }

  ptr = os::GetExportAddress(lib, "hsa_ext_program_finalize_impl");
  if (ptr != NULL) {
    assert(finalizer_api.hsa_ext_program_finalize_fn ==
               (decltype(::hsa_ext_program_finalize)*)hsa_ext_null &&
           "Duplicate load of extension import.");
    finalizer_api.hsa_ext_program_finalize_fn =
        (decltype(::hsa_ext_program_finalize)*)ptr;
  }
  
  // Initialize Version of Api Table
  finalizer_api.version.major_id = HSA_FINALIZER_API_TABLE_MAJOR_VERSION;
  finalizer_api.version.minor_id = sizeof(::FinalizerExtTable);
  finalizer_api.version.step_id = HSA_FINALIZER_API_TABLE_STEP_VERSION;
 
  // Update handle of table of HSA extensions
  hsa_internal_api_table().CloneExts(&finalizer_api,
                                    core::HsaApiTable::HSA_EXT_FINALIZER_API_TABLE_ID);

  ptr = os::GetExportAddress(lib, "Load");
  if (ptr != NULL) {
    ((Load_t)ptr)(&core::hsa_internal_api_table().hsa_api);
  }

  return true;
}

}  // namespace core
}  // namespace rocr

//---------------------------------------------------------------------------//
//   Exported extension stub functions
//---------------------------------------------------------------------------//

hsa_status_t hsa_ext_program_create(
    hsa_machine_model_t machine_model, hsa_profile_t profile,
    hsa_default_float_rounding_mode_t default_float_rounding_mode,
    const char* options, hsa_ext_program_t* program) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.finalizer_api
      .hsa_ext_program_create_fn(machine_model, profile,
                                 default_float_rounding_mode, options, program);
}

hsa_status_t hsa_ext_program_destroy(hsa_ext_program_t program) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.finalizer_api
      .hsa_ext_program_destroy_fn(program);
}

hsa_status_t hsa_ext_program_add_module(hsa_ext_program_t program,
                                        hsa_ext_module_t module) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.finalizer_api
      .hsa_ext_program_add_module_fn(program, module);
}

hsa_status_t hsa_ext_program_iterate_modules(
    hsa_ext_program_t program,
    hsa_status_t (*callback)(hsa_ext_program_t program, hsa_ext_module_t module,
                             void* data),
    void* data) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.finalizer_api
      .hsa_ext_program_iterate_modules_fn(program, callback, data);
}

hsa_status_t hsa_ext_program_get_info(hsa_ext_program_t program,
                                      hsa_ext_program_info_t attribute,
                                      void* value) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.finalizer_api
      .hsa_ext_program_get_info_fn(program, attribute, value);
}

hsa_status_t hsa_ext_program_finalize(
    hsa_ext_program_t program, hsa_isa_t isa, int32_t call_convention,
    hsa_ext_control_directives_t control_directives, const char* options,
    hsa_code_object_type_t code_object_type, hsa_code_object_t* code_object) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.finalizer_api
      .hsa_ext_program_finalize_fn(program, isa, call_convention,
                                   control_directives, options,
                                   code_object_type, code_object);
}

hsa_status_t hsa_ext_image_get_capability(
    hsa_agent_t agent, hsa_ext_image_geometry_t geometry,
    const hsa_ext_image_format_t* image_format, uint32_t* capability_mask) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_get_capability_fn(agent, geometry, image_format,
                                       capability_mask);
}

hsa_status_t hsa_ext_image_data_get_info(
    hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor,
    hsa_access_permission_t access_permission,
    hsa_ext_image_data_info_t* image_data_info) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_data_get_info_fn(agent, image_descriptor,
                                      access_permission, image_data_info);
}

hsa_status_t hsa_ext_image_create(
    hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor,
    const void* image_data, hsa_access_permission_t access_permission,
    hsa_ext_image_t* image) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_create_fn(agent, image_descriptor, image_data,
                               access_permission, image);
}

hsa_status_t hsa_ext_image_import(hsa_agent_t agent, const void* src_memory,
                                  size_t src_row_pitch, size_t src_slice_pitch,
                                  hsa_ext_image_t dst_image,
                                  const hsa_ext_image_region_t* image_region) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_import_fn(agent, src_memory, src_row_pitch,
                               src_slice_pitch, dst_image, image_region);
}

hsa_status_t hsa_ext_image_export(hsa_agent_t agent, hsa_ext_image_t src_image,
                                  void* dst_memory, size_t dst_row_pitch,
                                  size_t dst_slice_pitch,
                                  const hsa_ext_image_region_t* image_region) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_export_fn(agent, src_image, dst_memory, dst_row_pitch,
                               dst_slice_pitch, image_region);
}

hsa_status_t hsa_ext_image_copy(hsa_agent_t agent, hsa_ext_image_t src_image,
                                const hsa_dim3_t* src_offset,
                                hsa_ext_image_t dst_image,
                                const hsa_dim3_t* dst_offset,
                                const hsa_dim3_t* range) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_copy_fn(agent, src_image, src_offset, dst_image,
                             dst_offset, range);
}

hsa_status_t hsa_ext_image_clear(hsa_agent_t agent, hsa_ext_image_t image,
                                 const void* data,
                                 const hsa_ext_image_region_t* image_region) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_clear_fn(agent, image, data, image_region);
}

hsa_status_t hsa_ext_image_destroy(hsa_agent_t agent, hsa_ext_image_t image) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_destroy_fn(agent, image);
}

hsa_status_t hsa_ext_sampler_create(
    hsa_agent_t agent, const hsa_ext_sampler_descriptor_t* sampler_descriptor,
    hsa_ext_sampler_t* sampler) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_sampler_create_fn(agent, sampler_descriptor, sampler);
}

hsa_status_t hsa_ext_sampler_create_v2(
    hsa_agent_t agent, const hsa_ext_sampler_descriptor_v2_t* sampler_descriptor,
    hsa_ext_sampler_t* sampler) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_sampler_create_v2_fn(agent, sampler_descriptor, sampler);
}

hsa_status_t hsa_ext_sampler_destroy(hsa_agent_t agent,
                                     hsa_ext_sampler_t sampler) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_sampler_destroy_fn(agent, sampler);
}

hsa_status_t hsa_ext_image_get_capability_with_layout(
    hsa_agent_t agent, hsa_ext_image_geometry_t geometry,
    const hsa_ext_image_format_t* image_format,
    hsa_ext_image_data_layout_t image_data_layout,
    uint32_t* capability_mask) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_get_capability_with_layout_fn(agent, geometry, image_format,
                                       image_data_layout, capability_mask);
}

hsa_status_t hsa_ext_image_data_get_info_with_layout(
    hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor,
    hsa_access_permission_t access_permission,
    hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    hsa_ext_image_data_info_t* image_data_info) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_data_get_info_with_layout_fn(agent, image_descriptor,
                                      access_permission, image_data_layout,
                                      image_data_row_pitch, image_data_slice_pitch,
                                      image_data_info);
}

hsa_status_t hsa_ext_image_create_with_layout(
    hsa_agent_t agent, const hsa_ext_image_descriptor_t* image_descriptor,
    const void* image_data, hsa_access_permission_t access_permission,
    hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    hsa_ext_image_t* image) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_ext_image_create_with_layout_fn(agent, image_descriptor, image_data,
                               access_permission, image_data_layout,
                               image_data_row_pitch, image_data_slice_pitch,
                               image);
}

hsa_status_t HSA_API hsa_ven_amd_pcs_iterate_configuration(
    hsa_agent_t agent, hsa_ven_amd_pcs_iterate_configuration_callback_t configuration_callback,
    void* callback_data) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.pcs_api
      .hsa_ven_amd_pcs_iterate_configuration_fn(agent, configuration_callback, callback_data);
}

hsa_status_t HSA_API hsa_ven_amd_pcs_create(
    hsa_agent_t agent, hsa_ven_amd_pcs_method_kind_t method, hsa_ven_amd_pcs_units_t units,
    size_t interval, size_t latency, size_t buffer_size,
    hsa_ven_amd_pcs_data_ready_callback_t data_ready_callback, void* client_callback_data,
    hsa_ven_amd_pcs_t* pc_sampling) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.pcs_api.hsa_ven_amd_pcs_create_fn(
      agent, method, units, interval, latency, buffer_size, data_ready_callback,
      client_callback_data, pc_sampling);
}

hsa_status_t HSA_API hsa_ven_amd_pcs_create_from_id(
    uint32_t pcs_id, hsa_agent_t agent, hsa_ven_amd_pcs_method_kind_t method,
    hsa_ven_amd_pcs_units_t units, size_t interval, size_t latency, size_t buffer_size,
    hsa_ven_amd_pcs_data_ready_callback_t data_ready_callback, void* client_callback_data,
    hsa_ven_amd_pcs_t* pc_sampling) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.pcs_api
      .hsa_ven_amd_pcs_create_from_id_fn(pcs_id, agent, method, units, interval, latency,
                                         buffer_size, data_ready_callback, client_callback_data,
                                         pc_sampling);
}

hsa_status_t HSA_API hsa_ven_amd_pcs_destroy(hsa_ven_amd_pcs_t pc_sampling) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.pcs_api.hsa_ven_amd_pcs_destroy_fn(
      pc_sampling);
}

hsa_status_t HSA_API hsa_ven_amd_pcs_start(hsa_ven_amd_pcs_t pc_sampling) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.pcs_api.hsa_ven_amd_pcs_start_fn(
      pc_sampling);
}

hsa_status_t HSA_API hsa_ven_amd_pcs_stop(hsa_ven_amd_pcs_t pc_sampling) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.pcs_api.hsa_ven_amd_pcs_stop_fn(
      pc_sampling);
}

hsa_status_t HSA_API hsa_ven_amd_pcs_flush(hsa_ven_amd_pcs_t pc_sampling) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.pcs_api.hsa_ven_amd_pcs_flush_fn(
      pc_sampling);
}

//---------------------------------------------------------------------------//
//  Stubs for internal extension functions
//---------------------------------------------------------------------------//

// Use the function pointer from local instance Image Extension
hsa_status_t hsa_amd_image_get_info_max_dim(hsa_agent_t component,
                                            hsa_agent_info_t attribute,
                                            void* value) {
  return rocr::core::Runtime::runtime_singleton_->extensions_.image_api
      .hsa_amd_image_get_info_max_dim_fn(component, attribute, value);
}
