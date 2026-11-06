/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Static values initialization from class Kernel.
const amd::Kernel::ArgAccQualV3Type amd::Kernel::kArgAccQualV3[] = {
    {"default", CL_KERNEL_ARG_ACCESS_NONE},
    {"read_only", CL_KERNEL_ARG_ACCESS_READ_ONLY},
    {"write_only", CL_KERNEL_ARG_ACCESS_WRITE_ONLY},
    {"read_write", CL_KERNEL_ARG_ACCESS_READ_WRITE}};

const amd::Kernel::ArgAddrSpaceQualV3Type amd::Kernel::kArgAddrSpaceQualV3[] = {
    {"private", CL_KERNEL_ARG_ADDRESS_PRIVATE},   {"global", CL_KERNEL_ARG_ADDRESS_GLOBAL},
    {"constant", CL_KERNEL_ARG_ADDRESS_CONSTANT}, {"local", CL_KERNEL_ARG_ADDRESS_LOCAL},
    {"generic", CL_KERNEL_ARG_ADDRESS_GLOBAL},    {"region", CL_KERNEL_ARG_ADDRESS_PRIVATE}};

const amd::Kernel::KernelFieldMapV3Type amd::Kernel::kKernelFieldMapV3[] = {
    {".symbol", KernelField::SymbolName},
    {".reqd_workgroup_size", KernelField::ReqdWorkGroupSize},
    {".workgroup_size_hint", KernelField::WorkGroupSizeHint},
    {".vec_type_hint", KernelField::VecTypeHint},
    {".device_enqueue_symbol", KernelField::DeviceEnqueueSymbol},
    {".kernarg_segment_size", KernelField::KernargSegmentSize},
    {".group_segment_fixed_size", KernelField::GroupSegmentFixedSize},
    {".private_segment_fixed_size", KernelField::PrivateSegmentFixedSize},
    {".kernarg_segment_align", KernelField::KernargSegmentAlign},
    {".wavefront_size", KernelField::WavefrontSize},
    {".sgpr_count", KernelField::NumSGPRs},
    {".vgpr_count", KernelField::NumVGPRs},
    {".max_flat_workgroup_size", KernelField::MaxFlatWorkGroupSize},
    {".sgpr_spill_count", KernelField::NumSpilledSGPRs},
    {".vgpr_spill_count", KernelField::NumSpilledVGPRs},
    {".kind", KernelField::Kind},
    {".workgroup_processor_mode", KernelField::WgpMode},
    {".uniform_work_group_size", KernelField::UniformWrokGroupSize},
    {".cluster_dims", KernelField::ClusterDims}};

const amd::Kernel::ArgValueKindV3Type amd::Kernel::kArgValueKindV3[] = {
    {"by_value", amd::KernelParameterDescriptor::ValueObject},
    {"global_buffer", amd::KernelParameterDescriptor::MemoryObject},
    {"dynamic_shared_pointer", amd::KernelParameterDescriptor::MemoryObject},
    {"sampler", amd::KernelParameterDescriptor::SamplerObject},
    {"image", amd::KernelParameterDescriptor::ImageObject},
    {"pipe", amd::KernelParameterDescriptor::MemoryObject},
    {"queue", amd::KernelParameterDescriptor::QueueObject},
    {"hidden_global_offset_x", amd::KernelParameterDescriptor::HiddenGlobalOffsetX},
    {"hidden_global_offset_y", amd::KernelParameterDescriptor::HiddenGlobalOffsetY},
    {"hidden_global_offset_z", amd::KernelParameterDescriptor::HiddenGlobalOffsetZ},
    {"hidden_none", amd::KernelParameterDescriptor::HiddenNone},
    {"hidden_printf_buffer", amd::KernelParameterDescriptor::HiddenPrintfBuffer},
    {"hidden_default_queue", amd::KernelParameterDescriptor::HiddenDefaultQueue},
    {"hidden_completion_action", amd::KernelParameterDescriptor::HiddenCompletionAction},
    {"hidden_multigrid_sync_arg", amd::KernelParameterDescriptor::HiddenMultiGridSync},
    {"hidden_heap_v1", amd::KernelParameterDescriptor::HiddenHeap},
    {"hidden_hostcall_buffer", amd::KernelParameterDescriptor::HiddenHostcallBuffer},
    {"hidden_block_count_x", amd::KernelParameterDescriptor::HiddenBlockCountX},
    {"hidden_block_count_y", amd::KernelParameterDescriptor::HiddenBlockCountY},
    {"hidden_block_count_z", amd::KernelParameterDescriptor::HiddenBlockCountZ},
    {"hidden_group_size_x", amd::KernelParameterDescriptor::HiddenGroupSizeX},
    {"hidden_group_size_y", amd::KernelParameterDescriptor::HiddenGroupSizeY},
    {"hidden_group_size_z", amd::KernelParameterDescriptor::HiddenGroupSizeZ},
    {"hidden_remainder_x", amd::KernelParameterDescriptor::HiddenRemainderX},
    {"hidden_remainder_y", amd::KernelParameterDescriptor::HiddenRemainderY},
    {"hidden_remainder_z", amd::KernelParameterDescriptor::HiddenRemainderZ},
    {"hidden_grid_dims", amd::KernelParameterDescriptor::HiddenGridDims},
    {"hidden_private_base", amd::KernelParameterDescriptor::HiddenPrivateBase},
    {"hidden_shared_base", amd::KernelParameterDescriptor::HiddenSharedBase},
    {"hidden_queue_ptr", amd::KernelParameterDescriptor::HiddenQueuePtr},
    {"hidden_dynamic_lds_size", amd::KernelParameterDescriptor::HiddenDynamicLdsSize}};

const amd::Kernel::ArgFieldMapV3Type amd::Kernel::kArgFieldMapV3[] = {
    {".name", ArgField::Name},
    {".type_name", ArgField::TypeName},
    {".size", ArgField::Size},
    {".value_kind", ArgField::ValueKind},
    {".pointee_align", ArgField::PointeeAlign},
    {".address_space", ArgField::AddrSpaceQual},
    {".access", ArgField::AccQual},
    {".actual_access", ArgField::ActualAccQual},
    {".is_const", ArgField::IsConst},
    {".is_restrict", ArgField::IsRestrict},
    {".is_volatile", ArgField::IsVolatile},
    {".is_pipe", ArgField::IsPipe},
    {".offset", ArgField::Offset}};


// Templated find function to retrieve the right value based on string
template <typename V, typename T, size_t N>
V amd::Kernel::FindValue(const T (&structure)[N], std::string_view name) {
  for (size_t idx = 0; idx < N; ++idx) {
    if (name == structure[idx].name) {
      return structure[idx].value;
    }
  }
  return V::MaxSize;
}

// Templated find function to retrieve cl_int values.
template <typename T, size_t N>
cl_int amd::Kernel::FindValue(const T (&structure)[N], std::string_view name) {
  for (size_t idx = 0; idx < N; ++idx) {
    if (name == structure[idx].name) {
      return structure[idx].value;
    }
  }
  return 0;
}
