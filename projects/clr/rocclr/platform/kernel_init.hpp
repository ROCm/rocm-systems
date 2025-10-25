/* Copyright (c) 2008 - 2024 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#if defined(USE_COMGR_LIBRARY)

// Static values initialization from class Kernel.
const amd::Kernel::ArgFieldMapType amd::Kernel::kArgFieldMap[] = {
    {"Name", ArgField::Name},
    {"TypeName", ArgField::TypeName},
    {"Size", ArgField::Size},
    {"Align", ArgField::Align},
    {"ValueKind", ArgField::ValueKind},
    {"PointeeAlign", ArgField::PointeeAlign},
    {"AddrSpaceQual", ArgField::AddrSpaceQual},
    {"AccQual", ArgField::AccQual},
    {"ActualAccQual", ArgField::ActualAccQual},
    {"IsConst", ArgField::IsConst},
    {"IsRestrict", ArgField::IsRestrict},
    {"IsVolatile", ArgField::IsVolatile},
    {"IsPipe", ArgField::IsPipe},
    {"Offset", ArgField::Offset}};

const amd::Kernel::ArgValueKindType amd::Kernel::kArgValueKind[] = {
    {"ByValue", amd::KernelParameterDescriptor::ValueObject},
    {"GlobalBuffer", amd::KernelParameterDescriptor::MemoryObject},
    {"DynamicSharedPointer", amd::KernelParameterDescriptor::MemoryObject},
    {"Sampler", amd::KernelParameterDescriptor::SamplerObject},
    {"Image", amd::KernelParameterDescriptor::ImageObject},
    {"Pipe", amd::KernelParameterDescriptor::MemoryObject},
    {"Queue", amd::KernelParameterDescriptor::QueueObject},
    {"HiddenGlobalOffsetX", amd::KernelParameterDescriptor::HiddenGlobalOffsetX},
    {"HiddenGlobalOffsetY", amd::KernelParameterDescriptor::HiddenGlobalOffsetY},
    {"HiddenGlobalOffsetZ", amd::KernelParameterDescriptor::HiddenGlobalOffsetZ},
    {"HiddenNone", amd::KernelParameterDescriptor::HiddenNone},
    {"HiddenPrintfBuffer", amd::KernelParameterDescriptor::HiddenPrintfBuffer},
    {"HiddenDefaultQueue", amd::KernelParameterDescriptor::HiddenDefaultQueue},
    {"HiddenCompletionAction", amd::KernelParameterDescriptor::HiddenCompletionAction},
    {"HiddenMultigridSyncArg", amd::KernelParameterDescriptor::HiddenMultiGridSync},
    {"HiddenHostcallBuffer", amd::KernelParameterDescriptor::HiddenHostcallBuffer}};

const amd::Kernel::ArgAccQualType amd::Kernel::kArgAccQual[] = {
    {"Default", CL_KERNEL_ARG_ACCESS_NONE},
    {"ReadOnly", CL_KERNEL_ARG_ACCESS_READ_ONLY},
    {"WriteOnly", CL_KERNEL_ARG_ACCESS_WRITE_ONLY},
    {"ReadWrite", CL_KERNEL_ARG_ACCESS_READ_WRITE}};

const amd::Kernel::ArgAddrSpaceQualType amd::Kernel::kArgAddrSpaceQual[] = {
    {"Private", CL_KERNEL_ARG_ADDRESS_PRIVATE},   {"Global", CL_KERNEL_ARG_ADDRESS_GLOBAL},
    {"Constant", CL_KERNEL_ARG_ADDRESS_CONSTANT}, {"Local", CL_KERNEL_ARG_ADDRESS_LOCAL},
    {"Generic", CL_KERNEL_ARG_ADDRESS_GLOBAL},    {"Region", CL_KERNEL_ARG_ADDRESS_PRIVATE}};

const amd::Kernel::AttrFieldMapType amd::Kernel::kAttrFieldMap[] = {
    {"ReqdWorkGroupSize", AttrField::ReqdWorkGroupSize},
    {"WorkGroupSizeHint", AttrField::WorkGroupSizeHint},
    {"VecTypeHint", AttrField::VecTypeHint},
    {"RuntimeHandle", AttrField::RuntimeHandle}};

const amd::Kernel::CodePropFieldMapType amd::Kernel::kCodePropFieldMap[] = {
    {"KernargSegmentSize", CodePropField::KernargSegmentSize},
    {"GroupSegmentFixedSize", CodePropField::GroupSegmentFixedSize},
    {"PrivateSegmentFixedSize", CodePropField::PrivateSegmentFixedSize},
    {"KernargSegmentAlign", CodePropField::KernargSegmentAlign},
    {"WavefrontSize", CodePropField::WavefrontSize},
    {"NumSGPRs", CodePropField::NumSGPRs},
    {"NumVGPRs", CodePropField::NumVGPRs},
    {"MaxFlatWorkGroupSize", CodePropField::MaxFlatWorkGroupSize},
    {"IsDynamicCallStack", CodePropField::IsDynamicCallStack},
    {"IsXNACKEnabled", CodePropField::IsXNACKEnabled},
    {"NumSpilledSGPRs", CodePropField::NumSpilledSGPRs},
    {"NumSpilledVGPRs", CodePropField::NumSpilledVGPRs}};

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
    {".uniform_work_group_size", KernelField::UniformWrokGroupSize}};

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
V amd::Kernel::FindValue(const T (&structure)[N], const std::string& name) {
  for (size_t idx = 0; idx < N; ++idx) {
    if (std::string(structure[idx].name) == name) {
      return structure[idx].value;
    }
  }
  return V::MaxSize;
}

// Templated find function to retrieve cl_int values.
template <typename T, size_t N>
cl_int amd::Kernel::FindValue(const T (&structure)[N], const std::string& name) {
  for (size_t idx = 0; idx < N; ++idx) {
    if (std::string(structure[idx].name) == name) {
      return structure[idx].value;
    }
  }
  return 0;
}

#endif  // defined(USE_COMGR_LIBRARY)