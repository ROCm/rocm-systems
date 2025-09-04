/* Copyright (c) 2008 - 2022 Advanced Micro Devices, Inc.

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

#include "platform/runtime.hpp"
#include "platform/program.hpp"
#include "platform/ndrange.hpp"
#include "platform/kernel_init.hpp"
#include "devkernel.hpp"
#include "utils/macros.hpp"
#include "utils/options.hpp"
#if defined(WITH_COMPILER_LIB)
#include "utils/bif_section_labels.hpp"
#include "utils/libUtils.h"
#endif
#include "comgrctx.hpp"

#include <map>
#include <string>
#include <sstream>

#if defined(WITH_COMPILER_LIB)
#include "hsailctx.hpp"
#endif

namespace amd::device {

// ================================================================================================
static constexpr clk_value_type_t ClkValueMapType[6][6] = {
    {T_CHAR, T_CHAR2, T_CHAR3, T_CHAR4, T_CHAR8, T_CHAR16},
    {T_SHORT, T_SHORT2, T_SHORT3, T_SHORT4, T_SHORT8, T_SHORT16},
    {T_INT, T_INT2, T_INT3, T_INT4, T_INT8, T_INT16},
    {T_LONG, T_LONG2, T_LONG3, T_LONG4, T_LONG8, T_LONG16},
    {T_FLOAT, T_FLOAT2, T_FLOAT3, T_FLOAT4, T_FLOAT8, T_FLOAT16},
    {T_DOUBLE, T_DOUBLE2, T_DOUBLE3, T_DOUBLE4, T_DOUBLE8, T_DOUBLE16},
};

#if defined(USE_COMGR_LIBRARY)

// ================================================================================================
amd_comgr_status_t getMetaBuf(const amd_comgr_metadata_node_t meta,
                   std::string* str) {
  size_t size = 0;
  amd_comgr_status_t status = amd::Comgr::get_metadata_string(meta, &size, NULL);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    str->resize(size-1);    // minus one to discount the null character
    status = amd::Comgr::get_metadata_string(meta, &size, &((*str)[0]));
  }

  return status;
}

// ================================================================================================
bool getValueFromIsaMeta(const std::string& isa, const char* key, std::string& retValue) {
  amd_comgr_metadata_node_t isaMeta;

  amd_comgr_status_t status = amd::Comgr::get_isa_metadata(isa.c_str(), &isaMeta);

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    ClPrint(amd::LOG_ERROR, amd::LOG_INIT, "getIsaMeta(%s) failed!", isa.c_str());
    return false;
  }

  amd_comgr_metadata_node_t valMeta;
  size_t size = 0;
  status = amd::Comgr::metadata_lookup(isaMeta, key, &valMeta);
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::get_metadata_string(valMeta, &size, NULL);
  }
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    retValue.resize(size - 1);
    status = amd::Comgr::get_metadata_string(valMeta, &size, &(retValue[0]));
  }

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = amd::Comgr::destroy_metadata(valMeta);
  }
  amd::Comgr::destroy_metadata(isaMeta);
  return (status == AMD_COMGR_STATUS_SUCCESS) ? true : false;
}

// ================================================================================================
static amd_comgr_status_t populateArgs(const amd_comgr_metadata_node_t key,
                                       const amd_comgr_metadata_node_t value, void* data) {
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t kind;
  std::string buf;

  // get the key of the argument field
  size_t size = 0;
  status = amd::Comgr::get_metadata_kind(key, &kind);
  if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(key, &buf);
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  ArgField itArgField = amd::Kernel::FindValue<ArgField>(amd::Kernel::kArgFieldMap, buf);
  if (itArgField == ArgField::MaxSize) {
    return AMD_COMGR_STATUS_ERROR;
  }

  // get the value of the argument field
  status = getMetaBuf(value, &buf);

  amd::KernelParameterDescriptor* lcArg = static_cast<amd::KernelParameterDescriptor*>(data);

  switch (itArgField) {
    case ArgField::Name:
      lcArg->name_ = buf;
      break;
    case ArgField::TypeName:
      lcArg->typeName_ = buf;
      break;
    case ArgField::Size:
      lcArg->size_ = atoi(buf.c_str());
      break;
    case ArgField::Align:
      lcArg->alignment_ = atoi(buf.c_str());
      break;
    case ArgField::ValueKind: {
      amd::KernelParameterDescriptor::Desc itValueKind =
          amd::Kernel::FindValue<amd::KernelParameterDescriptor::Desc>(amd::Kernel::kArgValueKind,
                                                                       buf);
      if (itValueKind == amd::KernelParameterDescriptor::Desc::MaxSize) {
        lcArg->info_.hidden_ = true;
        return AMD_COMGR_STATUS_ERROR;
      }
      lcArg->info_.oclObject_ = itValueKind;
      switch (lcArg->info_.oclObject_) {
        case amd::KernelParameterDescriptor::MemoryObject:
          if (buf.compare("DynamicSharedPointer") == 0) {
            lcArg->info_.shared_ = true;
          }
          break;
        case amd::KernelParameterDescriptor::HiddenGlobalOffsetX:
        case amd::KernelParameterDescriptor::HiddenGlobalOffsetY:
        case amd::KernelParameterDescriptor::HiddenGlobalOffsetZ:
        case amd::KernelParameterDescriptor::HiddenPrintfBuffer:
        case amd::KernelParameterDescriptor::HiddenHostcallBuffer:
        case amd::KernelParameterDescriptor::HiddenDefaultQueue:
        case amd::KernelParameterDescriptor::HiddenCompletionAction:
        case amd::KernelParameterDescriptor::HiddenMultiGridSync:
        case amd::KernelParameterDescriptor::HiddenDynamicLdsSize:
        case amd::KernelParameterDescriptor::HiddenNone:
          lcArg->info_.hidden_ = true;
          break;
      }
    } break;
    case ArgField::PointeeAlign:
      lcArg->info_.arrayIndex_ = atoi(buf.c_str());
      break;
    case ArgField::AddrSpaceQual: {
      cl_int itAddrSpaceQual = amd::Kernel::FindValue(amd::Kernel::kArgAddrSpaceQual, buf);
      if (itAddrSpaceQual == static_cast<cl_int>(0)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      lcArg->addressQualifier_ = itAddrSpaceQual;
    } break;
    case ArgField::AccQual: {
      cl_int itAccQual = amd::Kernel::FindValue(amd::Kernel::kArgAccQual, buf);
      if (itAccQual == static_cast<cl_int>(0)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      lcArg->accessQualifier_ = itAccQual;
      lcArg->info_.readOnly_ =
          (lcArg->accessQualifier_ == CL_KERNEL_ARG_ACCESS_READ_ONLY) ? true : false;
    } break;
    case ArgField::ActualAccQual: {
      cl_int itAccQual = amd::Kernel::FindValue(amd::Kernel::kArgAccQual, buf);
      if (itAccQual == static_cast<cl_int>(0)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      // lcArg->mActualAccQual = itAccQual->second;
    } break;
    case ArgField::IsConst:
      lcArg->typeQualifier_ |= (buf.compare("true") == 0) ? CL_KERNEL_ARG_TYPE_CONST : 0;
      break;
    case ArgField::IsRestrict:
      lcArg->typeQualifier_ |= (buf.compare("true") == 0) ? CL_KERNEL_ARG_TYPE_RESTRICT : 0;
      break;
    case ArgField::IsVolatile:
      lcArg->typeQualifier_ |= (buf.compare("true") == 0) ? CL_KERNEL_ARG_TYPE_VOLATILE : 0;
      break;
    case ArgField::IsPipe:
      lcArg->typeQualifier_ |= (buf.compare("true") == 0) ? CL_KERNEL_ARG_TYPE_PIPE : 0;
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t populateAttrs(const amd_comgr_metadata_node_t key,
                                        const amd_comgr_metadata_node_t value, void* data) {
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t kind;
  size_t size = 0;
  std::string buf;

  // get the key of the argument field
  status = amd::Comgr::get_metadata_kind(key, &kind);
  if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(key, &buf);
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  AttrField itAttrField = amd::Kernel::FindValue<AttrField>(amd::Kernel::kAttrFieldMap, buf);
  if (itAttrField == AttrField::MaxSize) {
    return AMD_COMGR_STATUS_ERROR;
  }

  device::Kernel* kernel = static_cast<device::Kernel*>(data);
  switch (itAttrField) {
    case AttrField::ReqdWorkGroupSize: {
      status = amd::Comgr::get_metadata_list_size(value, &size);
      if (size == 3 && status == AMD_COMGR_STATUS_SUCCESS) {
        std::vector<size_t> wrkSize;
        for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
          amd_comgr_metadata_node_t workgroupSize;
          status = amd::Comgr::index_list_metadata(value, i, &workgroupSize);

          if (status == AMD_COMGR_STATUS_SUCCESS &&
              getMetaBuf(workgroupSize, &buf) == AMD_COMGR_STATUS_SUCCESS) {
            wrkSize.push_back(atoi(buf.c_str()));
          }
          amd::Comgr::destroy_metadata(workgroupSize);
        }
        if (!wrkSize.empty()) {
          kernel->setReqdWorkGroupSize(wrkSize[0], wrkSize[1], wrkSize[2]);
        }
      }
    } break;
    case AttrField::WorkGroupSizeHint: {
      status = amd::Comgr::get_metadata_list_size(value, &size);
      if (status == AMD_COMGR_STATUS_SUCCESS && size == 3) {
        std::vector<size_t> hintSize;
        for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
          amd_comgr_metadata_node_t workgroupSizeHint;
          status = amd::Comgr::index_list_metadata(value, i, &workgroupSizeHint);

          if (status == AMD_COMGR_STATUS_SUCCESS &&
              getMetaBuf(workgroupSizeHint, &buf) == AMD_COMGR_STATUS_SUCCESS) {
            hintSize.push_back(atoi(buf.c_str()));
          }
          amd::Comgr::destroy_metadata(workgroupSizeHint);
        }
        if (!hintSize.empty()) {
          kernel->setWorkGroupSizeHint(hintSize[0], hintSize[1], hintSize[2]);
        }
      }
    } break;
    case AttrField::VecTypeHint:
      if (getMetaBuf(value, &buf) == AMD_COMGR_STATUS_SUCCESS) {
        kernel->setVecTypeHint(buf);
      }
      break;
    case AttrField::RuntimeHandle:
      if (getMetaBuf(value, &buf) == AMD_COMGR_STATUS_SUCCESS) {
        kernel->setRuntimeHandle(buf);
      }
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }

  return status;
}

static amd_comgr_status_t populateCodeProps(const amd_comgr_metadata_node_t key,
                                            const amd_comgr_metadata_node_t value, void* data) {
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t kind;
  std::string buf;

  // get the key of the argument field
  status = amd::Comgr::get_metadata_kind(key, &kind);
  if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(key, &buf);
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  CodePropField itCodePropField =
      amd::Kernel::FindValue<CodePropField>(amd::Kernel::kCodePropFieldMap, buf);
  if (itCodePropField == CodePropField::MaxSize) {
    return AMD_COMGR_STATUS_ERROR;
  }

  // get the value of the argument field
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(value, &buf);
  }

  device::Kernel* kernel = static_cast<device::Kernel*>(data);
  switch (itCodePropField) {
    case CodePropField::KernargSegmentSize:
      kernel->SetKernargSegmentByteSize(atoi(buf.c_str()));
      break;
    case CodePropField::GroupSegmentFixedSize:
      kernel->SetWorkgroupGroupSegmentByteSize(atoi(buf.c_str()));
      break;
    case CodePropField::PrivateSegmentFixedSize:
      kernel->SetWorkitemPrivateSegmentByteSize(atoi(buf.c_str()));
      break;
    case CodePropField::KernargSegmentAlign:
      kernel->SetKernargSegmentAlignment(atoi(buf.c_str()));
      break;
    case CodePropField::WavefrontSize:
      kernel->workGroupInfo()->wavefrontSize_ = atoi(buf.c_str());
      break;
    case CodePropField::NumSGPRs:
      kernel->workGroupInfo()->usedSGPRs_ = atoi(buf.c_str());
      break;
    case CodePropField::NumVGPRs:
      kernel->workGroupInfo()->usedVGPRs_ = atoi(buf.c_str());
      break;
    case CodePropField::MaxFlatWorkGroupSize:
      kernel->workGroupInfo()->size_ = atoi(buf.c_str());
      break;
    case CodePropField::IsDynamicCallStack: {
      size_t mIsDynamicCallStack = (buf.compare("true") == 0);
    } break;
    case CodePropField::IsXNACKEnabled: {
      size_t mIsXNACKEnabled = (buf.compare("true") == 0);
    } break;
    case CodePropField::NumSpilledSGPRs: {
      size_t mNumSpilledSGPRs = atoi(buf.c_str());
    } break;
    case CodePropField::NumSpilledVGPRs: {
      size_t mNumSpilledVGPRs = atoi(buf.c_str());
    } break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t populateArgsV3(const amd_comgr_metadata_node_t key,
                                         const amd_comgr_metadata_node_t value, void* data) {
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t kind;
  std::string buf;

  // get the key of the argument field
  size_t size = 0;
  status = amd::Comgr::get_metadata_kind(key, &kind);
  if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(key, &buf);
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  ArgField itArgField = amd::Kernel::FindValue<ArgField>(amd::Kernel::kArgFieldMapV3, buf);
  if (itArgField == ArgField::MaxSize) {
    return AMD_COMGR_STATUS_ERROR;
  }

  // get the value of the argument field
  status = getMetaBuf(value, &buf);

  amd::KernelParameterDescriptor* lcArg = static_cast<amd::KernelParameterDescriptor*>(data);

  switch (itArgField) {
    case ArgField::Name:
      lcArg->name_ = buf;
      break;
    case ArgField::TypeName:
      lcArg->typeName_ = buf;
      break;
    case ArgField::Size:
      lcArg->size_ = atoi(buf.c_str());
      break;
    case ArgField::Offset:
      lcArg->offset_ = atoi(buf.c_str());
      break;
    case ArgField::ValueKind: {
      amd::KernelParameterDescriptor::Desc itArgValue =
          amd::Kernel::FindValue<amd::KernelParameterDescriptor::Desc>(amd::Kernel::kArgValueKindV3,
                                                                       buf);
      if (itArgValue == amd::KernelParameterDescriptor::MaxSize) {
        LogPrintfError("Unknown Kernel arg metadata: %s", buf.c_str());
        LogError("This may be due to running HIP app that requires a new HIP runtime version");
        LogError("Please update the display driver");
        return AMD_COMGR_STATUS_ERROR;
      }
      lcArg->info_.oclObject_ = itArgValue;
      if (lcArg->info_.oclObject_ == amd::KernelParameterDescriptor::MemoryObject) {
        if (buf.compare("dynamic_shared_pointer") == 0) {
          lcArg->info_.shared_ = true;
        }
      } else if ((lcArg->info_.oclObject_ >= amd::KernelParameterDescriptor::HiddenNone) &&
                 (lcArg->info_.oclObject_ < amd::KernelParameterDescriptor::HiddenLast)) {
        lcArg->info_.hidden_ = true;
      }
    } break;
    case ArgField::PointeeAlign:
      lcArg->info_.arrayIndex_ = atoi(buf.c_str());
      break;
    case ArgField::AddrSpaceQual: {
      cl_int itAddrSpaceQual = amd::Kernel::FindValue(amd::Kernel::kArgAddrSpaceQualV3, buf);
      if (itAddrSpaceQual == static_cast<cl_int>(0)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      lcArg->addressQualifier_ = itAddrSpaceQual;
    } break;
    case ArgField::AccQual: {
      cl_int itAccQual = amd::Kernel::FindValue(amd::Kernel::kArgAccQualV3, buf);
      if (itAccQual == static_cast<cl_int>(0)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      lcArg->accessQualifier_ = itAccQual;
      if (!lcArg->info_.isReadOnlyByCompiler) {
        lcArg->info_.readOnly_ =
            (lcArg->accessQualifier_ == CL_KERNEL_ARG_ACCESS_READ_ONLY) ? true : false;
      }
    } break;
    case ArgField::ActualAccQual: {
      cl_int itAccQual = amd::Kernel::FindValue(amd::Kernel::kArgAccQualV3, buf);
      if (itAccQual == static_cast<cl_int>(0)) {
        return AMD_COMGR_STATUS_ERROR;
      }
      lcArg->info_.isReadOnlyByCompiler = true;
      lcArg->info_.readOnly_ = (itAccQual == CL_KERNEL_ARG_ACCESS_READ_ONLY) ? true : false;
    } break;
    case ArgField::IsConst:
      lcArg->typeQualifier_ |= (buf.compare("1") == 0) ? CL_KERNEL_ARG_TYPE_CONST : 0;
      break;
    case ArgField::IsRestrict:
      lcArg->typeQualifier_ |= (buf.compare("1") == 0) ? CL_KERNEL_ARG_TYPE_RESTRICT : 0;
      break;
    case ArgField::IsVolatile:
      lcArg->typeQualifier_ |= (buf.compare("1") == 0) ? CL_KERNEL_ARG_TYPE_VOLATILE : 0;
      break;
    case ArgField::IsPipe:
      lcArg->typeQualifier_ |= (buf.compare("1") == 0) ? CL_KERNEL_ARG_TYPE_PIPE : 0;
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t populateKernelMetaV3(const amd_comgr_metadata_node_t key,
                                               const amd_comgr_metadata_node_t value, void* data) {
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t kind;
  size_t size = 0;
  std::string buf;
  // get the key of the argument field
  status = amd::Comgr::get_metadata_kind(key, &kind);
  if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
    status = getMetaBuf(key, &buf);
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  KernelField itKernelField =
      amd::Kernel::FindValue<KernelField>(amd::Kernel::kKernelFieldMapV3, buf);
  if (itKernelField == KernelField::MaxSize) {
    return AMD_COMGR_STATUS_ERROR;
  }

  if (itKernelField != KernelField::ReqdWorkGroupSize &&
      itKernelField != KernelField::WorkGroupSizeHint) {
    status = getMetaBuf(value, &buf);
  }
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return AMD_COMGR_STATUS_ERROR;
  }

  device::Kernel* kernel = static_cast<device::Kernel*>(data);
  switch (itKernelField) {
    case KernelField::ReqdWorkGroupSize:
      status = amd::Comgr::get_metadata_list_size(value, &size);
      if (size == 3 && status == AMD_COMGR_STATUS_SUCCESS) {
        std::vector<size_t> wrkSize;
        for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
          amd_comgr_metadata_node_t workgroupSize;
          status = amd::Comgr::index_list_metadata(value, i, &workgroupSize);

          if (status == AMD_COMGR_STATUS_SUCCESS &&
              getMetaBuf(workgroupSize, &buf) == AMD_COMGR_STATUS_SUCCESS) {
            wrkSize.push_back(atoi(buf.c_str()));
          }
          amd::Comgr::destroy_metadata(workgroupSize);
        }
        if (!wrkSize.empty()) {
          kernel->setReqdWorkGroupSize(wrkSize[0], wrkSize[1], wrkSize[2]);
        }
      }
      break;
    case KernelField::WorkGroupSizeHint:
      status = amd::Comgr::get_metadata_list_size(value, &size);
      if (status == AMD_COMGR_STATUS_SUCCESS && size == 3) {
        std::vector<size_t> hintSize;
        for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
          amd_comgr_metadata_node_t workgroupSizeHint;
          status = amd::Comgr::index_list_metadata(value, i, &workgroupSizeHint);

          if (status == AMD_COMGR_STATUS_SUCCESS &&
              getMetaBuf(workgroupSizeHint, &buf) == AMD_COMGR_STATUS_SUCCESS) {
            hintSize.push_back(atoi(buf.c_str()));
          }
          amd::Comgr::destroy_metadata(workgroupSizeHint);
        }
        if (!hintSize.empty()) {
          kernel->setWorkGroupSizeHint(hintSize[0], hintSize[1], hintSize[2]);
        }
      }
      break;
    case KernelField::VecTypeHint:
      kernel->setVecTypeHint(buf);
      break;
    case KernelField::DeviceEnqueueSymbol:
      kernel->setRuntimeHandle(buf);
      break;
    case KernelField::KernargSegmentSize:
      kernel->SetKernargSegmentByteSize(atoi(buf.c_str()));
      break;
    case KernelField::GroupSegmentFixedSize:
      kernel->SetWorkgroupGroupSegmentByteSize(atoi(buf.c_str()));
      break;
    case KernelField::PrivateSegmentFixedSize:
      kernel->SetWorkitemPrivateSegmentByteSize(atoi(buf.c_str()));
      break;
    case KernelField::KernargSegmentAlign:
      kernel->SetKernargSegmentAlignment(atoi(buf.c_str()));
      break;
    case KernelField::WavefrontSize:
      kernel->workGroupInfo()->wavefrontSize_ = atoi(buf.c_str());
      break;
    case KernelField::NumSGPRs:
      kernel->workGroupInfo()->usedSGPRs_ = atoi(buf.c_str());
      break;
    case KernelField::NumVGPRs:
      kernel->workGroupInfo()->usedVGPRs_ = atoi(buf.c_str());
      break;
    case KernelField::MaxFlatWorkGroupSize:
      kernel->workGroupInfo()->size_ = atoi(buf.c_str());
      break;
    case KernelField::NumSpilledSGPRs: {
      size_t mNumSpilledSGPRs = atoi(buf.c_str());
    } break;
    case KernelField::NumSpilledVGPRs: {
      size_t mNumSpilledVGPRs = atoi(buf.c_str());
    } break;
    case KernelField::SymbolName:
      kernel->SetSymbolName(buf);
      break;
    case KernelField::Kind:
      kernel->SetKernelKind(buf);
      break;
    case KernelField::WgpMode:
      // The compiler currently serializes this boolean field as "0"/"1" instead
      // of "false"/"true"; consider both "true" and "1" truthy values.
      kernel->SetWGPMode(buf.compare("true") == 0 || buf.compare("1") == 0);
      break;
    case KernelField::UniformWrokGroupSize:
      kernel->setUniformWorkGroupSize(buf.compare("1") == 0);
      break;
    default:
      return AMD_COMGR_STATUS_ERROR;
  }

  return status;
}
#endif

// ================================================================================================
Kernel::Kernel(const amd::Device& dev, const std::string& name, const Program& prog)
    : dev_(dev), name_(name), prog_(prog), signature_(nullptr) {
  // Instead of memset(&workGroupInfo_, '\0', sizeof(workGroupInfo_));
  // Due to std::string not being able to be memset to 0
  workGroupInfo_.size_ = 0;
  workGroupInfo_.compileSize_[0] = 0;
  workGroupInfo_.compileSize_[1] = 0;
  workGroupInfo_.compileSize_[2] = 0;
  workGroupInfo_.localMemSize_ = 0;
  workGroupInfo_.preferredSizeMultiple_ = 0;
  workGroupInfo_.privateMemSize_ = 0;
  workGroupInfo_.scratchRegs_ = 0;
  workGroupInfo_.wavefrontPerSIMD_ = 0;
  workGroupInfo_.wavefrontSize_ = 0;
  workGroupInfo_.availableGPRs_ = 0;
  workGroupInfo_.usedGPRs_ = 0;
  workGroupInfo_.availableSGPRs_ = 0;
  workGroupInfo_.usedSGPRs_ = 0;
  workGroupInfo_.availableVGPRs_ = dev.info().availableVGPRs_;
  workGroupInfo_.usedVGPRs_ = 0;
  workGroupInfo_.availableLDSSize_ = 0;
  workGroupInfo_.usedLDSSize_ = 0;
  workGroupInfo_.availableStackSize_ = 0;
  workGroupInfo_.usedStackSize_ = 0;
  workGroupInfo_.compileSizeHint_[0] = 0;
  workGroupInfo_.compileSizeHint_[1] = 0;
  workGroupInfo_.compileSizeHint_[2] = 0;
  workGroupInfo_.compileVecTypeHint_ = "";
  workGroupInfo_.isWGPMode_ = false;
  workGroupInfo_.uniformWorkGroupSize_ = false;
  workGroupInfo_.wavesPerSimdHint_ = 0;
  workGroupInfo_.constMemSize_ = 0;
  workGroupInfo_.maxDynamicSharedSizeBytes_ = 0;
}

// ================================================================================================
bool Kernel::createSignature(const parameters_t& params, uint32_t numParameters, uint32_t version) {
  std::stringstream attribs;
  if (workGroupInfo_.compileSize_[0] != 0) {
    attribs << "reqd_work_group_size(";
    for (size_t i = 0; i < 3; ++i) {
      if (i != 0) {
        attribs << ",";
      }

      attribs << workGroupInfo_.compileSize_[i];
    }
    attribs << ")";
  }
  if (workGroupInfo_.compileSizeHint_[0] != 0) {
    attribs << " work_group_size_hint(";
    for (size_t i = 0; i < 3; ++i) {
      if (i != 0) {
        attribs << ",";
      }

      attribs << workGroupInfo_.compileSizeHint_[i];
    }
    attribs << ")";
  }

  if (!workGroupInfo_.compileVecTypeHint_.empty()) {
    attribs << " vec_type_hint(" << workGroupInfo_.compileVecTypeHint_ << ")";
  }

  // Destroy old signature if it was allocated before
  // (offline devices path)
  delete signature_;
  signature_ = new amd::KernelSignature(params, attribs.str(), numParameters, version);
  if (NULL != signature_) {
    return true;
  }
  return false;
}

// ================================================================================================
Kernel::~Kernel() { delete signature_; }

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
std::string Kernel::openclMangledName(const std::string& name) {
  const oclBIFSymbolStruct* bifSym = findBIF30SymStruct(symOpenclKernel);
  assert(bifSym && "symbol not found");
  return std::string("&") + bifSym->str[bif::PRE] + name + bifSym->str[bif::POST];
}
#endif

// ================================================================================================
void Kernel::FindLocalWorkSize(size_t workDim, const amd::NDRange& gblWorkSize,
                               amd::NDRange& lclWorkSize) const {
  // Initialize the default workgoup info
  // Check if the kernel has the compiled sizes
  if (workGroupInfo()->compileSize_[0] == 0) {
    // Find the default local workgroup size, if it wasn't specified
    if (lclWorkSize[0] == 0) {
      // Find threads per group
      size_t thrPerGrp = workGroupInfo()->size_;

      // Check if kernel uses images
      if (flags_.imageEna_ &&
          // and thread group is a multiple value of wavefronts
          ((thrPerGrp % workGroupInfo()->wavefrontSize_) == 0) &&
          // and it's 2 or 3-dimensional workload
          (workDim > 1) && (((gblWorkSize[0] % 16) == 0) && ((gblWorkSize[1] % 16) == 0))) {
        // Use 8x8 workgroup size if kernel has image writes
        if (flags_.imageWriteEna_ || (thrPerGrp != device().info().preferredWorkGroupSize_)) {
          lclWorkSize[0] = 8;
          lclWorkSize[1] = 8;
        } else {
          lclWorkSize[0] = 16;
          lclWorkSize[1] = 16;
        }
        if (workDim == 3) {
          lclWorkSize[2] = 1;
        }
      } else {
        size_t tmp = thrPerGrp;
        // Split the local workgroup into the most efficient way
        for (uint d = 0; d < workDim; ++d) {
          size_t div = tmp;
          for (; (gblWorkSize[d] % div) != 0; div--);
          lclWorkSize[d] = div;
          tmp /= div;
        }

        if (!workGroupInfo()->uniformWorkGroupSize_) {
          // Assuming DWORD access
          const uint cacheLineMatch = device().info().globalMemCacheLineSize_ >> 2;

          // Check if we couldn't find optimal workload
          if (((lclWorkSize.product() % workGroupInfo()->wavefrontSize_) != 0) ||
              // or size is too small for the cache line
              (lclWorkSize[0] < cacheLineMatch)) {
            size_t maxSize = 0;
            size_t maxDim = 0;
            for (uint d = 0; d < workDim; ++d) {
              if (maxSize < gblWorkSize[d]) {
                maxSize = gblWorkSize[d];
                maxDim = d;
              }
            }
            // Use X dimension as high priority. Runtime will assume that
            // X dimension is more important for the address calculation
            if ((maxDim != 0) && (gblWorkSize[0] >= (cacheLineMatch / 2))) {
              lclWorkSize[0] = cacheLineMatch;
              thrPerGrp /= cacheLineMatch;
              lclWorkSize[maxDim] = thrPerGrp;
              for (uint d = 1; d < workDim; ++d) {
                if (d != maxDim) {
                  lclWorkSize[d] = 1;
                }
              }
            } else {
              // Check if a local workgroup has the most optimal size
              if (thrPerGrp > maxSize) {
                thrPerGrp = maxSize;
              }
              lclWorkSize[maxDim] = thrPerGrp;
              for (uint d = 0; d < workDim; ++d) {
                if (d != maxDim) {
                  lclWorkSize[d] = 1;
                }
              }
            }
          }
        }
      }
    }
  } else {
    for (uint d = 0; d < workDim; ++d) {
      lclWorkSize[d] = workGroupInfo()->compileSize_[d];
    }
  }
}

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
static inline uint32_t GetOclArgumentTypeOCL(const aclArgData* argInfo, bool* isHidden) {
  if (argInfo->argStr[0] == '_' && argInfo->argStr[1] == '.') {
    *isHidden = true;
    if (strcmp(&argInfo->argStr[2], "global_offset_0") == 0) {
      return amd::KernelParameterDescriptor::HiddenGlobalOffsetX;
    } else if (strcmp(&argInfo->argStr[2], "global_offset_1") == 0) {
      return amd::KernelParameterDescriptor::HiddenGlobalOffsetY;
    } else if (strcmp(&argInfo->argStr[2], "global_offset_2") == 0) {
      return amd::KernelParameterDescriptor::HiddenGlobalOffsetZ;
    } else if (strcmp(&argInfo->argStr[2], "printf_buffer") == 0) {
      return amd::KernelParameterDescriptor::HiddenPrintfBuffer;
    } else if (strcmp(&argInfo->argStr[2], "hostcall_buffer") == 0) {
      return amd::KernelParameterDescriptor::HiddenHostcallBuffer;
    } else if (strcmp(&argInfo->argStr[2], "vqueue_pointer") == 0) {
      return amd::KernelParameterDescriptor::HiddenDefaultQueue;
    } else if (strcmp(&argInfo->argStr[2], "aqlwrap_pointer") == 0) {
      return amd::KernelParameterDescriptor::HiddenCompletionAction;
    }
    return amd::KernelParameterDescriptor::HiddenNone;
  }
  switch (argInfo->type) {
    case ARG_TYPE_POINTER:
      return amd::KernelParameterDescriptor::MemoryObject;
    case ARG_TYPE_QUEUE:
      return amd::KernelParameterDescriptor::QueueObject;
    case ARG_TYPE_VALUE:
      return (argInfo->arg.value.data == DATATYPE_struct)
                 ? amd::KernelParameterDescriptor::ReferenceObject
                 : amd::KernelParameterDescriptor::ValueObject;
    case ARG_TYPE_IMAGE:
      return amd::KernelParameterDescriptor::ImageObject;
    case ARG_TYPE_SAMPLER:
      return amd::KernelParameterDescriptor::SamplerObject;
    case ARG_TYPE_ERROR:
    default:
      return amd::KernelParameterDescriptor::HiddenNone;
  }
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
static inline clk_value_type_t GetOclTypeOCL(const aclArgData* argInfo, size_t size = 0) {
  uint sizeType;
  uint numElements;
  if (argInfo->type == ARG_TYPE_QUEUE) {
    return T_QUEUE;
  } else if (argInfo->type == ARG_TYPE_POINTER || argInfo->type == ARG_TYPE_IMAGE) {
    return T_POINTER;
  } else if (argInfo->type == ARG_TYPE_VALUE) {
    switch (argInfo->arg.value.data) {
      case DATATYPE_i8:
      case DATATYPE_u8:
        sizeType = 0;
        numElements = size;
        break;
      case DATATYPE_i16:
      case DATATYPE_u16:
        sizeType = 1;
        numElements = size / 2;
        break;
      case DATATYPE_i32:
      case DATATYPE_u32:
        sizeType = 2;
        numElements = size / 4;
        break;
      case DATATYPE_i64:
      case DATATYPE_u64:
        sizeType = 3;
        numElements = size / 8;
        break;
      case DATATYPE_f16:
        sizeType = 4;
        numElements = size / 2;
        break;
      case DATATYPE_f32:
        sizeType = 4;
        numElements = size / 4;
        break;
      case DATATYPE_f64:
        sizeType = 5;
        numElements = size / 8;
        break;
      case DATATYPE_struct:
      case DATATYPE_opaque:
      case DATATYPE_ERROR:
      default:
        return T_VOID;
    }

    switch (numElements) {
      case 1:
        return ClkValueMapType[sizeType][0];
      case 2:
        return ClkValueMapType[sizeType][1];
      case 3:
        return ClkValueMapType[sizeType][2];
      case 4:
        return ClkValueMapType[sizeType][3];
      case 8:
        return ClkValueMapType[sizeType][4];
      case 16:
        return ClkValueMapType[sizeType][5];
      default:
        return T_VOID;
    }
  } else if (argInfo->type == ARG_TYPE_SAMPLER) {
    return T_SAMPLER;
  } else {
    return T_VOID;
  }
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
static inline size_t GetArgAlignmentOCL(const aclArgData* argInfo) {
  switch (argInfo->type) {
    case ARG_TYPE_POINTER:
      return sizeof(void*);
    case ARG_TYPE_VALUE:
      switch (argInfo->arg.value.data) {
        case DATATYPE_i8:
        case DATATYPE_u8:
          return 1;
        case DATATYPE_u16:
        case DATATYPE_i16:
        case DATATYPE_f16:
          return 2;
        case DATATYPE_u32:
        case DATATYPE_i32:
        case DATATYPE_f32:
          return 4;
        case DATATYPE_i64:
        case DATATYPE_u64:
        case DATATYPE_f64:
          return 8;
        case DATATYPE_struct:
          return 128;
        case DATATYPE_ERROR:
        default:
          return -1;
      }
    case ARG_TYPE_IMAGE:
      return sizeof(cl_mem);
    case ARG_TYPE_SAMPLER:
      return sizeof(cl_sampler);
    default:
      return -1;
  }
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
static inline size_t GetArgPointeeAlignmentOCL(const aclArgData* argInfo) {
  if (argInfo->type == ARG_TYPE_POINTER) {
    return argInfo->arg.pointer.align;
  }
  return 1;
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
static inline bool GetReadOnlyOCL(const aclArgData* argInfo) {
  if (argInfo->type == ARG_TYPE_POINTER) {
    return (argInfo->arg.pointer.type == ACCESS_TYPE_RO) ? true : false;
  } else if (argInfo->type == ARG_TYPE_IMAGE) {
    return (argInfo->arg.image.type == ACCESS_TYPE_RO) ? true : false;
  }
  return false;
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
inline static int GetArgSizeOCL(const aclArgData* argInfo) {
  switch (argInfo->type) {
    case ARG_TYPE_POINTER:
      return sizeof(void*);
    case ARG_TYPE_VALUE:
      switch (argInfo->arg.value.data) {
        case DATATYPE_i8:
        case DATATYPE_u8:
        case DATATYPE_struct:
          return 1 * argInfo->arg.value.numElements;
        case DATATYPE_u16:
        case DATATYPE_i16:
        case DATATYPE_f16:
          return 2 * argInfo->arg.value.numElements;
        case DATATYPE_u32:
        case DATATYPE_i32:
        case DATATYPE_f32:
          return 4 * argInfo->arg.value.numElements;
        case DATATYPE_i64:
        case DATATYPE_u64:
        case DATATYPE_f64:
          return 8 * argInfo->arg.value.numElements;
        case DATATYPE_ERROR:
        default:
          return -1;
      }
    case ARG_TYPE_IMAGE:
    case ARG_TYPE_SAMPLER:
    case ARG_TYPE_QUEUE:
      return sizeof(void*);
    default:
      return -1;
  }
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
static inline cl_kernel_arg_address_qualifier GetOclAddrQualOCL(const aclArgData* argInfo) {
  if (argInfo->type == ARG_TYPE_POINTER) {
    switch (argInfo->arg.pointer.memory) {
      case PTR_MT_UAV_CONSTANT:
      case PTR_MT_CONSTANT_EMU:
      case PTR_MT_CONSTANT:
        return CL_KERNEL_ARG_ADDRESS_CONSTANT;
      case PTR_MT_UAV:
      case PTR_MT_GLOBAL:
      case PTR_MT_SCRATCH_EMU:
        return CL_KERNEL_ARG_ADDRESS_GLOBAL;
      case PTR_MT_LDS_EMU:
      case PTR_MT_LDS:
        return CL_KERNEL_ARG_ADDRESS_LOCAL;
      case PTR_MT_ERROR:
      default:
        LogError("Unsupported address type");
        return CL_KERNEL_ARG_ADDRESS_PRIVATE;
    }
  } else if ((argInfo->type == ARG_TYPE_IMAGE) || (argInfo->type == ARG_TYPE_QUEUE)) {
    return CL_KERNEL_ARG_ADDRESS_GLOBAL;
  }

  // default for all other cases
  return CL_KERNEL_ARG_ADDRESS_PRIVATE;
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
static inline cl_kernel_arg_access_qualifier GetOclAccessQualOCL(const aclArgData* argInfo) {
  if (argInfo->type == ARG_TYPE_IMAGE) {
    switch (argInfo->arg.image.type) {
      case ACCESS_TYPE_RO:
        return CL_KERNEL_ARG_ACCESS_READ_ONLY;
      case ACCESS_TYPE_WO:
        return CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
      default:
        return CL_KERNEL_ARG_ACCESS_READ_WRITE;
    }
  }
  return CL_KERNEL_ARG_ACCESS_NONE;
}
#endif

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
static inline cl_kernel_arg_type_qualifier GetOclTypeQualOCL(const aclArgData* argInfo) {
  cl_kernel_arg_type_qualifier rv = CL_KERNEL_ARG_TYPE_NONE;
  if (argInfo->type == ARG_TYPE_POINTER) {
    if (argInfo->arg.pointer.isVolatile) {
      rv |= CL_KERNEL_ARG_TYPE_VOLATILE;
    }
    if (argInfo->arg.pointer.isRestrict) {
      rv |= CL_KERNEL_ARG_TYPE_RESTRICT;
    }
    if (argInfo->arg.pointer.isPipe) {
      rv |= CL_KERNEL_ARG_TYPE_PIPE;
    }
    if (argInfo->isConst) {
      rv |= CL_KERNEL_ARG_TYPE_CONST;
    }
    switch (argInfo->arg.pointer.memory) {
      case PTR_MT_CONSTANT:
      case PTR_MT_UAV_CONSTANT:
      case PTR_MT_CONSTANT_EMU:
        rv |= CL_KERNEL_ARG_TYPE_CONST;
        break;
      default:
        break;
    }
  }
  return rv;
}
#endif

// ================================================================================================
#if defined(USE_COMGR_LIBRARY)
bool Kernel::GetAttrCodePropMetadata() {
  amd_comgr_metadata_node_t kernelMetaNode;
  if (!prog().getKernelMetadata(name(), &kernelMetaNode)) {
    DevLogPrintfError("Cannot get program kernel metadata for %s \n", name().c_str());
    return false;
  }

  // Set the workgroup information for the kernel
  workGroupInfo_.availableLDSSize_ = device().info().localMemSizePerCU_;
  workGroupInfo_.availableSGPRs_ = 104;
  workGroupInfo_.availableVGPRs_ = 256;

  // extract the attribute metadata if there is any
  amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;

  switch (codeObjectVer()) {
    case 2: {
      amd_comgr_metadata_node_t symbolName;
      status = amd::Comgr::metadata_lookup(kernelMetaNode, "SymbolName", &symbolName);
      if (status == AMD_COMGR_STATUS_SUCCESS) {
        std::string name;
        status = getMetaBuf(symbolName, &name);
        amd::Comgr::destroy_metadata(symbolName);
        SetSymbolName(name);
      }

      amd_comgr_metadata_node_t attrMeta;
      if (status == AMD_COMGR_STATUS_SUCCESS) {
        if (amd::Comgr::metadata_lookup(kernelMetaNode, "Attrs", &attrMeta) ==
            AMD_COMGR_STATUS_SUCCESS) {
          status =
              amd::Comgr::iterate_map_metadata(attrMeta, populateAttrs, static_cast<void*>(this));
          amd::Comgr::destroy_metadata(attrMeta);
        }
      }

      // extract the code properties metadata
      amd_comgr_metadata_node_t codePropsMeta;
      if (status == AMD_COMGR_STATUS_SUCCESS) {
        status = amd::Comgr::metadata_lookup(kernelMetaNode, "CodeProps", &codePropsMeta);
      }

      if (status == AMD_COMGR_STATUS_SUCCESS) {
        status = amd::Comgr::iterate_map_metadata(codePropsMeta, populateCodeProps,
                                                  static_cast<void*>(this));
        amd::Comgr::destroy_metadata(codePropsMeta);
      }
    } break;
    default:
      status = amd::Comgr::iterate_map_metadata(kernelMetaNode, populateKernelMetaV3,
                                                static_cast<void*>(this));
  }

  if (status != AMD_COMGR_STATUS_SUCCESS) {
    LogError("Comgr Api failed with Status: \n");
    return false;
  }

  InitParameters(kernelMetaNode);

  return true;
}

bool Kernel::GetPrintfStr(std::vector<std::string>* printfStr) {
  const amd_comgr_metadata_node_t programMD = prog().metadata();
  amd_comgr_metadata_node_t printfMeta;

  amd_comgr_status_t status = amd::Comgr::metadata_lookup(
      programMD, codeObjectVer() == 2 ? "Printf" : "amdhsa.printf", &printfMeta);
  if (status != AMD_COMGR_STATUS_SUCCESS) {
    return true;  // printf string metadata is not provided so just exit
  }

  // handle the printf string
  size_t printfSize = 0;
  status = amd::Comgr::get_metadata_list_size(printfMeta, &printfSize);

  if (status == AMD_COMGR_STATUS_SUCCESS) {
    std::string buf;
    for (size_t i = 0; i < printfSize; ++i) {
      amd_comgr_metadata_node_t str;
      status = amd::Comgr::index_list_metadata(printfMeta, i, &str);

      if (status == AMD_COMGR_STATUS_SUCCESS) {
        status = getMetaBuf(str, &buf);
        amd::Comgr::destroy_metadata(str);
      }

      if (status != AMD_COMGR_STATUS_SUCCESS) {
        DevLogPrintfError("Comgr API failed with status: %d \n", status);
        amd::Comgr::destroy_metadata(printfMeta);
        return false;
      }

      printfStr->push_back(buf);
    }
  }

  amd::Comgr::destroy_metadata(printfMeta);
  return (status == AMD_COMGR_STATUS_SUCCESS);
}

void Kernel::InitParameters(const amd_comgr_metadata_node_t kernelMD) {
  // Iterate through the arguments and insert into parameterList
  device::Kernel::parameters_t params;
  device::Kernel::parameters_t hiddenParams;
  size_t offset = 0;

  amd_comgr_metadata_node_t argsMeta;
  bool hsaArgsMeta = false;
  size_t argsSize = 0;

  amd_comgr_status_t status =
      amd::Comgr::metadata_lookup(kernelMD, (codeObjectVer() == 2) ? "Args" : ".args", &argsMeta);
  // Assume no arguments if lookup fails.
  if (status == AMD_COMGR_STATUS_SUCCESS) {
    hsaArgsMeta = true;
    status = amd::Comgr::get_metadata_list_size(argsMeta, &argsSize);
  }

  for (size_t i = 0; i < argsSize; ++i) {
    amd::KernelParameterDescriptor desc = {};

    amd_comgr_metadata_node_t argsNode;
    amd_comgr_metadata_kind_t kind = AMD_COMGR_METADATA_KIND_NULL;
    bool hsaArgsNode = false;

    status = amd::Comgr::index_list_metadata(argsMeta, i, &argsNode);

    if (status == AMD_COMGR_STATUS_SUCCESS) {
      hsaArgsNode = true;
      status = amd::Comgr::get_metadata_kind(argsNode, &kind);
    }
    if (kind != AMD_COMGR_METADATA_KIND_MAP) {
      status = AMD_COMGR_STATUS_ERROR;
    }
    if (status == AMD_COMGR_STATUS_SUCCESS) {
      void* data = static_cast<void*>(&desc);
      if (codeObjectVer() == 2) {
        status = amd::Comgr::iterate_map_metadata(argsNode, populateArgs, data);
      } else if (codeObjectVer() >= 3) {
        status = amd::Comgr::iterate_map_metadata(argsNode, populateArgsV3, data);
      }
    }

    if (hsaArgsNode) {
      amd::Comgr::destroy_metadata(argsNode);
    }

    if (status != AMD_COMGR_STATUS_SUCCESS) {
      if (hsaArgsMeta) {
        amd::Comgr::destroy_metadata(argsMeta);
      }
      return;
    }

    // COMGR has unclear/undefined order of the fields filling.
    // Correct the types for the abstraciton layer after all fields are available
    if (desc.info_.oclObject_ != amd::KernelParameterDescriptor::ValueObject) {
      switch (desc.info_.oclObject_) {
        case amd::KernelParameterDescriptor::MemoryObject:
        case amd::KernelParameterDescriptor::ImageObject:
          desc.type_ = T_POINTER;
          if (desc.info_.shared_) {
            if (desc.info_.arrayIndex_ == 0) {
              LogWarning("Missing DynamicSharedPointer alignment");
              desc.info_.arrayIndex_ = 128; /* worst case alignment */
            }
          } else {
            desc.info_.arrayIndex_ = 1;
          }
          break;
        case amd::KernelParameterDescriptor::SamplerObject:
          desc.type_ = T_SAMPLER;
          desc.addressQualifier_ = CL_KERNEL_ARG_ADDRESS_PRIVATE;
          break;
        case amd::KernelParameterDescriptor::QueueObject:
          desc.type_ = T_QUEUE;
          break;
        default:
          desc.type_ = T_VOID;
          break;
      }
    }

    if ((desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) ||
        (desc.typeQualifier_ & CL_KERNEL_ARG_TYPE_PIPE)) {
      // LC doesn't report correct address qualifier for images and pipes,
      // hence overwrite it
      // We will remove this when newer LC is ready
      desc.addressQualifier_ = CL_KERNEL_ARG_ADDRESS_GLOBAL;
    } else {
      // According to CL spec, otherwise must be CL_KERNEL_ARG_ACCESS_NONE,
      desc.accessQualifier_ = CL_KERNEL_ARG_ACCESS_NONE;
    }

    size_t size = desc.size_;

    // Allocate the hidden arguments, but abstraction layer will skip them
    if (desc.info_.hidden_) {
      if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::HiddenCompletionAction &&
          !amd::IS_HIP) {
        setDynamicParallelFlag(true);
      }
      if (codeObjectVer() == 2) {
        desc.offset_ = amd::alignUp(offset, desc.alignment_);
        offset += size;
      }
      hiddenParams.push_back(desc);
      continue;
    }

    // These objects have forced data size to uint64_t
    if (codeObjectVer() == 2) {
      if ((desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) ||
          (desc.info_.oclObject_ == amd::KernelParameterDescriptor::SamplerObject) ||
          (desc.info_.oclObject_ == amd::KernelParameterDescriptor::QueueObject)) {
        offset = amd::alignUp(offset, sizeof(uint64_t));
        desc.offset_ = offset;
        offset += sizeof(uint64_t);
      } else {
        offset = amd::alignUp(offset, desc.alignment_);
        desc.offset_ = offset;
        offset += size;
      }
    }

    params.push_back(desc);

    if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) {
      flags_.imageEna_ = true;
      if (desc.accessQualifier_ != CL_KERNEL_ARG_ACCESS_READ_ONLY) {
        flags_.imageWriteEna_ = true;
      }
    }
  }

  if (hsaArgsMeta) {
    amd::Comgr::destroy_metadata(argsMeta);
  }

  // Save the number of OCL arguments
  uint32_t numParams = params.size();
  // Append the hidden arguments to the OCL arguments
  params.insert(params.end(), hiddenParams.begin(), hiddenParams.end());
  createSignature(params, numParams, amd::KernelSignature::ABIVersion_2);
}
#endif  // defined(USE_COMGR_LIBRARY)

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
void Kernel::InitParameters(const aclArgData* aclArg, uint32_t argBufferSize) {
  // Iterate through the arguments and insert into parameterList
  device::Kernel::parameters_t params;
  device::Kernel::parameters_t hiddenParams;
  amd::KernelParameterDescriptor desc;
  size_t offset = 0;
  size_t offsetStruct = argBufferSize;

  for (uint i = 0; aclArg->struct_size != 0; i++, aclArg++) {
    size_t size = GetArgSizeOCL(aclArg);
    size_t alignment = GetArgAlignmentOCL(aclArg);
    bool isHidden = false;
    desc.info_.oclObject_ = GetOclArgumentTypeOCL(aclArg, &isHidden);

    // Allocate the hidden arguments, but abstraction layer will skip them
    if (isHidden) {
      offset = amd::alignUp(offset, alignment);
      desc.offset_ = offset;
      desc.size_ = size;
      offset += size;
      hiddenParams.push_back(desc);
      continue;
    }

    desc.name_ = aclArg->argStr;
    desc.typeName_ = aclArg->typeStr;
    desc.type_ = GetOclTypeOCL(aclArg, size);

    desc.addressQualifier_ = GetOclAddrQualOCL(aclArg);
    desc.accessQualifier_ = GetOclAccessQualOCL(aclArg);
    desc.typeQualifier_ = GetOclTypeQualOCL(aclArg);
    desc.info_.arrayIndex_ = GetArgPointeeAlignmentOCL(aclArg);
    desc.size_ = size;

    // Check if HSAIL expects data by reference and allocate it behind
    if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ReferenceObject) {
      desc.offset_ = offsetStruct;
      // Align the offset reference
      offset = amd::alignUp(offset, sizeof(size_t));
      patchReferences_.insert({desc.offset_, offset});
      offsetStruct += size;
      // Adjust the offset of arguments
      offset += sizeof(size_t);
    } else {
      // These objects have forced data size to uint64_t
      if ((desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) ||
          (desc.info_.oclObject_ == amd::KernelParameterDescriptor::SamplerObject) ||
          (desc.info_.oclObject_ == amd::KernelParameterDescriptor::QueueObject)) {
        offset = amd::alignUp(offset, sizeof(uint64_t));
        desc.offset_ = offset;
        offset += sizeof(uint64_t);
      } else {
        offset = amd::alignUp(offset, alignment);
        desc.offset_ = offset;
        offset += size;
      }
    }
    // Update read only flag
    desc.info_.readOnly_ = GetReadOnlyOCL(aclArg);

    params.push_back(desc);

    if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) {
      flags_.imageEna_ = true;
      if (desc.accessQualifier_ != CL_KERNEL_ARG_ACCESS_READ_ONLY) {
        flags_.imageWriteEna_ = true;
      }
    }
  }
  // Save the number of OCL arguments
  uint32_t numParams = params.size();
  // Append the hidden arguments to the OCL arguments
  params.insert(params.end(), hiddenParams.begin(), hiddenParams.end());
  createSignature(params, numParams, amd::KernelSignature::ABIVersion_1);
}
#endif

// ================================================================================================
#if defined(USE_COMGR_LIBRARY)
void Kernel::InitPrintf(const std::vector<std::string>& printfInfoStrings) {
  size_t HIPPrintfInfoID = 0;
  for (auto str : printfInfoStrings) {
    std::vector<std::string> tokens;

    size_t end, pos = 0;
    do {
      end = str.find_first_of(':', pos);
      tokens.push_back(str.substr(pos, end - pos));
      pos = end + 1;
    } while (end != std::string::npos);

    if (tokens.size() < 2) {
      LogPrintfError("Invalid PrintInfo string: \"%s\"", str.c_str());
      continue;
    }

    pos = 0;
    size_t printfInfoID;

    if (amd::IS_HIP) {
      printfInfoID = HIPPrintfInfoID++;
      printf_.resize(HIPPrintfInfoID);
      pos++;
    } else {
      printfInfoID = std::stoi(tokens[pos++]);
      if (printf_.size() <= printfInfoID) {
        printf_.resize(printfInfoID + 1);
      }
    }

    PrintfInfo& info = printf_[printfInfoID];

    size_t numSizes = std::stoi(tokens[pos++]);
    end = pos + numSizes;

    // ensure that we have the correct number of tokens
    if (tokens.size() < end + 1 /*last token is the fmtString*/) {
      LogPrintfError("Invalid PrintInfo string: \"%s\"", str.c_str());
      continue;
    }

    // push the argument sizes
    while (pos < end) {
      info.arguments_.push_back(std::stoi(tokens[pos++]));
    }

    // FIXME: We should not need this! [
    std::string fmt;
    // Format string itself might contain ':' characters
    for (int i = 0; pos < tokens.size(); i++) {
      if (i) fmt += ':';
      fmt += tokens[pos++];
    }

    bool need_nl = true;

    for (pos = 0; pos < fmt.size(); ++pos) {
      char symbol = fmt[pos];
      need_nl = true;
      if (symbol == '\\') {
        switch (fmt[pos + 1]) {
          case 'a':
            pos++;
            symbol = '\a';
            break;
          case 'b':
            pos++;
            symbol = '\b';
            break;
          case 'f':
            pos++;
            symbol = '\f';
            break;
          case 'n':
            pos++;
            symbol = '\n';
            need_nl = false;
            break;
          case 'r':
            pos++;
            symbol = '\r';
            break;
          case 'v':
            pos++;
            symbol = '\v';
            break;
          case '7':
            if (fmt[pos + 2] == '2') {
              pos += 2;
              symbol = '\72';
            }
            break;
          default:
            break;
        }
      }
      info.fmtString_.push_back(symbol);
    }
    if (need_nl && !amd::IS_HIP) {
      info.fmtString_ += "\n";
    }
    // ]
  }
}
#endif  // defined(USE_COMGR_LIBRARY)

// ================================================================================================
#if defined(WITH_COMPILER_LIB)
void Kernel::InitPrintf(const aclPrintfFmt* aclPrintf) {
  uint index = 0, HIPIndex = 0;
  for (; aclPrintf->struct_size != 0; aclPrintf++) {
    if (amd::IS_HIP) {
      index = HIPIndex++;
      printf_.resize(HIPIndex);
    } else {
      index = aclPrintf->ID;
      if (printf_.size() <= index) {
        printf_.resize(index + 1);
      }
    }

    PrintfInfo& info = printf_[index];
    const std::string& pfmt = aclPrintf->fmtStr;
    bool need_nl = true;
    for (size_t pos = 0; pos < pfmt.size(); ++pos) {
      char symbol = pfmt[pos];
      need_nl = true;
      if (symbol == '\\') {
        switch (pfmt[pos + 1]) {
          case 'a':
            pos++;
            symbol = '\a';
            break;
          case 'b':
            pos++;
            symbol = '\b';
            break;
          case 'f':
            pos++;
            symbol = '\f';
            break;
          case 'n':
            pos++;
            symbol = '\n';
            need_nl = false;
            break;
          case 'r':
            pos++;
            symbol = '\r';
            break;
          case 'v':
            pos++;
            symbol = '\v';
            break;
          case '7':
            if (pfmt[pos + 2] == '2') {
              pos += 2;
              symbol = '\72';
            }
            break;
          default:
            break;
        }
      }
      info.fmtString_.push_back(symbol);
    }
    if (need_nl && !amd::IS_HIP) {
      info.fmtString_ += "\n";
    }
    uint32_t* tmp_ptr = const_cast<uint32_t*>(aclPrintf->argSizes);
    for (uint i = 0; i < aclPrintf->numSizes; i++, tmp_ptr++) {
      info.arguments_.push_back(*tmp_ptr);
    }
  }
}
#endif  // defined(WITH_COMPILER_LIB)
}  // namespace amd::device
