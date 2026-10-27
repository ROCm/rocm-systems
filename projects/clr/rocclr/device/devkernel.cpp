/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "platform/runtime.hpp"
#include "platform/program.hpp"
#include "platform/ndrange.hpp"
#include "platform/kernel_init.hpp"
#include "devkernel.hpp"
#include "utils/macros.hpp"
#include "utils/options.hpp"
#include "comgrctx.hpp"
#include "device/codeobject/msgpack.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <map>
#include <string>
#include <sstream>

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

// The kernel metadata is the msgpack map of one amdhsa.kernels entry, read in place from the
// code object (Program::createKernelMetadataMap). Field handling follows comgr's
// iterate_map_metadata, which the runtime used before: comgr reported every scalar as text
// (integers in decimal, booleans as "1"/"0") and ignored the callback's status, so an unknown
// field or a value that can't be used is ignored; only malformed msgpack fails the kernel.
// Values are read without allocating.

namespace {

//! A scalar metadata value: a string (a view into the note) or an integer or boolean (1/0)
struct MetaValue {
  std::string_view str;  //!< String value; empty for a number
  uint64_t num = 0;      //!< Integer or boolean value
  bool isNum = false;    //!< The value is an integer or a boolean

  //! The integer value; a string is read as atoi read its text (leading digits, else 0)
  int toInt() const {
    if (isNum) return static_cast<int>(num);
    int value = 0;
    std::from_chars(str.data(), str.data() + str.size(), value);
    return value;
  }

  //! True for 1 or "1", as the text comparison with "1" was
  bool isOne() const { return isNum ? (num == 1) : (str == "1"); }
};

}  // namespace

// ================================================================================================
//! Reads a scalar value. A map, an array or another type is skipped and @p scalar is false.
//! @returns false only if the metadata is malformed.
static bool readMetaValue(codeobject::MsgPackReader& reader, MetaValue& value, bool& scalar) {
  bool flag = false;
  scalar = true;
  if (reader.readString(&value.str)) {
    return true;
  }
  value.isNum = true;
  if (reader.readUint(&value.num)) {
    return true;
  }
  if (reader.readBool(&flag)) {
    value.num = flag ? 1 : 0;
    return true;
  }
  scalar = false;
  return reader.skip();
}

// ================================================================================================
//! Reads an array of 3 integers (reqd_workgroup_size, workgroup_size_hint, cluster_dims).
//! Any other value is skipped and @p found is false. @returns false only if malformed.
static bool readMetaDim3(codeobject::MsgPackReader& reader, std::array<size_t, 3>& dims,
                         bool& found) {
  found = false;
  codeobject::MsgPackReader array = reader;
  uint32_t count = 0;
  if (!array.readArraySize(&count) || count != 3) {
    return reader.skip();
  }
  reader = array;
  for (size_t i = 0; i < 3; ++i) {
    MetaValue value;
    bool scalar = false;
    if (!readMetaValue(reader, value, scalar)) {
      return false;
    }
    dims[i] = scalar ? value.toInt() : 0;
  }
  found = true;
  return true;
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
//! Applies one field of an .args entry to @p lcArg; the reader is at the field's value.
//! @returns false only if the metadata is malformed.
static bool populateArgsV3(std::string_view key, codeobject::MsgPackReader& reader,
                           amd::KernelParameterDescriptor* lcArg) {
  ArgField itArgField = amd::Kernel::FindValue<ArgField>(amd::Kernel::kArgFieldMapV3, key);
  if (itArgField == ArgField::MaxSize) {
    return reader.skip();
  }

  // get the value of the argument field
  MetaValue value;
  bool scalar = false;
  if (!readMetaValue(reader, value, scalar)) {
    return false;
  }
  if (!scalar) {
    return true;
  }

  switch (itArgField) {
    case ArgField::Name:
      lcArg->name_.assign(value.str);
      break;
    case ArgField::TypeName:
      lcArg->typeName_.assign(value.str);
      break;
    case ArgField::Size:
      lcArg->size_ = value.toInt();
      break;
    case ArgField::Offset:
      lcArg->offset_ = value.toInt();
      break;
    case ArgField::ValueKind: {
      amd::KernelParameterDescriptor::Desc itArgValue =
          amd::Kernel::FindValue<amd::KernelParameterDescriptor::Desc>(amd::Kernel::kArgValueKindV3,
                                                                       value.str);
      if (itArgValue == amd::KernelParameterDescriptor::MaxSize) {
        LogPrintfError("Unknown Kernel arg metadata: %.*s", static_cast<int>(value.str.size()),
                       value.str.data());
        LogError("This may be due to running HIP app that requires a new HIP runtime version");
        LogError("Please update the display driver");
        return true;
      }
      lcArg->info_.oclObject_ = itArgValue;
      if (lcArg->info_.oclObject_ == amd::KernelParameterDescriptor::MemoryObject) {
        if (value.str == "dynamic_shared_pointer") {
          lcArg->info_.shared_ = true;
        }
      } else if ((lcArg->info_.oclObject_ >= amd::KernelParameterDescriptor::HiddenNone) &&
                 (lcArg->info_.oclObject_ < amd::KernelParameterDescriptor::HiddenLast)) {
        lcArg->info_.hidden_ = true;
      }
    } break;
    case ArgField::PointeeAlign:
      lcArg->info_.arrayIndex_ = value.toInt();
      break;
    case ArgField::AddrSpaceQual: {
      cl_int itAddrSpaceQual = amd::Kernel::FindValue(amd::Kernel::kArgAddrSpaceQualV3, value.str);
      if (itAddrSpaceQual == static_cast<cl_int>(0)) {
        return true;
      }
      lcArg->addressQualifier_ = itAddrSpaceQual;
    } break;
    case ArgField::AccQual: {
      cl_int itAccQual = amd::Kernel::FindValue(amd::Kernel::kArgAccQualV3, value.str);
      if (itAccQual == static_cast<cl_int>(0)) {
        return true;
      }
      lcArg->accessQualifier_ = itAccQual;
      if (!lcArg->info_.isReadOnlyByCompiler) {
        lcArg->info_.readOnly_ =
            (lcArg->accessQualifier_ == CL_KERNEL_ARG_ACCESS_READ_ONLY) ? true : false;
      }
    } break;
    case ArgField::ActualAccQual: {
      cl_int itAccQual = amd::Kernel::FindValue(amd::Kernel::kArgAccQualV3, value.str);
      if (itAccQual == static_cast<cl_int>(0)) {
        return true;
      }
      lcArg->info_.isReadOnlyByCompiler = true;
      lcArg->info_.readOnly_ = (itAccQual == CL_KERNEL_ARG_ACCESS_READ_ONLY) ? true : false;
    } break;
    case ArgField::IsConst:
      lcArg->typeQualifier_ |= value.isOne() ? CL_KERNEL_ARG_TYPE_CONST : 0;
      break;
    case ArgField::IsRestrict:
      lcArg->typeQualifier_ |= value.isOne() ? CL_KERNEL_ARG_TYPE_RESTRICT : 0;
      break;
    case ArgField::IsVolatile:
      lcArg->typeQualifier_ |= value.isOne() ? CL_KERNEL_ARG_TYPE_VOLATILE : 0;
      break;
    case ArgField::IsPipe:
      lcArg->typeQualifier_ |= value.isOne() ? CL_KERNEL_ARG_TYPE_PIPE : 0;
      break;
    default:
      break;
  }
  return true;
}

// ================================================================================================
//! Applies one field of the kernel's map to @p kernel; the reader is at the field's value.
//! @returns false only if the metadata is malformed.
static bool populateKernelMetaV3(std::string_view key, codeobject::MsgPackReader& reader,
                                 device::Kernel* kernel) {
  KernelField itKernelField =
      amd::Kernel::FindValue<KernelField>(amd::Kernel::kKernelFieldMapV3, key);
  if (itKernelField == KernelField::MaxSize) {
    return reader.skip();
  }

  std::array<size_t, 3> dims = {};
  MetaValue value;
  bool found = false;
  if (itKernelField == KernelField::ReqdWorkGroupSize ||
      itKernelField == KernelField::WorkGroupSizeHint ||
      itKernelField == KernelField::ClusterDims) {
    if (!readMetaDim3(reader, dims, found)) {
      return false;
    }
  } else if (!readMetaValue(reader, value, found)) {
    return false;
  }
  if (!found) {
    return true;
  }

  switch (itKernelField) {
    case KernelField::ReqdWorkGroupSize:
      kernel->setReqdWorkGroupSize(dims[0], dims[1], dims[2]);
      break;
    case KernelField::WorkGroupSizeHint:
      kernel->setWorkGroupSizeHint(dims[0], dims[1], dims[2]);
      break;
    case KernelField::ClusterDims:
      kernel->setClusterSize(dims[0], dims[1], dims[2]);
      break;
    case KernelField::VecTypeHint:
      kernel->setVecTypeHint(value.str);
      break;
    case KernelField::DeviceEnqueueSymbol:
      kernel->setRuntimeHandle(value.str);
      break;
    case KernelField::KernargSegmentSize:
      kernel->SetKernargSegmentByteSize(value.toInt());
      break;
    case KernelField::GroupSegmentFixedSize:
      kernel->SetWorkgroupGroupSegmentByteSize(value.toInt());
      break;
    case KernelField::PrivateSegmentFixedSize:
      kernel->SetWorkitemPrivateSegmentByteSize(value.toInt());
      break;
    case KernelField::KernargSegmentAlign:
      kernel->SetKernargSegmentAlignment(value.toInt());
      break;
    case KernelField::WavefrontSize:
      kernel->workGroupInfo()->wavefrontSize_ = value.toInt();
      break;
    case KernelField::NumSGPRs:
      kernel->workGroupInfo()->usedSGPRs_ = value.toInt();
      break;
    case KernelField::NumVGPRs:
      kernel->workGroupInfo()->usedVGPRs_ = value.toInt();
      break;
    case KernelField::MaxFlatWorkGroupSize:
      kernel->workGroupInfo()->size_ = value.toInt();
      break;
    case KernelField::NumSpilledSGPRs:
    case KernelField::NumSpilledVGPRs:
      break;
    case KernelField::SymbolName:
      kernel->SetSymbolName(value.str);
      break;
    case KernelField::Kind:
      kernel->SetKernelKind(value.str);
      break;
    case KernelField::WgpMode:
      // The compiler currently serializes this boolean field as "0"/"1" instead
      // of "false"/"true"; consider both "true" and "1" truthy values.
      kernel->SetWGPMode(value.isOne() || value.str == "true");
      break;
    case KernelField::UniformWrokGroupSize:
      kernel->setUniformWorkGroupSize(value.isOne());
      break;
    default:
      break;
  }
  return true;
}

// ================================================================================================
Kernel::Kernel(const amd::Device& dev, const std::string& name, const Program& prog)
    : dev_(dev), name_(name), prog_(prog), signature_(nullptr) {
  // Instead of memset(&workGroupInfo_, '\0', sizeof(workGroupInfo_));
  // Due to std::string not being able to be memset to 0
  workGroupInfo_.size_ = 0;
  workGroupInfo_.compileSize_[0] = 0;
  workGroupInfo_.compileSize_[1] = 0;
  workGroupInfo_.compileSize_[2] = 0;
  workGroupInfo_.clusterSize_[0] = 1;
  workGroupInfo_.clusterSize_[1] = 1;
  workGroupInfo_.clusterSize_[2] = 1;
  workGroupInfo_.runtimeClusterSize_[0] = 0;
  workGroupInfo_.runtimeClusterSize_[1] = 0;
  workGroupInfo_.runtimeClusterSize_[2] = 0;
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
  workGroupInfo_.kernelMaxDynamicSharedSizeBytes_ = 0;
  workGroupInfo_.kernelPreferredShmemCarveout_ = -1;
  workGroupInfo_.nonPortableClusterSizeAllowed_ = 0;
  workGroupInfo_.clusterSchedulingPolicyPreference_ = 0;
  workGroupInfo_.hasClusterAttr_ = false;
  workGroupInfo_.hasFuncMaxDynamicSharedSize_ = false;
  workGroupInfo_.hasFuncPreferredShmemCarveout_ = false;
  workGroupInfo_.groupMemCarveout_ = -1;
  workGroupInfo_.coarseMemCarveout_ = 0; // preference = None
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
  if (workGroupInfo_.clusterSize_[0] != 0) {
    attribs << "cluster_dims(";
    for (size_t i = 0; i < 3; ++i) {
      if (i != 0) {
        attribs << ",";
      }
      attribs << workGroupInfo_.clusterSize_[i];
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
bool Kernel::GetAttrCodePropMetadata() {
  std::string_view kernelMeta;
  if (!prog().getKernelMetadata(name(), &kernelMeta)) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_KERN,
             "Cannot get program kernel metadata for %s \n", name().c_str());
    return false;
  }

  // Set the workgroup information for the kernel
  workGroupInfo_.availableLDSSize_ = device().info().localMemSizePerCU_;
  workGroupInfo_.availableSGPRs_ = 104;
  workGroupInfo_.availableVGPRs_ = 256;

  // One pass over the kernel's map, in place: .args and the other fields. The signature is
  // built after it, since attributes it reports (e.g. reqd_workgroup_size) can follow .args.
  // A kernel without .args has no arguments.
  codeobject::MsgPackReader reader(kernelMeta.data(), kernelMeta.size());
  device::Kernel::parameters_t params;
  device::Kernel::parameters_t hiddenParams;
  uint32_t numFields = 0;
  bool ok = reader.readMapSize(&numFields);
  for (uint32_t i = 0; ok && i < numFields; ++i) {
    std::string_view key;
    ok = reader.readString(&key);
    if (!ok) {
      break;
    }
    if (key == ".args") {
      ok = InitParameters(reader, params, hiddenParams);
    } else {
      ok = populateKernelMetaV3(key, reader, this);
    }
  }

  if (!ok) {
    LogPrintfError("Malformed code object metadata for kernel %s", name().c_str());
    return false;
  }

  // Save the number of OCL arguments
  uint32_t numParams = params.size();
  // Append the hidden arguments to the OCL arguments
  params.insert(params.end(), hiddenParams.begin(), hiddenParams.end());
  createSignature(params, numParams, amd::KernelSignature::ABIVersion_LC);
  return true;
}

bool Kernel::InitParameters(codeobject::MsgPackReader& reader, parameters_t& params,
                            parameters_t& hiddenParams) {
  uint32_t argsSize = 0;
  if (!reader.readArraySize(&argsSize)) {
    return false;
  }

  // The count comes from the note: never reserve more than the note can hold
  params.reserve(std::min<size_t>(argsSize, reader.remaining()));
  hiddenParams.reserve(amd::KernelParameterDescriptor::Desc::MaxSize);

  for (uint32_t i = 0; i < argsSize; ++i) {
    amd::KernelParameterDescriptor desc = {};

    uint32_t numFields = 0;
    bool ok = reader.readMapSize(&numFields);
    for (uint32_t f = 0; ok && f < numFields; ++f) {
      std::string_view key;
      ok = reader.readString(&key) && populateArgsV3(key, reader, &desc);
    }
    if (!ok) {
      return false;
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

    // Allocate the hidden arguments, but abstraction layer will skip them
    if (desc.info_.hidden_) {
      if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::HiddenCompletionAction &&
          !amd::IS_HIP) {
        setDynamicParallelFlag(true);
      }
      hiddenParams.push_back(desc);
      continue;
    }

    params.push_back(desc);

    if (desc.info_.oclObject_ == amd::KernelParameterDescriptor::ImageObject) {
      flags_.imageEna_ = true;
      if (desc.accessQualifier_ != CL_KERNEL_ARG_ACCESS_READ_ONLY) {
        flags_.imageWriteEna_ = true;
      }
    }
  }
  return true;
}

// ================================================================================================
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
}  // namespace amd::device
