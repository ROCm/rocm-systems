/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

inline std::ostream& operator<<(std::ostream& os, const hipTextureFilterMode& s) {
  switch (s) {
    case hipFilterModePoint:
      os << "hipFilterModePoint";
      break;
    case hipFilterModeLinear:
      os << "hipFilterModeLinear";
      break;
    default:
      os << "hipFilterModePoint";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipTextureReadMode& s) {
  switch (s) {
    case hipReadModeElementType:
      os << "hipReadModeElementType";
      break;
    case hipReadModeNormalizedFloat:
      os << "hipReadModeNormalizedFloat";
      break;
    default:
      os << "hipReadModeElementType";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipTextureAddressMode& s) {
  switch (s) {
    case hipAddressModeWrap:
      os << "hipAddressModeWrap";
      break;
    case hipAddressModeClamp:
      os << "hipAddressModeClamp";
      break;
    case hipAddressModeMirror:
      os << "hipAddressModeMirror";
      break;
    case hipAddressModeBorder:
      os << "hipAddressModeBorder";
      break;
    default:
      os << "hipAddressModeWrap";
  };
  return os;
}


inline std::ostream& operator<<(std::ostream& os, const hipMemcpyKind& s) {
  switch (s) {
    case hipMemcpyHostToHost:
      os << "hipMemcpyHostToHost";
      break;
    case hipMemcpyHostToDevice:
      os << "hipMemcpyHostToDevice";
      break;
    case hipMemcpyDeviceToHost:
      os << "hipMemcpyDeviceToHost";
      break;
    case hipMemcpyDeviceToDevice:
      os << "hipMemcpyDeviceToDevice";
      break;
    case hipMemcpyDefault:
      os << "hipMemcpyDefault";
      break;
    case hipMemcpyDeviceToDeviceNoCU:
      os << "hipMemcpyDeviceToDeviceNoCU";
      break;
    default:
      os << "hipMemcpyDefault";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipChannelFormatKind& s) {
  switch (s) {
    case hipChannelFormatKindSigned:
      os << "hipChannelFormatKindSigned";
      break;
    case hipChannelFormatKindUnsigned:
      os << "hipMemcpyHostToDevice";
      break;
    case hipChannelFormatKindFloat:
      os << "hipChannelFormatKindFloat";
      break;
    case hipChannelFormatKindNone:
      os << "hipChannelFormatKindNone";
      break;
    default:
      os << "hipChannelFormatKindNone";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipArray_Format& s) {
  switch (s) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
      os << "HIP_AD_FORMAT_UNSIGNED_INT8";
      break;
    case HIP_AD_FORMAT_UNSIGNED_INT16:
      os << "HIP_AD_FORMAT_UNSIGNED_INT16";
      break;
    case HIP_AD_FORMAT_UNSIGNED_INT32:
      os << "HIP_AD_FORMAT_UNSIGNED_INT32";
      break;
    case HIP_AD_FORMAT_SIGNED_INT8:
      os << "HIP_AD_FORMAT_SIGNED_INT8";
      break;
    case HIP_AD_FORMAT_SIGNED_INT16:
      os << "HIP_AD_FORMAT_SIGNED_INT16";
      break;
    case HIP_AD_FORMAT_SIGNED_INT32:
      os << "HIP_AD_FORMAT_SIGNED_INT32";
      break;
    case HIP_AD_FORMAT_HALF:
      os << "HIP_AD_FORMAT_HALF";
      break;
    case HIP_AD_FORMAT_FLOAT:
      os << "HIP_AD_FORMAT_FLOAT";
      break;
    default:
      os << "HIP_AD_FORMAT_FLOAT";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipResourceViewFormat& s) {
  switch (s) {
    case hipResViewFormatNone:
      os << "hipResViewFormatNone";
      break;
    case hipResViewFormatUnsignedChar1:
      os << "hipResViewFormatUnsignedChar1";
      break;
    case hipResViewFormatUnsignedChar2:
      os << "hipResViewFormatUnsignedChar2";
      break;
    case hipResViewFormatUnsignedChar4:
      os << "hipResViewFormatUnsignedChar4";
      break;
    case hipResViewFormatSignedChar1:
      os << "hipResViewFormatSignedChar1";
      break;
    case hipResViewFormatSignedChar2:
      os << "hipResViewFormatSignedChar2";
      break;
    case hipResViewFormatSignedChar4:
      os << "hipResViewFormatSignedChar4";
      break;
    case hipResViewFormatUnsignedShort1:
      os << "hipResViewFormatUnsignedShort1";
      break;
    case hipResViewFormatUnsignedShort2:
      os << "hipResViewFormatUnsignedShort2";
      break;
    case hipResViewFormatUnsignedShort4:
      os << "hipResViewFormatUnsignedShort4";
      break;
    case hipResViewFormatSignedShort1:
      os << "hipResViewFormatSignedShort1";
      break;
    case hipResViewFormatSignedShort2:
      os << "hipResViewFormatSignedShort2";
      break;
    case hipResViewFormatSignedShort4:
      os << "hipResViewFormatSignedShort4";
      break;
    case hipResViewFormatUnsignedInt1:
      os << "hipResViewFormatUnsignedInt1";
      break;
    case hipResViewFormatUnsignedInt2:
      os << "hipResViewFormatUnsignedInt2";
      break;
    case hipResViewFormatUnsignedInt4:
      os << "hipResViewFormatUnsignedInt4";
      break;
    case hipResViewFormatSignedInt1:
      os << "hipResViewFormatSignedInt1";
      break;
    case hipResViewFormatSignedInt2:
      os << "hipResViewFormatSignedInt2";
      break;
    case hipResViewFormatSignedInt4:
      os << "hipResViewFormatSignedInt4";
      break;
    case hipResViewFormatHalf1:
      os << "hipResViewFormatHalf1";
      break;
    case hipResViewFormatHalf2:
      os << "hipResViewFormatHalf2";
      break;
    case hipResViewFormatHalf4:
      os << "hipResViewFormatHalf4";
      break;
    case hipResViewFormatFloat1:
      os << "hipResViewFormatFloat1";
      break;
    case hipResViewFormatFloat2:
      os << "hipResViewFormatFloat2";
      break;
    case hipResViewFormatFloat4:
      os << "hipResViewFormatFloat4";
      break;
    case hipResViewFormatUnsignedBlockCompressed1:
      os << "hipResViewFormatUnsignedBlockCompressed1";
      break;
    case hipResViewFormatUnsignedBlockCompressed2:
      os << "hipResViewFormatUnsignedBlockCompressed2";
      break;
    case hipResViewFormatUnsignedBlockCompressed3:
      os << "hipResViewFormatUnsignedBlockCompressed3";
      break;
    case hipResViewFormatUnsignedBlockCompressed4:
      os << "hipResViewFormatUnsignedBlockCompressed4";
      break;
    case hipResViewFormatSignedBlockCompressed4:
      os << "hipResViewFormatSignedBlockCompressed4";
      break;
    case hipResViewFormatUnsignedBlockCompressed5:
      os << "hipResViewFormatUnsignedBlockCompressed5";
      break;
    case hipResViewFormatSignedBlockCompressed5:
      os << "hipResViewFormatSignedBlockCompressed5";
      break;
    case hipResViewFormatUnsignedBlockCompressed6H:
      os << "hipResViewFormatUnsignedBlockCompressed6H";
      break;
    case hipResViewFormatSignedBlockCompressed6H:
      os << "hipResViewFormatSignedBlockCompressed6H";
      break;
    case hipResViewFormatUnsignedBlockCompressed7:
      os << "hipResViewFormatUnsignedBlockCompressed7";
      break;
    default:
      os << "hipResViewFormatNone";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipFunction_attribute& s) {
  switch (s) {
    case HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
      os << "HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK";
      break;
    case HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
      os << "HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES";
      break;
    case HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
      os << "HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES";
      break;
    case HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
      os << "HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES";
      break;
    case HIP_FUNC_ATTRIBUTE_NUM_REGS:
      os << "HIP_FUNC_ATTRIBUTE_NUM_REGS";
      break;
    case HIP_FUNC_ATTRIBUTE_PTX_VERSION:
      os << "HIP_FUNC_ATTRIBUTE_PTX_VERSION";
      break;
    case HIP_FUNC_ATTRIBUTE_BINARY_VERSION:
      os << "HIP_FUNC_ATTRIBUTE_BINARY_VERSION";
      break;
    case HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA:
      os << "HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA";
      break;
    case HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
      os << "HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES";
      break;
    case HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
      os << "HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT";
      break;
    case HIP_FUNC_ATTRIBUTE_MAX:
      os << "HIP_FUNC_ATTRIBUTE_MAX";
      break;
    default:
      os << "HIP_FUNC_ATTRIBUTE_MAX";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hiprtcResult& s) {
  switch (s) {
    case HIPRTC_SUCCESS:
      os << "HIPRTC_SUCCESS";
      break;
    case HIPRTC_ERROR_OUT_OF_MEMORY:
      os << "HIPRTC_ERROR_OUT_OF_MEMORY";
      break;
    case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:
      os << "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
      break;
    case HIPRTC_ERROR_INVALID_INPUT:
      os << "HIPRTC_ERROR_INVALID_INPUT";
      break;
    case HIPRTC_ERROR_INVALID_PROGRAM:
      os << "HIPRTC_ERROR_INVALID_PROGRAM";
      break;
    case HIPRTC_ERROR_INVALID_OPTION:
      os << "HIPRTC_ERROR_INVALID_OPTION";
      break;
    case HIPRTC_ERROR_COMPILATION:
      os << "HIPRTC_ERROR_COMPILATION";
      break;
    case HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:
      os << "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
      break;
    case HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
      os << "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
      break;
    case HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
      os << "IPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
      break;
    case HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
      os << "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
      break;
    case HIPRTC_ERROR_INTERNAL_ERROR:
      os << "HIPRTC_ERROR_INTERNAL_ERROR";
      break;
    default:
      os << "HIPRTC_ERROR_INTERNAL_ERROR";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipJitOption& s) {
  hiprtcJIT_option option = (hiprtcJIT_option)((int)s);

  switch (option) {
    case HIPRTC_JIT_MAX_REGISTERS:
      os << "HIPRTC_JIT_MAX_REGISTERS";
      break;
    case HIPRTC_JIT_THREADS_PER_BLOCK:
      os << "HIPRTC_JIT_THREADS_PER_BLOCK";
      break;
    case HIPRTC_JIT_WALL_TIME:
      os << "HIPRTC_JIT_WALL_TIME";
      break;
    case HIPRTC_JIT_INFO_LOG_BUFFER:
      os << "HIPRTC_JIT_INFO_LOG_BUFFER";
      break;
    case HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
      os << "HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES";
      break;
    case HIPRTC_JIT_ERROR_LOG_BUFFER:
      os << "HIPRTC_JIT_ERROR_LOG_BUFFER";
      break;
    case HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
      os << "HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES";
      break;
    case HIPRTC_JIT_OPTIMIZATION_LEVEL:
      os << "HIPRTC_JIT_OPTIMIZATION_LEVEL";
      break;
    case HIPRTC_JIT_TARGET_FROM_HIPCONTEXT:
      os << "HIPRTC_JIT_TARGET_FROM_HIPCONTEXT";
      break;
    case HIPRTC_JIT_TARGET:
      os << "HIPRTC_JIT_TARGET";
      break;
    case HIPRTC_JIT_FALLBACK_STRATEGY:
      os << "HIPRTC_JIT_FALLBACK_STRATEGY";
      break;
    case HIPRTC_JIT_GENERATE_DEBUG_INFO:
      os << "HIPRTC_JIT_GENERATE_DEBUG_INFO";
      break;
    case HIPRTC_JIT_CACHE_MODE:
      os << "HIPRTC_JIT_CACHE_MODE";
      break;
    case HIPRTC_JIT_NEW_SM3X_OPT:
      os << "HIPRTC_JIT_NEW_SM3X_OPT";
      break;
    case HIPRTC_JIT_FAST_COMPILE:
      os << "HIPRTC_JIT_FAST_COMPILE";
      break;
    case HIPRTC_JIT_GLOBAL_SYMBOL_NAMES:
      os << "HIPRTC_JIT_GLOBAL_SYMBOL_NAMES";
      break;
    case HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS:
      os << "HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS";
      break;
    case HIPRTC_JIT_GLOBAL_SYMBOL_COUNT:
      os << "HIPRTC_JIT_GLOBAL_SYMBOL_COUNT";
      break;
    case HIPRTC_JIT_LTO:
      os << "HIPRTC_JIT_LTO";
      break;
    case HIPRTC_JIT_FTZ:
      os << "HIPRTC_JIT_FTZ";
      break;
    case HIPRTC_JIT_PREC_DIV:
      os << "HIPRTC_JIT_PREC_DIV";
      break;
    case HIPRTC_JIT_PREC_SQRT:
      os << "HIPRTC_JIT_PREC_SQRT";
      break;
    case HIPRTC_JIT_FMA:
      os << "HIPRTC_JIT_FMA";
      break;
    case HIPRTC_JIT_NUM_OPTIONS:
      os << "HIPRTC_JIT_NUM_OPTIONS";
      break;
    default:
      os << "HIPRTC_JIT_MAX_REGISTERS";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipFuncCache_t& s) {
  switch (s) {
    case hipFuncCachePreferNone:
      os << "hipFuncCachePreferNone";
      break;
    case hipFuncCachePreferShared:
      os << "hipFuncCachePreferShared";
      break;
    case hipFuncCachePreferL1:
      os << "hipFuncCachePreferL1";
      break;
    case hipFuncCachePreferEqual:
      os << "hipFuncCachePreferEqual";
      break;
    default:
      os << "hipFuncCachePreferNone";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipSharedMemConfig& s) {
  switch (s) {
    case hipSharedMemBankSizeDefault:
      os << "hipSharedMemBankSizeDefault";
      break;
    case hipSharedMemBankSizeFourByte:
      os << "hipSharedMemBankSizeFourByte";
      break;
    case hipSharedMemBankSizeEightByte:
      os << "hipSharedMemBankSizeEightByte";
      break;
    default:
      os << "hipSharedMemBankSizeDefault";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipDataType& s) {
  switch (s) {
    case HIP_R_16F:
      os << "HIP_R_16F";
      break;
    case HIP_R_32F:
      os << "HIP_R_32F";
      break;
    case HIP_R_64F:
      os << "HIP_R_64F";
      break;
    case HIP_C_16F:
      os << "HIP_C_16F";
      break;
    case HIP_C_32F:
      os << "HIP_C_32F";
      break;
    case HIP_C_64F:
      os << "HIP_C_64F";
      break;
    case HIP_R_8I:
      os << "HIP_R_8I";
      break;
    case HIP_C_8I:
      os << "HIP_C_8I";
      break;
    case HIP_R_8U:
      os << "HIP_R_8U";
      break;
    case HIP_C_8U:
      os << "HIP_C_8U";
      break;
    case HIP_R_32I:
      os << "HIP_R_32I";
      break;
    case HIP_C_32I:
      os << "HIP_C_32I";
      break;
    case HIP_R_32U:
      os << "HIP_R_32U";
      break;
    case HIP_C_32U:
      os << "HIP_C_32U";
      break;
    case HIP_R_16BF:
      os << "HIP_R_16BF";
      break;
    case HIP_C_16BF:
      os << "HIP_C_16BF";
      break;
    case HIP_R_4I:
      os << "HIP_R_4I";
      break;
    case HIP_C_4I:
      os << "HIP_C_4I";
      break;
    case HIP_R_4U:
      os << "HIP_R_4U";
      break;
    case HIP_C_4U:
      os << "HIP_C_4U";
      break;
    case HIP_R_16I:
      os << "HIP_R_16I";
      break;
    case HIP_C_16I:
      os << "HIP_C_16I";
      break;
    case HIP_R_16U:
      os << "HIP_R_16U";
      break;
    case HIP_C_16U:
      os << "HIP_C_16U";
      break;
    case HIP_R_64I:
      os << "HIP_R_64I";
      break;
    case HIP_C_64I:
      os << "HIP_C_64I";
      break;
    case HIP_R_64U:
      os << "HIP_R_64U";
      break;
    case HIP_C_64U:
      os << "HIP_C_64U";
      break;
    case HIP_R_8F_E4M3:
      os << "HIP_R_8F_E4M3";
      break;
    case HIP_R_8F_E5M2:
      os << "HIP_R_8F_E5M2";
      break;
    case HIP_R_8F_E4M3_FNUZ:
      os << "HIP_R_8F_E4M3_FNUZ";
      break;
    case HIP_R_8F_E5M2_FNUZ:
      os << "HIP_R_8F_E5M2_FNUZ";
      break;
    default:
      os << "HIP_R_16F";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipLibraryPropertyType& s) {
  switch (s) {
    case HIP_LIBRARY_MAJOR_VERSION:
      os << "HIP_LIBRARY_MAJOR_VERSION";
      break;
    case HIP_LIBRARY_MINOR_VERSION:
      os << "HIP_LIBRARY_MINOR_VERSION";
      break;
    case HIP_LIBRARY_PATCH_LEVEL:
      os << "HIP_LIBRARY_PATCH_LEVEL";
      break;
    default:
      os << "HIP_LIBRARY_MAJOR_VERSION";
  };
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hip_api_id_t& s) {
  os << hip_api_name(s);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hip_api_id_t* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipTextureDesc& s) {
  os << '{' << '{' << s.addressMode[0] << ',' << s.addressMode[1] << ',' << s.addressMode[2] << '}'
     << ',' << s.filterMode << ',' << s.readMode << ',' << s.sRGB << ',' << '{' << s.borderColor[0]
     << ',' << s.borderColor[1] << ',' << s.borderColor[2] << ',' << s.borderColor[3] << '}' << ','
     << s.normalizedCoords << ',' << s.mipmapFilterMode << ',' << s.mipmapLevelBias << ','
     << s.minMipmapLevelClamp << ',' << s.maxMipmapLevelClamp << '}';
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipTextureDesc* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}


inline std::ostream& operator<<(std::ostream& os, const dim3& s) {
  os << '{' << s.x << ',' << s.y << ',' << s.z << '}';
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const dim3* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipChannelFormatDesc& s) {
  os << '{' << s.x << ',' << s.y << ',' << s.z << ',' << s.w << ',' << s.f << '}';
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipChannelFormatDesc* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipMipmappedArray& s) {
  os << '{' << s.data << ',' << s.desc << ',' << s.width << ',' << s.height << ',' << s.depth
     << '}';
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipMipmappedArray* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}


inline std::ostream& operator<<(std::ostream& os, const hipResourceDesc& s) {
  os << '{' << s.resType << ',' << '{';

  switch (s.resType) {
    case hipResourceTypeLinear:
      os << s.res.linear.devPtr << ',' << s.res.linear.desc << ',' << s.res.linear.sizeInBytes;
      break;
    case hipResourceTypePitch2D:
      os << s.res.pitch2D.devPtr << ',' << s.res.pitch2D.desc << ',' << s.res.pitch2D.width << ','
         << s.res.pitch2D.height << ',' << s.res.pitch2D.pitchInBytes;
      break;
    case hipResourceTypeArray:
      os << s.res.array.array;
      break;
    case hipResourceTypeMipmappedArray:
      os << s.res.mipmap.mipmap;
      break;
    default:
      break;
  }

  os << '}';

  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipResourceDesc* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const textureReference& s) {
  os << '{' << s.normalized << ',' << s.readMode << ',' << s.filterMode << ',' << '{'
     << s.addressMode[0] << ',' << s.addressMode[1] << ',' << s.addressMode[2] << '}' << ','
     << s.channelDesc << ',' << s.sRGB << ',' << s.maxAnisotropy << ',' << s.mipmapFilterMode << ','
     << s.mipmapLevelBias << ',' << s.minMipmapLevelClamp << ',' << s.maxMipmapLevelClamp << ','
     << s.textureObject << '}';
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const textureReference* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}


inline std::ostream& operator<<(std::ostream& os, const hipError_t& s) {
  os << hipGetErrorName(s);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipError_t* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipResourceViewDesc& s) {
  os << '{' << s.format << ',' << s.width << ',' << s.height << ',' << s.depth << ','
     << s.firstMipmapLevel << ',' << s.lastMipmapLevel << ',' << s.firstLayer << ',' << s.lastLayer
     << '}';
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipResourceViewDesc* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const HIP_ARRAY_DESCRIPTOR& s) {
  os << '{' << s.Width << ',' << s.Height << ',' << s.Format << ',' << s.NumChannels << '}';
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const HIP_ARRAY_DESCRIPTOR* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const HIP_ARRAY3D_DESCRIPTOR& s) {
  os << '{' << s.Width << ',' << s.Height << ',' << s.Depth << ',' << s.Format << ','
     << s.NumChannels << ',' << s.Flags << '}';
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const HIP_ARRAY3D_DESCRIPTOR* s) {
  if (s) {
    os << *s;
  } else {
    os << "nullptr";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipExtent& s) {
  os << '{' << s.width << ',' << s.height << ',' << s.depth << '}';
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipIpcEventHandle_t& s) {
  // TODO fill in later
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipIpcEventHandle_t* s) {
  // TODO fill in later
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipPitchedPtr& p) {
  os << "pitchPtr:" << std::hex << reinterpret_cast<uintptr_t>(p.ptr);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipPitchedPtr* p) {
  if (p) {
    os << *p;
  } else {
    os << "nullptr";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipMemLocation& s) {
  os << "{type=";
  switch (s.type) {
    case hipMemLocationTypeInvalid:
      os << "hipMemLocationTypeInvalid";
      break;
    case hipMemLocationTypeDevice:
      os << "hipMemLocationTypeDevice";
      break;
    case hipMemLocationTypeHost:
      os << "hipMemLocationTypeHost";
      break;
    case hipMemLocationTypeHostNuma:
      os << "hipMemLocationTypeHostNuma";
      break;
    case hipMemLocationTypeHostNumaCurrent:
      os << "hipMemLocationTypeHostNumaCurrent";
      break;
    default:
      os << static_cast<int>(s.type);
      break;
  }
  os << ", id=" << s.id << "}";
  return os;
}
