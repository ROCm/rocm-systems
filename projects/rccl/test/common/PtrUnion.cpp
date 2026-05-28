/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "PtrUnion.hpp"
#include "api_trace.h"
namespace RcclUnitTesting
{
  size_t DataTypeToBytes(ncclDataType_t const dataType)
  {
    switch (dataType)
    {
    case ncclInt8:   return 1;
    case ncclUint8:  return 1;
    case ncclFloat8e4m3:return 1;
    case ncclFloat8e5m2:return 1;
    case ncclInt32:  return 4;
    case ncclUint32: return 4;
    case ncclInt64:  return 8;
    case ncclUint64: return 8;
    case ncclFloat16: return 2;
    case ncclFloat32: return 4;
    case ncclFloat64: return 8;
    case ncclBfloat16: return 2;
    default:
      ERROR("Unsupported datatype (%d)\n", dataType);
      exit(0);
    }
  }

  ErrCode PtrUnion::Attach(void *ptr)
  {
    this->ptr = ptr;
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::Attach(PtrUnion ptrUnion)
  {
    this->ptr = ptrUnion.ptr;
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::AllocateGpuMem(size_t const numBytes, bool const useManagedMem, bool const userRegistered)
  {
    if (numBytes)
    {
      if (userRegistered)
      {
        if (ncclMemAlloc((void**)&I1, numBytes) != ncclSuccess)
        {
          ERROR("Unable to allocate user managed GPU memory (%lu bytes)\n", numBytes);
          return TEST_FAIL;
        }
      }
      else
      {
        if (useManagedMem)
        {
          if (hipMallocManaged(&I1, numBytes) != hipSuccess)
          {
            ERROR("Unable to allocate managed memory of GPU memory (%lu bytes)\n", numBytes);
            return TEST_FAIL;
          }
        }
        else
        {
          if (hipMalloc(&I1, numBytes) != hipSuccess)
          {
            ERROR("Unable to allocate memory of GPU memory (%lu bytes)\n", numBytes);
            return TEST_FAIL;
          }
        }
      }

    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::AllocateCpuMem(size_t const numBytes)
  {
    if (numBytes)
    {
      this->ptr = calloc(numBytes, 1);
      if (!ptr)
      {
        ERROR("Unable to allocate memory (%lu bytes)\n", numBytes);
        return TEST_FAIL;
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::FreeGpuMem(bool const userRegistered)
  {
    if (this->ptr != nullptr)
    {
      if (userRegistered)
        ncclMemFree(this->ptr);
      else
        hipFree(this->ptr);
      this->ptr = nullptr;
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::FreeCpuMem()
  {
    if (this->ptr != nullptr)
    {
      free(this->ptr);
      this->ptr = nullptr;
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::ClearGpuMem(size_t const numBytes)
  {
    if (hipMemset(this->ptr, 0, numBytes) != hipSuccess)
    {
      ERROR("Unable to call hipMemset\n");
      return TEST_FAIL;
    }
    hipStreamSynchronize(NULL);
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::ClearCpuMem(size_t const numBytes)
  {
    memset(this->ptr, 0, numBytes);
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::FillPattern(ncclDataType_t const dataType,
                                size_t         const numElements,
                                int            const globalRank,
                                bool           const isGpuMem)
  {
    PtrUnion temp;
    size_t const numBytes = numElements * DataTypeToBytes(dataType);

    // If this is GPU memory, create a CPU temp buffer otherwise fill CPU memory directly
    if (isGpuMem)
      temp.AllocateCpuMem(numBytes);
    else
      temp.Attach(this->ptr);

    for (int i = 0; i < numElements; i++)
    {
      // Due to floating-point math not being commutative, the ordering in which ranks are added will matter.
      // For lower-precision data types, we initialize all ranks to the same value to avoid this
      int    valueI = (dataType == ncclFloat8e4m3 || dataType == ncclFloat8e5m2)? (i % 16) :(globalRank + i) % 256;
      double valueF = 1.0L/((double)valueI+1.0L);
      temp.Set(dataType, i, valueI, valueF);
    }

    // If this is GPU memory, copy from CPU temp buffer
    if (isGpuMem)
    {
      if (hipMemcpy(this->ptr, temp.ptr, numBytes, hipMemcpyHostToDevice) != hipSuccess)
      {
        ERROR("Unable to fill input with pattern for rank %d\n", globalRank);
        return TEST_FAIL;
      }
      temp.FreeCpuMem();
    }

    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::Set(ncclDataType_t const dataType, int const idx, int valueI, double valueF)
  {
    switch (dataType)
    {
    case ncclInt8:     I1[idx] = valueI; break;
    case ncclUint8:    U1[idx] = valueI; break;
    case ncclInt32:    I4[idx] = valueI; break;
    case ncclUint32:   U4[idx] = valueI; break;
    case ncclInt64:    I8[idx] = valueI; break;
    case ncclUint64:   U8[idx] = valueI; break;
    case ncclFloat8e4m3:  F1[idx] = rccl_float8(valueF); break;
    case ncclFloat16:  F2[idx] = __float2half(static_cast<float>(valueF)); break;
    case ncclFloat32:  F4[idx] = valueF; break;
    case ncclFloat64:  F8[idx] = valueF; break;
    case ncclFloat8e5m2:  B1[idx] = rccl_bfloat8(valueF); break;
    case ncclBfloat16: B2[idx] = hip_bfloat16(static_cast<float>(valueF)); break;
    default:
      ERROR("Unsupported datatype\n");
      return TEST_FAIL;
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::Get(ncclDataType_t const dataType, int const idx, int& valueI, double& valueF) const
  {
    switch (dataType)
    {
    case ncclInt8:     valueI = I1[idx]; break;
    case ncclUint8:    valueI = I1[idx]; break;
    case ncclInt32:    valueI = I4[idx]; break;
    case ncclUint32:   valueI = U4[idx]; break;
    case ncclInt64:    valueI = I8[idx]; break;
    case ncclUint64:   valueI = U8[idx]; break;
    case ncclFloat8e4m3:  valueF = float(F1[idx]); break; 
    case ncclFloat16:  valueF = __half2float(F2[idx]); break;
    case ncclFloat32:  valueF = F4[idx]; break;
    case ncclFloat64:  valueF = F8[idx]; break;
    case ncclFloat8e5m2:  valueF = float(B1[idx]); break; 
    case ncclBfloat16: valueF = B2[idx]; break;
    default:
      ERROR("Unsupported datatype\n");
      return TEST_FAIL;
    }
    return TEST_SUCCESS;
  }

  // Multiplies in-place each element by scalarsPerRank[rank]
  ErrCode PtrUnion::Scale(ncclDataType_t const  dataType,
                          size_t         const  numElements,
                          PtrUnion       const& scalarsPerRank,
                          int            const  rank)
  {
    // If no scalars are provided do nothing
    if (scalarsPerRank.ptr == nullptr) return TEST_SUCCESS;

    for (size_t idx = 0; idx < numElements; ++idx)
    {
      switch (dataType)
      {
      case ncclInt8:     I1[idx] *= scalarsPerRank.I1[rank]; break;
      case ncclUint8:    U1[idx] *= scalarsPerRank.U1[rank]; break;
      case ncclInt32:    I4[idx] *= scalarsPerRank.I4[rank]; break;
      case ncclUint32:   U4[idx] *= scalarsPerRank.U4[rank]; break;
      case ncclInt64:    I8[idx] *= scalarsPerRank.I8[rank]; break;
      case ncclUint64:   U8[idx] *= scalarsPerRank.U8[rank]; break;
      case ncclFloat8e4m3:  F1[idx]  = rccl_float8((float)F1[idx] * (float)scalarsPerRank.F1[rank]); break;
      case ncclFloat16:  F2[idx]  = __float2half(__half2float(F2[idx]) * __half2float(scalarsPerRank.F2[rank])); break;
      case ncclFloat32:  F4[idx] *= scalarsPerRank.F4[rank]; break;
      case ncclFloat64:  F8[idx] *= scalarsPerRank.F8[rank]; break;
      case ncclFloat8e5m2:  B1[idx]  = rccl_bfloat8((float)B1[idx] * (float)scalarsPerRank.B1[rank]); break;
      case ncclBfloat16: B2[idx] *= scalarsPerRank.B2[rank]; break;
      default:
        ERROR("Unsupported datatype\n");
        return TEST_FAIL;
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::Reduce(ncclDataType_t const  dataType,
                           size_t         const  numElements,
                           PtrUnion       const& inputCpu,
                           ncclRedOp_t    const  op)
  {
    if (inputCpu.ptr == nullptr)
    {
      ERROR("Input pointer to Reduce should not be nullptr\n");
      return TEST_FAIL;
    }

    for (size_t idx = 0; idx < numElements; ++idx)
    {
      switch (dataType)
      {
      case ncclInt8:     I1[idx] = ReduceOp(op, I1[idx], inputCpu.I1[idx]); break;
      case ncclUint8:    U1[idx] = ReduceOp(op, U1[idx], inputCpu.U1[idx]); break;
      case ncclInt32:    I4[idx] = ReduceOp(op, I4[idx], inputCpu.I4[idx]); break;
      case ncclUint32:   U4[idx] = ReduceOp(op, U4[idx], inputCpu.U4[idx]); break;
      case ncclInt64:    I8[idx] = ReduceOp(op, I8[idx], inputCpu.I8[idx]); break;
      case ncclUint64:   U8[idx] = ReduceOp(op, U8[idx], inputCpu.U8[idx]); break;
      case ncclFloat8e4m3:  F1[idx] = rccl_float8(ReduceOp(op, float(F1[idx]), float(inputCpu.F1[idx]))); break;
      case ncclFloat16:  F2[idx] = __float2half(ReduceOp(op, __half2float(F2[idx]), __half2float(inputCpu.F2[idx]))); break;
      case ncclFloat32:  F4[idx] = ReduceOp(op, F4[idx], inputCpu.F4[idx]); break;
      case ncclFloat64:  F8[idx] = ReduceOp(op, F8[idx], inputCpu.F8[idx]); break;
      case ncclFloat8e5m2:  B1[idx] = rccl_bfloat8(ReduceOp(op, float(B1[idx]), float(inputCpu.B1[idx]))); break;
      case ncclBfloat16: B2[idx] = ReduceOp(op, B2[idx], inputCpu.B2[idx]); break;
      default:
        ERROR("Unsupported datatype\n");
        return TEST_FAIL;
      }
    }
    return TEST_SUCCESS;
  }


  ErrCode PtrUnion::DivideByInt(ncclDataType_t const dataType,
                                size_t         const numElements,
                                int            const divisor)
  {
    for (size_t idx = 0; idx < numElements; ++idx)
    {
      switch (dataType)
      {
      case ncclInt8:     I1[idx] /= divisor; break;
      case ncclUint8:    U1[idx] /= divisor; break;
      case ncclInt32:    I4[idx] /= divisor; break;
      case ncclUint32:   U4[idx] /= divisor; break;
      case ncclInt64:    I8[idx] /= divisor; break;
      case ncclUint64:   U8[idx] /= divisor; break;
      case ncclFloat8e4m3:  F1[idx] = (rccl_float8((float)(F1[idx]) / divisor)); break;
      case ncclFloat16:  F2[idx] = __float2half(__half2float(F2[idx])/divisor); break;
      case ncclFloat32:  F4[idx] /= divisor; break;
      case ncclFloat64:  F8[idx] /= divisor; break;
      case ncclFloat8e5m2:  B1[idx] = (rccl_bfloat8((float)(B1[idx]) / divisor)); break;
      case ncclBfloat16: B2[idx] = (hip_bfloat16((float)(B2[idx]) / divisor)); break;
      default:
        ERROR("Unsupported datatype\n");
        return TEST_FAIL;
      }
    }
    return TEST_SUCCESS;
  }

  ErrCode PtrUnion::IsEqual(ncclDataType_t const  dataType,
                            size_t         const  numElements,
                            PtrUnion       const& expected,
                            bool           const  verbose,
                            bool&                 isMatch)
  {
    isMatch = true;
    size_t idx = 0;
    for (idx = 0; idx < numElements; ++idx)
    {
      switch (dataType)
      {
      case ncclInt8:    isMatch = (I1[idx] == expected.I1[idx]); break;
      case ncclUint8:   isMatch = (U1[idx] == expected.U1[idx]); break;
      case ncclInt32:   isMatch = (I4[idx] == expected.I4[idx]); break;
      case ncclUint32:  isMatch = (U4[idx] == expected.U4[idx]); break;
      case ncclInt64:   isMatch = (I8[idx] == expected.I8[idx]); break;
      case ncclUint64:  isMatch = (U8[idx] == expected.U8[idx]); break;
      case ncclFloat8e4m3: isMatch = (fabs(float(F1[idx]) - float(expected.F1[idx])) < 9e-2); break;
      case ncclFloat16: isMatch = (fabs(__half2float(F2[idx]) - __half2float(expected.F2[idx])) < 9e-2); break;
      case ncclFloat32: isMatch = (fabs(F4[idx] - expected.F4[idx]) < 1e-5); break;
      case ncclFloat64: isMatch = (fabs(F8[idx] - expected.F8[idx]) < 1e-12); break;
      case ncclFloat8e5m2: isMatch = (fabs(float(B1[idx]) - float(expected.B1[idx])) < 9e-2); break;
      case ncclBfloat16: isMatch = (fabs((float)B2[idx] - (float)expected.B2[idx]) < 9e-2); break;
      default:
        ERROR("Unsupported datatype\n");
        return TEST_FAIL;
      }
      if (!isMatch) break;
    }

    if (verbose && !isMatch)
    {
      switch (dataType)
      {
      case ncclInt8:
        ERROR("Expected output: %d.  Actual output: %d at index %lu\n", expected.I1[idx], I1[idx], idx); break;
      case ncclUint8:
        ERROR("Expected output: %u.  Actual output: %u at index %lu\n", expected.U1[idx], U1[idx], idx); break;
      case ncclInt32:
        ERROR("Expected output: %d.  Actual output: %d at index %lu\n", expected.I4[idx], I4[idx], idx); break;
      case ncclUint32:
        ERROR("Expected output: %u.  Actual output: %u at index %lu\n", expected.U4[idx], U4[idx], idx); break;
      case ncclInt64:
        ERROR("Expected output: %ld.  Actual output: %ld at index %lu\n", expected.I8[idx], I8[idx], idx); break;
      case ncclUint64:
        ERROR("Expected output: %lu.  Actual output: %lu at index %lu\n", expected.U8[idx], U8[idx], idx); break;
      case ncclFloat8e4m3:
        ERROR("Expected output: %f.  Actual output: %f at index %lu\n", (float)expected.F1[idx], (float)F1[idx], idx);
      case ncclFloat16:
        ERROR("Expected output: %f.  Actual output: %f at index %lu\n", __half2float(expected.F2[idx]), __half2float(F2[idx]), idx); break;
      case ncclFloat32:
        ERROR("Expected output: %f.  Actual output: %f at index %lu\n", expected.F4[idx], F4[idx], idx); break;
      case ncclFloat64:
        ERROR("Expected output: %lf.  Actual output: %lf at index %lu\n", expected.F8[idx], F8[idx], idx); break;
      case ncclFloat8e5m2:
        ERROR("Expected output: %f.  Actual output: %f at index %lu\n", (float)expected.B1[idx], (float)B1[idx], idx);
      case ncclBfloat16:
        ERROR("Expected output: %f.  Actual output: %f at index %lu\n", (float)expected.B2[idx], (float)B2[idx], idx); break;
      default:
        break;
      }
    }
    return TEST_SUCCESS;
  }

  std::string PtrUnion::ToString(ncclDataType_t const  dataType,
                                 size_t         const  numElements) const
  {
    std::stringstream ss;
    for (int i = 0; i < numElements; i++)
    {
      if (i) ss <<  " ";
      switch (dataType)
      {
      case ncclInt8:     ss << I1[i]; break;
      case ncclUint8:    ss << U1[i]; break;
      case ncclInt32:    ss << I4[i]; break;
      case ncclUint32:   ss << U4[i]; break;
      case ncclInt64:    ss << I8[i]; break;
      case ncclUint64:   ss << U8[i]; break;
      case ncclFloat8e4m3:  ss << (float)F1[i]; break;
      case ncclFloat16:  ss << __half2float(F2[i]); break;
      case ncclFloat32:  ss << F4[i]; break;
      case ncclFloat64:  ss << F8[i]; break;
      case ncclFloat8e5m2:  ss << (float)B1[i]; break;
      case ncclBfloat16: ss << (float)B2[i]; break;
      default: break;
      }
    }
    return ss.str();
  }
}
