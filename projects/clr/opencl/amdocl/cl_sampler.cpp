/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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

#include "cl_common.hpp"

#include "platform/context.hpp"
#include "platform/sampler.hpp"


/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_Samplers
 *
 *  A sampler object describes how to sample an image when the image is read
 *  in the kernel. The built-in functions to read from an image in a kernel
 *  take a sampler as an argument. The sampler arguments to the image read
 *  function can be sampler objects created using OpenCL functions and passed
 *  as argument values to the kernel or can be samplers declared inside
 *  a kernel.
 *
 *  @{
 */

/*! \brief Create a sampler object.
 *
 *  \param context must be a valid OpenCL context.
 *
 *  \param specifies a list of sampler property names and their corresponding
 *  values. Each sampler property name is immediately followed by the
 *  corresponding desired value. The list is terminated with 0. If a supported
 *  property and its value is not specified in sampler_properties, its default
 *  value will be used. sampler_properties can be NULL in which case the default
 *  values for supported sampler properties will be used.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero sampler object and \a errcode_ret is set to
 *  CL_SUCCESS if the sampler object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context.
 *  - CL_INVALID_VALUE if the property name in sampler_properties is not a
 *    supported property name, if the value specified for a supported property
 *    name is not valid, or if the same property name is specified more than
 *    once
 *  - CL_INVALID_OPERATION if images are not supported by any device associated
 *    with context
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 2.0r19
 */
RUNTIME_ENTRY_RET(cl_sampler, clCreateSamplerWithProperties,
                  (cl_context context, const cl_sampler_properties* sampler_properties,
                   cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return (cl_sampler)0;
  }

  cl_bool normalizedCoords = CL_TRUE;
  cl_addressing_mode addressingMode = CL_ADDRESS_CLAMP;
  cl_filter_mode filterMode = CL_FILTER_NEAREST;
#ifndef CL_FILTER_NONE
#define CL_FILTER_NONE 0x1142
#endif
  cl_filter_mode mipFilterMode = CL_FILTER_NONE;
  float minLod = 0.f;
  float maxLod = CL_MAXFLOAT;

  const struct SamplerProperty {
    cl_sampler_properties name;
    union {
      cl_sampler_properties raw;
      cl_bool normalizedCoords;
      cl_addressing_mode addressingMode;
      cl_filter_mode filterMode;
      cl_float lod;
    } value;
  }* p = reinterpret_cast<const SamplerProperty*>(sampler_properties);

  if (p != NULL)
    while (p->name != 0) {
      switch (p->name) {
        case CL_SAMPLER_NORMALIZED_COORDS:
          normalizedCoords = p->value.normalizedCoords;
          break;
        case CL_SAMPLER_ADDRESSING_MODE:
          addressingMode = p->value.addressingMode;
          break;
        case CL_SAMPLER_FILTER_MODE:
          filterMode = p->value.filterMode;
          break;
        case CL_SAMPLER_MIP_FILTER_MODE:
          mipFilterMode = p->value.filterMode;
          break;
        case CL_SAMPLER_LOD_MIN:
          minLod = p->value.lod;
          break;
        case CL_SAMPLER_LOD_MAX:
          maxLod = p->value.lod;
          break;
        default:
          *not_null(errcode_ret) = CL_INVALID_VALUE;
          LogWarning("invalid property name");
          return (cl_sampler)0;
      }
      ++p;
    }

  // Check sampler validity
  // Check addressing mode
  switch (addressingMode) {
    case CL_ADDRESS_NONE:
    case CL_ADDRESS_CLAMP_TO_EDGE:
    case CL_ADDRESS_CLAMP:
      break;
    case CL_ADDRESS_REPEAT:
      if (!normalizedCoords) {
        // repeat mode cannot be used with unnormalized coordinates
        *not_null(errcode_ret) = CL_INVALID_VALUE;
        LogWarning("invalid combination for sampler");
        return (cl_sampler)0;
      }
      break;
    case CL_ADDRESS_MIRRORED_REPEAT:
      if (!normalizedCoords) {
        // repeat mode cannot be used with unnormalized coordinates
        *not_null(errcode_ret) = CL_INVALID_VALUE;
        LogWarning("invalid combination for sampler");
        return (cl_sampler)0;
      }
      break;
    default:
      *not_null(errcode_ret) = CL_INVALID_VALUE;
      LogWarning("invalid addressing mode");
      return (cl_sampler)0;
  }
  // Check filter mode
  switch (filterMode) {
    case CL_FILTER_NEAREST:
    case CL_FILTER_LINEAR:
      break;
    default:
      *not_null(errcode_ret) = CL_INVALID_VALUE;
      LogWarning("invalid filter mode");
      return (cl_sampler)0;
  }
  switch (mipFilterMode) {
    case CL_FILTER_NONE:
    case CL_FILTER_NEAREST:
    case CL_FILTER_LINEAR:
      break;
    default:
      *not_null(errcode_ret) = CL_INVALID_VALUE;
      LogWarning("invalid filter mode");
      return (cl_sampler)0;
  }
  // Create instance of Sampler
  amd::Sampler* sampler =
      new amd::Sampler(*as_amd(context),
                       normalizedCoords == CL_TRUE,  // To get rid of VS warning C4800
                       addressingMode, filterMode, mipFilterMode, minLod, maxLod);
  if (!sampler) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    LogWarning("not enough host memory");
    return (cl_sampler)0;
  }

  if (!sampler->create()) {
    delete sampler;
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    LogWarning("Runtime failed sampler creation!");
    return as_cl<amd::Sampler>(0);
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<amd::Sampler>(sampler);
}
RUNTIME_EXIT

/*! \brief Create a sampler object.
 *
 *  \param context must be a valid OpenCL context.
 *
 *  \param addressing_mode specifies how out of range image coordinates are
 *  handled when reading from an image. This can be set to CL_ADDRESS_REPEAT,
 *  CL_ADDRESS_CLAMP_TO_EDGE, CL_ADDRESS_CLAMP and CL_ADDRESS_NONE.
 *
 *  \param filter_mode specifies the type of filter that must be applied when
 *  reading an image. This can be CL_FILTER_NEAREST or CL_FILTER_LINEAR.
 *
 *  \param normalized_coords determines if the image coordinates specified are
 *  normalized (if \a normalized_coords is not zero) or not (if
 *  \a normalized_coords is zero).
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero sampler object and \a errcode_ret is set to
 *  CL_SUCCESS if the sampler object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context.
 *  - CL_INVALID_VALUE if \a addressing_mode, \a filter_mode or
 *    \a normalized_coords or combination of these argument values are not
 *    valid.
 *  - CL_INVALID_OPERATION if images are not supported by any device associated
 *    with context
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_sampler, clCreateSampler,
                  (cl_context context, cl_bool normalized_coords,
                   cl_addressing_mode addressing_mode, cl_filter_mode filter_mode,
                   cl_int* errcode_ret)) {
  const cl_sampler_properties sprops[] = {CL_SAMPLER_NORMALIZED_COORDS,
                                          static_cast<cl_sampler_properties>(normalized_coords),
                                          CL_SAMPLER_ADDRESSING_MODE,
                                          static_cast<cl_sampler_properties>(addressing_mode),
                                          CL_SAMPLER_FILTER_MODE,
                                          static_cast<cl_sampler_properties>(filter_mode),
                                          0};
  return clCreateSamplerWithProperties(context, sprops, errcode_ret);
}
RUNTIME_EXIT

/*! \brief Increment the sampler reference count.
 *
 *  clCreateSampler does an implicit retain.
 *
 *  \return CL_SUCCESS if the function is executed successfully. It returns
 *  CL_INVALID_SAMPLER if \a sampler is not a valid sampler object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clRetainSampler, (cl_sampler sampler)) {
  if (!is_valid(sampler)) {
    return CL_INVALID_SAMPLER;
  }
  as_amd(sampler)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Decrement the sampler reference count.
 *
 *  The sampler object is deleted after the reference count becomes zero and
 *  commands queued for execution on a command-queue(s) that use sampler have
 *  finished.
 *
 *  \return CL_SUCCESS if the function is executed successfully. It returns
 *  CL_INVALID_SAMPLER if \a sampler is not a valid sampler object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clReleaseSampler, (cl_sampler sampler)) {
  if (!is_valid(sampler)) {
    return CL_INVALID_SAMPLER;
  }
  as_amd(sampler)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Return information about the sampler object.
 *
 *  \param sampler specifies the sampler being queried.
 *
 *  \param param_name specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result
 *  being queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data copied
 *  to \a param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type and
 *    \a param_value is not NULL
 *  - CL_INVALID_SAMPLER if \a sampler is a not a valid sampler object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetSamplerInfo,
              (cl_sampler sampler, cl_sampler_info param_name, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  if (!is_valid(sampler)) {
    return CL_INVALID_SAMPLER;
  }

  switch (param_name) {
    case CL_SAMPLER_REFERENCE_COUNT: {
      cl_uint count = as_amd(sampler)->referenceCount();
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SAMPLER_CONTEXT: {
      cl_context context = as_cl(&as_amd(sampler)->context());
      return amd::clGetInfo(context, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SAMPLER_ADDRESSING_MODE: {
      cl_addressing_mode addressing = as_amd(sampler)->addressingMode();
      return amd::clGetInfo(addressing, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SAMPLER_FILTER_MODE: {
      cl_filter_mode filter = as_amd(sampler)->filterMode();
      return amd::clGetInfo(filter, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SAMPLER_NORMALIZED_COORDS: {
      cl_bool normalized = as_amd(sampler)->normalizedCoords();
      return amd::clGetInfo(normalized, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SAMPLER_MIP_FILTER_MODE: {
      cl_filter_mode mipFilter = as_amd(sampler)->mipFilter();
      return amd::clGetInfo(mipFilter, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SAMPLER_LOD_MIN: {
      cl_float minLod = as_amd(sampler)->minLod();
      return amd::clGetInfo(minLod, param_value_size, param_value, param_value_size_ret);
    }
    case CL_SAMPLER_LOD_MAX: {
      cl_float maxLod = as_amd(sampler)->maxLod();
      return amd::clGetInfo(maxLod, param_value_size, param_value, param_value_size_ret);
    }
    default:
      break;
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
