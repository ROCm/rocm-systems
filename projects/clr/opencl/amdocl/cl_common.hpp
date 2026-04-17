/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef CL_COMMON_HPP_
#define CL_COMMON_HPP_

#ifdef _WIN32
#include <CL/cl_d3d11.h>
#include <CL/cl_d3d10.h>
#include <CL/cl_dx9_media_sharing.h>
#endif
#include <CL/cl_icd.h>

#include "top.hpp"
#include "utils/util.hpp"

// ICDDispatchedObject must be defined before vdi_common.hpp, which references it.
typedef struct _cl_icd_dispatch cl_icd_dispatch;

// Maps OpenCL opaque handle types to their amd::* implementations
#define KHR_CL_TYPES_DO(F)                \
  F(cl_context,       amd::Context)       \
  F(cl_event,         amd::Event)         \
  F(cl_command_queue, amd::CommandQueue)  \
  F(cl_kernel,        amd::Kernel)        \
  F(cl_program,       amd::Program)       \
  F(cl_device_id,     amd::Device)        \
  F(cl_mem,           amd::Memory)        \
  F(cl_sampler,       amd::Sampler)

#define AMD_CL_TYPES_DO(F)                    \
  F(cl_counter_amd,     amd::Counter)         \
  F(cl_perfcounter_amd, amd::PerfCounter)     \
  F(cl_threadtrace_amd, amd::ThreadTrace)

#define CL_TYPES_DO(F)  \
  KHR_CL_TYPES_DO(F)    \
  AMD_CL_TYPES_DO(F)

// Define AMD-specific cl_* struct types (KHR types come from <CL/cl_icd.h>)
#define DECLARE_AMD_CL_TYPE(CL, AMD)    \
  typedef struct _##CL {                \
    cl_icd_dispatch* dispatch;          \
  }* CL;

AMD_CL_TYPES_DO(DECLARE_AMD_CL_TYPE);

#undef DECLARE_AMD_CL_TYPE

namespace amd {

// Define the cl_*_type tokens for type checking.
//

#define DEFINE_CL_TOKENS(CL, ignored) T##CL,

enum cl_token { Tinvalid = 0, CL_TYPES_DO(DEFINE_CL_TOKENS) numTokens };

#undef DEFINE_CL_TOKENS

const size_t RuntimeObjectAlignment = NextPowerOfTwo<numTokens>::value;

//! \cond ignore
template <typename T> struct as_internal {
  typedef void type;
};

template <typename T> struct as_external {
  typedef void type;
};

template <typename T> struct class_token {
  static const cl_token value = Tinvalid;
};

#define DEFINE_CL_TRAITS(CL, AMD)                         \
                                                          \
  template <> struct class_token<AMD> {                   \
    static const cl_token value = T##CL;                  \
  };                                                      \
                                                          \
  template <> struct as_internal<_##CL> {                 \
    typedef AMD type;                                     \
  };                                                      \
  template <> struct as_internal<const _##CL> {           \
    typedef AMD const type;                               \
  };                                                      \
                                                          \
  template <> struct as_external<AMD> {                   \
    typedef _##CL type;                                   \
  };                                                      \
  template <> struct as_external<const AMD> {             \
    typedef _##CL const type;                             \
  };

CL_TYPES_DO(DEFINE_CL_TRAITS);

#undef DEFINE_CL_TRAITS
//! \endcond

struct ICDDispatchedObject {
#ifdef __HIP_PLATFORM_AMD__
  static inline cl_icd_dispatch icdVendorDispatch_[] = {0};
#else
  static cl_icd_dispatch icdVendorDispatch_[];
#endif
  const cl_icd_dispatch* const dispatch_;

 protected:
  ICDDispatchedObject() : dispatch_(icdVendorDispatch_) {}

 public:
  static bool isValidHandle(const void* handle) { return handle != NULL; }

  const void* handle() const { return static_cast<const ICDDispatchedObject*>(this); }
  void* handle() { return static_cast<ICDDispatchedObject*>(this); }

  template <typename T> static const T* fromHandle(const void* handle) {
    return static_cast<const T*>(reinterpret_cast<const ICDDispatchedObject*>(handle));
  }
  template <typename T> static T* fromHandle(void* handle) {
    return static_cast<T*>(reinterpret_cast<ICDDispatchedObject*>(handle));
  }
};

}  // namespace amd

template <typename CL> typename amd::as_internal<CL>::type* as_amd(CL* cl_obj) {
  return cl_obj == NULL ? NULL
                        : amd::ICDDispatchedObject::fromHandle<typename amd::as_internal<CL>::type>(
                              static_cast<void*>(cl_obj));
}

template <typename AMD> typename amd::as_external<AMD>::type* as_cl(AMD* amd_obj) {
  return amd_obj == NULL ? NULL
                         : static_cast<typename amd::as_external<AMD>::type*>(amd_obj->handle());
}

template <typename CL> bool is_valid(CL* handle) {
  return amd::as_internal<CL>::type::isValidHandle(handle);
}

#include "vdi_common.hpp"
#include "cl_type_map.hpp"

namespace amd {

//! Helper function to check "properties" parameter in various functions
int checkContextProperties(const cl_context_properties* properties, bool* offlineDevices);

template <typename T> static inline cl_int clGetInfo(T& field, size_t param_value_size,
                                                     void* param_value,
                                                     size_t* param_value_size_ret) {
  const void* valuePtr;
  size_t valueSize;

  std::tie(valuePtr, valueSize) =
      detail::ParamInfo<typename std::remove_const<T>::type>::get(field);

  *not_null(param_value_size_ret) = valueSize;

  cl_int ret = CL_SUCCESS;
  if (param_value != NULL && param_value_size < valueSize) {
    if ((param_value_size == 0) || !std::is_pointer<T>() ||
        !std::is_same<typename std::remove_const<typename std::remove_pointer<T>::type>::type,
                      char>()) {
      return CL_INVALID_VALUE;
    }
    // For char* and char[] params, we will at least fill up to
    // param_value_size, then return an error.
    valueSize = param_value_size;
    static_cast<char*>(param_value)[--valueSize] = '\0';
    ret = CL_INVALID_VALUE;
  }

  if (param_value != NULL) {
    ::memcpy(param_value, valuePtr, valueSize);
    if (param_value_size > valueSize) {
      ::memset(static_cast<address>(param_value) + valueSize, '\0', param_value_size - valueSize);
    }
  }

  return ret;
}

static inline cl_int clSetEventWaitList(Command::EventWaitList& eventWaitList,
                                        const amd::HostQueue& hostQueue,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event* event_wait_list) {
  if ((num_events_in_wait_list == 0 && event_wait_list != NULL) ||
      (num_events_in_wait_list != 0 && event_wait_list == NULL)) {
    return CL_INVALID_EVENT_WAIT_LIST;
  }

  while (num_events_in_wait_list-- > 0) {
    cl_event event = *event_wait_list++;
    Event* amdEvent = as_amd(event);
    if (!is_valid(event)) {
      return CL_INVALID_EVENT_WAIT_LIST;
    }
    if (&hostQueue.context() != &amdEvent->context()) {
      return CL_INVALID_CONTEXT;
    }
    if ((amdEvent->command().queue() != &hostQueue) && !amdEvent->notifyCmdQueue()) {
      return CL_INVALID_EVENT_WAIT_LIST;
    }
    eventWaitList.push_back(amdEvent);
  }
  return CL_SUCCESS;
}

//! Common function declarations for CL-external graphics API interop
cl_int clEnqueueAcquireExtObjectsAMD(cl_command_queue command_queue, cl_uint num_objects,
                                     const cl_mem* mem_objects, cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list, cl_event* event,
                                     cl_command_type cmd_type);
cl_int clEnqueueReleaseExtObjectsAMD(cl_command_queue command_queue, cl_uint num_objects,
                                     const cl_mem* mem_objects, cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list, cl_event* event,
                                     cl_command_type cmd_type);
static inline cl_int clDXTranslateErrorCode(cl_int err) {
  return err == CL_INVALID_GL_OBJECT ? CL_INVALID_MEM_OBJECT : err;
}

}  // namespace amd

extern "C" {

#if defined(CL_VERSION_1_1)
extern CL_API_ENTRY cl_int CL_API_CALL clSetCommandQueueProperty(
    cl_command_queue command_queue, cl_command_queue_properties properties, cl_bool enable,
    cl_command_queue_properties* old_properties) CL_API_SUFFIX__VERSION_1_0;
#endif  // CL_VERSION_1_1

extern CL_API_ENTRY cl_mem CL_API_CALL clConvertImageAMD(cl_context context, cl_mem image,
                                                         const cl_image_format* image_format,
                                                         cl_int* errcode_ret);

extern CL_API_ENTRY cl_mem CL_API_CALL clCreateBufferFromImageAMD(cl_context context, cl_mem image,
                                                                  cl_int* errcode_ret);

extern CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithAssemblyAMD(cl_context context,
                                                                          cl_uint count,
                                                                          const char** strings,
                                                                          const size_t* lengths,
                                                                          cl_int* errcode_ret);

}  // extern "C"

//! \endcond

#endif /*CL_COMMON_HPP_*/
