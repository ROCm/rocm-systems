/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_BACKEND_TYPE_HPP_
#define LIBRARY_SRC_BACKEND_TYPE_HPP_

/**
 * @file backend_type.hpp
 * Defines the Backend derived class types and contains the DISPATCH macros.
 *
 * The type information is required to be known at compile time because
 * we use static dispatch to produce compile time polymorphism.
 *
 * The device cannot use runtime polymorphism because calls through virtual
 * functions are not supported at this time.
 */

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)

namespace rocshmem {

/**
 * @brief Enumerates the Backend derived classes.
 *
 * @note Derived classes which use Backend as a base class must add
 * themselves to this enum class to support static polymorphism.
 */
//enum class BackendType { GDA_BACKEND, RO_BACKEND, IPC_BACKEND };

/**
 * @brief Helper macro for some dispatch calls
 */
#define PAIR(A, B) A, B

/**
 * @brief Device static dispatch method call.
 */
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
#define DISPATCH(Func)                     \
  switch(this->btype) {                    \
  case BackendType::GDA_BACKEND:           \
    static_cast<GDAContext *>(this)->Func; \
    break;                                 \
  case BackendType::RO_BACKEND:            \
    static_cast<ROContext *>(this)->Func;  \
    break;                                 \
  case BackendType::IPC_BACKEND:           \
  default:                                 \
    static_cast<IPCContext *>(this)->Func; \
    break;                                 \
  }
#elif defined(USE_GDA)
#define DISPATCH(Func)                     \
  static_cast<GDAContext *>(this)->Func;
#elif defined(USE_RO)
#define DISPATCH(Func)                     \
  static_cast<ROContext *>(this)->Func;
#elif defined(USE_IPC)
#define DISPATCH(Func)                     \
  static_cast<IPCContext *>(this)->Func;
#endif

/**
 * @brief Device static dispatch method call with a return value.
 */
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
#define DISPATCH_RET(Func)                                \
  if (this->btype == BackendType::GDA_BACKEND) {          \
    auto ret1 = static_cast<GDAContext *>(this)->Func;    \
    return ret1;                                          \
  } else if(this->btype == BackendType::RO_BACKEND) {     \
    auto ret2 = static_cast<ROContext *>(this)->Func;     \
    return ret2;                                          \
  } else {                                                \
    auto ret3 = static_cast<IPCContext *>(this)->Func;    \
    return ret3;                                          \
  }
#elif defined(USE_GDA)
#define DISPATCH_RET(Func)                              \
  auto ret_val = static_cast<GDAContext *>(this)->Func; \
  return ret_val;
#elif defined(USE_RO)
#define DISPATCH_RET(Func)                              \
  auto ret_val = static_cast<ROContext *>(this)->Func;  \
  return ret_val;
#elif defined(USE_IPC)
#define DISPATCH_RET(Func)                              \
  auto ret_val = static_cast<IPCContext *>(this)->Func; \
  return ret_val;
#endif

/**
 * @brief Device static dispatch method call with a return type of pointer.
 */
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
#define DISPATCH_RET_PTR(Func)                       \
  void *ret_val{nullptr};                            \
  switch(this->btype) {                              \
  case BackendType::GDA_BACKEND:                     \
    ret_val = static_cast<GDAContext *>(this)->Func; \
    break;                                           \
  case BackendType::RO_BACKEND:                      \
    ret_val = static_cast<ROContext *>(this)->Func;  \
    break;                                           \
  case BackendType::IPC_BACKEND:                     \
  default:                                           \
    ret_val = static_cast<IPCContext *>(this)->Func; \
    break;                                           \
  }                                                  \
  return ret_val;
#elif defined(USE_GDA)
#define DISPATCH_RET_PTR(Func)                     \
  void *ret_val{nullptr};                          \
  ret_val = static_cast<GDAContext *>(this)->Func; \
  return ret_val;
#elif defined(USE_RO)
#define DISPATCH_RET_PTR(Func)                     \
  void *ret_val{nullptr};                          \
  ret_val = static_cast<ROContext *>(this)->Func;  \
  return ret_val;
#elif defined(USE_IPC)
#define DISPATCH_RET_PTR(Func)                     \
  void *ret_val{nullptr};                          \
  ret_val = static_cast<IPCContext *>(this)->Func; \
  return ret_val;
#endif

/**
 * @brief Host static dispatch method call.
 *
 * @note There is no need to lock-unlock on host since we are using
 * MPI_THREAD_MULTIPLE (for RMA and AMO operations) and the ordering and
 * threading semantics of collectives in OpenSHMEM match those of MPI.
 */
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
#define HOST_DISPATCH(Func)                    \
  switch(this->btype) {                        \
  case BackendType::GDA_BACKEND:               \
    static_cast<GDAHostContext *>(this)->Func; \
    break;                                     \
  case BackendType::RO_BACKEND:                \
    static_cast<ROHostContext *>(this)->Func;  \
    break;                                     \
  case BackendType::IPC_BACKEND:               \
  default:                                     \
    static_cast<IPCHostContext *>(this)->Func; \
    break;                                     \
  }
#elif defined(USE_GDA)
#define HOST_DISPATCH(Func) static_cast<GDAHostContext *>(this)->Func;
#elif defined(USE_RO)
#define HOST_DISPATCH(Func) static_cast<ROHostContext *>(this)->Func;
#elif defined(USE_IPC)
#define HOST_DISPATCH(Func) static_cast<IPCHostContext *>(this)->Func;
#endif

/**
 * @brief Host static dispatch method call with return value.
 *
 * @note There is no need to lock-unlock on host since we are using
 * MPI_THREAD_MULTIPLE (for RMA and AMO operations) and the ordering and
 * threading semantics of collectives in OpenSHMEM match those of MPI.
 */
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
#define HOST_DISPATCH_RET(Func)                               \
  if (this->btype == BackendType::GDA_BACKEND) {              \
    auto ret1 = static_cast<GDAHostContext *>(this)->Func;    \
    return ret1;                                              \
  } else if (this->btype == BackendType::RO_BACKEND) {        \
    auto ret2 = static_cast<ROHostContext *>(this)->Func;     \
    return ret2;                                              \
  } else {                                                    \
    auto ret3 = static_cast<IPCHostContext *>(this)->Func;    \
    return ret3;                                              \
  }
#elif defined(USE_GDA)
#define HOST_DISPATCH_RET(Func)                             \
  auto ret_val = static_cast<GDAHostContext *>(this)->Func; \
  return ret_val;
#elif defined(USE_RO)
#define HOST_DISPATCH_RET(Func)                             \
  auto ret_val = static_cast<ROHostContext *>(this)->Func;  \
  return ret_val;
#elif defined(USE_IPC)
#define HOST_DISPATCH_RET(Func)                             \
  auto ret_val = static_cast<IPCHostContext *>(this)->Func; \
  return ret_val;
#endif

/**
 * @brief Host static dispatch method call with a return type of pointer.
 */
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
#define HOST_DISPATCH_RET_PTR(Func)                      \
  void *ret_val{nullptr};                                \
  switch(this->btype) {                                  \
  case BackendType::GDA_BACKEND:                         \
    ret_val = static_cast<GDAHostContext *>(this)->Func; \
    break;                                               \
  case BackendType::RO_BACKEND:                          \
    ret_val = static_cast<ROHostContext *>(this)->Func;  \
    break;                                               \
  case BackendType::IPC_BACKEND:                         \
  default:                                               \
    ret_val = static_cast<IPCHostContext *>(this)->Func; \
    break;                                               \
  }                                                      \
  return ret_val;
#elif defined(USE_GDA)
#define HOST_DISPATCH_RET_PTR(Func)                    \
  void *ret_val{nullptr};                              \
  ret_val = static_cast<GDAHostContext *>(this)->Func; \
  return ret_val;
#elif defined(USE_RO)
#define HOST_DISPATCH_RET_PTR(Func)                    \
  void *ret_val{nullptr};                              \
  ret_val = static_cast<ROHostContext *>(this)->Func;  \
  return ret_val;
#elif defined(USE_IPC)
#define HOST_DISPATCH_RET_PTR(Func)                    \
  void *ret_val{nullptr};                              \
  ret_val = static_cast<IPCHostContext *>(this)->Func; \
  return ret_val;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_BACKEND_TYPE_HPP_
