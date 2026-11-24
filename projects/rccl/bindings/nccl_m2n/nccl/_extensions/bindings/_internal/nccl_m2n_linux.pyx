# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

from libc.stdint cimport intptr_t

import os
import threading

from nccl._extensions._runtime import bundled_library, is_rocm
from .utils import FunctionNotFoundError


###############################################################################
# Extern
###############################################################################

cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()
    void* dlsym(void*, const char*)
    int dlclose(void*)

    enum:
        RTLD_NOW
        RTLD_GLOBAL

###############################################################################
# Library resolution
###############################################################################

def _candidate_library_paths() -> list[str]:
    explicit = os.environ.get("NCCL_M2N_LIBRARY")
    if explicit:
        return [explicit]

    # With no explicit override, prefer the native library bundled with this
    # facade before environment and SONAME fallbacks.
    bundled = bundled_library("nccl_m2n")
    candidates = [bundled] if bundled is not None else []

    home = os.environ.get("NCCL_M2N_HOME")
    if home:
        candidates.append(os.path.join(home, "lib", "libnccl_m2n.so"))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for subdir in ("lib", "lib64"):
            candidates.append(os.path.join(conda_prefix, subdir, "libnccl_m2n.so"))

    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_var)
        if root:
            for subdir in ("lib", "lib64"):
                candidates.append(os.path.join(root, subdir, "libnccl_m2n.so"))

    candidates.append("libnccl_m2n.so")
    return candidates


cdef void* load_library() except* with gil:
    if not is_rocm():
        # On ROCm libnccl_m2n resolves librccl.so.1 through its own DT_NEEDED.
        from cuda.pathfinder import load_nvidia_dynamic_lib

        load_nvidia_dynamic_lib("nccl")

    cdef void* handle = NULL
    cdef bytes path_bytes
    cdef char* err_msg
    errors = []

    for path in _candidate_library_paths():
        if path != "libnccl_m2n.so" and not os.path.exists(path):
            errors.append(f"{path}: not found")
            continue
        path_bytes = path.encode()
        handle = dlopen(path_bytes, RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            if is_rocm():
                from nccl._extensions._rocm_check import check_loaded_library

                check_loaded_library(path)
            return handle
        err_msg = dlerror()
        if err_msg != NULL:
            errors.append(f"{path}: {err_msg.decode()}")
        else:
            errors.append(f"{path}: dlopen failed")

    detail = ""
    if is_rocm():
        from nccl._extensions._rocm_check import load_failure_detail

        detail = f" ({load_failure_detail()})"
    raise RuntimeError(
        "Failed to dlopen libnccl_m2n.so. Set NCCL_M2N_LIBRARY to the "
        "shared library path or NCCL_M2N_HOME to an install prefix. Tried: "
        + "; ".join(errors) + detail
    )


###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_nccl_m2n_init = False
cdef void* __library_handle = NULL

cdef void* __ncclM2nInit = NULL
cdef void* __ncclM2nFinalize = NULL
cdef void* __ncclM2nGroupStart = NULL
cdef void* __ncclM2nGroupEnd = NULL
cdef void* __ncclM2nGroupAbort = NULL
cdef void* __ncclM2nGetLastError = NULL
cdef void* __ncclReshardWithWindow = NULL
cdef void* __ncclReshard = NULL


cdef int _check_or_init_nccl_m2n() except -1 nogil:
    global __py_nccl_m2n_init
    if __py_nccl_m2n_init:
        return 0

    cdef void* handle = NULL
    cdef void* init_fn = NULL
    cdef void* finalize_fn = NULL
    cdef void* group_start_fn = NULL
    cdef void* group_end_fn = NULL
    cdef void* group_abort_fn = NULL
    cdef void* get_last_error_fn = NULL
    cdef void* reshard_with_window_fn = NULL
    cdef void* reshard_fn = NULL

    with gil, __symbol_lock:
        if __py_nccl_m2n_init:
            return 0

        global __ncclM2nInit
        global __ncclM2nFinalize
        global __ncclM2nGroupStart
        global __ncclM2nGroupEnd
        global __ncclM2nGroupAbort
        global __ncclM2nGetLastError
        global __ncclReshardWithWindow
        global __ncclReshard

        handle = load_library()
        init_fn = dlsym(handle, 'ncclM2nInit')
        finalize_fn = dlsym(handle, 'ncclM2nFinalize')
        group_start_fn = dlsym(handle, 'ncclM2nGroupStart')
        group_end_fn = dlsym(handle, 'ncclM2nGroupEnd')
        group_abort_fn = dlsym(handle, 'ncclM2nGroupAbort')
        get_last_error_fn = dlsym(handle, 'ncclM2nGetLastError')
        reshard_with_window_fn = dlsym(handle, 'ncclReshardWithWindow')
        reshard_fn = dlsym(handle, 'ncclReshard')

        missing = []
        if init_fn == NULL:
            missing.append("ncclM2nInit")
        if finalize_fn == NULL:
            missing.append("ncclM2nFinalize")
        if group_start_fn == NULL:
            missing.append("ncclM2nGroupStart")
        if group_end_fn == NULL:
            missing.append("ncclM2nGroupEnd")
        if group_abort_fn == NULL:
            missing.append("ncclM2nGroupAbort")
        if get_last_error_fn == NULL:
            missing.append("ncclM2nGetLastError")
        if reshard_with_window_fn == NULL:
            missing.append("ncclReshardWithWindow")
        if reshard_fn == NULL:
            missing.append("ncclReshard")
        if missing:
            dlclose(handle)
            raise FunctionNotFoundError(
                "libnccl_m2n.so does not provide the complete M2N v2 API; "
                "missing: " + ", ".join(missing)
            )

        global __library_handle
        __library_handle = handle
        __ncclM2nInit = init_fn
        __ncclM2nFinalize = finalize_fn
        __ncclM2nGroupStart = group_start_fn
        __ncclM2nGroupEnd = group_end_fn
        __ncclM2nGroupAbort = group_abort_fn
        __ncclM2nGetLastError = get_last_error_fn
        __ncclReshardWithWindow = reshard_with_window_fn
        __ncclReshard = reshard_fn

        __py_nccl_m2n_init = True
        return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nccl_m2n()
    cdef dict data = {}

    global __ncclM2nInit
    data["__ncclM2nInit"] = <intptr_t>__ncclM2nInit

    global __ncclM2nFinalize
    data["__ncclM2nFinalize"] = <intptr_t>__ncclM2nFinalize

    global __ncclM2nGroupStart
    data["__ncclM2nGroupStart"] = <intptr_t>__ncclM2nGroupStart

    global __ncclM2nGroupEnd
    data["__ncclM2nGroupEnd"] = <intptr_t>__ncclM2nGroupEnd

    global __ncclM2nGroupAbort
    data["__ncclM2nGroupAbort"] = <intptr_t>__ncclM2nGroupAbort

    global __ncclM2nGetLastError
    data["__ncclM2nGetLastError"] = <intptr_t>__ncclM2nGetLastError

    global __ncclReshardWithWindow
    data["__ncclReshardWithWindow"] = <intptr_t>__ncclReshardWithWindow

    global __ncclReshard
    data["__ncclReshard"] = <intptr_t>__ncclReshard

    func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]


###############################################################################
# Wrapper functions
###############################################################################

cdef ncclResult_t _ncclM2nInit(ncclM2nHandle_t* handle, const ncclM2nConfig_t* config) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclM2nInit
    _check_or_init_nccl_m2n()
    if __ncclM2nInit == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclM2nInit is not found")
    return (<ncclResult_t (*)(ncclM2nHandle_t*, const ncclM2nConfig_t*) noexcept nogil>__ncclM2nInit)(handle, config)


cdef ncclResult_t _ncclM2nFinalize(ncclM2nHandle_t handle) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclM2nFinalize
    _check_or_init_nccl_m2n()
    if __ncclM2nFinalize == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclM2nFinalize is not found")
    return (<ncclResult_t (*)(ncclM2nHandle_t) noexcept nogil>__ncclM2nFinalize)(handle)


cdef ncclResult_t _ncclM2nGroupStart() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclM2nGroupStart
    _check_or_init_nccl_m2n()
    if __ncclM2nGroupStart == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclM2nGroupStart is not found")
    return (<ncclResult_t (*)() noexcept nogil>__ncclM2nGroupStart)()


cdef ncclResult_t _ncclM2nGroupEnd() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclM2nGroupEnd
    _check_or_init_nccl_m2n()
    if __ncclM2nGroupEnd == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclM2nGroupEnd is not found")
    return (<ncclResult_t (*)() noexcept nogil>__ncclM2nGroupEnd)()


cdef ncclResult_t _ncclM2nGroupAbort() except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclM2nGroupAbort
    _check_or_init_nccl_m2n()
    if __ncclM2nGroupAbort == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclM2nGroupAbort is not found")
    return (<ncclResult_t (*)() noexcept nogil>__ncclM2nGroupAbort)()


cdef const char* _ncclM2nGetLastError() noexcept nogil:
    global __ncclM2nGetLastError
    _check_or_init_nccl_m2n()
    if __ncclM2nGetLastError == NULL:
        return NULL
    return (<const char* (*)() noexcept nogil>__ncclM2nGetLastError)()


cdef ncclResult_t _ncclReshardWithWindow(ncclM2nHandle_t handle, ncclComm_t comm, ncclWindow_t window, const ncclDistTensor_t* src, const ncclDistTensor_t* dst, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclReshardWithWindow
    _check_or_init_nccl_m2n()
    if __ncclReshardWithWindow == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclReshardWithWindow is not found")
    return (<ncclResult_t (*)(ncclM2nHandle_t, ncclComm_t, ncclWindow_t, const ncclDistTensor_t*, const ncclDistTensor_t*, cudaStream_t) noexcept nogil>__ncclReshardWithWindow)(
        handle, comm, window, src, dst, stream)


cdef ncclResult_t _ncclReshard(ncclM2nHandle_t handle, ncclComm_t comm, const ncclDistTensor_t* src, const ncclDistTensor_t* dst, cudaStream_t stream) except?_NCCLRESULT_T_INTERNAL_LOADING_ERROR nogil:
    global __ncclReshard
    _check_or_init_nccl_m2n()
    if __ncclReshard == NULL:
        with gil:
            raise FunctionNotFoundError("function ncclReshard is not found")
    return (<ncclResult_t (*)(ncclM2nHandle_t, ncclComm_t, const ncclDistTensor_t*, const ncclDistTensor_t*, cudaStream_t) noexcept nogil>__ncclReshard)(
        handle, comm, src, dst, stream)
