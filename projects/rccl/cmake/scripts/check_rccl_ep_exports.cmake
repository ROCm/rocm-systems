# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# See LICENSE.txt for license information
#
# Post-build check for ENABLE_RCCL_EP_IN_LIBRCCL: librccl.so has to export rccl_ep's C ABI.
# The NCCL 2.31.2 sync once dropped the device-linker rule that compiles rccl_ep into librccl.so;
# the option then built nothing and no build or test noticed, since the rccl_ep tests compile the
# C ABI into their own binary. This fails the build instead.
#   cmake -DLIB=<librccl.so> -DNM=<nm> -P check_rccl_ep_exports.cmake
execute_process(COMMAND "${NM}" -D --defined-only "${LIB}"
                OUTPUT_VARIABLE _syms ERROR_VARIABLE _err RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "check_rccl_ep_exports: ${NM} -D ${LIB} failed (${_rc}): ${_err}")
endif()
set(_missing "")
foreach(_sym ep_unique_id_size ep_get_unique_id ep_create ep_configure ep_window_bytes ep_barrier ep_plan
             ep_dispatch ep_recv_counts ep_expert_counts ep_expand_build ep_expand_scatter ep_combine ep_destroy)
  if(NOT _syms MATCHES "[ \t]${_sym}(\n|$)")
    list(APPEND _missing ${_sym})
  endif()
endforeach()
if(_missing)
  list(JOIN _missing ", " _missing)
  message(FATAL_ERROR
    "ENABLE_RCCL_EP_IN_LIBRCCL is ON, but ${LIB} does not export ${_missing}. "
    "The device-linker rule that compiles rccl_ep_capi.hip into librccl.so (cmake/DeviceLinker.cmake) "
    "is missing or no longer runs.")
endif()
