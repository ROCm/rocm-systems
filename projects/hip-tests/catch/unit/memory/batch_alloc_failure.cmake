# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Build allocator failure injection only for isolated test children.
if(HIP_PLATFORM MATCHES "amd" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(
    hip_batch_alloc_failure
    SHARED
    EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/batch_alloc_failure.cc"
  )
  target_compile_features(hip_batch_alloc_failure PRIVATE cxx_std_17)
  target_link_libraries(hip_batch_alloc_failure PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(
    hip_batch_alloc_failure
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
  foreach(
    batch_target
    MemoryTest1
    hipMemcpyBatchAsync
    hipMemcpy3DBatchAsync
    hipExtMemcpyBatchAsync
  )
    if(TARGET ${batch_target})
      add_dependencies(${batch_target} hip_batch_alloc_failure)
    endif()
  endforeach()
  install(
    TARGETS hip_batch_alloc_failure
    LIBRARY
      DESTINATION "${CMAKE_INSTALL_DATADIR}/hip/${CATCH_BUILD_DIR}"
      OPTIONAL
  )
endif()
