find_package(ROCmCMakeBuildTools 0.11.0 CONFIG QUIET PATHS "${ROCM_PATH}")
include(ROCMInstallTargets)

# hip_add_executable: Create HIP executable with EXCLUDE_FROM_ALL
# Required: NAME, SOURCE
# Optional: ADD_TO_TARGET, COMPILE_FLAGS, LINKER_LIBS
function(hip_add_executable)
  cmake_parse_arguments(ARG "" "NAME;SOURCE;ADD_TO_TARGET;COMPILE_FLAGS" "LINKER_LIBS" ${ARGN})

  if(NOT ARG_NAME OR NOT ARG_SOURCE)
    message(FATAL_ERROR "hip_add_executable: NAME and SOURCE required")
  endif()

  if(HIP_PLATFORM STREQUAL "amd")
    set_source_files_properties(${ARG_SOURCE} PROPERTIES LANGUAGE HIP)
  elseif(HIP_PLATFORM STREQUAL "nvidia")
    set_source_files_properties(${ARG_SOURCE} PROPERTIES LANGUAGE CUDA)
  endif()

  add_executable(${ARG_NAME} EXCLUDE_FROM_ALL ${ARG_SOURCE})

  if(ARG_COMPILE_FLAGS)
    target_compile_options(${ARG_NAME} PRIVATE ${ARG_COMPILE_FLAGS})
  endif()

  if(ARG_LINKER_LIBS)
    target_link_libraries(${ARG_NAME} ${ARG_LINKER_LIBS})
  endif()

  # Link CUDA libraries for NVIDIA platform
  if(HIP_PLATFORM STREQUAL "nvidia")
    target_link_libraries(${ARG_NAME} CUDA::cuda_driver)
  endif()

  if(ARG_ADD_TO_TARGET)
    add_dependencies(${ARG_ADD_TO_TARGET} ${ARG_NAME})
  endif()

  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${ARG_NAME})
endfunction()

# hip_add_library: Create HIP library with EXCLUDE_FROM_ALL
# Required: NAME, TYPE (STATIC/SHARED/OBJECT), SOURCES
# Optional: LINKER_LIBS
function(hip_add_library)
  cmake_parse_arguments(ARG "" "NAME;TYPE" "SOURCES;LINKER_LIBS" ${ARGN})

  if(NOT ARG_NAME OR NOT ARG_TYPE OR NOT ARG_SOURCES)
    message(FATAL_ERROR "hip_add_library: NAME, TYPE, and SOURCES required")
  endif()

  add_library(${ARG_NAME} EXCLUDE_FROM_ALL ${ARG_TYPE} ${ARG_SOURCES})

  if(ARG_LINKER_LIBS)
    target_link_libraries(${ARG_NAME} ${ARG_LINKER_LIBS})
  endif()

  if(HIP_PLATFORM STREQUAL "amd")
    set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE HIP)
  elseif(HIP_PLATFORM STREQUAL "nvidia")
    set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE CUDA)
  endif()
endfunction()

# hip_add_device_code: Generate HIP device code files
# Required: TARGET, SOURCE
# Optional: OUTPUT_FORMAT (code|spv|spv-bundle|assembly), OUTPUT_NAME, OFFLOAD_ARCH_OVERRIDE, COMPILE_OPTIONS
function(hip_add_device_code)
  cmake_parse_arguments(ARG "" "TARGET;SOURCE;OUTPUT_FORMAT;OUTPUT_NAME;OFFLOAD_ARCH_OVERRIDE" "COMPILE_OPTIONS" ${ARGN})

  if(NOT ARG_TARGET OR NOT ARG_SOURCE)
    message(FATAL_ERROR "hip_add_device_code: TARGET and SOURCE required")
  endif()

  if(NOT ARG_OUTPUT_NAME)
    set(ARG_OUTPUT_NAME ${ARG_TARGET})
  endif()
  if(NOT ARG_OUTPUT_FORMAT)
    set(ARG_OUTPUT_FORMAT "code")
  endif()

  if(HIP_PLATFORM STREQUAL "nvidia")
    if(ARG_OUTPUT_FORMAT MATCHES "spv|assembly")
      message(FATAL_ERROR "Format '${ARG_OUTPUT_FORMAT}' is AMD-only")
    endif()

    set(OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${ARG_OUTPUT_NAME}")
    add_custom_command(
      OUTPUT ${OUTPUT_PATH}.fatbin ${OUTPUT_PATH}.code
      COMMAND ${CMAKE_CUDA_COMPILER} --fatbin -x cu ${ARG_SOURCE} ${ARG_COMPILE_OPTIONS}
              -o ${OUTPUT_PATH}.fatbin
              -I${HIP_PATH}/include -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${ARG_OUTPUT_NAME}.fatbin ${OUTPUT_PATH}.code
      DEPENDS ${ARG_SOURCE}
      COMMENT "Generating ${ARG_OUTPUT_NAME}.fatbin"
      VERBATIM)
    add_custom_target(${ARG_TARGET} DEPENDS ${OUTPUT_PATH}.code)
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${OUTPUT_PATH}.fatbin ${OUTPUT_PATH}.code)

  elseif(HIP_PLATFORM STREQUAL "amd")
    set(OFFLOAD_ARCH ${ARG_OFFLOAD_ARCH_OVERRIDE})
    if(NOT OFFLOAD_ARCH)
      set(OFFLOAD_ARCH ${OFFLOAD_ARCH_STR})
    endif()

    # Set flags and extension based on format
    if(ARG_OUTPUT_FORMAT STREQUAL "spv")
      set(COMPILER_FLAGS --offload-device-only --offload-arch=amdgcnspirv --no-gpu-bundle-output)
      set(FILE_EXTENSION ".spv")
    elseif(ARG_OUTPUT_FORMAT STREQUAL "spv-bundle")
      set(COMPILER_FLAGS --offload-device-only --offload-arch=amdgcnspirv)
      set(FILE_EXTENSION ".spv")
    elseif(ARG_OUTPUT_FORMAT STREQUAL "assembly")
      set(COMPILER_FLAGS -S ${OFFLOAD_ARCH})
      set(FILE_EXTENSION ".s")
    else() # code
      set(COMPILER_FLAGS --offload-device-only ${OFFLOAD_ARCH})
      set(FILE_EXTENSION ".code")
    endif()

    set(OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${ARG_OUTPUT_NAME}${FILE_EXTENSION}")
    add_custom_command(
      OUTPUT ${OUTPUT_PATH}
      COMMAND ${CMAKE_HIP_COMPILER} ${COMPILER_FLAGS} -x hip ${ARG_SOURCE} ${ARG_COMPILE_OPTIONS}
              -o ${OUTPUT_PATH}
              -I${HIP_PATH}/include -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
      DEPENDS ${ARG_SOURCE}
      COMMENT "Generating ${ARG_OUTPUT_NAME}${FILE_EXTENSION}"
      VERBATIM COMMAND_EXPAND_LISTS)

    add_custom_target(${ARG_TARGET} DEPENDS ${OUTPUT_PATH})
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${OUTPUT_PATH})
  endif()
endfunction()
