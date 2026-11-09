if(CLR_HRR_CODEGEN_CHECK)
  # Regenerates into the binary tree and verifies the checked-in files, which remain the compilation inputs.
  set(HRR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../hrr")
  set(HRR_CLR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
  set(HRR_HIP_SOURCE_DIR "${HIP_COMMON_DIR}")
  include("${HRR_CLR_SOURCE_DIR}/cmake/HrrCodegen.cmake")
  set(HRR_CODEGEN_CHECK_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/hrr")
  hrr_add_codegen_check(
    TARGET hrr_codegen_check
    HRR_SOURCE_DIR "${HRR_SOURCE_DIR}"
    CLR_SOURCE_DIR "${HRR_CLR_SOURCE_DIR}"
    HIP_SOURCE_DIR "${HRR_HIP_SOURCE_DIR}"
    OUTPUT_DIR "${HRR_CODEGEN_CHECK_DIR}")
  add_dependencies(amdhip64 hrr_codegen_check)
endif()

target_sources(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_metadata.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_scrub.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_generated.cpp
)

# The shared HRR wire format lives in projects/hrr/include, included as "hrr/hrr_api_args.h".
target_include_directories(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../hrr/include
)
