# HRR In-Tree Capture Layer
# Adds capture source files to the amdhip64 target.
# No separate library — capture is compiled into amdhip64 directly.
# Regenerate HRR sources in the binary tree and verify that the checked-in
# generated files are current. The checked-in files remain the compilation inputs.
set(HRR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../hrr")
set(HRR_CLR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(HRR_HIP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../hip")
include("${HRR_CLR_SOURCE_DIR}/cmake/HrrCodegen.cmake")
set(HRR_CODEGEN_CHECK_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/hrr")
hrr_add_codegen_check(
  TARGET hrr_codegen_check
  HRR_SOURCE_DIR "${HRR_SOURCE_DIR}"
  CLR_SOURCE_DIR "${HRR_CLR_SOURCE_DIR}"
  HIP_SOURCE_DIR "${HRR_HIP_SOURCE_DIR}"
  OUTPUT_DIR "${HRR_CODEGEN_CHECK_DIR}")


add_dependencies(amdhip64 hrr_codegen_check)

target_sources(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_metadata.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_generated.cpp
)

# Capture headers live here; the shared HRR archive wire format now lives in
# projects/hrr/include and is included as "hrr/hrr_api_args.h".
# Playback tools (hrr-playback, hrr_reader) now live in projects/hrr.
target_include_directories(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../hrr/include
)
