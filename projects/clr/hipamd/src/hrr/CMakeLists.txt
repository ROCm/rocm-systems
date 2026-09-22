# HRR in-tree capture layer. The generated capture shim and wire-format
# header are build artifacts so a CLR build never writes into the source tree.
set(HRR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../hrr")
set(HRR_CLR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(HRR_HIP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../hip")

include("${HRR_CLR_SOURCE_DIR}/cmake/HrrCodegen.cmake")
set(HRR_CAPTURE_CODEGEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/hrr")
hrr_add_codegen(
  TARGET hrr_capture_codegen
  HRR_SOURCE_DIR "${HRR_SOURCE_DIR}"
  CLR_SOURCE_DIR "${HRR_CLR_SOURCE_DIR}"
  HIP_SOURCE_DIR "${HRR_HIP_SOURCE_DIR}"
  OUTPUT_DIR "${HRR_CAPTURE_CODEGEN_DIR}"
  CAPTURE
)

set_source_files_properties("${HRR_CAPTURE_CODEGEN_DIR}/hip_capture_generated.cpp" PROPERTIES GENERATED TRUE)
target_sources(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_metadata.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_writer.cpp
  ${HRR_CAPTURE_CODEGEN_DIR}/hip_capture_generated.cpp
)
add_dependencies(amdhip64 hrr_capture_codegen)

target_include_directories(amdhip64 PRIVATE
  ${HRR_CAPTURE_CODEGEN_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
