add_executable(clinfo clinfo.cpp)

target_compile_definitions(clinfo PRIVATE CL_TARGET_OPENCL_VERSION=220 HAVE_CL2_HPP)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
find_package(AMD_ICD)

#todo: to be updated to use header files from other repos
target_include_directories(clinfo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../khronos/headers/opencl2.2")

find_library(AMD_ICD_LIBRARY OpenCL HINTS "${AMD_ICD_LIBRARY_DIR}")
target_link_libraries(clinfo PRIVATE ${AMD_ICD_LIBRARY})

INSTALL(TARGETS clinfo
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
