if(TARGET OpenCL)
  set(CLINFO_OPENCL_LIB OpenCL)
else()
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
  find_package(AMD_ICD)
  find_library(AMD_ICD_LIBRARY OpenCL HINTS "${AMD_ICD_LIBRARY_DIR}")
  if(AMD_ICD_LIBRARY)
    set(CLINFO_OPENCL_LIB ${AMD_ICD_LIBRARY})
  else()
    message(WARNING "OpenCL ICD loader not found, skipping clinfo build.")
    return()
  endif()
endif()

add_executable(clinfo clinfo.cpp)

target_compile_definitions(clinfo PRIVATE CL_TARGET_OPENCL_VERSION=220 HAVE_CL2_HPP)

#todo: to be updated to use header files from other repos
target_include_directories(clinfo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../khronos/headers/opencl2.2")

target_link_libraries(clinfo PRIVATE ${CLINFO_OPENCL_LIB})

INSTALL(TARGETS clinfo
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
