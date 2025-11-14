project(libMangment)

cmake_minimum_required(VERSION 3.10)

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
  endif()
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()
# Find hip
find_package(hip)

# Set codddmpiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

# additionla things end

add_library(CommonLib Common.cpp)
target_include_directories(CommonLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(MultiplyKernel.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${CMAKE_CURRENT_SOURCE_DIR}/MultiplyKernel.cpp -o MultiplyKernel.code)

add_library(LibMul Multiplication.cpp)
target_include_directories(LibMul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LibMul PUBLIC CommonLib)

add_dependencies(LibMul MultiplyKernel.code)

add_library(LibAdd Addition.cpp)
target_include_directories(LibAdd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LibAdd PUBLIC CommonLib)

#add_executable(app Application.cpp)
#target_link_libraries(app PRIVATE LibMul LibAdd)

add_custom_target(build_libMangment)

# Create the excutable
if(TARGET build_libMangment)
set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
set(EXCLUDE_OPTION )
endif()
add_executable(libMangnt ${EXCLUDE_OPTION} Application.cpp)

# Link with HIP
target_link_libraries(libMangnt hip::host LibMul LibAdd)

if(TARGET build_libMangment)
  add_dependencies(build_libMangment libMangnt)
endif()


