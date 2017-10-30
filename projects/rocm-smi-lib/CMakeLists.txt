#
# Minimum version of cmake required
#
cmake_minimum_required(VERSION 2.8.0)

project(rocm_smi)


if("${ROCM_SMI_BLD_BITS}" STREQUAL 64)
  set (ONLY64STR "64")
  set (IS64BIT 1)
else()
  set (ONLY64STR "")
  set (IS64BIT 0)
endif()

string(TOLOWER "${ROCM_SMI_BUILD_TYPE}" tmp)
if("${tmp}" STREQUAL release)
  set(BUILD_TYPE "Release")
  set(ISDEBUG 0)
else()
  set(BUILD_TYPE "Debug")
  set(ISDEBUG 1)
endif()

set(CMAKE_CXX_FLAGS "-std=c++11 ")
if(ISDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
endif()


if (IS64BIT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64  -msse -msse2")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
endif()

set(ROCM_SMI "rocm_smi${ONLY64STR}")

set(CMAKE_BUILD_TYPE Debug)
set(SRC_DIR "src")
set(INC_DIR "include/rocm_smi")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SMI_SRC_LIST "${SRC_DIR}/rocm_smi_device.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_main.cc")
set(SMI_SRC_LIST ${SMI_SRC_LIST} "${SRC_DIR}/rocm_smi_monitor.cc")

set(SMI_INC_LIST "${INC_DIR}/rocm_smi_device.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_main.h")
set(SMI_INC_LIST ${SMI_INC_LIST} "${INC_DIR}/rocm_smi_monitor.h")


# rocm_smi_device.h 
#      rocm_smi.h rocm_smi_main.h rocm_smi_monitor.h")

set(SMI_EXAMPLE_EXE "rocm_smi_ex")

add_executable(${SMI_EXAMPLE_EXE} "example/rocm_smi_example.cc")
target_link_libraries(${SMI_EXAMPLE_EXE} ${ROCM_SMI})
add_library(${ROCM_SMI} SHARED ${SMI_SRC_LIST} ${SMI_INC_LIST})
install(TARGETS ${ROCM_SMI} 
        LIBRARY DESTINATION ${PROJECT_BINARY_DIR}
        RUNTIME DESTINATION ${PROJECT_BINARY_DIR})


