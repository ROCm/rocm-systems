# ############################################################################################################################################
# ROCProfiler General Requirements
# ############################################################################################################################################
find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import lxml"
  RESULT_VARIABLE CPP_HEADER_PARSER
  OUTPUT_QUIET)

if(NOT ${CPP_HEADER_PARSER} EQUAL 0)
  message(FATAL_ERROR "\
The \"lxml\" Python3 package is not installed. \
Please install it using the following command: \"${Python3_EXECUTABLE} -m pip install lxml\".\
")
endif()

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import CppHeaderParser"
  RESULT_VARIABLE CPP_HEADER_PARSER
  OUTPUT_QUIET)

if(NOT ${CPP_HEADER_PARSER} EQUAL 0)
  message(FATAL_ERROR "\
The \"CppHeaderParser\" Python3 package is not installed. \
Please install it using the following command: \"${Python3_EXECUTABLE} -m pip install CppHeaderParser\".\
")
endif()

# Setting Default Binary output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Getting HSA Include Directory
get_property(HSA_RUNTIME_INCLUDE_DIRECTORIES TARGET hsa-runtime64::hsa-runtime64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(HSA_H hsa.h
  PATHS ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
  PATH_SUFFIXES hsa
  NO_DEFAULT_PATH
  REQUIRED)
get_filename_component(HSA_RUNTIME_INC_PATH ${HSA_H} DIRECTORY)

find_library(AQLPROFILE_LIB "libhsa-amd-aqlprofile64.so" HINTS ${CMAKE_PREFIX_PATH} PATHS ${ROCM_PATH} PATH_SUFFIXES lib)

if(NOT AQLPROFILE_LIB)
  message(FATAL_ERROR "AQL_PROFILE not installed. Please install hsa-amd-aqlprofile!")
endif()

# ############################################################################################################################################
# Adding Old Library Files
# ############################################################################################################################################
set (OLD_LIB_SRC
  ${LIB_DIR}/core/rocprofiler.cpp
  ${LIB_DIR}/core/gpu_command.cpp
  ${LIB_DIR}/core/proxy_queue.cpp
  ${LIB_DIR}/core/simple_proxy_queue.cpp
  ${LIB_DIR}/core/intercept_queue.cpp
  ${LIB_DIR}/core/metrics.cpp
  ${LIB_DIR}/core/activity.cpp
  ${LIB_DIR}/util/hsa_rsrc_factory.cpp
)

# ############################################################################################################################################
# Configuring Basic/Derived Counters
# ############################################################################################################################################
set(COUNTERS_DIR ${PROJECT_SOURCE_DIR}/src/core/counters)

execute_process(
  COMMAND ${Python3_EXECUTABLE} ${COUNTERS_DIR}/basic/xml_parser_basic.py ${COUNTERS_DIR}/basic ${CMAKE_CURRENT_BINARY_DIR}/basic_counter.cpp
  COMMENT "Generating basic_counter.cpp...")

# execute_process(
# COMMAND ${Python3_EXECUTABLE} ${COUNTERS_DIR}/derived/xml_parser_derived.py ${COUNTERS_DIR}/derived ${CMAKE_CURRENT_BINARY_DIR}/derived_counter.cpp
# COMMENT "Generating derived_counter.cpp...")

# ############################################################################################################################################
# ROCProfiler Tracer HIP/HSA Parsing
# ############################################################################################################################################
get_property(HIP_INCLUDE_DIRECTORIES TARGET hip::amdhip64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(HIP_RUNTIME_API_H hip_runtime_api.h
  PATHS ${HIP_INCLUDE_DIRECTORIES}
  PATH_SUFFIXES hip
  NO_DEFAULT_PATH
  REQUIRED)

# # Generate the HSA wrapper functions header
add_custom_command(
  OUTPUT hsa_prof_str.h hsa_prof_str.inline.h
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/hsaap.py ${CMAKE_CURRENT_BINARY_DIR} "${HSA_RUNTIME_INC_PATH}" > /dev/null
  DEPENDS ${PROJECT_SOURCE_DIR}/script/hsaap.py
  "${HSA_RUNTIME_INC_PATH}/hsa.h" "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h"
  "${HSA_RUNTIME_INC_PATH}/hsa_ext_image.h" "${HSA_RUNTIME_INC_PATH}/hsa_api_trace.h"
  COMMENT "Generating hsa_prof_str.h,hsa_prof_str.inline.h...")

# # Generate the HSA pretty printers
add_custom_command(
  OUTPUT hsa_ostream_ops.h
  COMMAND ${CMAKE_C_COMPILER} -E "${HSA_RUNTIME_INC_PATH}/hsa.h" -o hsa.h.i
  COMMAND ${CMAKE_C_COMPILER} -E "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h" -o hsa_ext_amd.h.i
  BYPRODUCTS hsa.h.i hsa_ext_amd.h.i
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py
  -in hsa.h.i,hsa_ext_amd.h.i -out hsa_ostream_ops.h > /dev/null
  DEPENDS ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py
  "${HSA_RUNTIME_INC_PATH}/hsa.h" "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h"
  COMMENT "Generating hsa_ostream_ops.h...")

get_property(HIP_INCLUDE_DIRECTORIES TARGET hip::amdhip64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(HIP_RUNTIME_API_H hip_runtime_api.h
  PATHS ${HIP_INCLUDE_DIRECTORIES}
  PATH_SUFFIXES hip
  NO_DEFAULT_PATH
  REQUIRED)

## Generate the HIP pretty printers
add_custom_command(
  OUTPUT hip_ostream_ops.h
  COMMAND ${CMAKE_C_COMPILER} "$<$<BOOL:${HIP_INCLUDE_DIRECTORIES}>:-I$<JOIN:${HIP_INCLUDE_DIRECTORIES},$<SEMICOLON>-I>>"
  -E "${HIP_RUNTIME_API_H}" -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 -o hip_runtime_api.h.i
  BYPRODUCTS hip_runtime_api.h.i
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py
  -in hip_runtime_api.h.i -out hip_ostream_ops.h > /dev/null
  DEPENDS ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py "${HIP_RUNTIME_API_H}"
  COMMENT "Generating hip_ostream_ops.h..."
  COMMAND_EXPAND_LISTS)

set(GENERATED_SOURCES
  hip_ostream_ops.h
  hsa_prof_str.h
  hsa_ostream_ops.h
  hsa_prof_str.inline.h)

# ############################################################################################################################################
# ROCProfiler API
# ############################################################################################################################################
# PC sampling uses libpciaccess as a fallback if the debugfs ioctl is
# unavailable
find_path(PCIACCESS_INCLUDE_DIR pciaccess.h REQUIRED)
find_library(PCIACCESS_LIBRARIES pciaccess REQUIRED)

set(PUBLIC_HEADERS rocprofiler.h)

foreach(header ${PUBLIC_HEADERS})
  install(FILES ${PROJECT_SOURCE_DIR}/include/rocprofiler/${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
    COMPONENT dev)
endforeach()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/rocprofiler/v2
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
        COMPONENT dev)

# Getting Source files for ROCProfiler, Hardware, HSA, Memory, Session, Counters, Utils
file(GLOB ROCPROFILER_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file(GLOB ROCPROFILER_PROFILER_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/profiler/profiler.cpp)
file(GLOB ROCPROFILER_TRACER_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/tracer/*.cpp)
file(GLOB ROCPROFILER_ROCTRACER_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/tracer/src/*.cpp)
file(GLOB ROCPROFILER_ATT_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/att/att.cpp)
file(GLOB ROCPROFILER_CLASS_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/rocmtool.cpp)
file(GLOB ROCPROFILER_SPM_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/spm/spm.cpp)


set(CORE_HARDWARE_DIR ${PROJECT_SOURCE_DIR}/src/core/hardware)
file(GLOB CORE_HARDWARE_SRC_FILES ${CORE_HARDWARE_DIR}/*.cpp)

set(CORE_HSA_QUEUES_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa/queues)
file(GLOB CORE_HSA_QUEUES_SRC_FILES ${CORE_HSA_QUEUES_DIR}/queue.cpp)

set(CORE_HSA_PACKETS_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa/packets)
file(GLOB CORE_HSA_PACKETS_SRC_FILES ${CORE_HSA_PACKETS_DIR}/packets_generator.cpp)

set(CORE_HSA_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa)
file(GLOB CORE_HSA_SRC_FILES ${CORE_HSA_DIR}/*.cpp)

set(CORE_MEMORY_DIR ${PROJECT_SOURCE_DIR}/src/core/memory)
file(GLOB CORE_MEMORY_SRC_FILES ${CORE_MEMORY_DIR}/*.cpp)

set(CORE_SESSION_DIR ${PROJECT_SOURCE_DIR}/src/core/session)
file(GLOB CORE_SESSION_SRC_FILES ${CORE_SESSION_DIR}/session.cpp)
file(GLOB CORE_FILTER_SRC_FILES ${CORE_SESSION_DIR}/filter.cpp)
file(GLOB CORE_DEVICE_PROFILING_SRC_FILES ${CORE_SESSION_DIR}/device_profiling.cpp)
file(GLOB CORE_COUNTERS_SAMPLER_SRC_FILES ${CORE_SESSION_DIR}/counters_sampler.cpp)

file(GLOB CORE_COUNTERS_SRC_FILES ${PROJECT_BINARY_DIR}/src/api/*_counter.cpp)
file(GLOB CORE_COUNTERS_PARENT_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/counters/*.cpp)
file(GLOB CORE_COUNTERS_METRICS_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/counters/metrics/*.cpp)
file(GLOB CORE_COUNTERS_MMIO_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/counters/mmio/*.cpp)

set(CORE_UTILS_DIR ${PROJECT_SOURCE_DIR}/src/utils)
file(GLOB CORE_UTILS_SRC_FILES ${CORE_UTILS_DIR}/*.cpp)

set(CORE_PC_SAMPLING_DIR ${PROJECT_SOURCE_DIR}/src/pcsampler)
file(GLOB CORE_PC_SAMPLING_FILES ${CORE_PC_SAMPLING_DIR}/core/*.cpp ${CORE_PC_SAMPLING_DIR}/gfxip/*.cpp ${CORE_PC_SAMPLING_DIR}/session/*.cpp)


#### V1 Library
# Compiling/Installing ROCProfiler API V1
add_library(${ROCPROFILER_TARGET} SHARED ${OLD_LIB_SRC})
set_target_properties(${ROCPROFILER_TARGET} PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VERSION 1.0.0
  SOVERSION 1)
# As ROCR hsa_api_trace header file is not usable unless AMD_INTERNAL_BUILD is defined
target_compile_definitions(${ROCPROFILER_TARGET} PUBLIC AMD_INTERNAL_BUILD)
target_include_directories(${ROCPROFILER_TARGET}
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/rocprofiler>
  PRIVATE
  ${LIB_DIR} ${ROOT_DIR}
  ${PROJECT_SOURCE_DIR}/include/rocprofiler)
target_link_libraries(${ROCPROFILER_TARGET} PRIVATE ${AQLPROFILE_LIB} hsa-runtime64::hsa-runtime64 c stdc++)
## Install libraries: Non versioned lib file in dev package
# install ( TARGETS ${ROCPROFILER_TARGET} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_COMPONENT runtime)
install ( TARGETS ${ROCPROFILER_TARGET} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
# install ( TARGETS ${ROCPROFILER_TARGET} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

#### V2 Library
# Compiling/Installing ROCProfiler API
add_library(rocprofiler-v2 SHARED
  ${ROCPROFILER_SRC_FILES}
  ${ROCPROFILER_CLASS_SRC_FILES}
  ${ROCPROFILER_PROFILER_SRC_FILES}
  ${ROCPROFILER_ATT_SRC_FILES}
  ${CORE_HARDWARE_SRC_FILES}
  ${CORE_HSA_SRC_FILES}
  ${ROCPROFILER_SPM_SRC_FILES}
  ${CORE_MEMORY_SRC_FILES}
  ${CORE_SESSION_SRC_FILES}
  ${CORE_FILTER_SRC_FILES}
  ${CORE_DEVICE_PROFILING_SRC_FILES}
  ${CORE_COUNTERS_SAMPLER_SRC_FILES}
  ${CORE_COUNTERS_PARENT_SRC_FILES}
  ${CORE_COUNTERS_METRICS_SRC_FILES}
  ${CORE_COUNTERS_MMIO_SRC_FILES}
  ${CORE_UTILS_SRC_FILES}
  ${CORE_HSA_PACKETS_SRC_FILES}
  ${CORE_HSA_QUEUES_SRC_FILES}
  ${ROCPROFILER_TRACER_SRC_FILES}
  ${ROCPROFILER_ROCTRACER_SRC_FILES}
  ${GENERATED_SOURCES}
  ${CORE_COUNTERS_SRC_FILES}
  ${CORE_PC_SAMPLING_FILES})
set_target_properties(rocprofiler-v2 PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  DEFINE_SYMBOL "ROCPROFILER_EXPORTS"
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exportmap
  OUTPUT_NAME rocprofiler64
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/v2
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})
 
# Add custom command to copy the v2 library to buil-dir as well
add_custom_command(TARGET rocprofiler-v2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/v2/librocprofiler64.so.2 ${CMAKE_BINARY_DIR}
)
# Add custom target to trigger the copy command
add_custom_target(copy_v2_library DEPENDS rocprofiler-v2)

target_compile_definitions(rocprofiler-v2
  # As ROCR hsa_api_trace header file is not usable unless AMD_INTERNAL_BUILD is defined
  PRIVATE AMD_INTERNAL_BUILD
  PROF_API_IMPL HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1)
target_include_directories(rocprofiler-v2
  PUBLIC
  ${HIP_INCLUDE_DIRECTORIES} ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/rocprofiler/v2>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  PRIVATE
  ${LIB_DIR} ${ROOT_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/tools)
if(ASAN)
  target_compile_options(rocprofiler-v2 PRIVATE -fsanitize=address)
  target_link_options(rocprofiler-v2 PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exportmap -Wl,--no-undefined,-fsanitize=address)
  target_link_libraries(rocprofiler-v2 PRIVATE ${AQLPROFILE_LIB} hsa-runtime64::hsa-runtime64 Threads::Threads atomic asan dl c stdc++ stdc++fs amd_comgr ${PCIACCESS_LIBRARIES})
else()
  target_link_options(rocprofiler-v2 PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exportmap -Wl,--no-undefined)
  target_link_libraries(rocprofiler-v2 PRIVATE ${AQLPROFILE_LIB} hsa-runtime64::hsa-runtime64 Threads::Threads atomic dl c stdc++ stdc++fs amd_comgr ${PCIACCESS_LIBRARIES})
endif()
## Install libraries: Non versioned lib file in dev package
# install(TARGETS rocprofiler-v2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev)
install(TARGETS rocprofiler-v2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
# install(TARGETS rocprofiler-v2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/librocprofiler64.so
  CONTENT "OUTPUT_FORMAT(elf64-x86-64)\nINPUT(librocprofiler64.so.1)")
install(FILES ${CMAKE_BINARY_DIR}/librocprofiler64.so DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  COMPONENT runtime)

file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/librocprofiler64v2.so
  CONTENT "OUTPUT_FORMAT(elf64-x86-64)\nINPUT(librocprofiler64.so.${PROJECT_VERSION_MAJOR})")
install(FILES ${CMAKE_BINARY_DIR}/librocprofiler64v2.so DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  COMPONENT runtime)

configure_file(${PROJECT_SOURCE_DIR}/src/core/counters/metrics/basic_counters.xml ${PROJECT_BINARY_DIR}/counters/basic_counters.xml COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/src/core/counters/metrics/derived_counters.xml ${PROJECT_BINARY_DIR}/counters/derived_counters.xml COPYONLY)

install(DIRECTORY
  ${PROJECT_BINARY_DIR}/counters
  DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
  COMPONENT runtime)

# ############################################################################################################################################
