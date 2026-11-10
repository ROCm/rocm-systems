#
#   Out-of-process collector library (librocprofiler-sdk-collector)
#
rocprofiler_activate_clang_tidy()

# The implementation, linked into the shared library and the unit tests.
add_library(rocprofiler-sdk-collector-static-library STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-collector-static-library ALIAS
            rocprofiler-sdk-collector-static-library)

target_sources(
    rocprofiler-sdk-collector-static-library
    PRIVATE clock.cpp daemon.cpp hip_records.cpp join.cpp launch.cpp read.cpp schema.cpp
            session.cpp)
add_subdirectory(ctf)

# The HIP APIs rocprofiler-sdk traces, and their argument names: the
# HIP_API_INFO_DEFINITION_* lines of its hip.def.cpp, with their version guards.
# hip_records.cpp expands them into the code that rebuilds a call's arguments.
set(_hip_def ${PROJECT_SOURCE_DIR}/source/lib/rocprofiler-sdk/hip/hip.def.cpp)
set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${_hip_def})
file(STRINGS ${_hip_def} _hip_def_lines)
set(_hip_api_list "// Generated from lib/rocprofiler-sdk/hip/hip.def.cpp. Do not edit.\n")
set(_hip_def_depth 0)
foreach(_line IN LISTS _hip_def_lines)
    if(_line MATCHES "^HIP_API_INFO_DEFINITION_(0|V)\\(")
        # (one line in hip.def.cpp ends with a stray semicolon)
        string(REGEX REPLACE "\\)[ \t\\]*;[ \t]*$" ")" _line "${_line}")
        string(APPEND _hip_api_list "${_line}\n")
    elseif(_line MATCHES "^#if HIP_RUNTIME_API_TABLE_STEP_VERSION")
        string(APPEND _hip_api_list "${_line}\n")
        math(EXPR _hip_def_depth "${_hip_def_depth} + 1")
    elseif(_line MATCHES "^#endif" AND _hip_def_depth GREATER 0)
        string(APPEND _hip_api_list "${_line}\n")
        math(EXPR _hip_def_depth "${_hip_def_depth} - 1")
    endif()
endforeach()
if(NOT _hip_def_depth EQUAL 0)
    message(FATAL_ERROR "Unbalanced version guards in ${_hip_def}")
endif()
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hip_api_list.def.tmp "${_hip_api_list}")
configure_file(
    ${CMAKE_CURRENT_BINARY_DIR}/hip_api_list.def.tmp
    ${CMAKE_CURRENT_BINARY_DIR}/rocprofiler-sdk-collector/hip_api_list.def COPYONLY)
target_include_directories(rocprofiler-sdk-collector-static-library
                           PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(
    rocprofiler-sdk-collector-static-library
    PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/include>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/include>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_link_libraries(
    rocprofiler-sdk-collector-static-library
    PUBLIC rocprofiler-sdk::rocprofiler-sdk-build-flags
           rocprofiler-sdk::rocprofiler-sdk-memcheck
           rocprofiler-sdk::rocprofiler-sdk-common-library
           rocprofiler-sdk::rocprofiler-sdk-hip-nolink
           rocprofiler-sdk::rocprofiler-sdk-json
           rocprofiler-sdk::rocprofiler-sdk-lttng-ctl
           ${CMAKE_DL_LIBS})

set_target_properties(
    rocprofiler-sdk-collector-static-library
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON)

add_library(rocprofiler-sdk-collector-shared-library SHARED)

foreach(_NAMESPACE rocprofiler-sdk rocprofiler-sdk-collector)
    foreach(_ALIAS library shared-library)
        add_library(${_NAMESPACE}::rocprofiler-sdk-collector-${_ALIAS} ALIAS
                    rocprofiler-sdk-collector-shared-library)
    endforeach()
endforeach()

target_sources(rocprofiler-sdk-collector-shared-library PRIVATE collector.cpp)

target_include_directories(
    rocprofiler-sdk-collector-shared-library
    PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/include>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_link_libraries(rocprofiler-sdk-collector-shared-library
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk-collector-static-library)

set_target_properties(
    rocprofiler-sdk-collector-shared-library
    PROPERTIES OUTPUT_NAME rocprofiler-sdk-collector
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               SKIP_BUILD_RPATH OFF
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               DEFINE_SYMBOL ROCPROFILER_COLLECTOR_EXPORTS)

install(
    TARGETS rocprofiler-sdk-collector-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT core)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
