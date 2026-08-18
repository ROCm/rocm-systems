find_program(hipify-perl_executable hipify-perl HINTS ${ROCM_PATH}/bin)
if(NOT hipify-perl_executable)
    message(FATAL_ERROR "hipify-perl not found")
endif()

set(PROFILER_HIPIFY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipify)

set(PROFILER_SRC_NAMES
    plugin.cc
    print_event.cc
    profiler_plugin_ce.cc
)

# event.h carries the cudaEvent_t/cudaStream_t members the CE sources need.
file(GLOB PROFILER_HDR_NAMES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h nccl/*.h)

set(PROFILER_HIPIFIED_SRC "")
set(PROFILER_HIPIFIED_HDR "")
foreach(_name IN LISTS PROFILER_SRC_NAMES PROFILER_HDR_NAMES)
    set(_in  ${CMAKE_CURRENT_SOURCE_DIR}/${_name})
    set(_out ${PROFILER_HIPIFY_DIR}/${_name})
    get_filename_component(_out_dir ${_out} DIRECTORY)
    add_custom_command(
        OUTPUT ${_out}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_out_dir}
        COMMAND ${hipify-perl_executable} -quiet-warnings ${_in} -o ${_out}
        MAIN_DEPENDENCY ${_in}
        COMMENT "Hipifying profiler/example/${_name}"
    )
    if(_name MATCHES "\\.cc$")
        list(APPEND PROFILER_HIPIFIED_SRC ${_out})
    else()
        list(APPEND PROFILER_HIPIFIED_HDR ${_out})
    endif()
endforeach()

# Sources include these by name, so they must exist before compiling.
add_custom_target(hipify-profiler-example DEPENDS ${PROFILER_HIPIFIED_HDR})

add_library(nccl-profiler-example SHARED ${PROFILER_HIPIFIED_SRC})
add_dependencies(nccl-profiler-example hipify-profiler-example)

target_include_directories(nccl-profiler-example PRIVATE
    ${PROFILER_HIPIFY_DIR}
    ${PROFILER_HIPIFY_DIR}/nccl
    ${ROCM_PATH}/include
)

target_compile_definitions(nccl-profiler-example PRIVATE __HIP_PLATFORM_AMD__)

# Match the Makefile output name.
set_target_properties(nccl-profiler-example PROPERTIES
    OUTPUT_NAME "rccl-profiler-example"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
)

add_custom_target(clean-profiler-lib
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/test/unit/plugins/librccl-profiler-example.so
    COMMENT "Cleaning librccl-profiler-example.so"
)
