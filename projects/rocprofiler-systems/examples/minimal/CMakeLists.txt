# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(rocprofiler-systems-minimal LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

add_executable(minimal-recursion ${CMAKE_CURRENT_SOURCE_DIR}/recursion.cpp)
# Keep recursive calls as real calls so the instrumenter sees every frame
# (noinline alone does not prevent tail-call / sibling-call optimization).
target_compile_options(minimal-recursion PRIVATE -fno-optimize-sibling-calls)

set(MINIMAL_EXAMPLES minimal-recursion)

# Minimal pthread example: exercises the pthread_create gotcha trace-args.
find_package(Threads REQUIRED)
add_executable(minimal-pthreads ${CMAKE_CURRENT_SOURCE_DIR}/pthreads.cpp)
target_link_libraries(minimal-pthreads PRIVATE Threads::Threads)
list(APPEND MINIMAL_EXAMPLES minimal-pthreads)

add_executable(minimal-sleep-interrupts ${CMAKE_CURRENT_SOURCE_DIR}/sleep_interrupts.cpp)
list(APPEND MINIMAL_EXAMPLES minimal-sleep-interrupts)

# Minimal tail-call main example: main() ends with a jmp to another function, not a ret.
add_executable(minimal-tail-call-main ${CMAKE_CURRENT_SOURCE_DIR}/tail_call_main.cpp)
# The tail call only exists when the compiler optimizes sibling calls, whatever the
# build type is. A stack protector makes rocprof-sys-instrument skip the function.
target_compile_options(
    minimal-tail-call-main
    PRIVATE -O2 -foptimize-sibling-calls -fno-stack-protector
)
list(APPEND MINIMAL_EXAMPLES minimal-tail-call-main)

# Minimal exit-in-main example: main() ends with a call to exit(), not a ret or a tail call.
add_executable(minimal-exit-in-main ${CMAKE_CURRENT_SOURCE_DIR}/exit_in_main.cpp)
target_compile_options(minimal-exit-in-main PRIVATE -O2 -fno-stack-protector)
list(APPEND MINIMAL_EXAMPLES minimal-exit-in-main)

# Minimal mixed main end example: main() has a tail call, a ret and a call to exit().
add_executable(minimal-mixed-main-end ${CMAKE_CURRENT_SOURCE_DIR}/mixed_main_end.cpp)
target_compile_options(
    minimal-mixed-main-end
    PRIVATE -O2 -foptimize-sibling-calls -fno-stack-protector
)
list(APPEND MINIMAL_EXAMPLES minimal-mixed-main-end)

# Minimal tail-call main example whose tail call jumps into a shared library.
add_library(
    minimal-tail-call-workload
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/tail_call_workload_lib.cpp
)
target_compile_options(minimal-tail-call-workload PRIVATE -O2 -fno-stack-protector)
add_executable(
    minimal-tail-call-main-shared
    ${CMAKE_CURRENT_SOURCE_DIR}/tail_call_main_shared.cpp
)
target_compile_options(
    minimal-tail-call-main-shared
    PRIVATE -O2 -foptimize-sibling-calls -fno-stack-protector
)
target_link_libraries(minimal-tail-call-main-shared PRIVATE minimal-tail-call-workload)
# The library is installed next to the executable, so find it there at run time.
set_target_properties(
    minimal-tail-call-main-shared
    minimal-tail-call-workload
    PROPERTIES INSTALL_RPATH "\$ORIGIN" BUILD_WITH_INSTALL_RPATH ON
)
list(APPEND MINIMAL_EXAMPLES minimal-tail-call-main-shared)

foreach(MINIMAL_EXAMPLE IN LISTS MINIMAL_EXAMPLES)
    if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET ${MINIMAL_EXAMPLE})
        install(
            TARGETS ${MINIMAL_EXAMPLE}
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
            COMPONENT rocprofiler-systems-examples
        )
    endif()
endforeach()

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET minimal-tail-call-workload)
    install(
        TARGETS minimal-tail-call-workload
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
