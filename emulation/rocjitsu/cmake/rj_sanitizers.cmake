# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compute compiler-specific sanitizer runtime link options.
#
# Usage:
#   rj_sanitizer_runtime_link_options(
#       <out-var>
#       COMPILER_ID <compiler-id>
#       SHARED <ON|OFF>
#       [SHARED_LIBRARIES <path>...]
#       SANITIZERS <address|undefined|thread|memory>...
#   )
#
# The caller is responsible for adding the common `-fsanitize=<kinds>` option.
# This helper only selects whether sanitizer runtime libraries are linked
# dynamically or statically. When shared runtime paths are available, it also
# adds rpaths so build-tree tools can run without setting LD_LIBRARY_PATH.
function(rj_sanitizer_runtime_link_options out_var)
    set(oneValueArgs COMPILER_ID SHARED)
    set(multiValueArgs SANITIZERS SHARED_LIBRARIES)
    cmake_parse_arguments(
        RJ_SANI
        ""
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    set(_options)
    if(RJ_SANI_SHARED)
        if(RJ_SANI_COMPILER_ID MATCHES "Clang|AppleClang")
            list(APPEND _options -shared-libsan)
        endif()
        if(RJ_SANI_SHARED_LIBRARIES AND UNIX AND NOT APPLE)
            set(_shared_library_dirs)
            foreach(_shared_library IN LISTS RJ_SANI_SHARED_LIBRARIES)
                get_filename_component(
                    _shared_library_dir
                    "${_shared_library}"
                    DIRECTORY
                )
                list(APPEND _shared_library_dirs "${_shared_library_dir}")
            endforeach()
            list(REMOVE_DUPLICATES _shared_library_dirs)
            foreach(_shared_library_dir IN LISTS _shared_library_dirs)
                list(APPEND _options "-Wl,-rpath,${_shared_library_dir}")
            endforeach()
        endif()
    elseif(RJ_SANI_COMPILER_ID STREQUAL "GNU")
        if(address IN_LIST RJ_SANI_SANITIZERS)
            list(APPEND _options -static-libasan)
        endif()
        if(undefined IN_LIST RJ_SANI_SANITIZERS)
            list(APPEND _options -static-libubsan)
        endif()
        if(thread IN_LIST RJ_SANI_SANITIZERS)
            list(APPEND _options -static-libtsan)
        endif()
    elseif(RJ_SANI_COMPILER_ID MATCHES "Clang|AppleClang")
        list(APPEND _options -static-libsan)
    endif()

    set(${out_var} ${_options} PARENT_SCOPE)
endfunction()

# Resolve shared sanitizer runtimes selected by the compiler.
#
# Usage:
#   rj_find_sanitizer_shared_libraries(
#       <libraries-out-var>
#       <asan-out-var>
#       <tsan-out-var>
#       COMPILER <compiler-path>
#       COMPILER_ID <compiler-id>
#       [COMPILER_ARG1 <compiler-arg1>]
#       [SYSTEM_PROCESSOR <processor>]
#       SANITIZERS <address|undefined|thread|memory>...
#   )
#
# The first result contains all resolved shared sanitizer runtime paths. The
# second result contains the ASan runtime path when AddressSanitizer is enabled.
# The third result contains the TSan runtime path when ThreadSanitizer is enabled.
function(
    rj_find_sanitizer_shared_libraries
    libraries_out_var
    asan_out_var
    tsan_out_var
)
    set(oneValueArgs COMPILER COMPILER_ARG1 COMPILER_ID SYSTEM_PROCESSOR)
    set(multiValueArgs SANITIZERS)
    cmake_parse_arguments(
        RJ_SANI_LIB
        ""
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    if(NOT RJ_SANI_LIB_SANITIZERS)
        set(${libraries_out_var} "" PARENT_SCOPE)
        set(${asan_out_var} "" PARENT_SCOPE)
        set(${tsan_out_var} "" PARENT_SCOPE)
        return()
    endif()

    set(_compiler_command "${RJ_SANI_LIB_COMPILER}")
    if(RJ_SANI_LIB_COMPILER_ARG1)
        list(APPEND _compiler_command ${RJ_SANI_LIB_COMPILER_ARG1})
    endif()

    set(_sanitizer_shared_libraries)
    set(_asan_shared_library "")
    set(_tsan_shared_library "")
    if(RJ_SANI_LIB_COMPILER_ID MATCHES "Clang")
        set(_sanitizer_arch "${RJ_SANI_LIB_SYSTEM_PROCESSOR}")
        if(NOT _sanitizer_arch)
            set(_sanitizer_arch "${CMAKE_SYSTEM_PROCESSOR}")
        endif()
        if(_sanitizer_arch STREQUAL "AMD64")
            set(_sanitizer_arch "x86_64")
        endif()
        execute_process(
            COMMAND ${_compiler_command} --print-resource-dir
            RESULT_VARIABLE _resource_dir_result
            OUTPUT_VARIABLE _resource_dir
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()

    foreach(_sanitizer IN LISTS RJ_SANI_LIB_SANITIZERS)
        set(_runtime_name)
        if(RJ_SANI_LIB_COMPILER_ID STREQUAL "GNU")
            if(_sanitizer STREQUAL "address")
                set(_runtime_name "libasan.so")
            elseif(_sanitizer STREQUAL "undefined")
                set(_runtime_name "libubsan.so")
            elseif(_sanitizer STREQUAL "thread")
                set(_runtime_name "libtsan.so")
            endif()
        elseif(RJ_SANI_LIB_COMPILER_ID MATCHES "Clang")
            if(_sanitizer STREQUAL "address")
                set(_runtime_name "libclang_rt.asan-${_sanitizer_arch}.so")
            elseif(_sanitizer STREQUAL "undefined")
                set(_runtime_name
                    "libclang_rt.ubsan_standalone-${_sanitizer_arch}.so"
                )
            elseif(_sanitizer STREQUAL "thread")
                set(_runtime_name "libclang_rt.tsan-${_sanitizer_arch}.so")
            endif()
        endif()

        if(NOT _runtime_name)
            continue()
        endif()

        execute_process(
            COMMAND ${_compiler_command} --print-file-name=${_runtime_name}
            RESULT_VARIABLE _runtime_result
            OUTPUT_VARIABLE _runtime_output
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(_runtime_library "")
        if(
            _runtime_result EQUAL 0
            AND IS_ABSOLUTE "${_runtime_output}"
            AND EXISTS "${_runtime_output}"
        )
            set(_runtime_library "${_runtime_output}")
        endif()

        if(
            NOT _runtime_library
            AND _resource_dir_result EQUAL 0
            AND IS_DIRECTORY "${_resource_dir}"
        )
            file(
                GLOB _runtime_candidates
                "${_resource_dir}/lib/*/${_runtime_name}"
            )
            list(SORT _runtime_candidates)
            if(_runtime_candidates)
                list(GET _runtime_candidates 0 _runtime_library)
            endif()
        endif()

        if(NOT _runtime_library)
            message(
                FATAL_ERROR
                "RJ_SANITIZER_RUNTIME=SHARED requires the ${_sanitizer} "
                "sanitizer shared runtime, but CMake could not locate "
                "${_runtime_name} with ${RJ_SANI_LIB_COMPILER}."
            )
        endif()

        list(APPEND _sanitizer_shared_libraries "${_runtime_library}")
        if(_sanitizer STREQUAL "address")
            set(_asan_shared_library "${_runtime_library}")
        elseif(_sanitizer STREQUAL "thread")
            set(_tsan_shared_library "${_runtime_library}")
        endif()
    endforeach()

    set(${libraries_out_var} "${_sanitizer_shared_libraries}" PARENT_SCOPE)
    set(${asan_out_var} "${_asan_shared_library}" PARENT_SCOPE)
    set(${tsan_out_var} "${_tsan_shared_library}" PARENT_SCOPE)
endfunction()
