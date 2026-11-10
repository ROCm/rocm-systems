# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# cmake -DBUILD=<build dir> -DWORK=<scratch dir> -DCC=<C compiler>
#       [-DCFLAGS=<flags>] [-DPKG_CONFIG=<pkg-config>] -P check_c_consumer.cmake
# Installs the library component, then builds and runs a C-only consumer
# through the CMake package and, where pkg-config is found, through amdcuid.pc.
# CFLAGS are the library build's own, so that a sanitizer build links its
# runtime into the consumer too.

function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE _result OUTPUT_VARIABLE _out ERROR_VARIABLE _out)
    if(NOT _result EQUAL 0)
        list(JOIN ARGN " " _command)
        message(FATAL_ERROR "${_command} failed:\n${_out}")
    endif()
endfunction()

file(REMOVE_RECURSE ${WORK})
set(_prefix ${WORK}/install)
run(${CMAKE_COMMAND} --install ${BUILD} --component library --prefix ${_prefix})

run(${CMAKE_COMMAND} -S ${CMAKE_CURRENT_LIST_DIR}/c_consumer -B ${WORK}/cmake
    -DCMAKE_C_COMPILER=${CC} -DCMAKE_PREFIX_PATH=${_prefix} "-DCMAKE_C_FLAGS=${CFLAGS}"
)
run(${CMAKE_COMMAND} --build ${WORK}/cmake)
run(${WORK}/cmake/c_consumer)

if(PKG_CONFIG)
    file(GLOB_RECURSE _pc ${_prefix}/amdcuid.pc)
    if(NOT _pc)
        message(FATAL_ERROR "no amdcuid.pc under ${_prefix}")
    endif()
    get_filename_component(_pc_dir ${_pc} DIRECTORY)
    set(ENV{PKG_CONFIG_PATH} ${_pc_dir})
    execute_process(
        COMMAND ${PKG_CONFIG} --cflags --libs amdcuid
        OUTPUT_VARIABLE _flags
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "pkg-config cannot read ${_pc}")
    endif()
    separate_arguments(_flags UNIX_COMMAND "${_flags}")
    separate_arguments(_cflags UNIX_COMMAND "${CFLAGS}")
    run(${CC} ${_cflags} ${CMAKE_CURRENT_LIST_DIR}/c_consumer/main.c ${_flags} -o ${WORK}/pc_consumer)
    run(${WORK}/pc_consumer)
endif()
