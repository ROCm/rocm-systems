# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Model an SDK with discoverable runtimes but an unusable HIP compiler.
file(REMOVE_RECURSE "${RJ_BINARY_DIR}")
set(sdk "${RJ_BINARY_DIR}/sdk")
file(MAKE_DIRECTORY "${sdk}/bin" "${sdk}/lib")
file(
    WRITE "${sdk}/bin/amdclang++"
    "#!/bin/sh\necho 'hip/hip_runtime.h not found' >&2\nexit 1\n"
)
file(
    CHMOD
    "${sdk}/bin/amdclang++"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
)
file(WRITE "${sdk}/lib/libamdhip64.so" "gfx950\ngfx942\n")
file(WRITE "${sdk}/lib/libhsa-runtime64.so" "")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -S "${RJ_SOURCE_DIR}" -B "${RJ_BINARY_DIR}/build"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${GOOGLETEST_SOURCE_DIR}"
        "-DFETCHCONTENT_SOURCE_DIR_FLATBUFFERS=${FLATBUFFERS_SOURCE_DIR}"
        "-DFETCHCONTENT_BASE_DIR=${RJ_BINARY_DIR}/third_party"
        "-DROCM_PATH=${sdk}" "-DCMAKE_LIBRARY_PATH=${sdk}/lib"
        -DBUILD_TESTING=ON
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Full test configuration failed:\n${output}\n${error}")
endif()
if(
    NOT output MATCHES "HIP compilation probe failed"
    OR output MATCHES "HIP tests enabled"
)
    message(
        FATAL_ERROR
        "HIP tests were not disabled by the failed probe:\n${output}"
    )
endif()

file(READ "${RJ_BINARY_DIR}/build/CMakeCache.txt" cache)
foreach(runtime IN ITEMS HIP_RUNTIME64 HSA_RUNTIME64)
    if(NOT cache MATCHES "${runtime}:FILEPATH=${sdk}/lib/")
        message(FATAL_ERROR "${runtime} was not discovered in the test SDK")
    endif()
endforeach()

file(READ "${RJ_BINARY_DIR}/build/CMakeFiles/TargetDirectories.txt" targets)
if(targets MATCHES "/(device_kernels|[^/]*hip[^/]*_target)\\.dir")
    message(
        FATAL_ERROR
        "Unusable HIP targets remain in the generated build:\n${targets}"
    )
endif()
if(NOT targets MATCHES "/rocjitsu_tests\\.dir")
    message(FATAL_ERROR "Host tests are missing from the generated build")
endif()
