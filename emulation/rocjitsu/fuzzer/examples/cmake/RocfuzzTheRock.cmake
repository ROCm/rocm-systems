# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

function(rocfuzz_discover_therock)
    set(ROCFUZZ_THEROCK_VENV
        "${CMAKE_CURRENT_SOURCE_DIR}/.venv"
        CACHE PATH "uv virtual environment containing the TheRock ROCm SDK wheels")
    set(ROCFUZZ_THEROCK_ROOT
        ""
        CACHE PATH "Expanded TheRock ROCm SDK root")
    set(ROCFUZZ_THEROCK_LIBRARIES_ROOT
        ""
        CACHE PATH "TheRock ROCm SDK runtime library payload root")

    if(NOT ROCFUZZ_THEROCK_ROOT)
        set(_rocfuzz_rocm_sdk "${ROCFUZZ_THEROCK_VENV}/bin/rocm-sdk")
        if(NOT EXISTS "${_rocfuzz_rocm_sdk}")
            message(FATAL_ERROR
                "Could not find ${_rocfuzz_rocm_sdk}. "
                "Run scripts/setup-therock.sh or pass -DROCFUZZ_THEROCK_ROOT=...")
        endif()

        execute_process(
            COMMAND "${_rocfuzz_rocm_sdk}" path --root
            OUTPUT_VARIABLE _rocfuzz_therock_root
            ERROR_VARIABLE _rocfuzz_therock_error
            RESULT_VARIABLE _rocfuzz_therock_result
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT _rocfuzz_therock_result EQUAL 0)
            message(FATAL_ERROR
                "rocm-sdk path --root failed:\n${_rocfuzz_therock_error}")
        endif()

        set(ROCFUZZ_THEROCK_ROOT "${_rocfuzz_therock_root}"
            CACHE PATH "Expanded TheRock ROCm SDK root" FORCE)
    endif()

    if(NOT EXISTS "${ROCFUZZ_THEROCK_ROOT}/include/hip/hip_runtime.h")
        message(FATAL_ERROR
            "ROCFUZZ_THEROCK_ROOT does not look like an expanded ROCm SDK: "
            "${ROCFUZZ_THEROCK_ROOT}")
    endif()

    list(PREPEND CMAKE_PREFIX_PATH
        "${ROCFUZZ_THEROCK_ROOT}"
        "${ROCFUZZ_THEROCK_ROOT}/lib/cmake")
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
    set(ROCFUZZ_THEROCK_ROOT "${ROCFUZZ_THEROCK_ROOT}" PARENT_SCOPE)

    if(NOT ROCFUZZ_THEROCK_LIBRARIES_ROOT)
        set(_rocfuzz_python "${ROCFUZZ_THEROCK_VENV}/bin/python")
        if(EXISTS "${_rocfuzz_python}")
            execute_process(
                COMMAND "${_rocfuzz_python}" -c
                        "import importlib.util, pathlib; spec = importlib.util.find_spec('_rocm_sdk_libraries'); print(pathlib.Path(spec.submodule_search_locations[0]) if spec and spec.submodule_search_locations else '')"
                OUTPUT_VARIABLE _rocfuzz_libraries_root
                RESULT_VARIABLE _rocfuzz_libraries_result
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(_rocfuzz_libraries_result EQUAL 0 AND _rocfuzz_libraries_root)
                set(ROCFUZZ_THEROCK_LIBRARIES_ROOT "${_rocfuzz_libraries_root}"
                    CACHE PATH "TheRock ROCm SDK runtime library payload root" FORCE)
            endif()
        endif()
    endif()
    if(NOT ROCFUZZ_THEROCK_LIBRARIES_ROOT)
        set(ROCFUZZ_THEROCK_LIBRARIES_ROOT "${ROCFUZZ_THEROCK_ROOT}"
            CACHE PATH "TheRock ROCm SDK runtime library payload root" FORCE)
    endif()

    set(_rocfuzz_runtime_rpath "")
    if(EXISTS "${ROCFUZZ_THEROCK_LIBRARIES_ROOT}/lib")
        list(APPEND _rocfuzz_runtime_rpath "${ROCFUZZ_THEROCK_LIBRARIES_ROOT}/lib")
    endif()
    list(APPEND _rocfuzz_runtime_rpath "${ROCFUZZ_THEROCK_ROOT}/lib")
    list(JOIN _rocfuzz_runtime_rpath ":" _rocfuzz_runtime_library_path_env)

    set(ROCFUZZ_THEROCK_LIBRARIES_ROOT
        "${ROCFUZZ_THEROCK_LIBRARIES_ROOT}" PARENT_SCOPE)
    set(ROCFUZZ_RUNTIME_RPATH "${_rocfuzz_runtime_rpath}" PARENT_SCOPE)
    set(ROCFUZZ_RUNTIME_LIBRARY_PATH_ENV
        "${_rocfuzz_runtime_library_path_env}" PARENT_SCOPE)

    message(STATUS "Using TheRock ROCm SDK: ${ROCFUZZ_THEROCK_ROOT}")
    message(STATUS "Using TheRock ROCm library payload: ${ROCFUZZ_THEROCK_LIBRARIES_ROOT}")
endfunction()
