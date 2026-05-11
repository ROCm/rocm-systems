# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(CMakeParseArguments)

function(_rj_run_python_venv_step)
    set(options)
    set(oneValueArgs DESCRIPTION WORKING_DIRECTORY)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    execute_process(
        COMMAND ${ARG_COMMAND}
        WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}"
        RESULT_VARIABLE result
        COMMAND_ECHO STDOUT
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${ARG_DESCRIPTION} failed with exit code ${result}")
    endif()
endfunction()

function(rj_setup_python_venv)
    set(options)
    set(oneValueArgs
        NAME
        BOOTSTRAP_PYTHON
        EDITABLE_PATH
        REQUIREMENTS_FILE
        OUT_VENV_DIR
        OUT_PYTHON
        OUT_PROGRAM_DIRS
        OUT_LIBRARY_DIRS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "" ${ARGN})

    if(NOT ARG_NAME)
        message(FATAL_ERROR "rj_setup_python_venv(NAME ...) requires NAME")
    endif()
    if(NOT ARG_BOOTSTRAP_PYTHON)
        message(FATAL_ERROR "rj_setup_python_venv(NAME ${ARG_NAME} ...) requires BOOTSTRAP_PYTHON")
    endif()
    if(NOT ARG_EDITABLE_PATH AND NOT ARG_REQUIREMENTS_FILE)
        message(FATAL_ERROR
            "rj_setup_python_venv(NAME ${ARG_NAME} ...) requires EDITABLE_PATH or REQUIREMENTS_FILE")
    endif()

    set(venv_dir "${CMAKE_BINARY_DIR}/venv/${ARG_NAME}")
    if(WIN32)
        set(venv_python "${venv_dir}/Scripts/python.exe")
        set(venv_bin_dir "${venv_dir}/Scripts")
    else()
        set(venv_python "${venv_dir}/bin/python3")
        set(venv_bin_dir "${venv_dir}/bin")
    endif()
    set(stamp_file "${venv_dir}/.${ARG_NAME}_venv.stamp")

    set(dependency_files)
    if(ARG_EDITABLE_PATH)
        list(APPEND dependency_files "${ARG_EDITABLE_PATH}/pyproject.toml")
    endif()
    if(ARG_REQUIREMENTS_FILE)
        list(APPEND dependency_files "${ARG_REQUIREMENTS_FILE}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${dependency_files})

    set(refresh_venv FALSE)
    if(NOT EXISTS "${venv_python}" OR NOT EXISTS "${stamp_file}")
        set(refresh_venv TRUE)
    else()
        foreach(file_path IN LISTS dependency_files)
            if("${file_path}" IS_NEWER_THAN "${stamp_file}")
                set(refresh_venv TRUE)
                break()
            endif()
        endforeach()
    endif()

    if(refresh_venv)
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/venv")
        _rj_run_python_venv_step(
            DESCRIPTION "Creating ${ARG_NAME} Python virtual environment"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMAND "${ARG_BOOTSTRAP_PYTHON}" -m venv "${venv_dir}")
        _rj_run_python_venv_step(
            DESCRIPTION "Upgrading pip in ${ARG_NAME} Python virtual environment"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMAND "${venv_python}" -m pip install --upgrade pip)
        if(ARG_EDITABLE_PATH)
            _rj_run_python_venv_step(
                DESCRIPTION "Installing editable package for ${ARG_NAME} Python virtual environment"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                COMMAND "${venv_python}" -m pip install -e "${ARG_EDITABLE_PATH}")
        endif()
        if(ARG_REQUIREMENTS_FILE)
            _rj_run_python_venv_step(
                DESCRIPTION "Installing requirements for ${ARG_NAME} Python virtual environment"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                COMMAND "${venv_python}" -m pip install -r "${ARG_REQUIREMENTS_FILE}")
        endif()
        file(TOUCH "${stamp_file}")
    endif()

    execute_process(
        COMMAND "${venv_python}" -c
            "import site; print('\\n'.join(site.getsitepackages()))"
        RESULT_VARIABLE site_result
        OUTPUT_VARIABLE site_packages_raw
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT site_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to query site-packages for ${ARG_NAME} Python virtual environment")
    endif()
    string(REPLACE "\n" ";" site_packages "${site_packages_raw}")

    set(program_dirs "${venv_bin_dir}")
    set(library_dirs)
    foreach(site_dir IN LISTS site_packages)
        file(GLOB rocm_program_dirs LIST_DIRECTORIES TRUE "${site_dir}/_rocm_sdk*/bin")
        file(GLOB rocm_library_dirs LIST_DIRECTORIES TRUE "${site_dir}/_rocm_sdk*/lib")
        list(APPEND program_dirs ${rocm_program_dirs})
        list(APPEND library_dirs ${rocm_library_dirs})
    endforeach()
    list(REMOVE_DUPLICATES program_dirs)
    list(REMOVE_DUPLICATES library_dirs)

    if(ARG_OUT_VENV_DIR)
        set(${ARG_OUT_VENV_DIR} "${venv_dir}" PARENT_SCOPE)
    endif()
    if(ARG_OUT_PYTHON)
        set(${ARG_OUT_PYTHON} "${venv_python}" PARENT_SCOPE)
    endif()
    if(ARG_OUT_PROGRAM_DIRS)
        set(${ARG_OUT_PROGRAM_DIRS} "${program_dirs}" PARENT_SCOPE)
    endif()
    if(ARG_OUT_LIBRARY_DIRS)
        set(${ARG_OUT_LIBRARY_DIRS} "${library_dirs}" PARENT_SCOPE)
    endif()
endfunction()