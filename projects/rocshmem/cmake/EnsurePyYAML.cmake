# Ensure PyYAML is available for YAML-based test tier label processing.
#
# Call rocshmem_ensure_pyyaml() once before any subdirectory that needs to
# parse YAML files with Python.  Results are stored in cache variables so the
# check runs only once across all subdirectories:
#
#   ROCSHMEM_PYYAML_AVAILABLE  - TRUE if pyyaml is importable
#   ROCSHMEM_PYTHON_EXECUTABLE - path to the Python interpreter that has pyyaml
#
# Strategy (in order of preference):
#   1. Use the system interpreter if it already has pyyaml.
#   2. Create an isolated venv inside the build tree and install pyyaml there.
#   3. Fall back to pip install --user if venv creation is not available.

function(rocshmem_ensure_pyyaml)
    if(DEFINED ROCSHMEM_PYYAML_CHECKED)
        return()
    endif()

    find_package(Python3 COMPONENTS Interpreter QUIET)

    if(NOT Python3_FOUND)
        message(WARNING "Python3 not found. Test tier labels will not be applied.")
        set(ROCSHMEM_PYYAML_AVAILABLE FALSE CACHE INTERNAL "PyYAML availability")
        set(ROCSHMEM_PYYAML_CHECKED TRUE CACHE INTERNAL "PyYAML check completed")
        return()
    endif()

    # 1. Check whether pyyaml is already available in the system interpreter.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import yaml"
        RESULT_VARIABLE _yaml_check
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(_yaml_check EQUAL 0)
        message(STATUS "PyYAML found in system interpreter")
        set(ROCSHMEM_PYYAML_AVAILABLE TRUE CACHE INTERNAL "PyYAML availability")
        set(ROCSHMEM_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE INTERNAL "Python interpreter with PyYAML")
        set(ROCSHMEM_PYYAML_CHECKED TRUE CACHE INTERNAL "PyYAML check completed")
        return()
    endif()

    # 2. Try to create a venv inside the build tree and install pyyaml into it.
    set(_venv_dir "${CMAKE_BINARY_DIR}/_pyyaml_venv")
    message(STATUS "PyYAML not found — attempting venv install at ${_venv_dir}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m venv "${_venv_dir}"
        RESULT_VARIABLE _venv_result
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(_venv_result EQUAL 0)
        set(_venv_python "${_venv_dir}/bin/python")
        execute_process(
            COMMAND ${_venv_python} -m pip install --quiet pyyaml
            RESULT_VARIABLE _pip_result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(_pip_result EQUAL 0)
            message(STATUS "PyYAML installed into venv at ${_venv_dir}")
            set(ROCSHMEM_PYYAML_AVAILABLE TRUE CACHE INTERNAL "PyYAML availability")
            set(ROCSHMEM_PYTHON_EXECUTABLE "${_venv_python}" CACHE INTERNAL "Python interpreter with PyYAML")
        else()
            message(WARNING "Failed to install PyYAML into venv. Test tier labels will not be applied.")
            set(ROCSHMEM_PYYAML_AVAILABLE FALSE CACHE INTERNAL "PyYAML availability")
        endif()
    else()
        # 3. venv not available — fall back to pip install --user.
        message(STATUS "venv not available — falling back to pip install --user pyyaml")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pip install --user pyyaml
            RESULT_VARIABLE _pip_result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(_pip_result EQUAL 0)
            message(STATUS "PyYAML installed via pip --user")
            set(ROCSHMEM_PYYAML_AVAILABLE TRUE CACHE INTERNAL "PyYAML availability")
            set(ROCSHMEM_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE INTERNAL "Python interpreter with PyYAML")
        else()
            message(WARNING "Failed to install PyYAML. Test tier labels will not be applied.")
            set(ROCSHMEM_PYYAML_AVAILABLE FALSE CACHE INTERNAL "PyYAML availability")
        endif()
    endif()

    set(ROCSHMEM_PYYAML_CHECKED TRUE CACHE INTERNAL "PyYAML check completed")
endfunction()
