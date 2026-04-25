# Generate py-interface and package targets

# CLANG installed must be 16.0 or above
# CLANG is only needed for generating new amdsmi_wrapper.py
# this is normally done in a docker container with a controlled clang and python-clang versions
set(clang_ver 16.0)
set(clang_ver_py 16.0.1)
set(ctypeslib_ver_py 2.4.0)

set(PY_BUILD_DIR "python_package")
# amdsmi part of this string is the directory containing all python files
# additionally defined in pyproject.toml
set(PY_PACKAGE_DIR "${PY_BUILD_DIR}/amdsmi")
set(PY_WRAPPER_INSTALL_DIR "${SHARE_INSTALL_PREFIX}" CACHE STRING "Wrapper installation directory")

# Filename of the .so that ships next to the wrapper.
#  - BUILD_PYTHON_LIB=ON  (default): libamd_smi_python.so (distinct SONAME, avoids
#    the inode-dedup conflict that occurs when two copies of libamd_smi.so coexist)
#  - BUILD_PYTHON_LIB=OFF: libamd_smi.so (legacy single-lib build; no .so installed
#    into share/ -- the wrapper falls back to <prefix>/lib/libamd_smi.so at runtime)
set(AMDSMI_PYTHON_LIB_NAME "libamd_smi.so")
if(BUILD_PYTHON_LIB)
    set(AMDSMI_PYTHON_LIB_NAME "libamd_smi_python.so")
endif()
set(AMDSMI_PYTHON_LIB_PATH "${PROJECT_BINARY_DIR}/src/${AMDSMI_PYTHON_LIB_NAME}")

# try to find clang of the right version
set(GOOD_CLANG_FOUND FALSE)
# only try to re-generate amdsmi_wrapper.py if BUILD_WRAPPER is on/true
if(BUILD_WRAPPER)
    find_program(clang NAMES clang REQUIRED)
    # extract clang version manually because find_package(clang) doesn't work
    execute_process(COMMAND ${clang} --version OUTPUT_VARIABLE clang_full_version_string)
    string(REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})
    if((CLANG_VERSION_STRING VERSION_GREATER clang_ver) OR (CLANG_VERSION_STRING VERSION_EQUAL clang_ver))
        message("GOOD CLANG VERSION: ${CLANG_VERSION_STRING}")
        set(GOOD_CLANG_FOUND TRUE)
    else()
        message(FATAL_ERROR "${clang} VERSION TOO OLD!: ${CLANG_VERSION_STRING}")
    endif()
endif()

if(NOT GOOD_CLANG_FOUND)
    # keep old wrapper because no clang found
    message(
        AUTHOR_WARNING
        "A wrapper will not be re-generated! Using old wrapper instead.\nSet -DBUILD_WRAPPER=ON to re-build the wrapper"
    )
    # Python3 is optional here (only needed if python_wheel is explicitly requested)
    find_package(Python3 3.6 COMPONENTS Interpreter)
    add_custom_command(
        OUTPUT amdsmi_wrapper.py ${PY_PACKAGE_DIR}/amdsmi_wrapper.py
        DEPENDS ${AMD_SMI} ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py
        COMMAND mkdir -p ${PY_PACKAGE_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py ${PY_PACKAGE_DIR}/
    )
else()
    find_package(Python3 3.6 COMPONENTS Interpreter Development REQUIRED)
    # --break-system-packages is needed for python 3.11
    # see: https://peps.python.org/pep-0668/
    if(NOT Python3_VERSION VERSION_LESS "3.11")
        set(Python3_BREAK_SYSTEM_PACKAGES "--break-system-packages")
    endif()
    add_custom_target(
        python_pre_reqs
        COMMAND ${Python3_EXECUTABLE} -m pip install ${Python3_BREAK_SYSTEM_PACKAGES} ctypeslib2==${ctypeslib_ver_py}
    )
    # generate new wrapper
    configure_file(${PROJECT_SOURCE_DIR}/tools/generator.py generator.py @ONLY COPYONLY)
    add_custom_command(
        OUTPUT
            amdsmi.h
            ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py
            amdsmi_wrapper.py
            ${PY_PACKAGE_DIR}/amdsmi_wrapper.py
        DEPENDS
            ${AMD_SMI}
            $<$<BOOL:${BUILD_PYTHON_LIB}>:${AMD_SMI}_python>
            python_pre_reqs
            generator.py
            ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h
        COMMAND cp ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h ./
        # Always pass libamd_smi.so to the generator so the resulting wrapper
        # uses '_libraries["libamd_smi.so"]' as the binding-dict key. The actual
        # .so loaded at runtime (libamd_smi_python.so or libamd_smi.so) is
        # decided by the wrapper's loader logic; symbols are identical between
        # the two .so files (same source, only SONAME differs).
        COMMAND
            ${Python3_EXECUTABLE} generator.py "$<$<BOOL:${ENABLE_ESMI_LIB}>:-e -DENABLE_ESMI_LIB>" -i amdsmi.h -l
            ${PROJECT_BINARY_DIR}/src/libamd_smi.so -o ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py
        COMMAND mkdir -p ${PY_PACKAGE_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py ${PY_PACKAGE_DIR}/
    )
endif()

# populate version string
configure_file(pyproject.toml.in ${PY_BUILD_DIR}/pyproject.toml @ONLY)
configure_file(_version.py.in ${PY_PACKAGE_DIR}/_version.py @ONLY)

add_custom_target(python_wrapper DEPENDS amdsmi_wrapper.py)

# hard-linking instead of copying avoids unnecessarry regeneration of packaged files
add_custom_command(
    OUTPUT
        ${PY_PACKAGE_DIR}/__init__.py
        ${PY_PACKAGE_DIR}/amdsmi_exception.py
        ${PY_PACKAGE_DIR}/amdsmi_interface.py
        ${PY_PACKAGE_DIR}/README.md
        ${PY_PACKAGE_DIR}/LICENSE
    DEPENDS python_wrapper
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_exception.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_interface.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROJECT_SOURCE_DIR}/LICENSE ${PY_PACKAGE_DIR}/
)

# copy libamd_smi.so to allow for a self-contained python package
add_custom_command(
    OUTPUT ${PY_PACKAGE_DIR}/${AMDSMI_PYTHON_LIB_NAME}
    DEPENDS ${AMDSMI_PYTHON_LIB_PATH}
    COMMAND cp "${AMDSMI_PYTHON_LIB_PATH}" ${PY_PACKAGE_DIR}/
)

add_custom_target(
    python_package
    ALL
    DEPENDS
        ${PY_BUILD_DIR}/pyproject.toml
        ${PY_PACKAGE_DIR}/_version.py
        ${PY_PACKAGE_DIR}/__init__.py
        ${PY_PACKAGE_DIR}/amdsmi_exception.py
        ${PY_PACKAGE_DIR}/amdsmi_interface.py
        ${PY_PACKAGE_DIR}/README.md
        ${PY_PACKAGE_DIR}/LICENSE
        ${PY_PACKAGE_DIR}/${AMDSMI_PYTHON_LIB_NAME}
)
if(BUILD_PYTHON_LIB)
    add_dependencies(python_package ${AMD_SMI}_python)
endif()

# Always attempt to build the wheel.  pip is optional: if it is unavailable
# or the build fails, the || true makes the step non-fatal so the rest of the
# package still builds.  The postinstall guard skips pip registration when no
# wheel is present.
if(Python3_Interpreter_FOUND)
    set(_PY3 ${Python3_EXECUTABLE})
else()
    set(_PY3 "python3")
endif()

# Generate a small cmake script so that pip wheel runs non-fatally at build
# time without relying on shell quoting or multi-line COMMAND strings.
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/run_pip_wheel.cmake
    CONTENT
        "set(ENV{PYTHONPATH} \"${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}\")
execute_process(
    COMMAND ${_PY3} -m pip wheel --no-deps
            -w ${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}
    RESULT_VARIABLE _PIP_RC
)
if(NOT _PIP_RC EQUAL 0)
    message(STATUS \"INFO: pip wheel build skipped (pip unavailable or failed)\")
endif()
"
)

add_custom_target(
    python_wheel
    ALL
    DEPENDS python_package
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/run_pip_wheel.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}
    COMMENT "Building amdsmi wheel (optional)"
)

install(
    CODE
        "
        file(GLOB _WHEEL_FILES
             \"${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}/amdsmi-*.whl\")
        if(_WHEEL_FILES)
            list(GET _WHEEL_FILES 0 _WHEEL_FILE)
            message(STATUS \"Installing wheel: \${_WHEEL_FILE}\")
            file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${PY_WRAPPER_INSTALL_DIR}\"
                 TYPE FILE FILES \"\${_WHEEL_FILE}\")
        else()
            message(STATUS \"No amdsmi wheel found in build dir -- pip registration skipped.\")
        endif()
    "
    COMPONENT dev
)

# Install the package directory (amdsmi/) under <prefix>/share/amd_smi/.
# *.so files are excluded here — libamd_smi_python.so is installed by
# src/CMakeLists.txt directly into <prefix>/share/amd_smi/amdsmi/ so it sits
# next to the wrapper at runtime and is found by the wrapper's load logic.
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}
    DESTINATION ${PY_WRAPPER_INSTALL_DIR}
    COMPONENT dev
    PATTERN "*.so" EXCLUDE
)
