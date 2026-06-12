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
# AMDSMI_PYTHON_LIB_NAME is overloaded: when BUILD_PYTHON_WHEEL=OFF it
# resolves to the system libamd_smi.so (the python_package staging tree
# just copies it for in-tree imports); when ON it resolves to the
# SONAME-isolated libamd_smi_python.so that ships inside the wheel.
set(AMDSMI_PYTHON_LIB_NAME "libamd_smi.so")
if(BUILD_PYTHON_WHEEL)
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
    # Still need Python3 for building the wheel package
    find_package(Python3 3.6 COMPONENTS Interpreter REQUIRED)
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
            $<$<BOOL:${BUILD_PYTHON_WHEEL}>:${AMD_SMI}_python>
            python_pre_reqs
            generator.py
            ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h
        COMMAND cp ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h ./
        COMMAND
            ${Python3_EXECUTABLE} generator.py "$<$<BOOL:${ENABLE_ESMI_LIB}>:-e -DENABLE_ESMI_LIB>" -i amdsmi.h -l
            ${AMDSMI_PYTHON_LIB_PATH} -o ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py
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
if(BUILD_PYTHON_WHEEL)
    add_dependencies(python_package ${AMD_SMI}_python)
endif()

# Build a wheel into the binary dir (no deps, local artifacts only).
# Gated on BUILD_PYTHON_WHEEL so the default ROCm CI build (no extra flags) does
# not produce a .whl artifact -- only the system .deb/.rpm layout is built.
if(BUILD_PYTHON_WHEEL)
    add_custom_target(
        python_wheel
        ALL
        DEPENDS python_package
        COMMAND
            ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR} ${Python3_EXECUTABLE} -m pip
            wheel --no-deps -w ${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}
        COMMENT "Building amdsmi wheel without dependencies"
    )
endif()

# Install the python module directly into the target Python's
# site-packages so `import amdsmi` works after `apt install` / `dnf install`
# without any sys.path / .pth tricks. The bundled libamd_smi_python.so is
# excluded -- the system package ships libamd_smi.so via /opt/rocm/lib and
# the wrapper resolves it through the dynamic linker's SONAME lookup
# (ld.so.conf.d entry installed by postinst). The bundled .so is only
# needed inside the pip wheel.
#
# We MUST install to the SYSTEM python's purelib, not whichever python
# find_package(Python3) happened to pick. On Ubuntu 20 / Debian 10 build
# containers the cmake-discovered interpreter can be a non-system python
# (e.g. /opt/python/cp3X-cp3X/bin/python3 in manylinux images) whose
# sysconfig returns a path that no user's python3 ever searches -- the
# package would install successfully and `import amdsmi` would still
# fail with "No module named 'amdsmi'".
#
# /usr/bin/python3 is the python that /opt/rocm/bin/amd-smi resolves to
# via its `#!/usr/bin/env python3` shebang at runtime, so its purelib is
# the only correct destination for the system DEB/RPM. We query it
# directly with sysconfig.get_paths()['purelib'].
#
# IMPORTANT: do NOT use sysconfig.get_paths()['purelib'] -- on
# Debian/Ubuntu the sysconfig "purelib" is /usr/lib/python3.X/site-packages
# which is NOT on the system python's sys.path. Debian-derived distros put
# system packages under /usr/lib/python3/dist-packages. The reliable query
# is site.getsitepackages(), which returns the actual on-sys.path locations
# we should install into. We prefer (in order):
#   1. /usr/lib/python3/dist-packages         (Debian/Ubuntu primary)
#   2. /usr/lib/python3.X/dist-packages       (Debian/Ubuntu versioned)
#   3. /usr/lib/python3.X/site-packages       (RHEL/SLES/Fedora primary)
# We avoid /usr/local/* because the system DEB/RPM owns /usr, and pip
# manages /usr/local separately.
set(_amdsmi_sitelib_query
    [[
import site, sys
candidates = [p for p in site.getsitepackages() if not p.startswith("/usr/local")]
order = [
    lambda p: p == "/usr/lib/python3/dist-packages",
    lambda p: p.endswith("/dist-packages"),
    lambda p: p.endswith("/site-packages"),
]
for pred in order:
    for p in candidates:
        if pred(p):
            print(p); sys.exit(0)
import sysconfig
print(sysconfig.get_paths()["purelib"])
]]
)
if(EXISTS "/usr/bin/python3")
    execute_process(
        COMMAND /usr/bin/python3 -c "${_amdsmi_sitelib_query}"
        OUTPUT_VARIABLE AMDSMI_SYSTEM_PYTHON_SITELIB
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _amdsmi_sitelib_rc
    )
    if(NOT _amdsmi_sitelib_rc EQUAL 0 OR AMDSMI_SYSTEM_PYTHON_SITELIB STREQUAL "")
        message(
            WARNING
            "Could not query /usr/bin/python3 site-packages; falling back to Python3_SITELIB=${Python3_SITELIB}"
        )
        set(AMDSMI_SYSTEM_PYTHON_SITELIB "${Python3_SITELIB}")
    endif()
else()
    set(AMDSMI_SYSTEM_PYTHON_SITELIB "${Python3_SITELIB}")
endif()
message(STATUS "amdsmi python module install path: ${AMDSMI_SYSTEM_PYTHON_SITELIB}")

install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}
    DESTINATION ${AMDSMI_SYSTEM_PYTHON_SITELIB}
    COMPONENT dev
    PATTERN "libamd_smi*.so" EXCLUDE
)
