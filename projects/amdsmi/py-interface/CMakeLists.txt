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
        ${PY_PACKAGE_DIR}/amdsmi_interface_utils.py
        ${PY_PACKAGE_DIR}/README.md
        ${PY_PACKAGE_DIR}/LICENSE
    DEPENDS python_wrapper
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_exception.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_interface.py ${PY_PACKAGE_DIR}/
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_interface_utils.py ${PY_PACKAGE_DIR}/
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
        ${PY_PACKAGE_DIR}/amdsmi_interface_utils.py
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
        # The wheel bundles libamd_smi_python.so, so it must never fall back
        # to a system libamd_smi.so. Flip the loader flag in the staged copy
        # (only the wheel ships this file) so a wheel that is missing its
        # bundled .so fails loudly instead of loading an unrelated system lib.
        COMMAND
            ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/disable_system_fallback.py
            ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}/amdsmi_wrapper.py
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
# via its `#!/usr/bin/env python3` shebang at runtime, so its site-packages
# is the only correct destination for the system DEB/RPM. We query it
# directly with the script below.
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
sitepkgs = site.getsitepackages()
candidates = [p for p in sitepkgs if not p.startswith("/usr/local")]
order = [
    lambda p: p == "/usr/lib/python3/dist-packages",
    lambda p: p.endswith("/dist-packages"),
    lambda p: p.endswith("/site-packages"),
]
for pred in order:
    for p in candidates:
        if pred(p):
            print(p); sys.exit(0)
# No /usr candidate matched (unusual container / venv layout). Fall back to
# the first directory site.getsitepackages() reported: unlike sysconfig
# purelib, every getsitepackages() entry is guaranteed to be on sys.path, so
# `import amdsmi` still resolves even if it lands under /usr/local.
if sitepkgs:
    print(sitepkgs[0]); sys.exit(0)
import sysconfig
print(sysconfig.get_paths()["purelib"])
]]
)
# The module must land where the system's DEFAULT python3 looks -- that is the
# interpreter /opt/rocm/bin/amd-smi runs under (its `#!/usr/bin/python3` shebang)
# and the one a plain `import amdsmi` uses. We must NOT detect from a
# non-default interpreter: RHEL build/CI images frequently repoint
# /usr/bin/python3 to a newer python via `alternatives` (e.g. to run a 3.7+
# build harness on an el8 base whose platform python is 3.6). Detecting from
# that switched interpreter would install into /usr/lib64/python3.NN/... where
# the *default* python3 -- and the CLI -- never look, so `import amdsmi` and
# `amd-smi` both fail with ModuleNotFoundError even though the package installed
# "successfully". (Verified on el8/el9: the bug reproduces exactly when
# build-python != default-python.)
#
# /usr/libexec/platform-python is RHEL/CentOS/Fedora's stable symlink to the
# OS's own python3; `alternatives --set python3` does NOT repoint it, so it is
# the reliable anchor for the DEFAULT interpreter's site-packages. Prefer it
# when present, else fall back to /usr/bin/python3 (correct on Debian/Ubuntu,
# SUSE, and AzureLinux, none of which ship platform-python and none of which the
# harness repoints).
set(_amdsmi_detect_python "/usr/bin/python3")
if(EXISTS "/usr/libexec/platform-python")
    set(_amdsmi_detect_python "/usr/libexec/platform-python")
endif()
# Packagers may override the destination explicitly; the empty default
# means "auto-detect from the default system python3 below".
set(AMDSMI_SYSTEM_PYTHON_SITELIB
    ""
    CACHE PATH
    "Install destination for the amdsmi Python module (system site-packages). Empty = auto-detect from the default system python3."
)
if(NOT AMDSMI_SYSTEM_PYTHON_SITELIB)
    if(EXISTS "${_amdsmi_detect_python}")
        execute_process(
            COMMAND ${_amdsmi_detect_python} -c "${_amdsmi_sitelib_query}"
            OUTPUT_VARIABLE _amdsmi_detected_sitelib
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _amdsmi_sitelib_rc
        )
        if(NOT _amdsmi_sitelib_rc EQUAL 0 OR _amdsmi_detected_sitelib STREQUAL "")
            message(
                WARNING
                "Could not query ${_amdsmi_detect_python} site-packages; falling back to Python3_SITELIB=${Python3_SITELIB}"
            )
            set(_amdsmi_detected_sitelib "${Python3_SITELIB}")
        endif()
    else()
        set(_amdsmi_detected_sitelib "${Python3_SITELIB}")
    endif()
    set(AMDSMI_SYSTEM_PYTHON_SITELIB
        "${_amdsmi_detected_sitelib}"
        CACHE PATH
        "Install destination for the amdsmi Python module (system site-packages). Empty = auto-detect from the default system python3."
        FORCE
    )
endif()
message(
    STATUS
    "amdsmi python module install path: ${AMDSMI_SYSTEM_PYTHON_SITELIB} (detected via ${_amdsmi_detect_python})"
)

# Derive the interpreter ABI (major.minor) that the install path is pinned to,
# for use as an RPM `python(abi) = X.Y` dependency (set in the root
# CMakeLists.txt). RHEL/SLES/Fedora site-packages paths are version-specific
# (e.g. /usr/lib64/python3.12/site-packages), so an RPM built against one
# interpreter installs its module where a DIFFERENT system python3 will never
# look -- `import amdsmi` then fails with ModuleNotFoundError even though the
# loose `python3 >= 3.6.8` dependency is satisfied. Pinning `python(abi)` to
# the exact minor makes the RPM installable only where the matching interpreter
# (and thus the baked path) exists. Debian's version-agnostic
# /usr/lib/python3/dist-packages carries no minor, so no ABI is derived and the
# .deb dependency is left untouched.
set(AMDSMI_SYSTEM_PYTHON_ABI "")
if(AMDSMI_SYSTEM_PYTHON_SITELIB MATCHES "python([0-9]+\\.[0-9]+)")
    set(AMDSMI_SYSTEM_PYTHON_ABI "${CMAKE_MATCH_1}")
endif()
set(AMDSMI_SYSTEM_PYTHON_ABI
    "${AMDSMI_SYSTEM_PYTHON_ABI}"
    CACHE STRING
    "Python major.minor ABI the RPM module path is pinned to (empty = version-agnostic path, e.g. Debian dist-packages)."
    FORCE
)

# Concrete interpreter the amd-smi CLI shebang must point at, for RPM distros
# where the module lives in a version-specific site-packages. The CLI ships with
# `#!/usr/bin/python3`, but on RHEL that symlink can be repointed (by
# `alternatives`, a venv on PATH, or simply a newer default in a later release)
# to a python minor whose site-packages does NOT contain amdsmi -- then the CLI
# dies with ModuleNotFoundError even though the package is installed. Pinning the
# shebang to the exact interpreter the module was built for makes the CLI immune
# to that. The value must be the REAL path of the detection interpreter, not a
# guessed /usr/bin/python3.X: on el8 `python(abi) = 3.6` is provided by
# platform-python, whose binary is /usr/libexec/platform-python3.6 -- there is NO
# /usr/bin/python3.6 -- so a hardcoded /usr/bin/python3.X shebang would be a
# guaranteed "bad interpreter" failure. Resolving the anchor's realpath yields
# the interpreter that the python(abi) dependency actually guarantees present
# (el8 -> /usr/libexec/platform-python3.6, el9 -> /usr/bin/python3.9,
# Azure -> /usr/bin/python3.12). Only set on RPM distros (ABI derived); Debian
# keeps the version-agnostic /usr/bin/python3 shebang (dist-packages serves every
# python3 minor there, and pinning would break across default-python transitions).
set(AMDSMI_CLI_SHEBANG_INTERP "")
if(AMDSMI_SYSTEM_PYTHON_ABI AND EXISTS "${_amdsmi_detect_python}")
    execute_process(
        COMMAND ${_amdsmi_detect_python} -c "import os,sys;print(os.path.realpath(sys.executable))"
        OUTPUT_VARIABLE _amdsmi_cli_interp
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _amdsmi_cli_interp_rc
    )
    if(_amdsmi_cli_interp_rc EQUAL 0 AND EXISTS "${_amdsmi_cli_interp}")
        set(AMDSMI_CLI_SHEBANG_INTERP "${_amdsmi_cli_interp}")
    endif()
endif()
set(AMDSMI_CLI_SHEBANG_INTERP
    "${AMDSMI_CLI_SHEBANG_INTERP}"
    CACHE STRING
    "Concrete interpreter path for the amd-smi CLI shebang on RPM distros (empty = keep #!/usr/bin/python3, e.g. Debian)."
    FORCE
)
if(AMDSMI_CLI_SHEBANG_INTERP)
    message(STATUS "amd-smi CLI shebang interpreter: ${AMDSMI_CLI_SHEBANG_INTERP}")
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}
    DESTINATION ${AMDSMI_SYSTEM_PYTHON_SITELIB}
    COMPONENT dev
    PATTERN "libamd_smi*.so" EXCLUDE
)

# Also stage the module under share/amd_smi (the historical location). site-
# packages above is the primary, importable location for the DEB/RPM and for
# `pip install`; this second copy is required for two things that do NOT go
# through site-packages:
#   1. The TheRock artifact flow: core-amdsmi's artifact manifest captures
#      share/amd_smi/** (and bin/, libexec/), but NOT the interpreter's
#      site-packages tree -- which, being an absolute system path like
#      /usr/lib/python3.X/site-packages, falls outside the /opt/rocm artifact
#      root entirely. Without this copy the module is absent from the artifact
#      and downstream consumers (rocprofiler-compute, omnistat) that
#      `sys.path.insert(ROCM_PATH + "/share/amd_smi")` get ModuleNotFoundError.
#   2. In-tree tooling/tests that resolve the module relative to ROCM_PATH.
# The bundled wheel-only .so is excluded here too; share/amd_smi resolves the
# system libamd_smi.so via the linker, exactly as site-packages does.
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}
    DESTINATION ${SHARE_INSTALL_PREFIX}
    COMPONENT dev
    PATTERN "libamd_smi*.so" EXCLUDE
)
