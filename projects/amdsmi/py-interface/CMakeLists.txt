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
    # The committed wrapper is authoritative: its loader and every ctypes binding
    # key on 'libamd_smi.so'. With BUILD_PYTHON_WHEEL=ON the generator is fed
    # -l libamd_smi_python.so, so ctypeslib would emit bindings keyed on
    # 'libamd_smi_python.so' while the loader still populates 'libamd_smi.so' --
    # every API call would then KeyError. The wheel reuses the committed wrapper
    # (only flipping the fallback flag), so it never needs regeneration. Reject
    # the combination rather than emit a broken wrapper.
    if(BUILD_PYTHON_WHEEL)
        message(
            FATAL_ERROR
            "BUILD_WRAPPER=ON is incompatible with BUILD_PYTHON_WHEEL=ON: regenerating the wrapper against libamd_smi_python.so produces bindings that mismatch the committed libamd_smi.so loader. Regenerate the wrapper with BUILD_PYTHON_WHEEL=OFF, then build the wheel from the committed wrapper."
        )
    endif()
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
        ${PY_BUILD_DIR}/setup.py
    DEPENDS python_wrapper
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${PY_BUILD_DIR}/
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
        ${PY_BUILD_DIR}/setup.py
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

# Build a relocatable pure-python wheel (py3-none-any, no bundled .so) and ship
# it under share/amd_smi/wheels. The DEB/RPM post-install scriptlet installs it
# into the target interpreter's site-packages, detecting that interpreter on the
# target host at install time rather than baking a system path at build time
# (which broke on build-host vs target-host python mismatches). Nothing here
# writes to an absolute system path, so a prefix-based / non-root
# `cmake --install` (e.g. TheRock staging) stays inside CMAKE_INSTALL_PREFIX.
# Skipped for the manylinux wheel build (BUILD_PYTHON_WHEEL=ON), which produces
# its own .so-bundled wheel.
if(NOT BUILD_PYTHON_WHEEL)
    set(_amdsmi_wheelhouse ${CMAKE_CURRENT_BINARY_DIR}/wheelhouse)
    set(_amdsmi_wheel_stamp ${_amdsmi_wheelhouse}/.built)
    # Model the wheel as a generated artifact (OUTPUT) so it is a real build
    # dependency, not just an ALL side effect. Depend on the staged module files
    # (not just the python_package target) so the wheel rebuilds when any
    # packaged .py changes under Ninja. VERBATIM + quoting keeps paths intact.
    add_custom_command(
        OUTPUT ${_amdsmi_wheel_stamp}
        DEPENDS
            python_package
            ${PROJECT_SOURCE_DIR}/tools/build_system_wheel.py
            ${PY_PACKAGE_DIR}/__init__.py
            ${PY_PACKAGE_DIR}/amdsmi_interface.py
            ${PY_PACKAGE_DIR}/amdsmi_interface_utils.py
            ${PY_PACKAGE_DIR}/amdsmi_exception.py
            ${PY_PACKAGE_DIR}/amdsmi_wrapper.py
            ${PY_PACKAGE_DIR}/_version.py
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_amdsmi_wheelhouse}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_amdsmi_wheelhouse}"
        COMMAND
            ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tools/build_system_wheel.py" --package-dir
            "${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}" --version "${amd_smi_lib_WHEEL_VERSION}" --output-dir
            "${_amdsmi_wheelhouse}"
        COMMAND ${CMAKE_COMMAND} -E touch "${_amdsmi_wheel_stamp}"
        VERBATIM
        COMMENT "Building relocatable amdsmi system wheel"
    )
    add_custom_target(python_system_wheel ALL DEPENDS ${_amdsmi_wheel_stamp})

    # Fail loudly at install time if the wheel was never built (e.g. a bare
    # `cmake --install` with no preceding `cmake --build`), instead of silently
    # shipping a package with no importable module.
    install(
        CODE
            "file(GLOB _amdsmi_whl \"${_amdsmi_wheelhouse}/amdsmi-*.whl\")
if(NOT _amdsmi_whl)
    message(FATAL_ERROR \"amdsmi system wheel missing from ${_amdsmi_wheelhouse}; run 'cmake --build' before install so the python_system_wheel target produces it.\")
endif()"
        COMPONENT dev
    )

    # Ship the wheel and the scriptlet-invoked installer under share/amd_smi.
    install(
        DIRECTORY ${_amdsmi_wheelhouse}/
        DESTINATION ${SHARE_INSTALL_PREFIX}/wheels
        COMPONENT dev
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        FILES_MATCHING
        PATTERN "amdsmi-*.whl"
    )
    install(
        PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_install_wheel.py
        DESTINATION ${SHARE_INSTALL_PREFIX}
        COMPONENT dev
    )
endif()

# Also stage the plain module tree under share/amd_smi (prefix-relative). The
# wheel above is the primary importable form for the DEB/RPM (its post-install
# scriptlet installs it into site-packages); this copy is required for things
# that do NOT go through site-packages:
#   1. The TheRock artifact flow: core-amdsmi's artifact manifest captures
#      share/amd_smi/** (and bin/, libexec/), but NOT the interpreter's
#      site-packages tree. Without this copy the module is absent from the
#      artifact and downstream consumers (rocprofiler-compute, omnistat) that
#      `sys.path.insert(ROCM_PATH + "/share/amd_smi")` get ModuleNotFoundError.
#   2. In-tree tooling/tests that resolve the module relative to ROCM_PATH.
#   3. The CLI: amdsmi_cli/amdsmi_init.py prefers this copy.
# The bundled wheel-only .so is excluded here too; share/amd_smi resolves the
# system libamd_smi.so via the linker.
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}
    DESTINATION ${SHARE_INSTALL_PREFIX}
    COMPONENT dev
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    PATTERN "libamd_smi*.so" EXCLUDE
)
