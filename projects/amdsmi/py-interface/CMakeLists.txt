# Generate py-interface and package targets

# CLANG installed must be 16.0 or above
# CLANG is only needed for generating new amdsmi_wrapper.py
# this is normally done in a docker container with a controlled clang and python-clang versions
set(clang_ver 16.0)
set(clang_ver_py 16.0.1)
set(ctypeslib_ver_py 2.3.4)

set(PY_BUILD_DIR "python_package")
# amdsmi part of this string is the directory containing all python files
# additionally defined in pyproject.toml
set(PY_PACKAGE_DIR "${PY_BUILD_DIR}/amdsmi")
set(PY_WRAPPER_INSTALL_DIR "${SHARE_INSTALL_PREFIX}" CACHE STRING "Wrapper installation directory")

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
    add_custom_command(
        OUTPUT amdsmi_wrapper.py ${PY_PACKAGE_DIR}/amdsmi_wrapper.py
        DEPENDS ${AMD_SMI} ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py
        COMMAND mkdir -p ${PY_PACKAGE_DIR}
        COMMAND  ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py ${PY_PACKAGE_DIR}/)
else()
    find_package(Python3 3.6 COMPONENTS Interpreter Development REQUIRED)
    # --break-system-packages is needed for python 3.11
    # see: https://peps.python.org/pep-0668/
    if(NOT Python3_VERSION VERSION_LESS "3.11")
        set(Python3_BREAK_SYSTEM_PACKAGES "--break-system-packages")
    endif()
    add_custom_target(python_pre_reqs COMMAND ${Python3_EXECUTABLE} -m pip install ${Python3_BREAK_SYSTEM_PACKAGES}
                                              ctypeslib2==${ctypeslib_ver_py})
    # generate new wrapper
    configure_file(${PROJECT_SOURCE_DIR}/tools/generator.py generator.py @ONLY COPYONLY)
    add_custom_command(
        OUTPUT amdsmi.h ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py amdsmi_wrapper.py
               ${PY_PACKAGE_DIR}/amdsmi_wrapper.py
        DEPENDS ${AMD_SMI} python_pre_reqs generator.py ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h
        COMMAND cp ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h ./
        COMMAND ${Python3_EXECUTABLE} generator.py "$<$<BOOL:${ENABLE_ESMI_LIB}>:-e -DENABLE_ESMI_LIB>" -i amdsmi.h -l
                ${PROJECT_BINARY_DIR}/src/libamd_smi.so -o ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py
        COMMAND mkdir -p ${PY_PACKAGE_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_wrapper.py ${PY_PACKAGE_DIR}/)
endif()

# populate version string
configure_file(pyproject.toml.in ${PY_BUILD_DIR}/pyproject.toml @ONLY)
configure_file(_version.py.in ${PY_PACKAGE_DIR}/_version.py @ONLY)
configure_file(setup.py.in ${PY_BUILD_DIR}/setup.py @ONLY)

add_custom_target(python_wrapper DEPENDS amdsmi_wrapper.py)

# hard-linking instead of copying avoids unnecessarry regeneration of packaged files
add_custom_command(
    OUTPUT ${PY_PACKAGE_DIR}/__init__.py ${PY_PACKAGE_DIR}/amdsmi_exception.py ${PY_PACKAGE_DIR}/amdsmi_interface.py
           ${PY_PACKAGE_DIR}/README.md ${PY_PACKAGE_DIR}/LICENSE
    DEPENDS python_wrapper
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_exception.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_interface.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROJECT_SOURCE_DIR}/LICENSE ${PY_PACKAGE_DIR}/)

# copy libamd_smi.so to allow for a self-contained python package
add_custom_command(OUTPUT ${PY_PACKAGE_DIR}/libamd_smi.so DEPENDS ${PROJECT_BINARY_DIR}/src/libamd_smi.so
                   COMMAND cp "${PROJECT_BINARY_DIR}/src/libamd_smi.so" ${PY_PACKAGE_DIR}/)

add_custom_target(
    python_package ALL
    DEPENDS ${PY_BUILD_DIR}/pyproject.toml
            ${PY_BUILD_DIR}/setup.py
            ${PY_PACKAGE_DIR}/_version.py
            ${PY_PACKAGE_DIR}/__init__.py
            ${PY_PACKAGE_DIR}/amdsmi_exception.py
            ${PY_PACKAGE_DIR}/amdsmi_interface.py
            ${PY_PACKAGE_DIR}/README.md
            ${PY_PACKAGE_DIR}/LICENSE
            ${PY_PACKAGE_DIR}/libamd_smi.so)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}/pyproject.toml
          ${CMAKE_CURRENT_BINARY_DIR}/${PY_BUILD_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}/_version.py
    DESTINATION ${PY_WRAPPER_INSTALL_DIR}
    COMPONENT dev)

install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}
    DESTINATION ${PY_WRAPPER_INSTALL_DIR}
    COMPONENT dev)
