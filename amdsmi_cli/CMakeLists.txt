message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                     CMake AMDSMI CLI Install                      ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

# Set CLI Build Directory
set(PY_PACKAGE_DIR "amdsmi_cli")
set(PY_CLI_INSTALL_DIR "${CMAKE_INSTALL_LIBEXECDIR}" CACHE STRING "CLI tool installation directory")

# populate version string
configure_file(_version.py.in ${PY_PACKAGE_DIR}/_version.py @ONLY)

# copy only if files are different
add_custom_command(
    OUTPUT ${PY_PACKAGE_DIR}/__init__.py
           ${PY_PACKAGE_DIR}/amdsmi_cli.py
           ${PY_PACKAGE_DIR}/amdsmi_commands.py
           ${PY_PACKAGE_DIR}/amdsmi_helpers.py
           ${PY_PACKAGE_DIR}/amdsmi_init.py
           ${PY_PACKAGE_DIR}/amdsmi_logger.py
           ${PY_PACKAGE_DIR}/amdsmi_parser.py
           ${PY_PACKAGE_DIR}/amdsmi_cli_exceptions.py
           ${PY_PACKAGE_DIR}/BDF.py
           ${PY_PACKAGE_DIR}/README.md
           ${PY_PACKAGE_DIR}/Release_Notes.md
    DEPENDS amdsmi_cli
    COMMAND mkdir -p ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_cli.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_commands.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_helpers.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_init.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_logger.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_parser.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/amdsmi_cli_exceptions.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/BDF.py ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${PY_PACKAGE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/Release_Notes.md ${PY_PACKAGE_DIR}/)

# The CLI requires the python amdsmi wrapper to be installed
add_custom_target(
    amdsmi_cli ALL
    DEPENDS python_package
            ${PY_PACKAGE_DIR}/__init__.py
            ${PY_PACKAGE_DIR}/_version.py
            ${PY_PACKAGE_DIR}/amdsmi_cli.py
            ${PY_PACKAGE_DIR}/amdsmi_commands.py
            ${PY_PACKAGE_DIR}/amdsmi_helpers.py
            ${PY_PACKAGE_DIR}/amdsmi_init.py
            ${PY_PACKAGE_DIR}/amdsmi_logger.py
            ${PY_PACKAGE_DIR}/amdsmi_parser.py
            ${PY_PACKAGE_DIR}/amdsmi_cli_exceptions.py
            ${PY_PACKAGE_DIR}/BDF.py
            ${PY_PACKAGE_DIR}/README.md
            ${PY_PACKAGE_DIR}/Release_Notes.md)

install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}
    DESTINATION ${PY_CLI_INSTALL_DIR}
    COMPONENT dev)

install(
    PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}/amdsmi_cli.py
    DESTINATION ${PY_CLI_INSTALL_DIR}/${PY_PACKAGE_DIR}
    COMPONENT dev)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# symlink amdsmi_cli.py to amd-smi
add_custom_target(
    link_amdsmi_cli ALL
    DEPENDS amdsmi_cli
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/bin/amd-smi
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../${PY_CLI_INSTALL_DIR}/${PY_PACKAGE_DIR}/amdsmi_cli.py
            ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/amd-smi)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/bin/amd-smi
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT dev)
