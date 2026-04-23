message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                     CMake AMDSMI CLI Install                      ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

# Set CLI Build Directory
set(PY_PACKAGE_DIR "amdsmi_cli")
set(PY_CLI_INSTALL_DIR "${CMAKE_INSTALL_LIBEXECDIR}" CACHE STRING "CLI tool installation directory")

# populate version string
configure_file(_version.py.in ${PY_PACKAGE_DIR}/_version.py @ONLY)

set(CLI_FILES
    __init__.py
    amdsmi_cli.py
    amdsmi_commands.py
    amdsmi_helpers.py
    amdsmi_init.py
    amdsmi_logger.py
    amdsmi_parser.py
    amdsmi_cli_exceptions.py
    amdsmi_rocm_smi_compat.py
    BDF.py
    README.md
    Release_Notes.md
)

set(CLI_OUTPUTS "")
foreach(file ${CLI_FILES})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    set(dst ${PY_PACKAGE_DIR}/${file})
    add_custom_command(
        OUTPUT ${dst}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
        COMMAND ${CMAKE_COMMAND} -E touch_nocreate ${dst}
        DEPENDS ${src}
    )
    list(APPEND CLI_OUTPUTS ${dst})
endforeach()

# The CLI requires the python amdsmi wrapper to be installed
add_custom_target(amdsmi_cli ALL DEPENDS python_package ${PY_PACKAGE_DIR}/_version.py ${CLI_OUTPUTS})

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR} DESTINATION ${PY_CLI_INSTALL_DIR} COMPONENT dev)

install(
    PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${PY_PACKAGE_DIR}/amdsmi_cli.py
    DESTINATION ${PY_CLI_INSTALL_DIR}/${PY_PACKAGE_DIR}
    COMPONENT dev
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# symlink amdsmi_cli.py to amd-smi
add_custom_target(
    link_amdsmi_cli
    ALL
    DEPENDS amdsmi_cli
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/bin/amd-smi
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND
        ${CMAKE_COMMAND} -E create_symlink ../${PY_CLI_INSTALL_DIR}/${PY_PACKAGE_DIR}/amdsmi_cli.py
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/amd-smi
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bin/amd-smi DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT dev)
