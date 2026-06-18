set(SUBCOMMAND_FILES
    __init__.py
    bad_pages.py
    default.py
    event.py
    fabric.py
    firmware.py
    list_devices.py
    metric.py
    monitor.py
    node.py
    partition.py
    process.py
    ras.py
    reset.py
    set_value.py
    static.py
    topology.py
    version.py
    xgmi.py
)

set(DST_DIR ${CMAKE_CURRENT_BINARY_DIR}/../${PY_PACKAGE_DIR}/subcommands)

set(SUBCOMMAND_OUTPUTS "")
foreach(file ${SUBCOMMAND_FILES})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    set(dst ${DST_DIR}/${file})
    add_custom_command(
        OUTPUT ${dst}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DST_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
        DEPENDS ${src}
    )
    list(APPEND SUBCOMMAND_OUTPUTS ${dst})
endforeach()

add_custom_target(amdsmi_cli_subcommands ALL DEPENDS python_package ${SUBCOMMAND_OUTPUTS})
