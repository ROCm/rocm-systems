# function to append content of IN_FILE to OUT_FILE
function(append_file IN_FILE OUT_FILE)
    file(READ "${IN_FILE}" CONTENTS)
    file(APPEND "${OUT_FILE}" "${CONTENTS}")
endfunction()

## Configure Copyright File for Debian Package
function(
    configure_pkg
    PACKAGE_NAME_T
    COMPONENT_NAME_T
    PACKAGE_VERSION_T
    MAINTAINER_NM_T
    MAINTAINER_EMAIL_T
)
    if("${COMPONENT_NAME_T}" STREQUAL "asan")
        set(LINTIAN_DOCS_DIR "${CMAKE_INSTALL_DOCDIR}-asan")
    else()
        set(LINTIAN_DOCS_DIR ${CMAKE_INSTALL_DOCDIR})
    endif()
    # Check If Debian Platform
    find_file(DEBIAN debian_version debconf.conf PATHS /etc)
    if(DEBIAN)
        set(BUILD_DEBIAN_PKGING_FLAG
            ON
            CACHE BOOL
            "Internal Status Flag to indicate Debian Packaging Build"
            FORCE
        )
        set_debian_pkg_cmake_flags(${PACKAGE_NAME_T} ${PACKAGE_VERSION_T}
                                  ${MAINTAINER_NM_T} ${MAINTAINER_EMAIL_T}
        )

        # Create debian directory in build tree
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/DEBIAN")

        # Configure the copyright file
        configure_file(
            "${CMAKE_SOURCE_DIR}/DEBIAN/copyright.in"
            "${CMAKE_BINARY_DIR}/DEBIAN/copyright"
            @ONLY
        )

        # Install copyright file
        install(
            FILES "${CMAKE_BINARY_DIR}/DEBIAN/copyright"
            DESTINATION ${LINTIAN_DOCS_DIR}
            COMPONENT ${COMPONENT_NAME_T}
        )

        # Configure the changelog file
        set(CHANGELOG_DATA_FILES
            "${CMAKE_SOURCE_DIR}/DEBIAN/changelog.in"
            "${CMAKE_SOURCE_DIR}/CHANGELOG.md"
        )
        set(CHANGELOG_DATA_APPENDED "${CMAKE_BINARY_DIR}/DEBIAN/changelog.in")
        file(WRITE "${CHANGELOG_DATA_APPENDED}" "")
        foreach(changelog_data ${CHANGELOG_DATA_FILES})
            append_file("${changelog_data}" "${CHANGELOG_DATA_APPENDED}")
        endforeach()
        configure_file(
            "${CHANGELOG_DATA_APPENDED}"
            "${CMAKE_BINARY_DIR}/DEBIAN/changelog.Debian"
            @ONLY
        )
        # Install Change Log
        find_program(DEB_GZIP_EXEC gzip)
        if(NOT DEB_GZIP_EXEC)
            message(FATAL_ERROR "gzip command not found: Failed to compress the changelog")
        endif()
        if(EXISTS "${CMAKE_BINARY_DIR}/DEBIAN/CHANGELOG.md")
            execute_process(
                COMMAND
                    ${DEB_GZIP_EXEC} -f -n -9 "${CMAKE_BINARY_DIR}/DEBIAN/CHANGELOG.md"
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/DEBIAN"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE output
                ERROR_VARIABLE error
            )
            if(NOT ${result} EQUAL 0)
                message(FATAL_ERROR "Failed to compress: ${error}")
            endif()
            install(
                FILES "${CMAKE_BINARY_DIR}/DEBIAN/${DEB_CHANGELOG_INSTALL_FILENM}"
                DESTINATION ${LINTIAN_DOCS_DIR}
                COMPONENT ${COMPONENT_NAME_T}
            )
        endif()
    endif()
endfunction()

# Set variables for changelog and copyright
# For Debian specific Packages
function(
    set_debian_pkg_cmake_flags
    DEB_PACKAGE_NAME_T
    DEB_PACKAGE_VERSION_T
    DEB_MAINTAINER_NM_T
    DEB_MAINTAINER_EMAIL_T
)
    # Setting configure flags
    set(DEB_PACKAGE_NAME "${DEB_PACKAGE_NAME_T}" CACHE STRING "Debian Package Name")
    set(DEB_PACKAGE_VERSION
        "${DEB_PACKAGE_VERSION_T}"
        CACHE STRING
        "Debian Package Version String"
    )
    set(DEB_MAINTAINER_NAME
        "${DEB_MAINTAINER_NM_T}"
        CACHE STRING
        "Debian Package Maintainer Name"
    )
    set(DEB_MAINTAINER_EMAIL
        "${DEB_MAINTAINER_EMAIL_T}"
        CACHE STRING
        "Debian Package Maintainer Email"
    )
    set(DEB_LICENSE "MIT" CACHE STRING "Debian Package License Type")
    set(DEB_CHANGELOG_INSTALL_FILENM
        "CHANGELOG.md.gz"
        CACHE STRING
        "Debian Package ChangeLog File Name"
    )
    # BUILD_ENABLE_LINTIAN_OVERRIDES not supported

    # Get TimeStamp
    find_program(DEB_DATE_TIMESTAMP_EXEC date)
    if(NOT DEB_DATE_TIMESTAMP_EXEC)
        message(
            FATAL_ERROR
            "date command not found: Failed to Configure the timestamp for Copyright/Changelog."
        )
    endif()
    set(DEB_TIMESTAMP_FORMAT_OPTION "-R")
    execute_process(
        COMMAND ${DEB_DATE_TIMESTAMP_EXEC} ${DEB_TIMESTAMP_FORMAT_OPTION}
        OUTPUT_VARIABLE TIMESTAMP_T
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(DEB_TIMESTAMP
        "${TIMESTAMP_T}"
        CACHE STRING
        "Current Time Stamp for Copyright/Changelog"
    )

    # Get Copyright Year
    set(DEB_YEAR_FORMAT_OPTION "+%Y")
    execute_process(
        COMMAND ${DEB_DATE_TIMESTAMP_EXEC} ${DEB_YEAR_FORMAT_OPTION}
        OUTPUT_VARIABLE DEB_COPYRIGHT_YEAR_T
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(DEB_COPYRIGHT_YEAR
        "${DEB_COPYRIGHT_YEAR_T}"
        CACHE STRING
        "Debian Package Copyright Year"
    )

    message(STATUS "DEB_PACKAGE_NAME             : ${DEB_PACKAGE_NAME}")
    message(STATUS "DEB_PACKAGE_VERSION          : ${DEB_PACKAGE_VERSION}")
    message(STATUS "DEB_MAINTAINER_NAME          : ${DEB_MAINTAINER_NAME}")
    message(STATUS "DEB_MAINTAINER_EMAIL         : ${DEB_MAINTAINER_EMAIL}")
    message(STATUS "DEB_COPYRIGHT_YEAR           : ${DEB_COPYRIGHT_YEAR}")
    message(STATUS "DEB_LICENSE                  : ${DEB_LICENSE}")
    message(STATUS "DEB_TIMESTAMP                : ${DEB_TIMESTAMP}")
    message(STATUS "DEB_CHANGELOG_INSTALL_FILENM : ${DEB_CHANGELOG_INSTALL_FILENM}")
endfunction()
