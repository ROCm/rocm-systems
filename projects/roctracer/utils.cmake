## Configure Copyright File for Debian Package
function( configure_pkg PACKAGE_NAME_T COMPONENT_NAME_T PACKAGE_VERSION_T MAINTAINER_NM_T MAINTAINER_EMAIL_T)
    # Check If Debian Platform
    find_file (DEBIAN debian_version debconf.conf PATHS /etc)
    if(DEBIAN)
      set( BUILD_DEBIAN_PKGING_FLAG ON CACHE BOOL "Internal Status Flag to indicate Debian Packaging Build" FORCE )
      set_debian_pkg_cmake_flags( ${PACKAGE_NAME_T} ${PACKAGE_VERSION_T}
                                  ${MAINTAINER_NM_T} ${MAINTAINER_EMAIL_T} )

      # Create debian directory in build tree
      file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/DEBIAN")

      # Configure the copyright file
      configure_file(
        "${CMAKE_SOURCE_DIR}/DEBIAN/copyright.in"
        "${CMAKE_BINARY_DIR}/DEBIAN/copyright"
        @ONLY
      )

      # Install copyright file
      install ( FILES "${CMAKE_BINARY_DIR}/DEBIAN/copyright"
	        DESTINATION "${CMAKE_INSTALL_DOCDIR}"
	        COMPONENT ${COMPONENT_NAME_T} )

      # Configure the changelog file
      configure_file(
        "${CMAKE_SOURCE_DIR}/DEBIAN/changelog.in"
        "${CMAKE_BINARY_DIR}/DEBIAN/changelog.Debian"
        @ONLY
      )
      if( BUILD_ENABLE_LINTIAN_OVERRIDES )
        if(BUILD_SHARED_LIBS STREQUAL "OFF")
          string(FIND ${DEB_OVERRIDES_INSTALL_FILENM} "static" OUT_VAR1)
          if(OUT_VAR1 EQUAL -1)
            set( DEB_OVERRIDES_INSTALL_FILENM "${DEB_OVERRIDES_INSTALL_FILENM}-static" )
          endif()
        else()
          if(ENABLE_ASAN_PACKAGING)
            string( FIND ${DEB_OVERRIDES_INSTALL_FILENM} "asan" OUT_VAR2)
            if(OUT_VAR2 EQUAL -1)
              set( DEB_OVERRIDES_INSTALL_FILENM "${DEB_OVERRIDES_INSTALL_FILENM}-asan" )
            endif()
          endif()
        endif()
        set( DEB_OVERRIDES_INSTALL_FILENM
          "${DEB_OVERRIDES_INSTALL_FILENM}" CACHE STRING "Debian Package Lintian Override File Name" FORCE)
        # Configure the changelog file
              configure_file(
                "${CMAKE_SOURCE_DIR}/DEBIAN/overrides.in"
                "${CMAKE_BINARY_DIR}/DEBIAN/${DEB_OVERRIDES_INSTALL_FILENM}"
                FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
                @ONLY
              )
      endif()

      # Install Change Log 
      find_program ( DEB_GZIP_EXEC gzip )
      if(EXISTS "${CMAKE_BINARY_DIR}/DEBIAN/changelog.Debian" )
        execute_process(
          COMMAND ${DEB_GZIP_EXEC} -f -n -9 "${CMAKE_BINARY_DIR}/DEBIAN/changelog.Debian"
          WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/DEBIAN"
          RESULT_VARIABLE result
          OUTPUT_VARIABLE output
          ERROR_VARIABLE error
        )
        if(NOT ${result} EQUAL 0)
          message(FATAL_ERROR "Failed to compress: ${error}")
        endif()
        install ( FILES "${CMAKE_BINARY_DIR}/DEBIAN/${DEB_CHANGELOG_INSTALL_FILENM}"
                  DESTINATION ${CMAKE_INSTALL_DOCDIR}
                  COMPONENT ${COMPONENT_NAME_T})
      endif()
    else()
        # License file
        install ( FILES ${LICENSE_FILE}
            DESTINATION ${CMAKE_INSTALL_DOCDIR} RENAME LICENSE.txt
            COMPONENT ${COMPONENT_NAME_T})
    endif()

       # Install lintian overrides
    if( BUILD_ENABLE_LINTIAN_OVERRIDES STREQUAL "ON" AND BUILD_DEBIAN_PKGING_FLAG STREQUAL "ON")
      set( OVERRIDE_FILE "${CMAKE_BINARY_DIR}/DEBIAN/${DEB_OVERRIDES_INSTALL_FILENM}" )
      install ( FILES ${OVERRIDE_FILE}
	  DESTINATION ${DEB_OVERRIDES_INSTALL_PATH}
          COMPONENT ${COMPONENT_NAME_T})

      message(STATUS "POTATO INSIDE OVERRIDE_FILE: ${OVERRIDE_FILE}")
      message(STATUS "POTATO INSIDE DEB_OVERRIDES_INSTALL_PATH: ${DEB_OVERRIDES_INSTALL_PATH}")
      message(STATUS "POTATO INSIDE COMPONENT_NAME_T: ${COMPONENT_NAME_T}")

    endif()
endfunction()

# Set variables for changelog and copyright
# For Debian specific Packages 
function( set_debian_pkg_cmake_flags DEB_PACKAGE_NAME_T DEB_PACKAGE_VERSION_T DEB_MAINTAINER_NM_T DEB_MAINTAINER_EMAIL_T )
    # Setting configure flags
    set( DEB_PACKAGE_NAME             "${DEB_PACKAGE_NAME_T}" CACHE STRING "Debian Package Name" )
    set( DEB_PACKAGE_VERSION          "${DEB_PACKAGE_VERSION_T}" CACHE STRING "Debian Package Version String" )
    set( DEB_MAINTAINER_NAME          "${DEB_MAINTAINER_NM_T}" CACHE STRING "Debian Package Maintainer Name" )
    set( DEB_MAINTAINER_EMAIL         "${DEB_MAINTAINER_EMAIL_T}" CACHE STRING "Debian Package Maintainer Email" )
    set( DEB_COPYRIGHT_YEAR           "2025" CACHE STRING "Debian Package Copyright Year" )
    set( DEB_LICENSE                  "MIT" CACHE STRING "Debian Package License Type" )
    set( DEB_CHANGELOG_INSTALL_FILENM "changelog.Debian.gz" CACHE STRING "Debian Package ChangeLog File Name" ) 

    if( BUILD_ENABLE_LINTIAN_OVERRIDES )
      set( DEB_OVERRIDES_INSTALL_FILENM "${DEB_PACKAGE_NAME}" CACHE STRING "Debian Package Lintian Override File Name" )
      set( DEB_OVERRIDES_INSTALL_PATH   "/usr/share/lintian/overrides/" CACHE STRING "Deb Pkg Lintian Override Install Loc" )
    endif()

    # Get TimeStamp
    find_program( DEB_DATE_TIMESTAMP_EXEC date )
    set ( DEB_TIMESTAMP_FORMAT_OPTION "-R" )
    execute_process (
        COMMAND ${DEB_DATE_TIMESTAMP_EXEC} ${DEB_TIMESTAMP_FORMAT_OPTION}
        OUTPUT_VARIABLE TIMESTAMP_T
    )
    set( DEB_TIMESTAMP                "${TIMESTAMP_T}" CACHE STRING "Current Time Stamp for Copyright/Changelog" )

    message(STATUS "DEB_PACKAGE_NAME             : ${DEB_PACKAGE_NAME}" )
    message(STATUS "DEB_PACKAGE_VERSION          : ${DEB_PACKAGE_VERSION}" )
    message(STATUS "DEB_MAINTAINER_NAME          : ${DEB_MAINTAINER_NAME}" )
    message(STATUS "DEB_MAINTAINER_EMAIL         : ${DEB_MAINTAINER_EMAIL}" )
    message(STATUS "DEB_COPYRIGHT_YEAR           : ${DEB_COPYRIGHT_YEAR}" )
    message(STATUS "DEB_LICENSE                  : ${DEB_LICENSE}" )
    message(STATUS "DEB_TIMESTAMP                : ${DEB_TIMESTAMP}" )
    message(STATUS "DEB_CHANGELOG_INSTALL_FILENM : ${DEB_CHANGELOG_INSTALL_FILENM}" )
endfunction()
