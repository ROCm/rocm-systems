# This module provides common functions used for building
# and packaging ROCm projects

option(CMAKE_VERBOSE_MAKEFILE "Enable verbose output" ON)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile commands for linters and autocompleters" ON)

function(generic_add_rocm)
    set(ROCM_DIR
        "/opt/rocm"
        CACHE STRING "ROCm directory.")
    if(DEFINED ENV{ROCM_RPATH} AND NOT DEFINED LIB_RUNPATH)
        set(LIB_RUNPATH "\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../lib64" PARENT_SCOPE)
    endif()

    set(CMAKE_INSTALL_PREFIX ${ROCM_DIR} CACHE STRING "Default installation directory.")
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE STRING "Default packaging prefix.")
    # add package search paths
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /usr/local PARENT_SCOPE)
    set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/lib64 /usr/lib/x86_64-linux-gnu PARENT_SCOPE)
endfunction()

function(generic_package)
    # Used by test and example CMakeLists
    set(SHARE_INSTALL_PREFIX "share/${CMAKE_PROJECT_NAME}" CACHE STRING "Tests and Example install directory")

    if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4.0)
        message("Compiler version is " ${CMAKE_CXX_COMPILER_VERSION})
        message(FATAL_ERROR "Require at least gcc-5.4.0")
    endif()

    if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2" PARENT_SCOPE)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG" PARENT_SCOPE)
    endif()

    # Add address sanitizer
    # derived from:
    # https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime/blob/e176056061bf11fdd98b58dd57deb4ac5625844d/amdocl/CMakeLists.txt#L27
    if(${ADDRESS_SANITIZER})
        set(ASAN_COMPILER_FLAGS "-fno-omit-frame-pointer -fsanitize=address")
        set(ASAN_LINKER_FLAGS "-fsanitize=address")

        if(BUILD_SHARED_LIBS)
            set(ASAN_COMPILER_FLAGS "${ASAN_COMPILER_FLAGS} -shared-libsan")
            set(ASAN_LINKER_FLAGS "${ASAN_LINKER_FLAGS} -shared-libsan")
        else()
            set(ASAN_LINKER_FLAGS "${ASAN_LINKER_FLAGS} -static-libsan")
        endif()

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_COMPILER_FLAGS}" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_COMPILER_FLAGS}" PARENT_SCOPE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ASAN_LINKER_FLAGS}" PARENT_SCOPE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ASAN_LINKER_FLAGS}" PARENT_SCOPE)
    else()
        ## Security breach mitigation flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFORTIFY_SOURCE=2 -fstack-protector-all -Wcast-align" PARENT_SCOPE)
        ## More security breach mitigation flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-z,noexecstack -Wl,-znoexecheap -Wl,-z,relro" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wtrampolines -Wl,-z,now -fPIE" PARENT_SCOPE)
    endif()

    # Clang does not set the build-id
    # similar to if(NOT CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--build-id=sha1" PARENT_SCOPE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=sha1" PARENT_SCOPE)
    endif()

    # configure packaging
    # cpack version is populated with CMAKE_PROJECT_VERSION implicitly
    set(CPACK_PACKAGE_NAME
        ${CMAKE_PROJECT_NAME}
        CACHE STRING "")
    set(CPACK_PACKAGE_VENDOR
        "Advanced Micro Devices, Inc."
        CACHE STRING "")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
        "Placeholder Tool"
        CACHE STRING "")
    set(CPACK_PACKAGE_DESCRIPTION
        "This package contains the AMD ${CPACK_PACKAGE_DESCRIPTION_SUMMARY}."
        CACHE STRING "")
    set(CPACK_PACKAGING_INSTALL_PREFIX
        "${CMAKE_INSTALL_PREFIX}"
        CACHE STRING "Default packaging prefix.")
    set(CPACK_RESOURCE_FILE_LICENSE
        "${CMAKE_CURRENT_SOURCE_DIR}/License.txt"
        CACHE STRING "")
    set(CPACK_RPM_PACKAGE_LICENSE
        "MIT"
        CACHE STRING "")
    set(CPACK_GENERATOR
        "DEB;RPM"
        CACHE STRING "Default packaging generators.")
    set(CPACK_DEB_COMPONENT_INSTALL ON PARENT_SCOPE)
    set(CPACK_RPM_COMPONENT_INSTALL ON PARENT_SCOPE)
    mark_as_advanced(CPACK_PACKAGE_NAME CPACK_PACKAGE_VENDOR CPACK_PACKAGE_CONTACT CPACK_PACKAGE_DESCRIPTION_SUMMARY
        CPACK_PACKAGE_DESCRIPTION CPACK_RESOURCE_FILE_LICENSE CPACK_RPM_PACKAGE_LICENSE CPACK_GENERATOR)

    # Debian package specific variables
    if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
        set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE} PARENT_SCOPE)
    else()
        set(CPACK_DEBIAN_PACKAGE_RELEASE "local" PARENT_SCOPE)
    endif()
    message("Using CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE}")
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT" PARENT_SCOPE)

    # RPM package specific variables
    if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
        set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE} PARENT_SCOPE)
    else()
        set(CPACK_RPM_PACKAGE_RELEASE "local" PARENT_SCOPE)
    endif()
    message("Using CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE}")
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT" PARENT_SCOPE)
    set(CPACK_RPM_PACKAGE_AUTOREQ 0 PARENT_SCOPE)
    set(CPACK_RPM_PACKAGE_AUTOPROV 0 PARENT_SCOPE)
    list(
        APPEND
        CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
        "/lib"
        "/usr/sbin"
        "/lib/systemd"
        "/lib/systemd/system"
        "/usr"
        "/opt")

    # PACKAGE-tests need PACKAGE
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "${CPACK_PACKAGE_NAME}" PARENT_SCOPE)
    set(CPACK_RPM_TESTS_PACKAGE_REQUIRES "${CPACK_PACKAGE_NAME}" PARENT_SCOPE)

    # Treat runtime group as package base.
    # Without it - the base package would be named 'rdc-runtime'
    # resulting in rdc-runtime*.deb and rdc-runtime*.rpm
    set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "${CPACK_PACKAGE_NAME}" PARENT_SCOPE)
    set(CPACK_RPM_RUNTIME_PACKAGE_NAME "${CPACK_PACKAGE_NAME}" PARENT_SCOPE)
endfunction()

# this function goes after 'include(CPack)'
function(generic_package_post)
    # PACKAGE package, no postfix
    cpack_add_component_group("runtime")
    cpack_add_component(dev GROUP runtime)
    cpack_add_component(unspecified GROUP runtime)

    # PACKAGE-tests package, -tests postfix
    cpack_add_component_group("tests")
    cpack_add_component(tests GROUP tests)
endfunction()
