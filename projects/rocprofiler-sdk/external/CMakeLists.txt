#
#
#   External dependencies
#
#
include(rocprofiler_utilities)

cmake_policy(PUSH)
cmake_policy(SET CMP0077 NEW) # option() honors normal variables

set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}][external] ")

set(BUILD_TESTING OFF)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_OBJECT_LIBS OFF) # Specific to PTL
set(BUILD_STATIC_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# filesystem library
if(ROCPROFILER_BUILD_GHC_FS)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/filesystem
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/gulrak/filesystem.git
        REPO_BRANCH "v1.5.14")

    target_compile_definitions(
        rocprofiler-sdk-cxx-filesystem
        INTERFACE $<BUILD_INTERFACE:ROCPROFILER_HAS_GHC_LIB_FILESYSTEM=1>)
    target_include_directories(
        rocprofiler-sdk-cxx-filesystem SYSTEM
        INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/filesystem/include>)
endif()

if(ROCPROFILER_BUILD_TESTS)
    if(ROCPROFILER_BUILD_GTEST)
        # checkout submodule if not already checked out or clone repo if no .gitmodules
        # file
        rocprofiler_checkout_git_submodule(
            RECURSIVE
            RELATIVE_PATH external/googletest
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            REPO_URL https://github.com/google/googletest.git
            REPO_BRANCH "main")

        set(BUILD_GMOCK OFF)
        set(INSTALL_GTEST OFF)
        add_subdirectory(googletest EXCLUDE_FROM_ALL)

        if(NOT TARGET GTest::gtest)
            message(FATAL_ERROR "missing GTest::gtest")
        endif()

        target_link_libraries(rocprofiler-sdk-gtest INTERFACE GTest::gtest)
        target_include_directories(
            rocprofiler-sdk-gtest SYSTEM
            INTERFACE ${CMAKE_CURRENT_LIST_DIR}/googletest/googletest/include)

        mark_as_advanced(INSTALL_GTEST)
        mark_as_advanced(BUILD_GMOCK)
        mark_as_advanced(GTEST_HAS_ABSL)
    else()
        find_package(GTest REQUIRED)
        target_link_libraries(rocprofiler-sdk-gtest INTERFACE GTest::gtest)
    endif()
endif()

if(ROCPROFILER_BUILD_GLOG)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/glog
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/google/glog.git
        REPO_BRANCH "master")

    # May want to use GFLAGS in the future
    set(WITH_GFLAGS OFF)
    set(WITH_GTEST OFF)
    set(WITH_UNWIND "none")
    add_subdirectory(glog EXCLUDE_FROM_ALL)

    target_link_libraries(rocprofiler-sdk-glog INTERFACE $<BUILD_INTERFACE:glog::glog>)
    target_include_directories(
        rocprofiler-sdk-glog SYSTEM
        INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/glog>
                  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/glog/src>)
else()
    find_package(glog REQUIRED)
    target_link_libraries(rocprofiler-sdk-glog INTERFACE glog::glog)
endif()

if(ROCPROFILER_BUILD_FMT)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/fmt
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/fmtlib/fmt.git
        REPO_BRANCH "master")

    set(FMT_TEST OFF)
    add_subdirectory(fmt EXCLUDE_FROM_ALL)

    target_link_libraries(rocprofiler-sdk-fmt INTERFACE $<BUILD_INTERFACE:fmt::fmt>)
    target_include_directories(
        rocprofiler-sdk-fmt SYSTEM
        INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/fmt/include>)
else()
    find_package(fmt REQUIRED)
    target_link_libraries(rocprofiler-sdk-fmt INTERFACE fmt::fmt)
endif()

if(NOT TARGET PTL::ptl-static)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_checkout_git_submodule(
        RELATIVE_PATH external/ptl
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/PTL.git
        REPO_BRANCH rocprofiler)

    set(PTL_USE_TBB OFF)
    set(PTL_USE_GPU OFF)
    set(PTL_USE_LOCKS ON)
    set(PTL_BUILD_EXAMPLES OFF)
    set(PTL_DEVELOPER_INSTALL OFF)
    add_subdirectory(ptl EXCLUDE_FROM_ALL)
endif()

rocprofiler_checkout_git_submodule(
    RECURSIVE
    RELATIVE_PATH external/yaml-cpp
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://github.com/jbeder/yaml-cpp.git
    REPO_BRANCH "master")

add_subdirectory(yaml-cpp EXCLUDE_FROM_ALL)

target_link_libraries(rocprofiler-sdk-yaml-cpp
                      INTERFACE $<BUILD_INTERFACE:yaml-cpp::yaml-cpp>)
target_include_directories(
    rocprofiler-sdk-yaml-cpp
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/yaml-cpp/include>)

# checkout submodule if not already checked out or clone repo if no .gitmodules file
rocprofiler_checkout_git_submodule(
    RECURSIVE
    RELATIVE_PATH external/cereal
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://github.com/jrmadsen/cereal.git
    REPO_BRANCH "rocprofiler")

target_compile_definitions(rocprofiler-sdk-cereal
                           INTERFACE $<BUILD_INTERFACE:CEREAL_THREAD_SAFE=1>)
target_include_directories(
    rocprofiler-sdk-cereal
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cereal/include>)

# doxygen-awesome
if(ROCPROFILER_BUILD_DOCS)
    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/doxygen-awesome-css
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_FILE Makefile
        REPO_URL https://github.com/jothepro/doxygen-awesome-css.git
        REPO_BRANCH "main")
endif()

# perfetto
rocprofiler_checkout_git_submodule(
    RECURSIVE
    RELATIVE_PATH external/perfetto
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    TEST_FILE meson.build
    REPO_URL https://github.com/google/perfetto
    REPO_BRANCH "v44.0")

add_library(rocprofiler-sdk-perfetto-static-library STATIC)
target_sources(
    rocprofiler-sdk-perfetto-static-library
    PRIVATE ${PROJECT_SOURCE_DIR}/external/perfetto/sdk/perfetto.h
            ${PROJECT_SOURCE_DIR}/external/perfetto/sdk/perfetto.cc)
target_include_directories(
    rocprofiler-sdk-perfetto-static-library SYSTEM
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/perfetto/sdk>)
set_target_properties(
    rocprofiler-sdk-perfetto-static-library
    PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME rocprofiler-sdk-perfetto)
target_link_libraries(
    rocprofiler-sdk-perfetto
    INTERFACE $<BUILD_INTERFACE:rocprofiler-sdk-perfetto-static-library>)

# ELFIO
rocprofiler_checkout_git_submodule(
    RECURSIVE
    RELATIVE_PATH external/elfio
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://github.com/serge1/ELFIO.git
    REPO_BRANCH "Release_3.12")

set(ELFIO_BUILD_EXAMPLES OFF)
set(ELFIO_BUILD_TESTS OFF)
add_subdirectory(elfio EXCLUDE_FROM_ALL)
if(TARGET rocprofiler-sdk-elfio)
    get_target_property(ELFIO_INCLUDE_DIR elfio::elfio INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(rocprofiler-sdk-elfio SYSTEM
                               INTERFACE ${ELFIO_INCLUDE_DIR})
endif()

#
# OTF2
#
add_subdirectory(otf2)

rocprofiler_checkout_git_submodule(
    RECURSIVE
    RELATIVE_PATH external/json
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://github.com/nlohmann/json.git
    REPO_BRANCH "develop")

#
# JSON
#
add_subdirectory(json)
target_include_directories(
    rocprofiler-sdk-json SYSTEM
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/json/include>)

#
# SQLite3
#
if(ROCPROFILER_BUILD_SQLITE3)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/sqlite
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_FILE configure
        REPO_URL https://github.com/sqlite/sqlite
        REPO_BRANCH "version-3.47.0")

    find_program(
        MAKE_COMMAND
        NAMES make gmake
        PATH_SUFFIXES bin REQUIRED)

    include(ExternalProject)
    externalproject_add(
        rocprofiler-sdk-sqlite-build
        PREFIX ${PROJECT_BINARY_DIR}/external/sqlite/build
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/sqlite
        BUILD_IN_SOURCE 0
        CONFIGURE_COMMAND
            <SOURCE_DIR>/configure --prefix=${PROJECT_BINARY_DIR}/external/sqlite/install
            --libdir=${PROJECT_BINARY_DIR}/external/sqlite/install/lib
            --enable-tempstore=yes --enable-all --disable-tcl --with-pic CFLAGS=-O3\ -g1
        BUILD_COMMAND ${MAKE_COMMAND} install -s
        INSTALL_COMMAND "")

    target_link_libraries(
        rocprofiler-sdk-sqlite3
        INTERFACE
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/sqlite/install/lib/libsqlite3.so>
        )
    target_include_directories(
        rocprofiler-sdk-sqlite3 SYSTEM
        INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/sqlite/install/include>
        )
    add_dependencies(rocprofiler-sdk-sqlite3 rocprofiler-sdk-sqlite-build)
else()
    find_package(SQLite3 REQUIRED)
    target_link_libraries(rocprofiler-sdk-sqlite3 INTERFACE SQLite::SQLite3)
endif()

#
# PyBind11
#
if(ROCPROFILER_BUILD_PYBIND11)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/pybind11
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/pybind/pybind11.git
        REPO_BRANCH "v2.9.2")

    target_include_directories(
        rocprofiler-sdk-pybind11 SYSTEM
        INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/pybind11/include>)
else()
    find_package(pybind11 REQUIRED)
    target_link_libraries(rocprofiler-sdk-pybind11 INTERFACE pybind11::pybind11)
endif()

#
# GOTCHA
#
if(ROCPROFILER_BUILD_GOTCHA)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/gotcha
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/GOTCHA.git
        REPO_BRANCH "rocprofiler")

    function(add_gotcha_subdirectory)
        set(GOTCHA_ENABLE_TESTS OFF)
        set(GOTCHA_ENABLE_WARNING_ERROR OFF)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofiler-sdk")
        add_subdirectory(gotcha EXCLUDE_FROM_ALL)
        add_dependencies(rocprofiler-sdk-gotcha gotcha)
        install(
            TARGETS gotcha
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk"
            OPTIONAL)
    endfunction()

    add_gotcha_subdirectory()

    target_link_libraries(rocprofiler-sdk-gotcha INTERFACE gotcha::gotcha)
else()
    find_package(gotcha REQUIRED)
    if(TARGET gotcha::gotcha)
        target_link_libraries(rocprofiler-sdk-gotcha INTERFACE gotcha::gotcha)
    elseif(TARGET gotcha)
        target_link_libraries(rocprofiler-sdk-gotcha INTERFACE gotcha)
    endif()
endif()

find_package(re2 QUIET)
if(re2_FOUND AND NOT ROCPROFILER_BUILD_RE2)
    message(STATUS "Found re2: ${re2_INCLUDE_DIRS}")
    target_link_libraries(rocprofiler-sdk-re2 INTERFACE re2::re2)
else()
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/abseil-cpp
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/abseil/abseil-cpp.git
        REPO_BRANCH "lts_2025_05_12")

    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/re2
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/google/re2.git
        REPO_BRANCH "2025-07-22")

    function(add_re2_subdirectory)
        set(ABSL_PROPAGATE_CXX_STD ON)
        set(ABSL_ENABLE_INSTALL ON)
        add_subdirectory(abseil-cpp EXCLUDE_FROM_ALL)
        add_subdirectory(re2 EXCLUDE_FROM_ALL)
        add_dependencies(rocprofiler-sdk-re2 re2::re2)
        install(
            TARGETS re2
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk"
            OPTIONAL)
    endfunction()

    add_re2_subdirectory()

    target_link_libraries(rocprofiler-sdk-re2 INTERFACE $<BUILD_INTERFACE:re2::re2>)
endif()

cmake_policy(POP)
