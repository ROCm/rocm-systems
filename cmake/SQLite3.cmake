include_guard(GLOBAL)

if(ROCPROFILER_BUILD_SQLITE3)
    message(STATUS "Building SQLite3 from source!")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/external/sqlite
    )
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/sqlite
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_FILE configure
        REPO_URL https://github.com/sqlite/sqlite.git
        REPO_BRANCH "version-3.45.3"
    )

    find_program(MAKE_COMMAND NAMES make gmake PATH_SUFFIXES bin REQUIRED)

    include(ExternalProject)
    ExternalProject_Add(
        rocprofiler-systems-sqlite-build
        PREFIX ${PROJECT_BINARY_DIR}/external/sqlite/build
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/sqlite
        BUILD_IN_SOURCE 0
        CONFIGURE_COMMAND
            <SOURCE_DIR>/configure --prefix=${PROJECT_BINARY_DIR}/external/sqlite/install
            --libdir=${PROJECT_BINARY_DIR}/external/sqlite/install/lib --disable-shared
            --with-tempstore=yes --enable-all --disable-tcl CFLAGS=-O3\ -g1
        BUILD_COMMAND ${MAKE_COMMAND} install -s
        INSTALL_COMMAND ""
    )

    target_link_libraries(
        rocprofiler-systems-sqlite3
        INTERFACE
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/sqlite/install/lib/libsqlite3.a>
    )
    target_include_directories(
        rocprofiler-systems-sqlite3
        SYSTEM
        INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/sqlite/install/include>
    )
    add_dependencies(rocprofiler-systems-sqlite3 rocprofiler-systems-sqlite-build)
else()
    message(STATUS "Using system SQLite3 library")
    find_package(SQLite3 REQUIRED)
    target_link_libraries(rocprofiler-systems-sqlite3 INTERFACE SQLite::SQLite3)
endif()
