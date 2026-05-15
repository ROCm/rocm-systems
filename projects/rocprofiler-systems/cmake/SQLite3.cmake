include_guard(GLOBAL)

if(ROCPROFSYS_BUILD_SQLITE3)
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

    set(SQLITE_BUILD_DIR ${PROJECT_BINARY_DIR}/external/sqlite/build)
    set(SQLITE_INSTALL_DIR ${PROJECT_BINARY_DIR}/external/sqlite/install)
    set(SQLITE_LIB ${SQLITE_INSTALL_DIR}/lib/libsqlite3.a)

    add_custom_command(
        OUTPUT ${SQLITE_LIB}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SQLITE_BUILD_DIR}
        COMMAND
            ${CMAKE_COMMAND} -E chdir ${SQLITE_BUILD_DIR}
            ${PROJECT_SOURCE_DIR}/external/sqlite/configure --prefix=${SQLITE_INSTALL_DIR}
            --libdir=${SQLITE_INSTALL_DIR}/lib --disable-shared --with-tempstore=yes
            --enable-all --disable-tcl
            CFLAGS=-O3\ -g1\ -fPIC\ -DSQLITE_DEFAULT_MEMSTATUS=0
        COMMAND ${CMAKE_COMMAND} -E chdir ${SQLITE_BUILD_DIR} ${MAKE_COMMAND} install -s
        COMMENT "Building SQLite3 from source"
        VERBATIM
    )

    add_custom_target(rocprofiler-systems-sqlite-build DEPENDS ${SQLITE_LIB})

    target_link_libraries(
        rocprofiler-systems-sqlite3
        INTERFACE $<BUILD_INTERFACE:${SQLITE_LIB}>
    )
    target_include_directories(
        rocprofiler-systems-sqlite3
        SYSTEM
        INTERFACE $<BUILD_INTERFACE:${SQLITE_INSTALL_DIR}/include>
    )
    add_dependencies(rocprofiler-systems-sqlite3 rocprofiler-systems-sqlite-build)
else()
    message(STATUS "Using system SQLite3 library")
    find_package(SQLite3 REQUIRED)
    target_link_libraries(rocprofiler-systems-sqlite3 INTERFACE SQLite::SQLite3)
endif()
