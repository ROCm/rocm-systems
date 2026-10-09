# Tests for the DXG (WSL) thunk.
#
# The export check needs no GPU, DXCore adapter or open thunk, so it runs
# anywhere the library can be built. The lifecycle test needs all three and
# skips without them. Runtime-level tests that need a live adapter live in
# rocrtst, which already has the WSL detection to gate them
# (rocrtstFunc.Heap_Reservation).

# What the built artifact's dynamic symbol table actually offers. A symbol can
# be lost to mangling or to local visibility while the header and the version
# script both still read correctly, and nothing at link time would notice,
# because ROCr binds every one of these by name at run time.
#
# The names are given here rather than parsed out of the version script, so
# deleting an entry there fails here rather than quietly narrowing the
# five-call contract shared by ROCr and rocprofiler-sdk.
#
# Requires nm, which a cross-build might not have.
if(CMAKE_NM)
    add_test(
        NAME dxg-dynamic-exports
        COMMAND
            ${CMAKE_COMMAND} -DLIBRARY=$<TARGET_FILE:${ROCDXG_TARGET}>
            -DNM=${CMAKE_NM} -DEXPECTED_VERSION=HSAKMT_1
            -DEXPECTED_SYMBOLS=hsaKmtOpenKFD,hsaKmtCloseKFD,hsaKmtAcquireSystemProperties,hsaKmtReleaseSystemProperties,hsaKmtGetNodeProperties
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/check_dynamic_exports.cmake
    )
else()
    message(
        STATUS
        "librocdxg tests: no nm found, skipping the dynamic-export test. "
        "Set CMAKE_NM to run it."
    )
endif()

# The open and snapshot reference counts behind that contract, driven through
# the built library. Unlike the check above this opens the thunk, so it needs
# /dev/dxg and libdxcore and reports itself skipped anywhere without them. It
# forks, so it is POSIX only.
if(UNIX)
    add_executable(dxg_lifecycle_test lifecycle_test.cpp)
    find_package(Threads REQUIRED)
    target_link_libraries(
        dxg_lifecycle_test
        PRIVATE ${ROCDXG_TARGET} Threads::Threads
    )
    target_compile_features(dxg_lifecycle_test PRIVATE cxx_std_17)
    add_test(NAME dxg-lifecycle COMMAND dxg_lifecycle_test)
    set_tests_properties(dxg-lifecycle PROPERTIES SKIP_RETURN_CODE 77)

    # A libdxcore.so the loader finds first and then rejects, for the open that
    # fails after it has opened /dev/dxg.
    add_library(dxg_lifecycle_dxcore_stub SHARED dxcore_stub.c)
    set_target_properties(
        dxg_lifecycle_dxcore_stub
        PROPERTIES
            OUTPUT_NAME dxcore
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dxcore_stub
    )
    add_test(
        NAME dxg-lifecycle-partial-open
        COMMAND dxg_lifecycle_test partial-open
    )
    set_tests_properties(
        dxg-lifecycle-partial-open
        PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT
                "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:dxg_lifecycle_dxcore_stub>"
    )
endif()
