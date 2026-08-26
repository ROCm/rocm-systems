# Pure-logic tests for the DXG (WSL) thunk. No GPU, DXCore adapter or open
# thunk is involved, so these run on any machine that can build the library.
#
# libhsakmt has no test framework; ../../../tests/unit_test_harness.h is the
# whole harness. These targets are build-only artifacts and are deliberately
# not installed.

add_executable(dxg-snapshot-refcount-test snapshot_refcount_test.cpp)
target_include_directories(
    dxg-snapshot-refcount-test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests
            ${CMAKE_CURRENT_SOURCE_DIR}/../../src/dxg
)
set_target_properties(
    dxg-snapshot-refcount-test
    PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
)

add_test(NAME dxg-snapshot-refcount COMMAND dxg-snapshot-refcount-test)

# What the built artifact's dynamic symbol table actually offers. Nothing else
# here links or loads librocdxg - the refcount test compiles a header on its
# own - so this is the only test that speaks for what ROCr's dlsym() will
# find. A symbol can be lost to mangling or to local visibility while the
# header and the version script both still read correctly, and nothing at link
# time would notice, because ROCr binds every one of these by name at run time.
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
