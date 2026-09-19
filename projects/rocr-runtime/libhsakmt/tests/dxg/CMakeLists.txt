# Tests for the DXG (WSL) thunk.
#
# The export check needs neither a GPU nor an open thunk, so it runs anywhere
# the library can be built. The heap-reservation test does need a DXCore
# adapter; it reports itself skipped when there is none, so a build on a
# machine without one still gets a clean ctest run.

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

# Regression test for ROCM-30547: the system heap's VA reservation had no
# fallback, so a refused reservation left system_heap_mgr_ null while init
# still reported success, and the next allocation that needed the pool
# dereferenced it.
#
# Linked against the library target the way tests/kfdtest and rocrtst link
# theirs, so CMake points the build-tree rpath at the artifact this build just
# produced - no path argument and no LD_LIBRARY_PATH needed to run it.
add_executable(dxg_heap_reservation_test heap_reservation_test.cpp)

target_include_directories(
    dxg_heap_reservation_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(dxg_heap_reservation_test PRIVATE ${ROCDXG_TARGET})

add_test(NAME dxg-heap-reservation COMMAND dxg_heap_reservation_test)

# Forks several children and lowers RLIMIT_AS in each, which is slower than the
# export check and depends on an adapter being present.
set_tests_properties(
    dxg-heap-reservation
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300
)
