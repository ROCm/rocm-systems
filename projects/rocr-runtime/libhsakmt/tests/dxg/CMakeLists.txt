# Build-time checks for the DXG (WSL) thunk. No GPU, DXCore adapter or open
# thunk is involved, so this runs on any machine that can build the library.

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
