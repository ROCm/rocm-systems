# VendorLttngRpath.cmake — post-install hook for the vendored LTTng-UST
# + userspace-rcu ExternalProjects.
#
# autotools/libtool unconditionally embeds --libdir as the RPATH of
# every installed .so. That path is the in-tree build prefix
# (<build>/_deps/lttng-prefix/lib), which will not exist at runtime
# when the libraries are repackaged under /opt/rocm/lib/.
#
# This script walks every .so in the vendored libdir and rewrites the
# RPATH to $ORIGIN so the loader resolves transitive vendored deps
# (e.g. liblttng-ust.so.1 -> liblttng-ust-common.so.1) relative to the
# library's own directory.
#
# Required input variables (passed via cmake -D):
#   LIB_DIR  — absolute path of the vendored install lib directory
#   PATCHELF — absolute path of the patchelf executable

if(NOT DEFINED LIB_DIR)
    message(FATAL_ERROR "VendorLttngRpath.cmake: LIB_DIR is required")
endif()
if(NOT DEFINED PATCHELF)
    message(FATAL_ERROR "VendorLttngRpath.cmake: PATCHELF is required")
endif()

file(GLOB _so_files
    "${LIB_DIR}/liblttng*.so*"
    "${LIB_DIR}/liburcu*.so*")

foreach(_so ${_so_files})
    # Only patch real ELF files (skip libtool .la, broken symlinks, etc.).
    if(IS_SYMLINK "${_so}")
        continue()
    endif()
    if(NOT _so MATCHES "\\.so($|\\.[0-9])")
        continue()
    endif()

    execute_process(
        COMMAND ${PATCHELF} --set-rpath "\$ORIGIN" "${_so}"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE  _err)
    if(NOT _rc EQUAL 0)
        message(WARNING
            "patchelf failed on ${_so}: rc=${_rc}\nstdout: ${_out}\nstderr: ${_err}")
    endif()
endforeach()
