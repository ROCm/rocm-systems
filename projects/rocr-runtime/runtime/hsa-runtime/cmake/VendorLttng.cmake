# VendorLttng.cmake — build LTTng-UST + userspace-rcu from vendored
# submodules, install into a private build-tree prefix, and expose paths
# for downstream targets to link against.
#
# The submodules live under <project>/external/{lttng-ust,userspace-rcu}.
# Both are autotools-based and require autoreconf, autoconf, automake,
# libtool, and pkg-config to be available at configure time.
#
# Outputs (cached, visible to all subsequent CMake code):
#   LTTNG_VENDORED_PREFIX      — install prefix in the build tree
#   LTTNG_VENDORED_INCLUDE_DIR — header include path
#   LTTNG_VENDORED_LIB_DIR     — runtime .so directory
#   LTTNG_VENDORED_PKGCONFIG   — directory containing the .pc files
#
# Targets:
#   urcu_vendored      — userspace-rcu install
#   lttng_ust_vendored — lttng-ust install (depends on urcu_vendored)
#
# Callers should `add_dependencies(<target> lttng_ust_vendored)` to
# guarantee the vendored libs are built before the consumer is linked.
#
# Required input variables (set by includer before include()):
#   LTTNG_VENDORED_URCU_SRC — absolute path of the userspace-rcu submodule
#   LTTNG_VENDORED_UST_SRC  — absolute path of the lttng-ust submodule

include_guard(GLOBAL)
include(ExternalProject)

if(NOT DEFINED LTTNG_VENDORED_URCU_SRC)
    message(FATAL_ERROR "VendorLttng.cmake: LTTNG_VENDORED_URCU_SRC must be set "
                        "to the absolute path of the userspace-rcu submodule.")
endif()
if(NOT DEFINED LTTNG_VENDORED_UST_SRC)
    message(FATAL_ERROR "VendorLttng.cmake: LTTNG_VENDORED_UST_SRC must be set "
                        "to the absolute path of the lttng-ust submodule.")
endif()

# Verify the submodules are actually checked out (a freshly-cloned repo
# without `git submodule update --init` will have empty directories).
if(NOT EXISTS "${LTTNG_VENDORED_URCU_SRC}/bootstrap")
    message(FATAL_ERROR
        "Vendored userspace-rcu submodule is not initialised at\n"
        "    ${LTTNG_VENDORED_URCU_SRC}\n"
        "Run: git submodule update --init --recursive")
endif()
if(NOT EXISTS "${LTTNG_VENDORED_UST_SRC}/bootstrap")
    message(FATAL_ERROR
        "Vendored lttng-ust submodule is not initialised at\n"
        "    ${LTTNG_VENDORED_UST_SRC}\n"
        "Run: git submodule update --init --recursive")
endif()

set(LTTNG_VENDORED_PREFIX      "${CMAKE_BINARY_DIR}/_deps/lttng-prefix"
    CACHE INTERNAL "Vendored LTTng install prefix")
set(LTTNG_VENDORED_INCLUDE_DIR "${LTTNG_VENDORED_PREFIX}/include"
    CACHE INTERNAL "Vendored LTTng include directory")
set(LTTNG_VENDORED_LIB_DIR     "${LTTNG_VENDORED_PREFIX}/lib"
    CACHE INTERNAL "Vendored LTTng lib directory")
set(LTTNG_VENDORED_PKGCONFIG   "${LTTNG_VENDORED_PREFIX}/lib/pkgconfig"
    CACHE INTERNAL "Vendored LTTng pkg-config directory")

# Parallel jobs for the autotools sub-builds.
if(NOT DEFINED LTTNG_VENDORED_PARALLEL)
    if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        set(LTTNG_VENDORED_PARALLEL $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    else()
        include(ProcessorCount)
        ProcessorCount(_lttng_nproc)
        if(_lttng_nproc EQUAL 0)
            set(LTTNG_VENDORED_PARALLEL 4)
        else()
            set(LTTNG_VENDORED_PARALLEL ${_lttng_nproc})
        endif()
    endif()
endif()

# Force --libdir to a known path: by default autotools picks lib64 on
# 64-bit RPM-style distros, lib on Debian-style. We always want lib so
# the install rules below find the .so files at a single deterministic
# location.
set(_lttng_libdir "${LTTNG_VENDORED_LIB_DIR}")

# ---------------------------------------------------------------------
# userspace-rcu
# ---------------------------------------------------------------------
ExternalProject_Add(urcu_vendored
    PREFIX            "${CMAKE_BINARY_DIR}/_deps/urcu-build"
    SOURCE_DIR        "${LTTNG_VENDORED_URCU_SRC}"
    BUILD_IN_SOURCE   1
    CONFIGURE_COMMAND ./bootstrap
              COMMAND ./configure
                      --prefix=${LTTNG_VENDORED_PREFIX}
                      --libdir=${_lttng_libdir}
                      --disable-static
                      --enable-shared
    BUILD_COMMAND     make -j${LTTNG_VENDORED_PARALLEL}
    INSTALL_COMMAND   make install
    BUILD_BYPRODUCTS  "${_lttng_libdir}/liburcu.so"
                      "${_lttng_libdir}/liburcu-bp.so"
                      "${_lttng_libdir}/liburcu-cds.so"
                      "${LTTNG_VENDORED_PKGCONFIG}/liburcu.pc")

# ---------------------------------------------------------------------
# lttng-ust  (depends on urcu)
# ---------------------------------------------------------------------
# PKG_CONFIG_PATH steers configure to the vendored urcu .pc files.
# CPPFLAGS/LDFLAGS provide explicit fallbacks for any sub-component that
# bypasses pkg-config. The -Wl,-rpath embeds the vendored prefix into
# the temporary build artefacts so internal test programs link cleanly;
# the final installed .so files have RPATH stripped/replaced when the
# top-level package install rule copies them to /opt/rocm/lib/lttng/.
ExternalProject_Add(lttng_ust_vendored
    DEPENDS           urcu_vendored
    PREFIX            "${CMAKE_BINARY_DIR}/_deps/lttng-ust-build"
    SOURCE_DIR        "${LTTNG_VENDORED_UST_SRC}"
    BUILD_IN_SOURCE   1
    CONFIGURE_COMMAND ./bootstrap
              COMMAND ${CMAKE_COMMAND} -E env
                      "PKG_CONFIG_PATH=${LTTNG_VENDORED_PKGCONFIG}"
                      "CPPFLAGS=-I${LTTNG_VENDORED_INCLUDE_DIR}"
                      "LDFLAGS=-L${_lttng_libdir} -Wl,-rpath,${_lttng_libdir}"
                      ./configure
                      --prefix=${LTTNG_VENDORED_PREFIX}
                      --libdir=${_lttng_libdir}
                      --disable-man-pages
                      --disable-static
                      --enable-shared
                      --disable-numa
                      --disable-examples
    BUILD_COMMAND     make -j${LTTNG_VENDORED_PARALLEL}
    INSTALL_COMMAND   make install
    BUILD_BYPRODUCTS  "${_lttng_libdir}/liblttng-ust.so"
                      "${_lttng_libdir}/liblttng-ust-common.so"
                      "${_lttng_libdir}/liblttng-ust-tracepoint.so"
                      "${LTTNG_VENDORED_PKGCONFIG}/lttng-ust.pc")

# ---------------------------------------------------------------------
# Post-install RPATH rewrite
# ---------------------------------------------------------------------
# autotools' libtool unconditionally bakes the build-tree --libdir as
# RPATH on every installed .so. That path won't exist when the package
# is installed to /opt/rocm/lib/lttng/. Rewrite each .so's RPATH to
# $ORIGIN so the loader resolves intra-vendored references via the
# installed directory itself.
find_program(LTTNG_VENDORED_PATCHELF patchelf)
if(NOT LTTNG_VENDORED_PATCHELF)
    message(FATAL_ERROR
        "patchelf is required to vendor LTTng-UST: the libtool-relinked "
        "shared libraries have a build-tree RPATH baked in, and that "
        "path will not exist at runtime once the package is installed. "
        "patchelf rewrites the RPATH to $ORIGIN. Install patchelf and "
        "re-run cmake (Debian/Ubuntu: apt-get install patchelf).")
endif()

# Wrap the install step with a CMake script that, after `make install`,
# walks every .so and sets RPATH to $ORIGIN. We use ExternalProject's
# step mechanism so this is part of the normal build dependency graph
# and re-runs whenever the install step re-runs.
ExternalProject_Add_Step(lttng_ust_vendored rpath_rewrite
    COMMAND ${CMAKE_COMMAND}
            -DLIB_DIR=${_lttng_libdir}
            -DPATCHELF=${LTTNG_VENDORED_PATCHELF}
            -P ${CMAKE_CURRENT_LIST_DIR}/VendorLttngRpath.cmake
    DEPENDEES install
    COMMENT   "Rewriting RPATH on vendored lttng-ust + urcu .so files to \$ORIGIN")
