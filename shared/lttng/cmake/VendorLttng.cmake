# VendorLttng.cmake — build LTTng-UST + userspace-rcu from vendored
# submodules, install into a private build-tree prefix, and expose paths
# for downstream targets to link against.
#
# Canonical location: shared/lttng/cmake/. The submodules live under
# shared/lttng/{lttng-ust,userspace-rcu} and are shared by both consumers
# (rocr-runtime, clr/hipamd). Each consumer still produces its own
# private vendored build (see comment on urcu_vendored below).
# Both submodules are autotools-based and require autoreconf, autoconf,
# automake, libtool, and pkg-config to be available at configure time.
# See shared/lttng/README.md for consumer integration details.
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

# Auto-init the vendored submodules if they're missing AND we're inside a
# git working tree. TheRock CI (and similar aggregator builds) does
# selective submodule init that doesn't include our nested external/
# submodules; this block bridges that gap so the build "just works"
# regardless of how the repo was checked out. Skipped if no .git is found
# (e.g. tarball release) — the caller gets the same actionable error
# message as before.
function(_lttng_vendor_init_submodule abs_path display_name)
    # Submodule is "present" if it has the autotools bootstrap script.
    if(EXISTS "${abs_path}/bootstrap")
        return()
    endif()

    # Walk up from abs_path looking for a .git entry (file or directory).
    # That gives us the working-tree root that owns this submodule.
    set(_repo_root "${abs_path}")
    while(NOT _repo_root STREQUAL "/")
        get_filename_component(_repo_root "${_repo_root}" DIRECTORY)
        if(EXISTS "${_repo_root}/.git")
            break()
        endif()
    endwhile()

    if(NOT EXISTS "${_repo_root}/.git")
        message(FATAL_ERROR
            "Vendored ${display_name} submodule is not initialised at\n"
            "    ${abs_path}\n"
            "and we cannot auto-init because no .git was found walking up "
            "from that path (this looks like a tarball release).\n"
            "Please run: git submodule update --init --recursive ${abs_path}")
    endif()

    # Compute the submodule path relative to the discovered repo root —
    # that's the form `git submodule update` accepts.
    file(RELATIVE_PATH _rel "${_repo_root}" "${abs_path}")

    find_program(GIT_EXECUTABLE git REQUIRED)
    message(STATUS
        "Vendored ${display_name} submodule is empty; "
        "auto-initialising via git (repo root: ${_repo_root}, path: ${_rel}) ...")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive ${_rel}
        WORKING_DIRECTORY ${_repo_root}
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE  _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR
            "Failed to auto-init vendored ${display_name} submodule:\n"
            "  rc:     ${_rc}\n"
            "  stdout: ${_out}\n"
            "  stderr: ${_err}\n"
            "Please run manually from ${_repo_root}:\n"
            "  git submodule update --init --recursive ${_rel}")
    endif()
    if(NOT EXISTS "${abs_path}/bootstrap")
        message(FATAL_ERROR
            "Auto-init of ${display_name} reported success but "
            "${abs_path}/bootstrap is still missing. git output was:\n"
            "  stdout: ${_out}\n"
            "  stderr: ${_err}")
    endif()
endfunction()

_lttng_vendor_init_submodule("${LTTNG_VENDORED_URCU_SRC}" "userspace-rcu")
_lttng_vendor_init_submodule("${LTTNG_VENDORED_UST_SRC}"  "lttng-ust")

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
# Stub pkg-config files (generated at CMake CONFIGURE time)
# ---------------------------------------------------------------------
# pkg_check_modules() runs at CMake configure time. The real .pc files
# generated by autotools' `make install` only exist after the
# ExternalProject build step runs — too late. On a fresh tree this
# manifests as "Package 'lttng-ust' not found" even though auto-init
# successfully cloned the submodule.
#
# Fix: write stub .pc files now, pointing at the FUTURE vendored install
# paths. The .so files don't need to exist at configure time, only at
# link time; ExternalProject_Add ordering plus the explicit
# add_dependencies() at the consumer site guarantees the real libraries
# are produced before anything tries to link against them.
#
# When ExternalProject's `make install` later runs, it overwrites these
# stubs with the real .pc files. That's fine: by then pkg_check_modules
# has already done its job at configure time, and the only thing the
# build/link step cares about is the .so search path the stub already
# encoded correctly.
file(MAKE_DIRECTORY "${LTTNG_VENDORED_PKGCONFIG}")

# CMake's IMPORTED-target validation rejects INTERFACE_INCLUDE_DIRECTORIES
# entries whose path doesn't exist at generate time. The vendored
# include/ tree won't exist until ExternalProject runs, so create it now
# (empty); the real headers land here at build time.
file(MAKE_DIRECTORY "${LTTNG_VENDORED_INCLUDE_DIR}")

file(WRITE "${LTTNG_VENDORED_PKGCONFIG}/lttng-ust.pc"
"prefix=${LTTNG_VENDORED_PREFIX}
exec_prefix=\${prefix}
libdir=${_lttng_libdir}
includedir=\${prefix}/include

Name: LTTng-UST
Description: vendored LTTng-UST (built via ExternalProject from external/lttng-ust submodule)
Version: 2.13.7
Libs: -L\${libdir} -llttng-ust -llttng-ust-common
Cflags: -I\${includedir}
")

file(WRITE "${LTTNG_VENDORED_PKGCONFIG}/liburcu-bp.pc"
"prefix=${LTTNG_VENDORED_PREFIX}
exec_prefix=\${prefix}
libdir=${_lttng_libdir}
includedir=\${prefix}/include

Name: liburcu-bp
Description: vendored userspace-rcu (brand-preserving)
Version: 0.14.0
Libs: -L\${libdir} -lurcu-bp
Cflags: -I\${includedir}
")

# ---------------------------------------------------------------------
# Stage pkg.m4 for autotools bootstrap
# ---------------------------------------------------------------------
# lttng-ust's bootstrap calls autoreconf -> aclocal, which needs the
# PKG_CHECK_MODULES macro defined in pkg.m4 (shipped with pkg-config /
# pkgconf). RHEL-family manylinux containers (TheRock CI) ship the
# pkg-config binary but often NOT the m4 macro file (split into a
# separate pkgconf-m4 / pkg-config-devel package). Without pkg.m4
# aclocal silently leaves PKG_CHECK_MODULES([URCU], ...) unexpanded,
# producing a configure script with "syntax error near unexpected token
# URCU" at run time.
#
# Ship our own copy of pkg.m4 (verbatim from pkg-config 0.29.2, MIT
# licensed) into a per-build aclocal directory and prepend it to
# ACLOCAL_PATH when invoking bootstrap. Independent of the host env.
# urcu's configure.ac uses PKG_CHECK_MODULES too, so apply to both.
set(LTTNG_VENDORED_ACLOCAL_DIR "${CMAKE_BINARY_DIR}/_deps/lttng-aclocal"
    CACHE INTERNAL "Directory holding staged pkg.m4 for vendored bootstrap")
file(MAKE_DIRECTORY "${LTTNG_VENDORED_ACLOCAL_DIR}")
file(WRITE "${LTTNG_VENDORED_ACLOCAL_DIR}/pkg.m4" [=[
# pkg.m4 - Macros to locate and use pkg-config.   -*- Autoconf -*-
# serial 12 (pkg-config-0.29.2)

dnl Copyright © 2004 Scott James Remnant <scott@netsplit.com>.
dnl Copyright © 2012-2015 Dan Nicholson <dbn.lists@gmail.com>
dnl
dnl This program is free software; you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation; either version 2 of the License, or
dnl (at your option) any later version.
dnl
dnl This program is distributed in the hope that it will be useful, but
dnl WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
dnl General Public License for more details.
dnl
dnl You should have received a copy of the GNU General Public License
dnl along with this program; if not, write to the Free Software
dnl Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
dnl MA 02110-1301, USA.
dnl
dnl As a special exception to the GNU General Public License, if you
dnl distribute this file as part of a program that contains a
dnl configuration script generated by Autoconf, you may include it under
dnl the same distribution terms that you use for the rest of that
dnl program.

dnl PKG_PREREQ(MIN-VERSION)
dnl -----------------------
dnl Since: 0.29
dnl
dnl Verify that the version of the pkg-config macros are at least
dnl MIN-VERSION. Unlike PKG_PROG_PKG_CONFIG, which checks the user's
dnl installed version of pkg-config, this checks the developer's version
dnl of pkg.m4 when generating configure.
dnl
dnl To ensure that this macro is defined, also add:
dnl m4_ifndef([PKG_PREREQ],
dnl     [m4_fatal([must install pkg-config 0.29 or later before running autoconf/autogen])])
dnl
dnl See the "Since" comment for each macro you use to see what version
dnl of the macros you require.
m4_defun([PKG_PREREQ],
[m4_define([PKG_MACROS_VERSION], [0.29.2])
m4_if(m4_version_compare(PKG_MACROS_VERSION, [$1]), -1,
    [m4_fatal([pkg.m4 version $1 or higher is required but ]PKG_MACROS_VERSION[ found])])
])dnl PKG_PREREQ

dnl PKG_PROG_PKG_CONFIG([MIN-VERSION])
dnl ----------------------------------
dnl Since: 0.16
dnl
dnl Search for the pkg-config tool and set the PKG_CONFIG variable to
dnl first found in the path. Checks that the version of pkg-config found
dnl is at least MIN-VERSION. If MIN-VERSION is not specified, 0.9.0 is
dnl used.
AC_DEFUN([PKG_PROG_PKG_CONFIG],
[m4_pattern_forbid([^_?PKG_[A-Z_]+$])
m4_pattern_allow([^PKG_CONFIG(_(PATH|LIBDIR|SYSROOT_DIR|ALLOW_SYSTEM_(CFLAGS|LIBS)))?$])
m4_pattern_allow([^PKG_CONFIG_(DISABLE_UNINSTALLED|TOP_BUILD_DIR|DEBUG_SPEW)$])
AC_ARG_VAR([PKG_CONFIG], [path to pkg-config utility])
AC_ARG_VAR([PKG_CONFIG_PATH], [directories to add to pkg-config's search path])
AC_ARG_VAR([PKG_CONFIG_LIBDIR], [path overriding pkg-config's built-in search path])

if test "x$ac_cv_env_PKG_CONFIG_set" != "xset"; then
	AC_PATH_TOOL([PKG_CONFIG], [pkg-config])
fi
if test -n "$PKG_CONFIG"; then
	_pkg_min_version=m4_default([$1], [0.9.0])
	AC_MSG_CHECKING([pkg-config is at least version $_pkg_min_version])
	if $PKG_CONFIG --atleast-pkgconfig-version $_pkg_min_version; then
		AC_MSG_RESULT([yes])
	else
		AC_MSG_RESULT([no])
		PKG_CONFIG=""
	fi
fi[]dnl
])dnl PKG_PROG_PKG_CONFIG

dnl PKG_CHECK_EXISTS(MODULES, [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
dnl -------------------------------------------------------------------
dnl Since: 0.18
dnl
dnl Check to see whether a particular set of modules exists. Similar to
dnl PKG_CHECK_MODULES(), but does not set variables or print errors.
dnl
dnl Please remember that m4 expands AC_REQUIRE([PKG_PROG_PKG_CONFIG])
dnl only at the first occurence in configure.ac, so if the first place
dnl it's called might be skipped (such as if it is within an "if", you
dnl have to call PKG_CHECK_EXISTS manually
AC_DEFUN([PKG_CHECK_EXISTS],
[AC_REQUIRE([PKG_PROG_PKG_CONFIG])dnl
if test -n "$PKG_CONFIG" && \
    AC_RUN_LOG([$PKG_CONFIG --exists --print-errors "$1"]); then
  m4_default([$2], [:])
m4_ifvaln([$3], [else
  $3])dnl
fi])dnl PKG_CHECK_EXISTS

dnl _PKG_CONFIG([VARIABLE], [COMMAND], [MODULES])
dnl ---------------------------------------------
dnl Internal wrapper calling pkg-config via PKG_CONFIG and setting
dnl pkg_failed based on the result.
m4_define([_PKG_CONFIG],
[if test -n "$$1"; then
    pkg_cv_[]$1="$$1"
 elif test -n "$PKG_CONFIG"; then
    PKG_CHECK_EXISTS([$3],
                     [pkg_cv_[]$1=`$PKG_CONFIG --[]$2 "$3" 2>/dev/null`
		      test "x$?" != "x0" && pkg_failed=yes ],
		     [pkg_failed=yes])
 else
    pkg_failed=untried
fi[]dnl
])dnl _PKG_CONFIG

dnl _PKG_SHORT_ERRORS_SUPPORTED
dnl ---------------------------
dnl Internal check to see if pkg-config supports short errors.
AC_DEFUN([_PKG_SHORT_ERRORS_SUPPORTED],
[AC_REQUIRE([PKG_PROG_PKG_CONFIG])
if $PKG_CONFIG --atleast-pkgconfig-version 0.20; then
        _pkg_short_errors_supported=yes
else
        _pkg_short_errors_supported=no
fi[]dnl
])dnl _PKG_SHORT_ERRORS_SUPPORTED


dnl PKG_CHECK_MODULES(VARIABLE-PREFIX, MODULES, [ACTION-IF-FOUND],
dnl   [ACTION-IF-NOT-FOUND])
dnl --------------------------------------------------------------
dnl Since: 0.4.0
dnl
dnl Note that if there is a possibility the first call to
dnl PKG_CHECK_MODULES might not happen, you should be sure to include an
dnl explicit call to PKG_PROG_PKG_CONFIG in your configure.ac
AC_DEFUN([PKG_CHECK_MODULES],
[AC_REQUIRE([PKG_PROG_PKG_CONFIG])dnl
AC_ARG_VAR([$1][_CFLAGS], [C compiler flags for $1, overriding pkg-config])dnl
AC_ARG_VAR([$1][_LIBS], [linker flags for $1, overriding pkg-config])dnl

pkg_failed=no
AC_MSG_CHECKING([for $1])

_PKG_CONFIG([$1][_CFLAGS], [cflags], [$2])
_PKG_CONFIG([$1][_LIBS], [libs], [$2])

m4_define([_PKG_TEXT], [Alternatively, you may set the environment variables $1[]_CFLAGS
and $1[]_LIBS to avoid the need to call pkg-config.
See the pkg-config man page for more details.])

if test $pkg_failed = yes; then
   	AC_MSG_RESULT([no])
        _PKG_SHORT_ERRORS_SUPPORTED
        if test $_pkg_short_errors_supported = yes; then
	        $1[]_PKG_ERRORS=`$PKG_CONFIG --short-errors --print-errors --cflags --libs "$2" 2>&1`
        else
	        $1[]_PKG_ERRORS=`$PKG_CONFIG --print-errors --cflags --libs "$2" 2>&1`
        fi
	# Put the nasty error message in config.log where it belongs
	echo "$$1[]_PKG_ERRORS" >&AS_MESSAGE_LOG_FD

	m4_default([$4], [AC_MSG_ERROR(
[Package requirements ($2) were not met:

$$1_PKG_ERRORS

Consider adjusting the PKG_CONFIG_PATH environment variable if you
installed software in a non-standard prefix.

_PKG_TEXT])dnl
        ])
elif test $pkg_failed = untried; then
     	AC_MSG_RESULT([no])
	m4_default([$4], [AC_MSG_FAILURE(
[The pkg-config script could not be found or is too old.  Make sure it
is in your PATH or set the PKG_CONFIG environment variable to the full
path to pkg-config.

$_PKG_TEXT

To find pkg-config, see <http://pkgconfig.freedesktop.org/>.])dnl
        ])
else
	$1[]_CFLAGS=$pkg_cv_[]$1[]_CFLAGS
	$1[]_LIBS=$pkg_cv_[]$1[]_LIBS
        AC_MSG_RESULT([yes])
	$3
fi[]dnl
])dnl PKG_CHECK_MODULES


dnl PKG_CHECK_MODULES_STATIC(VARIABLE-PREFIX, MODULES, [ACTION-IF-FOUND],
dnl   [ACTION-IF-NOT-FOUND])
dnl ---------------------------------------------------------------------
dnl Since: 0.29
dnl
dnl Checks for existence of MODULES and gathers its build flags with
dnl static libraries enabled. Sets VARIABLE-PREFIX_CFLAGS from --cflags
dnl and VARIABLE-PREFIX_LIBS from --libs.
AC_DEFUN([PKG_CHECK_MODULES_STATIC],
[AC_REQUIRE([PKG_PROG_PKG_CONFIG])dnl
_save_PKG_CONFIG=$PKG_CONFIG
PKG_CONFIG="$PKG_CONFIG --static"
PKG_CHECK_MODULES($@)
PKG_CONFIG=$_save_PKG_CONFIG[]dnl
])dnl PKG_CHECK_MODULES_STATIC


dnl PKG_INSTALLDIR([DIRECTORY])
dnl -------------------------
dnl Since: 0.27
dnl
dnl Substitutes the variable pkgconfigdir as the location where a module
dnl should install pkg-config .pc files. By default the directory is
dnl $libdir/pkgconfig, but the default can be changed by passing
dnl DIRECTORY. The user can override through the --with-pkgconfigdir
dnl parameter.
AC_DEFUN([PKG_INSTALLDIR],
[m4_pushdef([pkg_default], [m4_default([$1], ['${libdir}/pkgconfig'])])
m4_pushdef([pkg_description],
    [pkg-config installation directory @<:@]pkg_default[@:>@])
AC_ARG_WITH([pkgconfigdir],
    [AS_HELP_STRING([--with-pkgconfigdir], pkg_description)],,
    [with_pkgconfigdir=]pkg_default)
AC_SUBST([pkgconfigdir], [$with_pkgconfigdir])
m4_popdef([pkg_default])
m4_popdef([pkg_description])
])dnl PKG_INSTALLDIR


dnl PKG_NOARCH_INSTALLDIR([DIRECTORY])
dnl -------------------------
dnl Since: 0.27
dnl
dnl Substitutes the variable noarch_pkgconfigdir as the location where a
dnl module should install arch-independent pkg-config .pc files. By
dnl default the directory is $datadir/pkgconfig, but the default can be
dnl changed by passing DIRECTORY. The user can override through the
dnl --with-noarch-pkgconfigdir parameter.
AC_DEFUN([PKG_NOARCH_INSTALLDIR],
[m4_pushdef([pkg_default], [m4_default([$1], ['${datadir}/pkgconfig'])])
m4_pushdef([pkg_description],
    [arch-independent pkg-config installation directory @<:@]pkg_default[@:>@])
AC_ARG_WITH([noarch-pkgconfigdir],
    [AS_HELP_STRING([--with-noarch-pkgconfigdir], pkg_description)],,
    [with_noarch_pkgconfigdir=]pkg_default)
AC_SUBST([noarch_pkgconfigdir], [$with_noarch_pkgconfigdir])
m4_popdef([pkg_default])
m4_popdef([pkg_description])
])dnl PKG_NOARCH_INSTALLDIR


dnl PKG_CHECK_VAR(VARIABLE, MODULE, CONFIG-VARIABLE,
dnl [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
dnl -------------------------------------------
dnl Since: 0.28
dnl
dnl Retrieves the value of the pkg-config variable for the given module.
AC_DEFUN([PKG_CHECK_VAR],
[AC_REQUIRE([PKG_PROG_PKG_CONFIG])dnl
AC_ARG_VAR([$1], [value of $3 for $2, overriding pkg-config])dnl

_PKG_CONFIG([$1], [variable="][$3]["], [$2])
AS_VAR_COPY([$1], [pkg_cv_][$1])

AS_VAR_IF([$1], [""], [$5], [$4])dnl
])dnl PKG_CHECK_VAR
]=])

# ACLOCAL_PATH passed to bootstrap. Prepend our staged dir so it shadows
# any older system pkg.m4; if the host has no ACLOCAL_PATH set, treat it
# as empty rather than letting the literal "$ENV{ACLOCAL_PATH}" leak
# through (CMake expands $ENV{} eagerly so this is safe — empty if unset).
set(_lttng_aclocal_path "${LTTNG_VENDORED_ACLOCAL_DIR}:$ENV{ACLOCAL_PATH}")

# ---------------------------------------------------------------------
# userspace-rcu
# ---------------------------------------------------------------------
ExternalProject_Add(urcu_vendored
    PREFIX            "${CMAKE_BINARY_DIR}/_deps/urcu-build"
    SOURCE_DIR        "${LTTNG_VENDORED_URCU_SRC}"
    BUILD_IN_SOURCE   1
    # The source dir is a git submodule shared across multiple build trees
    # (e.g. TheRock builds rocr-runtime / clr twice — once nested in an
    # amd-llvm runtimes ExternalProject and once as a standalone packaging
    # sub-project — and both invoke this vendored build with different
    # --prefix paths). Without cleaning, the .la files left behind by build
    # #1 carry build #1's libdir baked in, and libtool refuses to install to
    # build #2's libdir with "cannot install to a directory not ending in
    # <build-1 libdir>". Run `make distclean` (best-effort: a fresh checkout
    # has no Makefile) to wipe stale generated state before bootstrap +
    # configure each invocation.
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env
                      "ACLOCAL_PATH=${_lttng_aclocal_path}"
                      sh -c "(make distclean >/dev/null 2>&1 || true) && ./bootstrap && ./configure \
                          --prefix=${LTTNG_VENDORED_PREFIX} \
                          --libdir=${_lttng_libdir} \
                          --disable-static \
                          --enable-shared"
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
# top-level package install rule copies them to /opt/rocm/lib/.
# bootstrap + configure run under one shell so ACLOCAL_PATH applies to
# bootstrap and the build flags apply to configure.
ExternalProject_Add(lttng_ust_vendored
    DEPENDS           urcu_vendored
    PREFIX            "${CMAKE_BINARY_DIR}/_deps/lttng-ust-build"
    SOURCE_DIR        "${LTTNG_VENDORED_UST_SRC}"
    BUILD_IN_SOURCE   1
    # See urcu_vendored above: distclean wipes any stale build state from a
    # prior build with a different --libdir so libtool's install can succeed.
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env
                      "ACLOCAL_PATH=${_lttng_aclocal_path}"
                      "PKG_CONFIG_PATH=${LTTNG_VENDORED_PKGCONFIG}"
                      "CPPFLAGS=-I${LTTNG_VENDORED_INCLUDE_DIR}"
                      "LDFLAGS=-L${_lttng_libdir} -Wl,-rpath,${_lttng_libdir}"
                      sh -c "(make distclean >/dev/null 2>&1 || true) && ./bootstrap && ./configure \
                          --prefix=${LTTNG_VENDORED_PREFIX} \
                          --libdir=${_lttng_libdir} \
                          --disable-man-pages \
                          --disable-static \
                          --enable-shared \
                          --disable-numa \
                          --disable-examples"
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
# is installed to /opt/rocm/lib/. Rewrite each .so's RPATH to
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
