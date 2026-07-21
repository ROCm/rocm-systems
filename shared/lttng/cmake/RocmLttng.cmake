# Copyright (c) Advanced Micro Devices, Inc.
#
# RocmLttng.cmake -- locate LTTng-UST for the ROCm curated tracepoint providers.
#
# LTTng-UST (and its userspace-rcu dependency) are NOT vendored in this repo.
# They are provided one of two ways and discovered via pkg-config:
#
#   1. TheRock build: configure TheRock with -DTHEROCK_ENABLE_LTTNG=ON, which
#      builds the lttng-ust + userspace-rcu "sysdeps" and puts their pkg-config
#      metadata on CMAKE_PREFIX_PATH / PKG_CONFIG_PATH for this sub-project.
#   2. Standalone build: a system LTTng-UST development install
#      (e.g. Debian/Ubuntu 'liblttng-ust-dev', RHEL 'lttng-ust-devel').
#
# On success this defines the imported target PkgConfig::LTTNG_UST, which the
# HIP/HSA tracepoint providers link against. Consumers should gate the include
# of this module on their own option (e.g. ROCR_ENABLE_LTTNG_UST /
# HIP_ENABLE_LTTNG_UST); to build without LTTng, leave that option OFF and do
# not include this module.
#
# The minimum version tracks the schema/API the providers are written against.

set(ROCM_LTTNG_UST_MIN_VERSION "2.13")

if(NOT COMMAND pkg_check_modules)
    find_package(PkgConfig REQUIRED)
endif()

pkg_check_modules(LTTNG_UST QUIET IMPORTED_TARGET
    "lttng-ust>=${ROCM_LTTNG_UST_MIN_VERSION}")

if(NOT LTTNG_UST_FOUND)
    message(FATAL_ERROR
        "LTTng-UST (>=${ROCM_LTTNG_UST_MIN_VERSION}) was not found via "
        "pkg-config, but LTTng tracepoints are enabled.\n"
        "  * TheRock build: reconfigure TheRock with -DTHEROCK_ENABLE_LTTNG=ON "
        "so the lttng-ust sysdep is built and exposed.\n"
        "  * Standalone build: install the LTTng-UST development package "
        "(Debian/Ubuntu: 'apt-get install liblttng-ust-dev', "
        "RHEL: 'dnf install lttng-ust-devel').\n"
        "  * Or disable LTTng tracepoints for this project "
        "(-DROCR_ENABLE_LTTNG_UST=OFF or -DHIP_ENABLE_LTTNG_UST=OFF).")
endif()

message(STATUS
    "Found LTTng-UST ${LTTNG_UST_VERSION} (target PkgConfig::LTTNG_UST)")
