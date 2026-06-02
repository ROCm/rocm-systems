################################################################################
##
## Shared Debian package version suffix for rocdxg and amdsmi.
##
## Example: -DPACKAGE_VERSION_SUFFIX=115  -> 1.1.2~115
##          -DPACKAGE_VERSION_SUFFIX=rc2  -> 1.1.2~rc2
##
################################################################################

set(PACKAGE_VERSION_SUFFIX "" CACHE STRING
    "Optional suffix for Debian package version (e.g. 115, rc2). Empty for release.")

function(apply_package_version_suffix BASE_VERSION OUT_VAR)
    set(_version "${BASE_VERSION}")
    if(NOT "${PACKAGE_VERSION_SUFFIX}" STREQUAL "")
        string(TOLOWER "${PACKAGE_VERSION_SUFFIX}" _suffix_lc)
        set(_version "${_version}~${_suffix_lc}")
        message(STATUS "Debian package version: ${_version}")
    endif()
    set(${OUT_VAR} "${_version}" PARENT_SCOPE)
endfunction()
