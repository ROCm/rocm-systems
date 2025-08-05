#
#
#

include_guard(GLOBAL)

# function for copying most of a target's properties to another target except for the link
# related properties. Function is potentially recursive -- it should not be used if there
# is a cyclic target dependency.
function(rocprofiler_config_nolink_target _DST _SRC)
    # skip if not a cmake target but process any extra args
    if(NOT TARGET "${_SRC}")
        foreach(_LIB ${ARGN})
            rocprofiler_config_nolink_target(${_DST} ${_LIB})
        endforeach()
        return()
    endif()

    set(_LINK_LIBRARIES)
    set(_INCLUDE_DIRS)
    set(_COMPILE_DEFS)
    set(_COMPILE_OPTS)
    set(_COMPILE_FEATS)

    set(_PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
    foreach(_PROPERTY INCLUDE_DIRECTORIES LINK_LIBRARIES COMPILE_DEFINITIONS
                      COMPILE_OPTIONS COMPILE_FEATURES)
        list(APPEND _PROPERTIES ${_PROPERTY} INTERFACE_${_PROPERTY})
    endforeach()

    foreach(_PROPERTY ${_PROPERTIES})
        # get the target property
        get_target_property(_VAR ${_SRC} ${_PROPERTY})

        if(NOT _VAR)
            continue()
        endif()

        if("${_PROPERTY}" MATCHES ".*LINK_LIBRARIES$")
            list(APPEND _LINK_LIBRARIES ${_VAR})
        elseif("${_PROPERTY}" MATCHES ".*INCLUDE_DIRECTORIES$")
            list(APPEND _INCLUDE_DIRS ${_VAR})
        elseif("${_PROPERTY}" MATCHES ".*COMPILE_DEFINITIONS$")
            list(APPEND _COMPILE_DEFS ${_VAR})
        elseif("${_PROPERTY}" MATCHES ".*COMPILE_OPTIONS$")
            list(APPEND _COMPILE_OPTS ${_VAR})
        elseif("${_PROPERTY}" MATCHES ".*COMPILE_FEATURES$")
            list(APPEND _COMPILE_FEATS ${_VAR})
        else()
            message(SEND_ERROR "Unexpected target property: ${_PROPERTY}")
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _LINK_LIBRARIES)
    list(REMOVE_DUPLICATES _INCLUDE_DIRS)
    list(REMOVE_DUPLICATES _COMPILE_DEFS)
    list(REMOVE_DUPLICATES _COMPILE_OPTS)
    list(REMOVE_DUPLICATES _COMPILE_FEATS)

    target_include_directories(${_DST} SYSTEM INTERFACE ${_INCLUDE_DIRS})
    target_compile_definitions(${_DST} INTERFACE ${_COMPILE_DEFS})
    target_compile_options(${_DST} INTERFACE ${_COMPILE_OPTS})
    target_compile_features(${_DST} INTERFACE ${_COMPILE_FEATS})

    foreach(_LIB ${_LINK_LIBRARIES} ${ARGN})
        rocprofiler_config_nolink_target(${_DST} ${_LIB})
    endforeach()
endfunction()
