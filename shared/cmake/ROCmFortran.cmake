# Answers one question, once per build tree: is a Fortran compiler available?
#
# Every ROCm project that ships Fortran bindings has to know, because the bindings
# are built when a Fortran compiler is present and skipped when one is not, a
# C-only site must never be forced to acquire a Fortran compiler. The answer is
# published as ROCM_HAVE_FORTRAN.
#
# rocm-libraries answers this at its monorepo root, which every project below it
# inherits. rocm-systems has no root CMakeLists to do the same: each project here
# is configured on its own, by TheRock or by packaging. So the probe lives in a
# shared module that each interested project includes, following the same
# ROCM_SYSTEMS_ROOT + include(... OPTIONAL) convention already used for
# shared/ctest/TestCategories.cmake:
#
#   if(NOT DEFINED ROCM_SYSTEMS_ROOT)
#     get_filename_component(ROCM_SYSTEMS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
#   endif()
#   include("${ROCM_SYSTEMS_ROOT}/shared/cmake/ROCmFortran.cmake" OPTIONAL)
#
# OPTIONAL matters: a project extracted from the monorepo and built on its own
# will not find this file, and must degrade to its own probe rather than fail.
#
# The name is deliberately neither repository's. A library from one monorepo is
# regularly built inside the other (rocm-libraries consumes HIP
# from rocm-systems), and a repo-specific name would have each side reading a
# variable the other never sets. ROCM_HAVE_FORTRAN is the name to settle on, and
# nothing should grow a second permanent spelling for the same fact.
#
# It is not the name in use everywhere yet. rocm-libraries publishes
# ROCM_LIBS_HAVE_FORTRAN from its root today, so its bindings read that as a
# second choice, after ROCM_HAVE_FORTRAN and never over it. That is a bridge with
# an end: when that root is switched over, the fallback in those files is dead
# code and goes. Saying so here, rather than asserting a migration that has not
# happened, is the difference between a plan and a claim.
#
# Nothing in this file is specific to rocm-systems; rocm-libraries can adopt it
# unchanged, which is what would retire the bridge.

# Already answered, by the monorepo root, by TheRock, or by an earlier include
# in this same build, so do not probe again and do not override it.
if(NOT DEFINED ROCM_HAVE_FORTRAN)

  # check_language() configures and compiles a throwaway project, which is cheap
  # once and wasteful N times. A global property carries the answer across the
  # directory scopes of one build tree, so N projects including this file pay for
  # the probe once. It cannot be a plain variable: include() runs in the caller's
  # scope, and sibling directories do not see each other's variables.
  get_property(_rocm_fortran_probed GLOBAL PROPERTY ROCM_FORTRAN_PROBED)

  if(_rocm_fortran_probed)
    get_property(ROCM_HAVE_FORTRAN GLOBAL PROPERTY ROCM_FORTRAN_AVAILABLE)
  else()
    include(CheckLanguage)

    # check_language() answers by writing the CACHE entry CMAKE_Fortran_COMPILER,
    # and rocm-cmake's ROCMChecks arms a variable_watch on exactly that name to
    # catch a project setting a toolchain variable behind the toolchain file's
    # back. A compiler probe is the one legitimate writer that watch cannot tell
    # apart from a violation, so it reports this one: a stray twelve-line banner
    # by default, and a hard configure failure under
    # -DROCM_ERROR_TOOLCHAIN_VAR=ON (or ROCMCHECKS_ERROR_TOOLCHAIN_VAR in the
    # environment), which is a switch CI turns on.
    #
    # So silence the checker across the probe and put it back afterwards, the
    # same save/set/restore rocRAND uses in cmake/Dependencies.cmake. Plain
    # variables rather than cache ones: they shadow the cache for this scope
    # only, which is what the watch callback reads, and the user's cache entries
    # are left exactly as they were. Both names have to be cleared, because the
    # callback escalates to SEND_ERROR on ROCM_ERROR_TOOLCHAIN_VAR before it
    # ever looks at the warn flag.
    set(_rocm_warn_toolchain_var "${ROCM_WARN_TOOLCHAIN_VAR}")
    set(_rocm_error_toolchain_var "${ROCM_ERROR_TOOLCHAIN_VAR}")
    set(ROCM_WARN_TOOLCHAIN_VAR OFF)
    set(ROCM_ERROR_TOOLCHAIN_VAR OFF)

    # Prefer ROCm's own amdflang over whatever CMake would otherwise find first.
    # check_language(Fortran) takes the first compiler on the search path, which
    # on a ROCm CI image is the system gfortran. That is how the bindings became
    # the one component of ROCm built by a different toolchain from everything
    # else, installed under fortran/gfortran/ while the documentation promises
    # fortran/amdflang/.
    #
    # amdflang sits beside the clang already compiling the project that included
    # this module, so the answer is next to CMAKE_CXX_COMPILER when there is one;
    # ROCM_PATH covers a configure that has enabled no C or C++. NO_DEFAULT_PATH
    # so a stray amdflang elsewhere on PATH cannot win over the ROCm at hand.
    #
    # An explicit CMAKE_Fortran_COMPILER, or FC in the environment, always wins:
    # building the bindings with gfortran is supported, and a .mod is
    # compiler-specific, so whoever named a compiler meant it.
    #
    # check_language() is a no-op once CMAKE_Fortran_COMPILER is defined, so this
    # runs before it and the probe never overrides the choice made here.
    if(NOT CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
      set(_rocm_fc_hints)
      foreach(_rocm_fc_cc "${CMAKE_CXX_COMPILER}" "${CMAKE_C_COMPILER}"
                          "${CMAKE_HIP_COMPILER}")
        if(_rocm_fc_cc)
          get_filename_component(_rocm_fc_dir "${_rocm_fc_cc}" DIRECTORY)
          list(APPEND _rocm_fc_hints "${_rocm_fc_dir}")
        endif()
      endforeach()
      if(ROCM_PATH)
        list(APPEND _rocm_fc_hints "${ROCM_PATH}/lib/llvm/bin" "${ROCM_PATH}/bin")
      endif()
      if(_rocm_fc_hints)
        find_program(_rocm_amdflang NAMES amdflang HINTS ${_rocm_fc_hints}
                     NO_DEFAULT_PATH)
        # Confirm it runs before committing to it. A ROCm tree can carry a
        # dangling amdflang symlink, and check_language() would not catch that:
        # it does nothing at all once the variable is set, so the failure would
        # surface much later as a broken enable_language, with no hint of where
        # the choice was made.
        if(_rocm_amdflang)
          execute_process(COMMAND "${_rocm_amdflang}" --version
                          RESULT_VARIABLE _rocm_fc_rc
                          OUTPUT_QUIET ERROR_QUIET)
          if(_rocm_fc_rc EQUAL 0)
            set(CMAKE_Fortran_COMPILER "${_rocm_amdflang}" CACHE FILEPATH
                "Fortran compiler used for the ROCm Fortran bindings" FORCE)
          endif()
          unset(_rocm_fc_rc)
        endif()
      endif()
      unset(_rocm_fc_hints)
      unset(_rocm_fc_cc)
      unset(_rocm_fc_dir)
    endif()

    check_language(Fortran)
    set(ROCM_WARN_TOOLCHAIN_VAR "${_rocm_warn_toolchain_var}")
    set(ROCM_ERROR_TOOLCHAIN_VAR "${_rocm_error_toolchain_var}")
    unset(_rocm_warn_toolchain_var)
    unset(_rocm_error_toolchain_var)

    if(CMAKE_Fortran_COMPILER)
      set(ROCM_HAVE_FORTRAN TRUE)
    else()
      set(ROCM_HAVE_FORTRAN FALSE)
      message(STATUS
        "ROCm: no Fortran compiler found - the Fortran bindings will be skipped")
    endif()
    set_property(GLOBAL PROPERTY ROCM_FORTRAN_PROBED TRUE)
    set_property(GLOBAL PROPERTY ROCM_FORTRAN_AVAILABLE "${ROCM_HAVE_FORTRAN}")
  endif()

  unset(_rocm_fortran_probed)

endif()

# No return() anywhere above, deliberately. A return() inside a file processed by
# include() returns from the INCLUDING scope before CMP0140, so it would silently
# truncate the caller's CMakeLists rather than just end this module.
