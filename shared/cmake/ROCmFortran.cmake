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
    check_language(Fortran)
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
