# Copy a Python script from IN to OUT, replacing its first line (the shebang)
# with "#!${INTERP}". Used to pin the amd-smi CLI to a concrete interpreter on
# RPM distros; see amdsmi_cli/CMakeLists.txt and py-interface/CMakeLists.txt.
#
# Invoked as:
#   cmake -DIN=<src> -DOUT=<dst> -DINTERP=/path/to/python -P pin_shebang.cmake
# INTERP is the interpreter path only (no "#!"); the "#!" is prepended here to
# avoid passing a literal "#" on the cmake command line, which is parsed as a
# comment/argument boundary on some shells and breaks -P.

if(NOT DEFINED IN OR NOT DEFINED OUT OR NOT DEFINED INTERP)
    message(FATAL_ERROR "pin_shebang.cmake requires -DIN, -DOUT and -DINTERP")
endif()

file(READ "${IN}" _contents)
# Drop the original first line (the shebang) and prepend the pinned one. The
# source always starts with `#!...`, so strip up to and including the first
# newline, then re-add our shebang. NOTE: a regex like "^[^\n]*\n" does NOT work
# here -- CMake's regex engine does not treat \n inside a bracket expression as a
# newline, so [^\n] matches every character and the whole body is deleted. Use a
# literal newline search + substring instead.
string(FIND "${_contents}" "\n" _first_nl)
if(_first_nl LESS 0)
    message(FATAL_ERROR "pin_shebang.cmake: ${IN} has no newline / no shebang line")
endif()
math(EXPR _after_nl "${_first_nl} + 1")
string(SUBSTRING "${_contents}" ${_after_nl} -1 _body)
file(WRITE "${OUT}" "#!${INTERP}\n${_body}")
