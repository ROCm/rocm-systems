# Helper script: append SRC content to DST. Used by gen-prof-api-str-header
# because `cmake -E cat A B > C` is not portable to MSBuild/CMD.
file(READ "${SRC}" _content)
file(APPEND "${DST}" "${_content}")
