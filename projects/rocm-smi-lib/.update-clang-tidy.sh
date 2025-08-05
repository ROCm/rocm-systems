#!/usr/bin/env bash

set -x # trace
set -e # exit immediately if command fails
set -u # exit if an undefined variable is found

awk '
BEGIN {
  print "# THIS FILE IS GENERATED FROM .clangd!"
  print "# Run ./.update-clang-tidy.sh to regenerate."
  print "Checks:"
}
/Add: \[$/{
a=1
  next
}
/]/{
  a=0
}
a{
gsub(/^\s+/,"  ")
 print
}

/Remove: \[$/{
r=1
  next
}
/]/{
  r=0
}
r{
  gsub(/^\s+/,"  -")
  print
}
' .clangd | tee .clang-tidy
