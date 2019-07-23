#!/usr/bin/python
import os, sys, re

OUT_H = 'inc/kfd_prof_str.h'
OUT_C = "src/kfd/kfd_wrapper.cpp"
API_HEADER = "hsakmt.h"

content_h = \
  '#ifndef KFD_PROF_STR_H_\n' + \
  '#define KFD_PROF_STR_H_\n' + \
  '#endif  \\\\ KFD_PROF_STR_H_\n'

content_c = 'namespace kfd { void fun() {}; }  // namespace kfd\n'

#############################################################
# main
# Usage
if len(sys.argv) != 3:
  print >>sys.stderr, "Usage:", sys.argv[0], " <rocTracer root> <KFD include path>"
  sys.exit(1)
else:
  ROOT = sys.argv[1] + '/'
  KFD_DIR = sys.argv[2] + '/'


out_h_file = ROOT + OUT_H
out_c_file = ROOT + OUT_C
print 'Generating: "' + out_h_file + '", ' + out_c_file + '"'
f = open(out_h_file, 'w')
f.write(content_h)
f.close()
f = open(out_c_file, 'w')
f.write(content_c)
f.close()
#############################################################
