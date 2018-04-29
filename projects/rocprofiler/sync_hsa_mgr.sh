#!/bin/sh -x
BDIR=`dirname $0`
for name in hsa_rsrc_factory.h hsa_rsrc_factory.cpp ; do
  cat $BDIR/src/util/$name | grep -v namespace > $BDIR/test/util/$name
done
