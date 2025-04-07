#!/bin/bash

# Note:
# * This script should reside in the artifacts directory
#   when executed.
# * This script may require root privilege

if [ $# -lt 1 ]; then
  echo "Need to specify a installation root directory (e.g., /etc/rdc)"
  exit 1
fi

INSTALL_DIR=$1
mkdir -p $INSTALL_DIR
cp -R client $INSTALL_DIR
chown -R rdc:rdc $INSTALL_DIR/client

