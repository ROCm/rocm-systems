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
cp -R server $INSTALL_DIR
mkdir -p $INSTALL_DIR/client/certs
cp client/certs/rdc_cacert.pem $INSTALL_DIR/client/certs
chmod 700 $INSTALL_DIR/server/private
chown -R rdc:rdc $INSTALL_DIR/server
chown -R rdc:rdc $INSTALL_DIR/client

