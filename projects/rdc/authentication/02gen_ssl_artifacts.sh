#!/bin/bash

# This script generates ssl keys and self-signed certificates

INSTALL_RT="artifacts"

generate_artifacts() {
  HOST=$1
  echo "**********************************"
  echo "*** Generating $HOST artifacts ***"
  echo "**********************************"

  mkdir -p ${INSTALL_RT}/${HOST}/private
  mkdir -p ${INSTALL_RT}/${HOST}/certs

  echo "Generate CSR..."
  openssl req -new -nodes -out rdc_csr.pem -config ../openssl.cnf
  echo "Sign Certificate..."
  openssl ca -out rdc_${HOST}_cert.pem -config ../openssl.cnf -infiles rdc_csr.pem
  mv rdc_${HOST}_cert.pem ${INSTALL_RT}/${HOST}/certs/
  mv key.pem ${INSTALL_RT}/${HOST}/private/rdc_${HOST}_cert.key
  cp rdc_cacert.pem ${INSTALL_RT}/${HOST}/certs/
}

pushd CA
echo
echo "**********************"
echo "IMPORTANT:"
echo " * Make sure to use the same hostname (wildcards accepted) each"
echo "   time when prompted for \"Common Name\""
echo " * Make sure to select \"y\" when you are asked whether you want"
echo "   to sign the certificates"
echo "**********************"
echo
generate_artifacts "server"
generate_artifacts "client"
rm rdc_cacert.pem
cp ../install_client.sh ../install_server.sh $INSTALL_RT

popd


