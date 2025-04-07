#!/bin/bash

# This script should be called only once to generate a root
# certificate

mkdir -p CA
pushd CA
mkdir private newcerts
chmod 700 private newcerts

#  Our next step is to create a database for the certificates we will sign:
echo '01' >serial
touch index.txt

# openssl_part1.cnf

# Create the Root Certificate
# This call of openssl encrypts the keys
# openssl req -new -x509 extensions v3_ca -keyout private/rdc_cakey.pem \
#	 -out rdc_cacert.pem -days 3650 -config ../openssl.cnf

# This call of openssl does not encrypt the keys
openssl req -new -x509 -nodes -extensions v3_ca -keyout private/rdc_cakey.pem \
	 -out rdc_cacert.pem -days 3650 -config ../openssl.cnf
# This generates:
# A private key in private/rdc_cakey.pem
# A root CA certificate in rdc_cacert.pem (distribute to clients)

popd

