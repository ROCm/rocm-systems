#!/usr/bin/env bash
#
# Copyright (c) 2026 - Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# rdcd container entrypoint.
#
# The image previously started rdcd with `-u` (grpc::InsecureServerCredentials),
# exposing unauthenticated, privileged GPU management RPCs to the network.
# Instead, generate self-signed mTLS certificates on first start (if absent) and
# run rdcd in authenticated mode.
#
# Mount your own certificates over ${RDC_ETC_DIR:-/etc/rdc} to use a real PKI.
set -euo pipefail

RDC_ETC="${RDC_ETC_DIR:-/etc/rdc}"
RDCD_BIN="${RDCD_BIN:-/opt/rocm/bin/rdcd}"
CN="${RDC_SERVER_CN:-localhost}"

SRV_KEY="$RDC_ETC/server/private/rdc_server_cert.key"
SRV_CRT="$RDC_ETC/server/certs/rdc_server_cert.pem"
CLI_KEY="$RDC_ETC/client/private/rdc_client_cert.key"
CLI_CRT="$RDC_ETC/client/certs/rdc_client_cert.pem"
CA_CRT="$RDC_ETC/client/certs/rdc_cacert.pem"

if [[ ! -s "$SRV_KEY" || ! -s "$SRV_CRT" || ! -s "$CA_CRT" ]]; then
  echo "rdcd-entrypoint: generating self-signed mTLS certificates under $RDC_ETC"
  install -d -m 0755 "$RDC_ETC/server/certs" "$RDC_ETC/client/certs"
  install -d -m 0700 "$RDC_ETC/server/private" "$RDC_ETC/client/private"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

  # Root CA
  openssl req -x509 -nodes -newkey rsa:4096 -sha512 -days 3650 \
    -keyout "$tmp/ca.key" -out "$tmp/ca.crt" -subj "/O=AMD/CN=RDC Root CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign"

  # Server certificate (mTLS server identity)
  openssl req -nodes -newkey rsa:4096 -sha512 -keyout "$SRV_KEY" \
    -out "$tmp/server.csr" -subj "/O=AMD/CN=$CN"
  openssl x509 -req -in "$tmp/server.csr" -CA "$tmp/ca.crt" -CAkey "$tmp/ca.key" \
    -CAcreateserial -days 3650 -sha512 -out "$SRV_CRT" \
    -extfile <(printf 'subjectAltName=DNS:%s,DNS:localhost,IP:127.0.0.1\nbasicConstraints=CA:FALSE\nextendedKeyUsage=serverAuth\n' "$CN")

  # Client certificate (so the bundled rdci can authenticate to rdcd)
  openssl req -nodes -newkey rsa:4096 -sha512 -keyout "$CLI_KEY" \
    -out "$tmp/client.csr" -subj "/O=AMD/CN=rdc-client"
  openssl x509 -req -in "$tmp/client.csr" -CA "$tmp/ca.crt" -CAkey "$tmp/ca.key" \
    -CAcreateserial -days 3650 -sha512 -out "$CLI_CRT" \
    -extfile <(printf 'basicConstraints=CA:FALSE\nextendedKeyUsage=clientAuth\n')

  # rdcd verifies client certs against this CA; rdci verifies the server against it
  cp "$tmp/ca.crt" "$CA_CRT"
  chmod 0644 "$SRV_CRT" "$CLI_CRT" "$CA_CRT"
  chmod 0600 "$SRV_KEY" "$CLI_KEY"
  echo "rdcd-entrypoint: certificate generation complete"
fi

exec "$RDCD_BIN" "$@"
