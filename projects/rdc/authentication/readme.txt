# How-to generate authentication files

1. Modify openssl.cnf to match your company info
2. ./01gen_root_cert.sh
3. ./02gen_ssl_artifacts.sh
4. cd CA/artifacts
5. sudo ./install_client.sh /etc/rdc
6. sudo ./install_server.sh /etc/rdc

For a full guide refer to:
https://rocm.docs.amd.com/projects/rdc/en/latest/install/handbook.html#generate-files-for-authentication
