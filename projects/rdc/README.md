# rdc
Radeon Data Center

## To run the rdcd and rdci from the build folder without authentication
```
sudo LD_LIBRARY_PATH=$PWD/rdc_libs/  ./server/rdcd -u
LD_LIBRARY_PATH=$PWD/rdc_libs/  ./rdci/rdci discovery -u
```

## To run the rdcd and rdci from the build folder with authentication
```
sudo LD_LIBRARY_PATH=$PWD/rdc_libs/  ./server/rdcd
LD_LIBRARY_PATH=$PWD/rdc_libs/  ./rdci/rdci discovery
```

## Troubleshooting
Check the ssl connection in rdci:
```
rdcd_hostname= "" # Set the rdcd you want to connect
openssl s_client -connect $rdcd_hostname:50051 -cert /etc/rdc/client/certs/rdc_client_cert.pem -key /etc/rdc/client/private/rdc_client_cert.key -CAfile /etc/rdc/client/certs/rdc_cacert.pem
```

