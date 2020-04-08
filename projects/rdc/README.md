# rdc
Radeon Data Center

## To run the rdcd and rdci from the build folder without authentication
Note: Only if RDC not installed requires the LD_LIBRARY_PATH.
```
sudo LD_LIBRARY_PATH=$PWD/rdc_libs/  ./server/rdcd -u
LD_LIBRARY_PATH=$PWD/rdc_libs/  ./rdci/rdci discovery -u -l
```

## To run the rdcd and rdci from the build folder with authentication
Note: Only if RDC not installed requires the LD_LIBRARY_PATH.
```
sudo LD_LIBRARY_PATH=$PWD/rdc_libs/  ./server/rdcd
LD_LIBRARY_PATH=$PWD/rdc_libs/  ./rdci/rdci discovery -l
```

## Troubleshooting
Enable the debug log:
```
sudo RDC_LOG=DEBUG ./server/rdcd
```

Check the ssl connection in rdci:
```
rdcd_hostname= "" # Set the rdcd you want to connect
openssl s_client -connect $rdcd_hostname:50051 -cert /etc/rdc/client/certs/rdc_client_cert.pem -key /etc/rdc/client/private/rdc_client_cert.key -CAfile /etc/rdc/client/certs/rdc_cacert.pem
```



