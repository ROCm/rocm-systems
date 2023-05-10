# Data Center Tool: Developer Handbook

The RDC tool is open source and available under the MIT License. This section is helpful for open source developers. Third-party integrators may also find this information useful.

## Prerequisites for Building RDC

NOTE: The RDC tool is tested on the following software versions. Earlier versions may not work.

•        CMake 3.15

•        g++ (5.4.0)

•        AMD ROCm, which includes AMD ROCm SMI Library

•        gRPC and protoc

The following components are required to build the latest documentation:

•        Doxygen (1.8.11)

•        Latex (pdfTeX 3.14159265-2.6-1.40.16)

```
$ sudo apt install libcap-dev
$ sudo apt install -y doxygen
```

## Build and Install RDC

To build and install, clone the RDC source code from GitHub and use CMake.

```
$ git clone <GitHub for RDC>
$ cd rdc
$ mkdir -p build; cd build
$ cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT"..
$ make
#Install library file and header and the default location is /opt/rocm
$ make install
```
 
## Build Documentation

You can generate PDF documentation after a successful build. The reference manual, refman.pdf, appears in the latex directory.

```
$ make doc
$ cd latex
$ make
```

## Build Unit Tests for RDC Tool

```
$ cd rdc/tests/rdc_tests
$ mkdir -p build; cd build
$ cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT"..
$ make

# To run the tests
 
$ cd build/rdctst_tests
$ ./rdctst
```
 
## Test

```
# Run rdcd daemon
$ LD_LIBRARY_PATH=$PWD/rdc_libs/  ./server/rdcd -u
 
# In another console run the RDC command-line
$ LD_LIBRARY_PATH=$PWD/rdc_libs/  ./rdci/rdci discovery -l -u
```

## Authentication

The RDC tool supports encrypted communications between clients and servers.

### Generate Files for Authentication

The communication between the client and server can be configured to be authenticated or unauthenticated. By default, authentication is enabled.

To disable authentication, when starting the server, use the "--unauth_comm" flag (or "-u" for short). You must also use “-u” in rdci to access unauth rdcd. The /lib/systemd/system/rdc.service file can be edited to pass arguments to rdcd on starting. On the client side, when calling rdc_channel_create(), the "secure" argument must be set to False.

### Scripts

RDC users manage their own keys and certificates. However, some scripts generate self-signed certificates in the RDC source tree in the authentication directory for test purposes. The following flowchart depicts how to generate the root certificates using the openssl command in 01gen_root_cert.sh:

A picture containing sign, drawing Description automatically generated

![Generation of root certificates using openssl command](../data/handbook_openssl.png)

Generation of root certificates using openssl command

The section where the default responses to openssl questions can be specified is included in openssl.conf. To locate the section, look for the following comment line:

```
 # < ** REPLACE VALUES IN THIS SECTION WITH APPROPRIATE VALUES FOR YOUR ORG. **>
```

It is helpful to modify this section with values appropriate for your organization if you expect to call this script many times. Additionally, you must replace the dummy values and update the alt_names section for your environment.

To generate the keys and certificates using these scripts, make the following calls:

```
$ 01gen_root_cert.sh
# provide answers to posed questions
$ 02gen_ssl_artifacts.sh
# provide answers to posed questions
```

At this point, the keys and certificates are in the newly created "CA/artifacts" directory. You must delete this directory if you need to rerun the scripts.

To install the keys and certificates, access the artifacts directory and run the install.sh script as root, specifying the install location. By default, RDC expects this to be in /etc/rdc:

```
$ cd CA/artifacts
$ sudo install_<client|server>.sh /etc/rdc
```

These files must be copied to and installed on all client and server machines that are expected to communicate with one another.

### Known Limitation

The RDC tool has the following authentication limitation:

The client and server are hardcoded to look for the openssl certificate and key files in /etc/rdc. There is no workaround available currently.
Verify Files for Authentication

Several SSL keys and certificates must be generated and installed on clients and servers for authentication to work properly. By default, the RDC server will look in the /etc/rdc folder for the following keys and certificates: 

### Client

```
$ sudo tree /etc/rdc
/etc/rdc
|-- client
|-- certs
| |-- rdc_cacert.pem
| |-- rdc_client_cert.pem
|-- private
|-- rdc_client_cert.key
```

NOTE: Machines that are clients and servers consist of both directory structures. 

### Server

```
$ sudo tree /etc/rdc
/etc/rdc
|-- server
|-- certs
| |-- rdc_cacert.pem
| |-- rdc_server_cert.pem
|-- private
|-- rdc_server_cert.key
```
