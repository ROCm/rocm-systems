
# Radeon Data Center Tools

TODO: Add general description of RDC, link to github site 


# Building RDC

#### Additional Required software for building
In order to build the RDC software, the following components are required. Note that the software versions listed are what was used in development. Earlier versions are not guaranteed to work:
TODO: see if protoc and c++ plugin can be installed from packages rather than built from
source. This will be necessary for CI integration.
* CMake (v3.5.0)
* g++ (5.4.0)
* ROCm
* [ROCm SMI Library](https://github.com/RadeonOpenCompute/rocm_smi_lib)
* [gRPC and protoc](https://github.com/grpc/grpc/blob/master/src/cpp/README.md#make)

In order to build the latest documentation, the following are required:
* DOxygen (1.8.11)
* latex (pdfTeX 3.14159265-2.6-1.40.16)

The source code for RDC is available on [ADD RDC GITHUB URL HERE]().

After the RDC library git repository has been cloned to a local Linux machine, building  is achieved by following the typical CMake build sequence. Specifically,
##### ```$ mk -p build```
##### ```$ cd build```
##### ```$ cmake -DROCM_DIR=<location of ROCm root, including ROCm SMI> <location of root of ROCm SMI library CMakeLists.txt>```
##### ```$ make```
##### ```# Install library file and header; default location is /opt/rocm```
##### ```$ make install```
The built library will appear in the `build` folder.

#### Building the Documentation
The documentation PDF file can be built with the following steps (continued from the steps above):
##### ```$ make doc```
##### ```$ cd latex```
##### ```$ make```
The reference manual, `refman.pdf` will be in the `latex` directory upon a successful build.

#### Building the Tests UPDATE FOR RDC
In order to verify the build and capability of ROCm SMI on your system and to see an example of how ROCm SMI can be used, you may build and run the tests that are available in the repo. To build the tests, follow these steps:

##### ```$ cd <ROCM SMI source root>```
##### ```$ mkdir lib```
##### ```$ cd lib```
##### ```$ cmake -DROCM_DIR=<ROCM SMI source root> <ROCm SMI source root>/tests/rocm_smi_test```
##### ```$ make

TODO: THE REMAINDER NEEDS TO BE TAILORED FOR RDC

To run the test, execute the program `rsmitst` that is built from the steps above.


###Installation

### Authentication
RDC supports encrypted communications between clients and servers. The communication can be configured to be authenticated or not authenticated. 

##### Unauthenticated Communications
By default, authentication is enabled. To disable authentication, when starting the server use the "--unauth_comm" flag (or "-u" for short). On the client side, when calling rdc_channel_create(), the "secure" argument should be set to false.

##### Public Key Infrastructure (PKI) Authentication
 A number of SSL keys and certificates must be generated and installed on the clients and servers for authentication to work properly. By default, the RDC server will look under /etc/rdc for the following keys and certificates:

- Servers
 ```sh
 $ sudo tree /etc/rdc
/etc/rdc
└── server
    ├── certs
    │   ├── rdc_cacert.pem
    │   └── rdc_server_cert.pem
    └── private
        └── rdc_server_cert.key
 ```

- Clients
 ```sh
 $ sudo tree /etc/rdc
/etc/rdc
└── client
    ├── certs
    │   ├── rdc_cacert.pem
    │   └── rdc_client_cert.pem
    └── private
        └── rdc_client_cert.key

 ```
Machines that are both clients and servers will have both directory structures.

RDC users would normally generate their own keys and certificates. However, there are scripts that will generate self-signed certficates in RDC source tree, under the "authentication" directory. The scripts call the openssl command to generate the required keys and certificates. The openssl command will query the caller for different identifying information. The calls to openssl will refer to the openssl.cnf file for configuration information. Included in this file is a section where default responses to the openssl questions can  be specified. Look for the comment line
```sh
# < ** REPLACE VALUES IN THIS SECTION WITH APPROPRIATE VALUES FOR YOUR ORG. **>
```
to find this section. It is helpful to modify this section with values appropriate for your organization if this script will be called many times.

Additionally, the alt_names section needs to be updated for your environment (instead of the dummy values that there initially).

To generate the keys and certficates using these scripts, make the following calls:

```sh
$ 01gen_root_cert.sh
# provide answers to posed questions
$ 02gen_ssl_artifacts.sh
# provide answers to posed questions
```
At this point, the keys and certficates will be in the newly created "CA/artifacts" directory. This directory should be deleted if you need to rerun the scripts.

To install the keys and certificates, cd into the artifacts directory and run the install.sh script as root, specifying the install location. By default, RDC will expect this to be in /etc/rdc:

```sh
$ cd CA/artifacts
$ sudo install_<client|server>.sh /etc/rdc
```
These files should be copied to and and installed on all client and server machines that are expected to communicate with one another.

##### Current Limitations
There are a few limitations on the authentication capabilities. These limitations are temporary and will be eliminated when the server has a configuration file where user preferences can be specified.
* The client and server are hard-coded to look for openssl certificate and key files in /etc/rdc.

# Starting RDC Server Daemon (RDCD)
In order for an RDC client application to monitor and/or control a remote system, the RDC server daemon, rdcd, must be running on the remote system. rdcd can be configured to run with full capabilities, or with monitoring-only capabilities. "Full capabilities" includes the abilty to set some system functions exposed by the RDC APIs and tools. Changing a system's configuration involves writing to system files. When rdcd is configured to run with full capabilities, it has the ability to write to these system files. Alternatively, rdcd can be run with control functionality disabled. In this case, rdcd does not have the ability to write to the control-related system files. Calls to RDC APIs (or tools that invoke these APIs) will result in a permission-related failure when configured for limited functionality. This reduced mode can be used to prevent someone from inadvertantly or maliciously putting a system into an unwanted state.

Which configuration is used depends on how rdcd is started, and certain settings in the rdc.service systemd configuration file.

###Starting rdcd with systemctl
rdcd can be started by using the systemctl command.

#####Starting rdcd with systemctl
When starting rdcd using systemctl, like this:

```sh
$ systemctl start rdc
```
systemctl will read /lib/systemd/system/rdc.service, which is installed with rdc. This file has 2 lines that control what capabilities with which rdcd will run. If left uncommented, rdcd will run with full capabilities, as shown below:

```sh
CapabilityBoundingSet=CAP_DAC_OVERRIDE
AmbientCapabilities=CAP_DAC_OVERRIDE
```

When these lines are commented with `#`, rdcd will run with monitor-only capabilty.


###Starting rdcd Directly
rdcd can also be started by directly invoking rdcd from the command line, like this:

```sh
# Start as user rdc
$ sudo -u rdc rdcd
```

```sh
# Start as root
$ sudo rdcd
```

Note how rdcd must be started as user "rdc" or root. Other regular user accounts will not work. This is because rdcd will need access to private SSL keys and certificates, owned by rdc.

When run from the command line, the rdc.service file mentioned in the previous section does not come into play. What determines the level of capability is the level of capability of the id under which rdcd is started. If rdcd is directly started as root, then rdcd will have monitor and control capability. If rdcd is directly started with a normal user account, then it will have monitor-only capability.



###Other Notes about rdcd
* Only 1 instance of rdcd should be running at a time.
* rdcd runs under the "rdc" user id


###Running with Monitor-Only Capability ("Limited Capability")

# Hello RDC

