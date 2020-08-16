
# Radeon Data Center Tools

# Running RDC

##### Additional Software Required for Running RDC
In order to run RDC, the following components are required.
Note that the software versions listed are what was used in development.
Earlier versions are not guaranteed to work:
* ROCm
* gRPC and protoc
  Unfortunately, gRPC must be built from source as no pre-built .deb or .rpm
  packages are available.
  See instructions for building gRPC/protoc below.
* [ROCm SMI Library](https://github.com/RadeonOpenCompute/rocm_smi_lib)

##### Building gRPC and protoc
gRPC libraries and the protoc compiler must be installed to both build RDC and
must also be available machines were RDC will run. To build and install gRPC,
follow these steps.
- Get gRPC required tool installation
    ``$ sudo apt-get install -y automake make g++ unzip``
    ``$ sudo apt-get install -y build-essential autoconf libtool pkg-config``
    ``$ sudo apt-get install -y libgflags-dev libgtest-dev``
    ``$ sudo apt-get install -y clang-5.0 libc++-dev curl``

**IMPORTANT** Building gRPC and protocol buffers using this method requires
CMake 3.15 or greater. If you use an earlier version of CMake than this, the
build will succeed (aside from an easily missed CMake message), but when gRPC
is installed, not all the required files will be there, so the RDC program
will fail to run.
- Download and build gRPC

    $ git clone -b v1.28.1 https://github.com/grpc/grpc
    $ cd grpc
    $ git submodule update --init
    $ mkdir -p cmake/build
    $ cd cmake/build
      # By default (without using the CMAKE_INSTALL_PREFIX option), the following
      # will install to /usr/local lib, include and bin directories
    $ cmake -DgRPC_INSTALL=ON -DBUILD_SHARED_LIBS=ON <-DCMAKE_INSTALL_PREFIX=<install dir>> ../..
    $ make
    $ sudo make install
    $ echo "<install dir>" | sudo tee /etc/ld.so.conf.d/grpc.conf
    $ sudo ldconfig

### Installation of RDC
RDC packages can be installed with dpkg or yum for various distros. After RDC
is installed, authentication must be set up before attempting to run an RDC
application.


### Authentication
RDC supports encrypted communications between clients and servers. The
communication can be configured to be authenticated or not authenticated.

##### Unauthenticated Communications
By default, authentication is enabled. To disable authentication, when starting
the server use the ``--unauth_comm`` flag (or ``-u`` for short). (The
``/lib/systemd/system/rdc.service`` file can be edited to pass arguments
to rdcd on starting.) On the client side,
when calling rdc_channel_create(), the "secure" argument should be set to false.

##### Public Key Infrastructure (PKI) Authentication
 A number of SSL keys and certificates must be generated and installed on the
 clients and servers for authentication to work properly. By default, the RDC
 server will look under ``/etc/rdc`` for the following keys and certificates:

- Servers

    sudo tree /etc/rdc
    /etc/rdc
    |-- server
        |-- certs
        |   |-- rdc_cacert.pem
        |   |-- rdc_server_cert.pem
        |-- private
            |-- rdc_server_cert.key


- Clients

    $ sudo tree /etc/rdc
    /etc/rdc
    |-- client
        |-- certs
        |   |-- rdc_cacert.pem
        |   |-- rdc_client_cert.pem
        |-- private
            |-- rdc_client_cert.key


Machines that are both clients and servers will have both directory
structures.

RDC users would normally generate their own keys and certificates. However,
there are scripts in the client installation file hierarchy (normally in
``/opt/rocm/rdc/authentication``) that will generate self-signed
certificates in RDC source tree, under the "authentication" directory.
The scripts call the openssl command to generate the required keys and
certificates. The openssl command will query the caller for different
identifying information. The calls to openssl will refer to the
openssl.cnf file for configuration information. Included in this file is
a section where default responses to the openssl questions can  be
specified. Look for the comment line

    # < ** REPLACE VALUES IN THIS SECTION WITH APPROPRIATE VALUES FOR YOUR ORG. **>

to find this section. It is helpful to modify this section with values
appropriate for your organization if this script will be called many times.

Additionally, the alt_names section needs to be updated for your environment
(instead of the dummy values that there initially).

To generate the keys and certificates using these scripts, make the following
calls:


    $ 01gen_root_cert.sh
    # provide answers to posed questions
    $ 02gen_ssl_artifacts.sh
    # provide answers to posed questions

At this point, the keys and certificates will be in the newly created
"CA/artifacts" directory. This directory should be deleted if you need to
rerun the scripts.

To install the keys and certificates, cd into the artifacts directory and run
the install.sh script as root, specifying the install location. By default,
RDC will expect this to be in /etc/rdc:


    $ cd CA/artifacts
    $ sudo install_<client|server>.sh /etc/rdc

These files should be copied to and and installed on all client and server
machines that are expected to communicate with one another.

##### Current Limitations
There are a few limitations on the authentication capabilities. These
limitations are temporary and will be eliminated when the server has a
configuration file where user preferences can be specified.
* The client and server are hard-coded to look for openssl certificate and key
files in /etc/rdc.

# Starting RDC Server Daemon (RDCD)
In order for an RDC client application to monitor and/or control a remote
system, the RDC server daemon, rdcd, must be running on the remote system.
rdcd can be configured to run with full capabilities, or with monitoring-only
capabilities. "Full capabilities" includes the ability to set some system
functions exposed by the RDC APIs and tools. Changing a system's configuration
involves writing to system files. When rdcd is configured to run with full
capabilities, it has the ability to write to these system files. Alternatively,
rdcd can be run with control functionality disabled. In this case, rdcd does
not have the ability to write to the control-related system files. Calls to RDC
APIs (or tools that invoke these APIs) will result in a permission-related
failure when configured for limited functionality. This reduced mode can be
used to prevent someone from inadvertently or maliciously putting a system
into an unwanted state.

Which configuration is used depends on how rdcd is started, and certain
settings in the rdc.service systemd configuration file.

###Starting rdcd with systemctl
rdcd can be started by using the systemctl command.

#####Starting rdcd with systemctl
When starting rdcd using systemctl, like this:


    systemctl start rdc

systemctl will read /lib/systemd/system/rdc.service, which is installed with
rdc. This file has 2 lines that control what capabilities with which rdcd
will run. If left uncommented, rdcd will run with full capabilities, as
shown below:

    CapabilityBoundingSet=CAP_DAC_OVERRIDE
    AmbientCapabilities=CAP_DAC_OVERRIDE


When these lines are commented with ``#``, rdcd will run with monitor-only
capabilty.


###Starting rdcd Directly
rdcd can also be started by directly invoking rdcd from the command line,
like this:

    # Start as user rdc
    sudo -u rdc rdcd

or

    # Start as root
    sudo rdcd


Note how rdcd must be started as user "rdc" or root. Other regular user
accounts will not normally work. This is because rdcd will need access to
private SSL keys and certificates, owned by rdc. In order to run rdcd
under a different account, the SSL keys would need to be accessible by
that account.

When run from the command line, the rdc.service file mentioned in the previous
section does not come into play. What determines the level of capability is
the level of capability of the id under which rdcd is started. If rdcd is
directly started as root, then rdcd will have monitor and control capability.
If rdcd is directly started with a normal user account, then it will have
monitor-only capability.

###Troubleshooting rdcd
* When rdcd is started using systemctl, we can view messages that can help debug
problems with either starting rdcd or with communications with a client using
the ``journalctl`` command. To view rdcd logs, issue the following command on
the server

    journalctl -u rdc

* Enable the debug log:

    sudo RDC_LOG=DEBUG ./server/rdcd

* Check the ssl connection in rdci:

    rdcd_hostname=<rdcd hostname>  # Set the rdcd hostname to which you want to connect
    openssl s_client -connect $rdcd_hostname:50051 -cert /etc/rdc/client/certs/rdc_client_cert.pem -key /etc/rdc/client/private/rdc_client_cert.key -CAfile /etc/rdc/client/certs/rdc_cacert.pem


