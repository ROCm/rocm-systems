.. meta::
  :description: The ROCm Data Center tool (RDC) addresses key infrastructure challenges regarding AMD GPUs in cluster and data center environments and simplifies their administration
  :keywords: ROCm Data Center usage, RDC usage, RDC user manual, ROCm Data Center user manual, RDC tutorial, ROCm Data Center tutorial, RDC user guide, ROCm Data Center user guide

.. _using-RDC:

***********
Using RDC
***********

This topic provides useful information for the following audience on using RDC:

* Administrators: RDC provides the cluster administrator with the capability of monitoring, validating, and configuring policies.
* HPC users: RDC provides GPU-centric feedback for their workload submissions.
* OEM: RDC adds GPU information to their existing cluster management software.
* Open source contributors: RDC is open source and accepts contributions from the community.

Starting RDC
============

You can start RDC from command line using ``systemctl`` command or directly as a user. Both these options are explained in the following sections. The capability of RDC can be configured by modifying the ``rdc.service`` system configuration file. RDC reads the ``rdc.service`` file from ``/etc/systemd/system``. If multiple RDC versions are installed, copy ``/opt/rocm-<x.y.z>/libexec/rdc/rdc.service`` from the desired RDC version, to the ``/etc/systemd/system`` folder.

Starting RDC using systemctl
-----------------------------

Here are the steps to start RDC using ``systemctl`` command, which runs RDC in the background:

1. Copy the service file:

   .. code-block:: shell

    sudo cp /opt/rocm/libexec/rdc/rdc.service /etc/systemd/system/

2. Configure capabilities:

   - Full capabilities: Uncomment the following lines in ``/etc/systemd/system/rdc.service``:

   .. code-block:: shell

    CapabilityBoundingSet=CAP_DAC_OVERRIDE
    AmbientCapabilities=CAP_DAC_OVERRIDE

   - Monitor-only capabilities: Comment out the preceding lines in ``/etc/systemd/system/rdc.service``.

3. Start the service:

   .. code-block:: shell

    sudo systemctl start rdc
    sudo systemctl status rdc

4. Modify RDCD options:

   Edit ``/opt/rocm/etc/rdc_options`` to append any additional RDCD parameters.

   .. code-block:: shell

    sudo nano /opt/rocm/etc/rdc_options

   Example configuration:

   .. code-block:: shell

    RDC_OPTS="-p 50051 -u -d"

   Flags:

   - `-p 50051` : Use port 50051
   - `-u` : Unauthenticated mode
   - `-d` : Enable debug messages

Starting RDC using command line as a user
------------------------------------------

While ``systemctl`` is the preferred way to start RDC, you can also start RDC directly from the command line as a user, which runs RDC in the user's current terminal. By default, the user is defined as ``rdc`` in the ``rdc.service`` file:

.. code-block:: shell

  [Service]
  User=rdc
  Group=rdc

To change the user, you can edit the ``User`` in the ``rdc.service`` file.
To start RDC server daemon (``rdcd``) as a user such as ``rdc`` or as ``root``, use:

.. code-block:: shell

  #Start as user rdc
  $ sudo -u rdc rdcd

  # Start as root
  $ sudo rdcd

The RDC capability is determined by the privilege of the user starting ``rdcd``. For example, ``rdcd`` running under a normal user account has monitor-only capability and ``rdcd`` running as root has full capability.

.. note::

  If a user other than rdc or root starts the ``rdcd`` daemon, the file ownership of the SSL keys mentioned in the :ref:`authentication <authentication>` section must be modified to allow read and write access.

.. _authentication:

Authentication
===============

RDC supports encrypted communications between clients and servers.

You can enable or disable authentication for the communication between the client and server. By default, authentication is enabled.

To disable authentication, use the ``--unauth_comm`` or ``-u`` flag when starting the server. You must also use ``-u`` in ``rdci`` to access unauthenticated ``rdcd``. You can edit the ``rdc.service`` file to specify arguments to be passed while starting ``rdcd``. On the client side, the ``secure`` argument must be set to ``False`` when calling ``rdc_channel_create()``.
The following sections provide information for setting up the ``rdcd`` server for authentication.

Generating keys and certificates using scripts
------------------------------------------------

RDC users manage their own keys and certificates. However, some scripts generate self-signed certificates in the RDC source tree in the authentication directory for test purposes. The following flowchart depicts how to generate the root certificates using the ``openssl`` command in ``01gen_root_cert.sh``:

.. figure:: ../data/handbook_openssl.png

    Generation of root certificates using openssl command

You can specify the default responses to ``openssl`` questions in a section in the ``openssl.conf`` file. To locate the section in the ``openssl.conf`` file, look for the following comment:

.. code-block:: shell

    # < ** REPLACE VALUES IN THIS SECTION WITH APPROPRIATE VALUES FOR YOUR ORG. **>

Modifying this section with values appropriate for your organization is helpful in cases where this script is called multiple times. Additionally, you must replace the dummy values and update the ``alt_names`` section for your environment.

To generate the keys and certificates using these scripts, use:

.. code-block:: shell

    $ 01gen_root_cert.sh
    # provide answers to posed questions
    $ 02gen_ssl_artifacts.sh
    # provide answers to posed questions

On running the preceding scripts, the keys and certificates are generated in the newly created ``CA/artifacts`` directory.

.. important::
  You must delete this directory before rerunning the scripts.

To install the keys and certificates, access the artifacts directory and run the ``install.sh`` script as root along with specifying the install location. The default install location is ``/etc/rdc``:

.. code-block:: shell

    $ cd CA/artifacts
    $ sudo install_<client|server>.sh /etc/rdc

These files must be copied and installed on all client and server machines expected to communicate with each other.

Known limitation
-----------------

The client and server are hardcoded to look for the ``openssl`` certificate and key files in ``/etc/rdc``. No workaround is available for this.

Keys and certificates for authentication
-----------------------------------------

Several SSL keys and certificates must be generated and installed on clients and servers for authentication to work properly. By default, the RDC server looks in the ``/etc/rdc`` folder for the following keys and certificates:

Client
+++++++

.. code-block:: shell

    $ sudo tree /etc/rdc
    /etc/rdc
    |-- client
    |-- certs
    | |-- rdc_cacert.pem
    | |-- rdc_client_cert.pem
    |-- private
    |-- rdc_client_cert.key

Server
+++++++

.. code-block:: shell

    $ sudo tree /etc/rdc
    /etc/rdc
    |-- server
    |-- certs
    | |-- rdc_cacert.pem
    | |-- rdc_server_cert.pem
    |-- private
    |-- rdc_server_cert.key

.. note::

  Machines acting as both client and server consist of both directory structures.

Modes of operation
===================

RDC supports two primary modes of operation: *Standalone* and *Embedded*. The feature set is similar in both the cases. Choose the mode that best fits your deployment needs.

The capability in each mode depends on the user privileges while starting the RDC tool. A normal user has access only to monitor (GPU telemetry) capabilities. A privileged user can run the tool with full capabilities. In the full capability mode, GPU configuration features can be invoked. The full capability mode might affect all the users and processes sharing the GPU.

Standalone mode
-----------------

Standalone mode allows you to run RDC independently with all its components installed.
This is the preferred mode of operation, as it does not have any external dependencies. To start RDC in standalone mode, ``rdcd`` must run on each compute node.

- Starting RDCD as a privileged user: A privileged user can run RDC with full capabilities.

  - With authentication:

    .. code-block:: shell

      sudo /opt/rocm/bin/rdcd

  - Without authentication:

    .. code-block:: shell

      sudo /opt/rocm/bin/rdcd -u

- Starting RDC as a normal user: A normal user can run RDC with monitor-only capabilities only.

  - With authentication:

    .. code-block:: shell

      /opt/rocm/bin/rdcd

  - Without authentication:

    .. code-block:: shell

      /opt/rocm/bin/rdcd -u

Embedded mode
--------------

Embedded mode integrates RDC directly into your existing management tools using its library format.
The embedded mode is especially useful for a monitoring agent running on the compute node. The monitoring agent can directly use the RDC library to achieve a fine-grain control on how and when to invoke the RDC features. For example, if the monitoring agent has a facility to synchronize across multiple nodes, it can synchronize GPU telemetry across these nodes.

The RDC daemon ``rdcd`` can be used as a reference code for this purpose. The dependency on ``gRPC`` is also eliminated, if the RDC library is directly used.

To run RDC in embedded mode, use:

.. code-block:: shell

  python your_management_tool.py --rdc_embedded

.. note::

  Ensure that the ``rdcd`` daemon is not running separately, when using embedded mode.

.. caution::

  RDC command-line ``rdci`` doesn't function in this mode. Third-party monitoring software is responsible for providing the user interface and remote access or monitoring.

Troubleshooting RDC
====================

The RDCD logs provide useful status and debugging information. The logs can also help debug problems like ``rdcd`` failing to start, communication issues with a client, and many more.

- View logs:

  When ``rdcd`` is started using ``systemctl``, you can view the logs using:

  .. code-block:: shell

    $ journalctl -u rdc

- Run RDCD with debug logs:

  .. code-block:: shell

    RDC_LOG=DEBUG /opt/rocm/bin/rdcd

  Logging levels supported: `ERROR`, `INFO`, `DEBUG`.

- Enable additional logging messages:

  .. code-block:: shell

    export RSMI_LOGGING=3

If the GPU reset fails, restart the server. Note that restarting the server also initiates ``rdcd``. You might then encounter the following two scenarios:

- ``rdcd`` returns the correct GPU information to ``rdci``

- ``rdcd`` returns the `No GPUs found on the system` error to ``rdci``. To resolve this error, restart ``rdcd`` using:

  .. code-block:: shell

    $ sudo systemctl restart rdcd

Known issues
-------------

- dmon fields return N/A

  **Reasons:**

  - Missing libraries:

    - Verify ``/opt/rocm/lib/rdc/librdc_*.so`` exists.
    - Ensure all related libraries such as ``rocprofiler``, ``rocruntime``, and others are present.

  - Unsupported GPU:

    - Most metrics work on MI300 and newer.
    - Limited metrics on MI200.
    - Consumer GPUs such as RX6800 have fewer supported metrics.

- HSA_STATUS_ERROR_OUT_OF_RESOURCES

  **Error message:**

  .. code-block:: shell

    terminate called after throwing an instance of 'std::runtime_error'
    what():  hsa error code: 4104 HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources. This error may also occur when the core runtime library needs to spawn threads or create internal OS-specific events.
    Aborted (core dumped)

  **Solution:**

  Follow these steps to check for missing groups:

  1. Ensure video and render groups exist.

     .. code-block:: shell

      sudo usermod -aG video,render $USER

  2. Logout and login to apply group changes.
