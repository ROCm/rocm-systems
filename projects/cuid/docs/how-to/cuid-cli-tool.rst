.. meta::
  :description: The CUID library provides a command-line interface (CLI) for generating CUIDs and querying devices for their CUIDs.
  :keywords: CUID tool, CUID command-line, CUID CLI

.. _cuid-cli-tool:

****************
Using CUID CLI
****************

The CUID library provides a command-line interface (CLI) for generating CUIDs and querying devices for their CUIDs. This topic discusses how to use this CUID CLI tool.

Options
========

.. |br| raw:: html

    <br />

The following table lists the CUID CLI tool options:

.. list-table:: CUID tool options
  :header-rows: 1

  * - Option
    - Description
    - Usage

  * - ``--generate-cuid``
    -
      * Generates CUID registry for the discovered devices. For devices with an existing CUID registry, this option refreshes the registry.

      * Can be used with an existing key or with ``generate-key`` or ``set-key`` for a new key.

      * Requires root privileges to run.
    - ``sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid``

  * - ``--generate-key``
    -
      * Generates a new random HMAC key.

      * Used in conjunction with the ``generate-cuid`` option to generate a CUID registry with a new random key.
    - ``sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid --generate-key``

  * - ``--set-key <key_file>``
    -
      * Sets 32-byte HMAC key from the specified file.

      * Used in conjunction with the ``generate-cuid`` option to generate a CUID registry with an existing key file.
    - ``sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid --set-key <path to the key file>``

  * - ``--notify-daemon``
    -
      * Notifies daemon to refresh the device registry.

      * Called by ``udev`` when any device-related changes occur.
    - ``/opt/rocm/core/bin/amdcuid_tool --notify-daemon``

  * - ``--list``
    - Lists all devices with their CUIDs
    - ``/opt/rocm/core/bin/amdcuid_tool --list``

  * - ``--type <device-type>``
    -
      * Lists the devices with their CUIDs filtered according to the specified device type.

      * Used in conjunction with ``list`` or ``query-device`` option.

      * The ``<device-type>`` value can be ``gpu``, ``cpu``, ``nic``, or ``platform``.
    - ``/opt/rocm/core/bin/amdcuid_tool --list --type gpu``

  * - ``--show-primary``
    -
      * Lists all devices with their primary CUIDs.

      * Used in conjunction with ``list`` or ``query-device`` option.

      * Requires root privileges to run.
    - ``sudo /opt/rocm/core/bin/amdcuid_tool --list --show-primary``

  * - ``--query-device <device-identifier>``
    - Finds device using the device path or BDF.
    -
      * Using device path: ``/opt/rocm/core/bin/amdcuid_tool --query-device /sys/class/drm/renderD128``

      * Using BDF: ``/opt/rocm/core/bin/amdcuid_tool --query-device 0000:03:00.0 --type gpu``

  * - ``--version``
    - Shows the CUID library version
    - ``/opt/rocm/core/bin/amdcuid_tool --version``

To see the complete list of CUID tool option, run ``--help``, ``-h`` command.

Tool usage
===========

This section lists commonly used CUID CLI commands by purpose.

Generating CUID
----------------

When running the tool for the first time, no CUIDs might be registered on the system. This is normally the case when ``daemonize`` in the ``amdcuid_daemon.conf`` file is set to ``false`` (default setting).

To generate the CUIDs, use the ``--generate-cuid`` option:

.. code-block:: shell

  $ sudo amdcuid_tool --generate-cuid
  Generating/refreshing CUID registry...

  Successfully generated: /tmp/cuid
  Successfully generated: /tmp/priv_cuid
  Discovered 290 device(s)

  CUID registry refreshed successfully!

.. note::

  Generating CUIDs requires root privileges, as protected hardware information is required to create the CUIDs.

Managing hash key
------------------

To generate publicly available CUIDs, the CUID library uses a hash key to process protected hardware information. Therefore, a hash key is created during CUID library installation and must be managed. While the key is auto-generated initially, users might want to use a key rotation system to remove stale keys and create new ones.

- To generate CUIDs using a new key, you can use the ``generate-key`` option while generating CUIDs:

  .. code-block:: shell

    $ sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid --generate-key
    Generating/refreshing CUID registry...

    Generated new HMAC key.
    Successfully generated: /tmp/cuid
    Successfully generated: /tmp/priv_cuid
    Discovered 290 device(s)

    CUID registry refreshed successfully!

- To generate CUIDs using an existing key, use the ``set-key`` option and specify the path to the key file:

  .. code-block:: shell

    $ sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid --set-key /etc/path/to/my/key
    Generating/refreshing CUID registry...

    HMAC key loaded from: /etc/path/to/my/key
    Successfully generated: /tmp/cuid
    Successfully generated: /tmp/priv_cuid
    Discovered 290 device(s)

    CUID registry refreshed successfully!

.. note::

  A new key will create new derived CUIDs for all the devices, while their primary CUIDs will always remain the same. For more information about primary and derived CUIDs, see :ref:`what-is-cuid`

Getting CUIDs
--------------

Once CUIDs are generated for devices using the daemon or the CLI tool, you can query a specific device for its CUID or list all devices with their CUIDs.

- To list the CUIDs for all the devices on the system, use the ``--list`` option:

  .. code-block:: shell

    $ amdcuid_tool --list
    Found 290 device(s):

    ---- PLATFORM Devices ----
    PLATFORM
      CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX

    ---- CPU Devices ----
    CPU #0
      CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
      Device Path:    /sys/devices/system/cpu/cpu140

    CPU #1
      CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
      Device Path:    /sys/devices/system/cpu/cpu153

    ---- GPU Devices ----
    GPU #0
      CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
      Device Path:    /sys/class/drm/renderD175

    GPU #1
      CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
      Device Path:    /sys/class/drm/renderD158

    ---- NIC Devices ----
    NIC #0
      CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
      Device Path:    /sys/class/net/ens14np0

    NIC #1
      CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
      Device Path:    /sys/class/net/docker0

- By default, only the derived CUIDs are displayed. Viewing primary CUIDs requires root privileges to protect potentially sensitive hardware information.

  To view primary CUIDs, use the ``--show-primary`` option with ``sudo``:

  .. code-block:: shell

    $ sudo amdcuid_tool --list --show-primary
    Found 290 device(s):

    ---- PLATFORM Devices ----
    PLATFORM
      Primary CUID:   YYYYYYYY-YYYY-YYYY-YYYY-YYYYYYYYYYYY
      CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX

    ---- CPU Devices ----
    CPU #0
      Primary CUID:   YYYYYYYY-YYYY-YYYY-YYYY-YYYYYYYYYYYY
      CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
      Device Path:    /sys/devices/system/cpu/cpu140

- To get the CUID of a specific device, use the ``--query-device`` option. You can either provide the BDF or the device path.

  .. code-block:: shell

    $ amdcuid_tool --query-device /sys/class/drm/renderD188

    Device Found:
    Type:           GPU
    CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
    Device Path:    /sys/class/drm/renderD188

  To view the primary CUID of the device, use the ``--show-primary`` option with ``sudo``:

  .. code-block:: shell

    $ sudo amdcuid_tool --query-device 0000:0c:00.0 --show-primary

    Device Found:
    Type:           GPU
    Primary CUID:   YYYYYYYY-YYYY-YYYY-YYYY-YYYYYYYYYYYY
    CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
    Device Path:    /sys/class/drm/renderD128
