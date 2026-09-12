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

  Successfully generated: /var/lib/amdcuid/cuid
  Successfully generated: /var/lib/amdcuid/priv_cuid
  Discovered 290 device(s)

  CUID registry refreshed successfully!

.. note::

  Generating CUIDs requires root privileges, as protected hardware information is required to create the CUIDs.

Where the records are stored
-----------------------------

The record store is ``/var/lib/amdcuid/``:

* ``/var/lib/amdcuid/cuid``: the world-readable records, holding derived CUIDs.
* ``/var/lib/amdcuid/priv_cuid``: the privileged records, holding primary CUIDs and hardware fingerprints. Not world-readable.

The directory is created root-owned and mode ``0755`` by the post-install script.

.. warning::

  These files were previously ``/tmp/cuid`` and ``/tmp/priv_cuid``. A world-writable directory is not a safe place for files a root-privileged refresh writes and every consumer trusts: a local user could pre-create the path, or the predictable temporary file beside it, and redirect a root write. That was a local privilege-escalation defect, and moving the store to ``/var/lib/amdcuid`` is its fix. Do not point the store back under ``/tmp``.

``AMDCUID_RECORD_DIR``
~~~~~~~~~~~~~~~~~~~~~~~

The environment variable ``AMDCUID_RECORD_DIR`` overrides the directory, and is honoured **only when the effective UID is not 0**:

.. code-block:: shell

  # Honoured: an ordinary user pointing its own store at a directory it can write.
  $ AMDCUID_RECORD_DIR=$HOME/.cache/amdcuid amdcuid_tool --list

  # Ignored: root always uses the compiled-in /var/lib/amdcuid.
  $ sudo AMDCUID_RECORD_DIR=/tmp/mine amdcuid_tool --generate-cuid
  Successfully generated: /var/lib/amdcuid/cuid

An unprivileged caller redirecting its own store is harmless, and is how a test (or a caller on a node where no privileged refresh has ever run) gets a store at all. For root the variable is ignored outright: letting the environment steer where a root-privileged refresh writes primary CUIDs and hardware fingerprints would reopen exactly the hole the move closed.

To relocate the store for packaging, set the compile-time default instead: ``cmake -DAMDCUID_RECORD_DIR=/some/path ...``.

.. note::

  When no serial number and no machine identity (``/etc/machine-id``, falling back to ``/var/lib/dbus/machine-id``) can be found for a device, the tool reports an **error** for that device. It does not fall back to a synthesised placeholder identifier, because every device that took that path produced the same value and callers were told it was unique.

Managing hash key
------------------

To generate publicly available CUIDs, the CUID library uses a hash key to process protected hardware information. Therefore, a hash key is created during CUID library installation and must be managed. While the key is auto-generated initially, users might want to use a key rotation system to remove stale keys and create new ones.

- To generate CUIDs using a new key, you can use the ``generate-key`` option while generating CUIDs:

  .. code-block:: shell

    $ sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid --generate-key
    Generating/refreshing CUID registry...

    Generated new HMAC key.
    Successfully generated: /var/lib/amdcuid/cuid
    Successfully generated: /var/lib/amdcuid/priv_cuid
    Discovered 290 device(s)

    CUID registry refreshed successfully!

- To generate CUIDs using an existing key, use the ``set-key`` option and specify the path to the key file:

  .. code-block:: shell

    $ sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid --set-key /etc/path/to/my/key
    Generating/refreshing CUID registry...

    HMAC key loaded from: /etc/path/to/my/key
    Successfully generated: /var/lib/amdcuid/cuid
    Successfully generated: /var/lib/amdcuid/priv_cuid
    Discovered 290 device(s)

    CUID registry refreshed successfully!

.. note::

  A new key will create new derived CUIDs for all the devices, while their primary CUIDs will always remain the same. For more information about primary and derived CUIDs, see :ref:`what-is-cuid`

.. note::

  Re-keying also writes the new 32-byte seed to each discovered PCI device's sysfs ``cuid_seed`` attribute, before the records are rebuilt. This is required for the re-key to take effect at all: derivation prefers the driver-published ``cuid_secondary`` over anything the library computes, so replacing only the key file and the records would leave every GPU whose driver publishes that attribute still serving its pre-re-key value. The kernel accepts exactly 32 bytes and rejects any other length.

  A device with no ``cuid_seed`` attribute is not an error: there is no driver-published value to retire. A device that has the attribute and refuses the write is reported, and the tool exits non-zero. The node has been re-keyed, but those devices are still serving their old derived CUIDs.

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
