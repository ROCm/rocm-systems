---
myst:
  html_meta:
    "description lang=en": "Learn how to use the CUID command line tool."
    "keywords": "lib, example, CLI, tool, component, unified, identifier"
---

# CUID CLI Tool Usage

This tool is a command line interface for generating CUIDs and querying devices for their CUID.

## Install the CUID Project

Refer to the [install instructions](../install/install.md)

## Get Started

The CUID CLI Tool allows users to quickly generate and find CUIDs for their devices. When running the tool without any options, or using the `-h` option, the following help text is displayed:

```shell-session
Usage: /opt/rocm/core/bin/amdcuid_tool [OPTIONS]

AMD Component Unified Identifier (CUID) Tool

Options:
  --generate-cuid              Generate/refresh CUID registry from discovered devices
                               Requires root privileges
                               Uses existing key or specify --generate-key/--set-key
  --generate-key               Generate a new random HMAC key (use with --generate-cuid)
  --set-key <key_file>         Set HMAC key from file (32 bytes, use with --generate-cuid)
  --notify-daemon              Notify daemon to refresh device registry (for udev integration)
  --list                       List all devices and their CUIDs
  --type <type>                Filter by device type (gpu, cpu, nic, platform)
                               Use with --list or --query-device
  --show-primary               Show primary CUIDs (requires root privileges)
                               Use with --list or --query-device
  --query-device <identifier>  Query specific device by device path or BDF
  --version                    Show library version
  --help, -h                   Show this help message

Examples:
  # Generate CUID registry with a new random key (requires root)
  sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid --generate-key

  # Generate CUID registry with existing key file
  sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid --set-key /path/to/hmac_key.bin

  # Generate CUID registry using previously set key
  sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid

  # Notify daemon of device changes (called by udev)
  /opt/rocm/core/bin/amdcuid_tool --notify-daemon

  # List all devices with their CUIDs
  /opt/rocm/core/bin/amdcuid_tool --list

  # List all GPUs with their CUIDs
  /opt/rocm/core/bin/amdcuid_tool --list --type gpu

  # List all devices with primary CUIDs (requires root)
  sudo /opt/rocm/core/bin/amdcuid_tool --list --show-primary

  # Query specific device by path
  /opt/rocm/core/bin/amdcuid_tool --query-device /sys/class/drm/renderD128

  # Query device by BDF
  /opt/rocm/core/bin/amdcuid_tool --query-device 0000:03:00.0 --type gpu
```

When running the tool for the first time, there may not be any CUIDs known on the system. This would especially be the case if the `daemonize` setting in the amdcuid_daemon.conf file is left as `false`. Users can instead detect and generate CUIDs on their own by using the --generate-cuid option like below:

```shell-session
$ sudo amdcuid_tool --generate-cuid
Generating/refreshing CUID registry...

Successfully generated: /tmp/cuid
Successfully generated: /tmp/priv_cuid
Discovered 290 device(s)

CUID registry refreshed successfully!
```

Users should note that generating CUIDs requires root privileges as protected hardware information is needed to create CUIDs.

## Managing the Hash Key

The library puts the hardware information through a hash to generate the CUIDs that are publicly seen. Therefore, a hash key is created as part of the installation and needs to be managed. While the initial key is auto generated, users may want to employ a key rotation system to remove old stale keys and create new keys. If, for whatever reason, a new key needs to be made, users can do so by using the `--generate-key` option when generating CUIDs:

```shell-session
$ sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid --generate-key
Generating/refreshing CUID registry...

Generated new HMAC key.
Successfully generated: /tmp/cuid
Successfully generated: /tmp/priv_cuid
Discovered 290 device(s)

CUID registry refreshed successfully!
```

If users already have a key file, they can set the key instead to the one in the file using the `--set-key` option:

```shell-session
$ sudo /opt/rocm/core/bin/amdcuid_tool --generate-cuid --set-key /etc/path/to/my/key
Generating/refreshing CUID registry...

HMAC key loaded from: /etc/path/to/my/key
Successfully generated: /tmp/cuid
Successfully generated: /tmp/priv_cuid
Discovered 290 device(s)

CUID registry refreshed successfully!
```

Users should note that a new key will necessarily create new **derived CUIDs** for all the devices. The **primary CUIDs** for devices will always remain the same however. For more information about **primary** and **derived** ids, refer to [What is CUID](../conceptual/what_is_cuid.md).

## Getting CUIDs

Once CUIDs have been generated, either by the daemon or by the CLI tool, users can then get or list CUIDs for their devices. To list all the CUIDs for all devices on the system, use the `--list` option:

```shell-session
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
...
---- GPU Devices ----
GPU #0
  CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
  Device Path:    /sys/class/drm/renderD175

GPU #1
  CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
  Device Path:    /sys/class/drm/renderD158
...
---- NIC Devices ----
NIC #0
  CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
  Device Path:    /sys/class/net/ens14np0

NIC #1
  CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
  Device Path:    /sys/class/net/docker0
```

By default, only the **derived CUIDs** are displayed. Viewing **primary CUIDs** requires root privileges in order to protect potentially sensitive hardware information. To view **primary CUIDs**, users should additionally add `sudo` and the `--show-primary` option on to the command like so:

```shell-session
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
...
```

Users may also obtain the CUID of a specific device by using the `--query-device` option. Users can either provide a BDF or a device path as depicted below:

```shell-session
$ amdcuid_tool --query-device /sys/class/drm/renderD188
Device Found:
  Type:           GPU
  CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
  Device Path:    /sys/class/drm/renderD188
```

As before, to view the **primary CUID** of a device, users should add the `--show-primary` option and root privileges are required:

```shell-session
$ sudo amdcuid_tool --query-device 0000:0c:00.0 --show-primary
Device Found:
  Type:           GPU
  Primary CUID:   YYYYYYYY-YYYY-YYYY-YYYY-YYYYYYYYYYYY
  CUID:           XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
  Device Path:    /sys/class/drm/renderD128
```
