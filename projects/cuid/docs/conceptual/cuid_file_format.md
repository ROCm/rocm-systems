---
myst:
  html_meta:
    "description lang=en": "CUID conceptual guide for what CUID files look like"
    "keywords": "component, unified, identifier, primary, derived, file, format"
---

# CUID File

Privileged applications and users can generate and share derived CUIDs with non-privileged users via files (the **CUID file**) or IPC. The files, one for unprivileged users and one for privileged users to protect the sensitive device informatio, are stored in `/tmp` or a `tmpfs` directory. They are updated by a daemon, boot script, or cron job.

## Example CUID File Format

```ini
[GPU:0]
derived_cuid=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
vendor_id=1022
device_id=1974
pci_class=0004
revision_id=01
unit_id=0002
device_node=/sys/class/drm/renderD128
last_update=1753987166

[CPU:0]
derived_cuid=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
vendor_id=1022
device_id=1974
revision_id=01
family=0019
model=0074
unit_id=0004
package_core_id=0:0
last_update=6980d761

[NIC:0]
derived_cuid=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
vendor_id=1414
device_id=0003
pci_class=0005
revision_id=10
device_node=/sys/class/net/enp131s0
last_update=1753987166

[PLATFORM]
derived_cuid=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
vendor_id=1573
last_update=1753987166
```

- **GPU:** `device_node` links the CUID to hardware.
- **CPU:** `package_core_id` identifies cores (`package_id:core_id`).
- **NIC:** `device_node` links to the network device.
- **PLATFORM:** Only the derived CUID is listed.

`last_update` is a Unix timestamp for the last modification.

## Including Primary CUID (Privileged Users)

The privileged CUID file will also contain the primary CUID and some sensitive device information, primarily the serial number (labeled as hardware fingerprint) Privileged users can use the privileged CUID file to map `primary_cuid` to `derived_cuid`:

```ini
[GPU:0]
primary_cuid=YYYYYYYY-YYYY-YYYY-YYYY-YYYYYYYYYYYY
derived_cuid=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
hardware_fingerprint=ZZZZZZZZ
vendor_id=1022
device_id=1974
pci_class=0004
revision_id=01
unit_id=0002
device_node=/sys/class/drm/renderD128
last_update=1753987166
```

> **Note:** The file contains sensitive information. Restrict access to `primary_cuid`.