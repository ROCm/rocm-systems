---
myst:
  html_meta:
    "description lang=en": "AMD CUID library documentation and API reference."
    "keywords": "amdcuid, cuid, compute, unit, id, identifier, library, api, amd"
---

# AMD CUID documentation

The AMD Compute Unit ID (CUID) library provides a stable, unique identifier for
AMD hardware devices — including GPUs, CPUs, NICs, and platform devices. CUIDs
are formatted as UUIDv8 values and are derived from hardware fingerprints using
HMAC, allowing for consistent device identification across reboots and driver
upgrades.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Install
* [Build and install from source](./install/install.md)
:::

:::{grid-item-card} Reference
* [C API](./reference/amdcuid-c-api.md)
:::

::::
