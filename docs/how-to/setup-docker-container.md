---
myst:
  html_meta:
    "description lang=en": "Docker container configuration and setup procedures for AMD SMI."
    "keywords": "api, smi, lib, system, management, interface, ROCm, docker, systemd, modprobe"
---

# Using AMD SMI in a Docker container

To ensure proper functionality of AMD SMI within a Docker container, the
following configuration options must be included. These settings are
particularly important for managing memory partitions, as partitioning depends
on loading and unloading drivers (with `systemd` dependencies):

* `--cap-add=SYS_MODULE`

  This option adds the `SYS_MODULE` capability to the container, allowing it to
  load and interact with kernel modules.

   ```{note}
   Granting `SYS_MODULE` increases the container's privileges and reduces
   isolation from the host. Use this option only with trusted containers and
   images.
   ```

* `-v /lib/modules:/lib/modules`

  By mounting the `/lib/modules/` directory into the container, the container
  gains access to the host's kernel modules, allowing it to load and interact
  with them. Without this access, operations requiring module loading like
  memory partitioning would fail.

For example:

```{image} ../data/how-to/setup-docker-container/docker-run-example.jpg
:alt: Command line example of running a Docker container for AMD SMI
:align: center
:width: 100%
```
