---
myst:
  html_meta:
    "description lang=en": "Explore the AMD SMI Go API."
    "keywords": "api, smi, lib, system, management, interface, ROCm, golang"
---

# AMD SMI Go API reference

The AMD SMI Go interface provides a convenient way to interact with AMD
hardware through a simple and accessible API. The API is compatible with Go
version 1.20 and higher and requires the AMD driver to be loaded for
initialization. Review the [prerequisites](#go_prereqs) before getting
started.

This section provides documentation for the AMD SMI Go API. Explore these
sections to understand the full scope of available functionalities and how to
implement them in your applications.

## GPU functions

```{eval-rst}
.. go-api-ref:: ../../goamdsmi.go
   :section: gpu
```

## CPU functions


```{eval-rst}
.. go-api-ref:: ../../goamdsmi.go
   :section: cpu
```
