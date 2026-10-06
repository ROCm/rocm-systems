Third-Party Notices for the amdsmi Python Wheel
===============================================

The Python files and `libamd_smi_python.so` in this wheel are licensed under
the MIT License (see `amdsmi/LICENSE`). The wheel also contains the following
third-party components. Their license texts are in this directory.

Compiled into libamd_smi_python.so
----------------------------------

| Component | Source | License | License text |
|-----------|--------|---------|--------------|
| shared_mutex | `third_party/shared_mutex` in the AMD SMI source tree | MIT; Copyright (c) 2018 Oleg Yamnikov | `shared_mutex-LICENSE.txt` |
| E-SMI In-Band Library | https://github.com/amd/esmi_ib_library, commit `d494a3194ceb4cc4dbb2debf9fcbe8773c6d3bef` | University of Illinois/NCSA Open Source License; Copyright (c) 2020-2023, Advanced Micro Devices, Inc. | `E-SMI-LICENSE.txt` |

Shared libraries in amdsmi.libs/
--------------------------------

The wheel build (`auditwheel repair`) copies these netlink libraries from the
AlmaLinux 8 packages installed in the build container:

| File in the wheel | Library | Package | License |
|-------------------|---------|---------|---------|
| `amdsmi.libs/libnl-3-47cd035c.so.200.26.0` | libnl 3.7.0 (core) | `libnl3-3.7.0-1.el8` | `LGPL-2.1-only` |
| `amdsmi.libs/libnl-genl-3-999b815b.so.200.26.0` | libnl 3.7.0 (generic netlink) | `libnl3-3.7.0-1.el8` | `LGPL-2.1-only` |
| `amdsmi.libs/libmnl-1831b10a.so.0.2.0` | libmnl 1.0.4 | `libmnl-1.0.4-6.el8` | `LGPL-2.1-or-later` |

The LGPL 2.1 text is in `LGPL-2.1.txt`. Copyright notices:

- libnl: Copyright (c) 2003-2012 Thomas Graf <tgraf@suug.ch> and contributors.
- libmnl: Copyright (c) 2008-2010 Pablo Neira Ayuso <pablo@netfilter.org> and
  contributors.

auditwheel does not change the code of these libraries. It adds a hash to each
file name and SONAME, updates the references between the three libraries to
those names, and sets the run-time search path (RPATH) of libnl-genl-3 to
`$ORIGIN`. `libamd_smi_python.so` links to them dynamically, so you can use
your own build of a library by saving it in `amdsmi.libs/` under the same file
name.

Source code
-----------

The corresponding source code is in these AlmaLinux 8 source packages, which
include the distribution's patches:

- https://vault.almalinux.org/8.10/BaseOS/Source/Packages/libnl3-3.7.0-1.el8.src.rpm
- https://vault.almalinux.org/8.10/BaseOS/Source/Packages/libmnl-1.0.4-6.el8.src.rpm

For at least three years after AMD last distributes this version of the wheel,
AMD will also give anyone who asks a complete copy of that source code, for no
more than the cost of performing the distribution. Send requests to
amd-smi.support@amd.com and include the amdsmi version.
