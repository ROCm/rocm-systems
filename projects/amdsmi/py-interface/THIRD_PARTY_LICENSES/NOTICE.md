Third-Party Notices for the amdsmi Python Wheel
===============================================

`libamd_smi_python.so` is MIT-licensed (see `amdsmi/LICENSE`) but links at
runtime against the Netlink libraries. When the wheel is repaired for the
manylinux platform tag (`tools/build_wheel.py --repair`), `auditwheel repair`
copies the shared libraries that are not on the manylinux baseline allowlist
into `amdsmi.libs/` so the wheel is self-contained. That currently copies three
third-party shared objects:

| Bundled file          | Upstream project                  | Upstream source                  | SPDX license        |
|-----------------------|-----------------------------------|----------------------------------|---------------------|
| `libnl-3.so.200`      | libnl                             | https://github.com/thom311/libnl | `LGPL-2.1-only`     |
| `libnl-genl-3.so.200` | libnl (generic-netlink component) | https://github.com/thom311/libnl | `LGPL-2.1-only`     |
| `libmnl.so.0`         | libmnl (Netfilter)                | https://git.netfilter.org/libmnl | `LGPL-2.1-or-later` |

Which exact build is bundled depends on the host that ran `auditwheel repair`;
the wheel's own `amdsmi.libs/` directory records what shipped in that release.

None of the three are modified from upstream — `auditwheel` copies them
verbatim from the build host's packages. `amdsmi` links to them dynamically
(`DT_NEEDED`, not static linking or source inclusion), which keeps this inside
LGPL v2.1 Section 6 ("Combined Works"). Section 6 still requires:

1. reproducing the license text and copyright notices — see `LGPL-2.1.txt` in
   this directory, the verbatim text published at
   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt; and
2. accompanying the work with the library's corresponding source, or a written
   offer valid for at least three years to provide it.

Source availability / written offer
-----------------------------------

libnl and libmnl are distributed as source by their upstream projects at the
URLs above and by every Linux distribution that packages them. As a written
offer under LGPL v2.1 Section 6(a): for three years from the publication date
of a given `amdsmi` wheel, AMD will, on written request to
amd-smi.support@amd.com identifying the wheel version and the SONAME of the
bundled library, either point the requester to the exact upstream source
release matching that build or provide a copy of it.

Copyright notices
-----------------

- libnl: Copyright (c) 2003-2012 Thomas Graf <tgraf@suug.ch> and contributors.
- libmnl: Copyright (c) 2008-2010 Pablo Neira Ayuso <pablo@netfilter.org> and
  contributors.

See each upstream repository for the full list of copyright holders.
