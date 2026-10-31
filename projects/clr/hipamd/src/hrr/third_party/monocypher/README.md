# Monocypher (vendored)

Monocypher 4.0.2, https://monocypher.org, tarball
`monocypher-4.0.2.tar.gz` (sha256 `38d07179738c0c90677dba3ceb7a7b8496bcfea758ba1a53e803fed30ae0879c`).

Licence: BSD-2-Clause OR CC0-1.0 (see `LICENCE.md`). No dependencies.

Copied unmodified from the tarball:

- `src/monocypher.c`, `src/monocypher.h` -> here
- `src/optional/monocypher-ed25519.c`, `.h` -> here (the RFC 8032 Ed25519 with SHA-512;
  plain Monocypher EdDSA uses BLAKE2b and does not interoperate with other Ed25519 code)
- `LICENCE.md`

To update, replace the four files from a newer tarball and compare with `diff -r`.
Used only by `hrr_archive_sign.cpp`; not installed.
