# Release Notes

## Documentation

Documentation for AMDSMI-CLI is available post install in /opt/<rocm_instance>/libexec/amdsmi_cli/README.md

## AMDSMI-CLI 23.3.1.0

- not all ecc fields are currently supported
- RHEL 8 & SLES 15 may have extra install steps

## AMDSMI-CLI 23.0.1.1

### Known Issues

- not all ecc fields are currently supported
- RHEL 8 & SLES 15 have extra install steps

## AMDSMI-CLI 23.0.1.0

### Known Issues

- not all ecc fields are currently supported
- RHEL 8 & SLES 15 have extra install steps

## AMDSMI-CLI 23.0.0.4

### Added

- AMDSMI-CLI tool enabled for Linux Baremetal & Guest
- Added CSV & Watch modifier
- Added topology subcommand

### Known Issues

- not all ecc fields are currently supported
- RHEL 8 & SLES 15 have extra install steps

## AMDSMI-CLI 0.0.2

### Added

- AMDSMI-CLI tool enabled for Linux Baremetal & Guest

### Known Issues

- ecc & ras subcommands will report N/A even if RAS is enabled
- process vram_mem's unit is listed as percentage vs bytes
- csv modifier does not work
- topology information is not yet enabled
- watch modifier not fully enabled
- limited guest support
