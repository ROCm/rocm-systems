---
myst:
  html_meta:
    "description lang=en": "How to build and install CUID from source."
    "keywords": "contribute, contributing, ROCm, develop, testing, component, unified, identifier"
---

# Building CUID

This section describes the prerequisites to build and install CUID from the source.

(build_reqs)=
## Dependencies

To build CUID, the following components are required:

* CMake (v3.14 or later)
* G++ (v5.0 or later)
* OpenSSL (v1.1 or later) (For Ubuntu or Debian)
* Bcrypt (Windows Native crypto library for Windows)

Furthermore, to build the documentation, the following are required:

* Doxygen (v1.9.1 or later)
* Python 3.12

## Build and Install Steps

1. Clone the CUID repository from the rocm-systems monorepo to your local machine.

    ```sh
    git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
    cd rocm-systems
    git sparse-checkout init --cone
    git sparse-checkout set projects/cuid
    git checkout develop
    cd rocm-systems/projects/cuid
    ```

2. Build the project by following the typical CMake build sequence (run as root or use `sudo` before `make install`). Note that the default install directory is `/opt/rocm/core`, but users can choose a different directory using the `-DCMAKE_INSTALL_PREFIX` option:

    ```sh
    mkdir build
    cd build
    cmake ..
    make -j $(nproc)
    ```

3. Configure the Daemon mode by editing and setting the `daemonize` variable in the amdcuid_daemon.conf file in the `daemon` directory. Setting the `daemonize` variable to `true` will install a systemd service and a set of udev rules for which to detect devices and automatically generate CUIDs. Setting `daemonize` to `false` will instead install a cron job which will only detect devices on system boot and generate CUIDs for the devices found at boot time. The default setting is `false`.

4. Perform the install and post install tasks by running the post install script that should now be located at `<install prefix>/share/amdcuid/amdcuid_postinst.sh`. This script needs to be run as root or with `sudo` as it restarts the systemd and udev services:

    ```sh
    make install
    <install prefix>/share/amdcuid/amdcuid_postinst.sh
    ```

5. Optionally, users can build the documentation with the commands below:

    ```sh
    pip install -r docs/sphinx/requirements.txt
    sphinx-build docs docs/_build/html
    ```

If attempting to uninstall the project, users, can follow the directions below:

    ```sh
    cd build
    sudo <install prefix>/share/amdcuid/amdcuid_prerm.sh
    sudo make uninstall
    ```