# amdsmi UALoE telemetry preview version

## Getting Started

### Prerequisites

1. **Install mock IFoE driver**
    ```bash
    # Build and install driver in mock mode
    $ cd linux && make CFG_MOCK=1 
    $ sudo insmod ifoe.ko && sudo insmod ifoe-cfg.ko && sudo insmod ifoe-cmd.ko
    $ ls -la /dev/cbl-*
    crw------- 1 root root 10, 121 Oct 30 12:28 /dev/cbl-cfg-ifoe.cfg.0

    ```
2. **Install AMDGPU driver**
    Follow the rocm installation guide to install the AMDGPU driver: [ROCm Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)

3. **ROCm supported GPU required**
        Ensure you have at least one ROCm supported GPU installed in your system. You can check GPU compatibility at: [ROCm Hardware Support](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)
4. **Install amdsmi library**
    ```bash
    % sudo rpm -Uvh amd-smi-lib-25.4.2.99999-local.x86_64.rpm
    ```

### Building the Example

1. **Build**
    ```bash
    $ g++ /opt/rocm/share/amd_smi/example/amd_smi_fabric_example.cc -I/opt/rocm/include/amd_smi -L/opt/rocm/lib -lamd_smi -o fabric_example
    ```

2. **Run the example**
    ```bash
    sudo ./fabric_example
    ```
