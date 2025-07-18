# ROCm™ Data Center Tool (RDC) 🚀

The ROCm™ Data Center Tool (RDC) simplifies administration and addresses key infrastructure challenges in AMD GPUs within cluster and datacenter environments. RDC offers a suite of features to enhance your GPU management and monitoring.

## 🌟 Main Features

- **GPU Telemetry** 📊
- **GPU Statistics for Jobs** 📈
- **Integration with Third-Party Tools** 🔗
- **Open Source** 🛠️

> [!NOTE]
> The published documentation is available at [ROCm Data Center Tool](https://rocm.docs.amd.com/projects/rdc/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `rdc/docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## 🛠️ Installation Guide

### 📋 Prerequisites

Before setting up RDC, ensure your system meets the following requirements:

- **Supported Platforms**: RDC runs on AMD ROCm-supported platforms. Refer to the [List of Supported Operating Systems](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems) for details.
- **Dependencies**:
  - **CMake** ≥ 3.15
  - **g++** (5.4.0)
  - **Doxygen** (1.8.11)
  - **LaTeX** (pdfTeX 3.14159265-2.6-1.40.16)
  - **gRPC and protoc**
  - **libcap-dev**
  - **AMD ROCm Platform** ([GitHub](https://github.com/ROCm/ROCm))
    - **AMDSMI Library** ([GitHub](https://github.com/ROCm/amdsmi))
    - **ROCK Kernel Driver** ([GitHub](https://github.com/ROCm/ROCK-Kernel-Driver))

### 🔐 Certificate Generation

For certificate generation, refer to the [**RDC Developer Handbook (Generate Files for Authentication)**](https://rocm.docs.amd.com/projects/rdc/en/latest/install/handbook.html#generate-files-for-authentication) or consult the concise guide located at `authentication/readme.txt`.

---

## 🚀 Running RDC

RDC supports two primary modes of operation: **Standalone** and **Embedded**. Choose the mode that best fits your deployment needs.

### 🗂️ Standalone Mode

Standalone mode allows RDC to run independently with all its components installed.

1. **Start RDCD with Authentication (Monitor-Only Capabilities):**

    ```bash
    /opt/rocm/bin/rdcd
    ```

2. **Start RDCD with Authentication (Full Capabilities):**

    ```bash
    sudo /opt/rocm/bin/rdcd
    ```

3. **Start RDCD without Authentication (Monitor-Only):**

    ```bash
    /opt/rocm/bin/rdcd -u
    ```

4. **Start RDCD without Authentication (Full Capabilities):**

    ```bash
    sudo /opt/rocm/bin/rdcd -u
    ```

### 🔗 Embedded Mode

Embedded mode integrates RDC directly into your existing management tools using its library format.

- **Run RDC in Embedded Mode:**

    ```bash
    python your_management_tool.py --rdc_embedded
    ```

**Note:** Ensure that the `rdcd` daemon is not running separately when using embedded mode.

### 🛠️ Starting RDCD Using systemd

1. **Copy the Service File:**

    ```bash
    sudo cp /opt/rocm/libexec/rdc/rdc.service /etc/systemd/system/
    ```

2. **Configure Capabilities:**

    - **Full Capabilities:** Ensure the following lines are **uncommented** in `/etc/systemd/system/rdc.service`:

        ```ini
        CapabilityBoundingSet=CAP_DAC_OVERRIDE
        AmbientCapabilities=CAP_DAC_OVERRIDE
        ```

    - **Monitor-Only Capabilities:** **Comment out** the above lines to restrict RDCD to monitoring.

3. **Start the Service:**

    ```bash
    sudo systemctl start rdc
    sudo systemctl status rdc
    ```

4. **Modify RDCD Options:**

    Edit `/opt/rocm/share/rdc/conf/rdc_options.conf` to append any additional RDCD parameters.

    ```bash
    sudo nano /opt/rocm/share/rdc/conf/rdc_options.conf
    ```

    **Example Configuration:**

    ```bash
    RDC_OPTS="-p 50051 -u -d"
    ```

    - **Flags:**
        - `-p 50051` : Use port 50051
        - `-u` : Unauthenticated mode
        - `-d` : Enable debug messages

---

## 🏗️ Building RDC from Source

If you prefer to build RDC from source, follow the steps below.

### 🔧 Building gRPC and protoc

**Important:** RDC requires gRPC and protoc to be built from source as pre-built packages are not available.

1. **Install Required Tools:**

    ```bash
    sudo apt-get update
    sudo apt-get install automake make cmake g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev curl libcap-dev
    ```

2. **Clone and Build gRPC:**

    ```bash
    git clone -b v1.67.1 https://github.com/grpc/grpc --depth=1 --shallow-submodules --recurse-submodules
    cd grpc
    export GRPC_ROOT=/opt/grpc
    cmake -B build \
        -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_SHARED_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,'$ORIGIN' \
        -DCMAKE_INSTALL_PREFIX="$GRPC_ROOT" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release
    make -C build -j $(nproc)
    sudo make -C build install
    echo "$GRPC_ROOT" | sudo tee /etc/ld.so.conf.d/grpc.conf
    sudo ldconfig
    cd ..
    ```

### 🔧 Building RDC

1. **Clone the RDC Repository:**

    ```bash
    git clone https://github.com/ROCm/rdc
    cd rdc
    ```

2. **Configure the Build:**

    ```bash
    cmake -B build -DGRPC_ROOT="$GRPC_ROOT"
    ```

    - **Optional Features:**
        - **Enable ROCm Profiler:**

            ```bash
            cmake -B build -DBUILD_PROFILER=ON
            ```

        - **Enable RVS:**

            ```bash
            cmake -B build -DBUILD_RVS=ON
            ```

        - **Build RDC Library Only (without rdci and rdcd):**

            ```bash
            cmake -B build -DBUILD_STANDALONE=OFF
            ```

        - **Build RDC Library Without ROCm Run-time:**

            ```bash
            cmake -B build -DBUILD_RUNTIME=OFF
            ```

3. **Build and Install:**

    ```bash
    make -C build -j $(nproc)
    sudo make -C build install
    ```

4. **Update System Library Path:**

    ```bash
    export RDC_LIB_DIR=/opt/rocm/lib/rdc
    export GRPC_LIB_DIR="/opt/grpc/lib"
    echo "${RDC_LIB_DIR}" | sudo tee /etc/ld.so.conf.d/x86_64-librdc_client.conf
    echo "${GRPC_LIB_DIR}" | sudo tee -a /etc/ld.so.conf.d/x86_64-librdc_client.conf
    sudo ldconfig
    ```

---

## 📊 Features Overview

### 🔍 Discovery

Locate and display information about GPUs present in a compute node.

**Example:**

```bash
rdci discovery <host_name> -l
```

**Output:**

```
2 GPUs found

+-----------+----------------------------------------------+
| GPU Index | Device Information                           |
+-----------+----------------------------------------------+
| 0         | Name: AMD Radeon Instinct MI50 Accelerator   |
| 1         | Name: AMD Radeon Instinct MI50 Accelerator   |
+-----------+----------------------------------------------+
```

## 👥 Groups

#### 🖥️ GPU Groups

Create, delete, and list logical groups of GPUs.

**Create a Group:**

```bash
rdci group -c GPU_GROUP
```

**Add GPUs to Group:**

```bash
rdci group -g 1 -a 0,1
```

**List Groups:**

```bash
rdci group -l
```

**Delete a Group:**

```bash
rdci group -d 1
```

#### 🗂️ Field Groups

Manage field groups to monitor specific GPU metrics.

**Create a Field Group:**

```bash
rdci fieldgroup -c <fgroup> -f 150,155
```

**List Field Groups:**

```bash
rdci fieldgroup -l
```

**Delete a Field Group:**

```bash
rdci fieldgroup -d 1
```
> [!IMPORTANT]
>### 🛑 Monitor Errors
>
>Define fields to monitor RAS ECC counters.
>
>- **Correctable ECC Errors:**
>
>    ```bash
>    312 RDC_FI_ECC_CORRECT_TOTAL
>    ```
>
>- **Uncorrectable ECC Errors:**
>
>    ```bash
>    313 RDC_FI_ECC_UNCORRECT_TOTAL
>   ```

### 📈 Device Monitoring

Monitor GPU fields such as temperature, power usage, and utilization.

**Command:**

```bash
rdci dmon -f <field_group> -g <gpu_group> -c 5 -d 1000
```

**Sample Output:**

```
1 group found

+-----------+-------------+---------------+
| GPU Index | TEMP (m°C)  | POWER (µW)    |
+-----------+-------------+---------------+
| 0         | 25000       | 520500        |
+-----------+-------------+---------------+
```

### 📊 Job Stats

Display GPU statistics for any given workload.

**Start Recording Stats:**

```bash
rdci stats -s 2 -g 1
```

**Stop Recording Stats:**

```bash
rdci stats -x 2
```

**Display Job Stats:**

```bash
rdci stats -j 2
```

**Sample Output:**

```
Summary:
Executive Status:

Start time: 1586795401
End time: 1586795445
Total execution time: 44

Energy Consumed (Joules): 21682
Power Usage (Watts): Max: 49 Min: 13 Avg: 34
GPU Clock (MHz): Max: 1000 Min: 300 Avg: 903
GPU Utilization (%): Max: 69 Min: 0 Avg: 2
Max GPU Memory Used (bytes): 524320768
Memory Utilization (%): Max: 12 Min: 11 Avg: 12
```

### 🩺 Diagnostic

Run diagnostics on a GPU group to ensure system health.

**Command:**

```bash
rdci diag -g <gpu_group>
```

**Sample Output:**

```
No compute process:  Pass
Node topology check:  Pass
GPU parameters check:  Pass
Compute Queue ready:  Pass
System memory check:  Pass
=============== Diagnostic Details ==================
No compute process:  No processes running on any devices.
Node topology check:  No link detected.
GPU parameters check:  GPU 0 Critical Edge temperature in range.
Compute Queue ready:  Run binary search task on GPU 0 Pass.
System memory check:  Max Single Allocation Memory Test for GPU 0 Pass. CPUAccessToGPUMemoryTest for GPU 0 Pass. GPUAccessToCPUMemoryTest for GPU 0 Pass.
```

---

## 🔌 Integration with Third-Party Tools

RDC integrates seamlessly with tools like **Prometheus**, **Grafana**, and **Reliability, Availability, and Serviceability (RAS)** to enhance monitoring and visualization.

### 🐍 Python Bindings

RDC provides a generic Python class `RdcReader` to simplify telemetry gathering.

**Sample Program:**

```python
from RdcReader import RdcReader
from RdcUtil import RdcUtil
from rdc_bootstrap import *
import time

default_field_ids = [
    rdc_field_t.RDC_FI_POWER_USAGE,
    rdc_field_t.RDC_FI_GPU_UTIL
]

class SimpleRdcReader(RdcReader):
    def __init__(self):
        super().__init__(ip_port=None, field_ids=default_field_ids, update_freq=1000000)

    def handle_field(self, gpu_index, value):
        field_name = self.rdc_util.field_id_string(value.field_id).lower()
        print(f"{value.ts} {gpu_index}:{field_name} {value.value.l_int}")

if __name__ == '__main__':
    reader = SimpleRdcReader()
    while True:
        time.sleep(1)
        reader.process()
```

**Running the Example:**

```bash
# Ensure RDC shared libraries are in the library path and RdcReader.py is in PYTHONPATH
python SimpleReader.py
```

### 📈 Prometheus Plugin

The Prometheus plugin allows you to monitor events and send alerts.

**Installation:**

1. **Install Prometheus Client:**

    ```bash
    pip install prometheus_client
    ```

2. **Run the Prometheus Plugin:**

    ```bash
    python rdc_prometheus.py
    ```

3. **Verify Plugin:**

    ```bash
    curl localhost:5000
    ```

**Integration Steps:**

1. **Download and Install Prometheus:**
    - [Prometheus GitHub](https://github.com/prometheus/prometheus)

2. **Configure Prometheus Targets:**
    - Modify `prometheus_targets.json` to point to your compute nodes.

    ```json
    [
      {
        "targets": [
          "rdc_test1.amd.com:5000",
          "rdc_test2.amd.com:5000"
        ]
      }
    ]
    ```

3. **Start Prometheus with Configuration File:**

    ```bash
    prometheus --config.file=/path/to/rdc_prometheus_example.yml
    ```

4. **Access Prometheus UI:**
    - Open [http://localhost:9090](http://localhost:9090) in your browser.

### 📊 Grafana Integration

Grafana provides advanced visualization capabilities for RDC metrics.

**Installation:**

1. **Download Grafana:**
    - [Grafana Download](https://grafana.com/grafana/download)

2. **Install Grafana:**
    - Follow the [Installation Instructions](https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/).

3. **Start Grafana Server:**

    ```bash
    sudo systemctl start grafana-server
    sudo systemctl status grafana-server
    ```

4. **Access Grafana:**
    - Open [http://localhost:3000](http://localhost:3000/) in your browser and log in with the default credentials (`admin`/`admin`).

**Configuration Steps:**

1. **Add Prometheus Data Source:**
    - Navigate to **Configuration → Data Sources → Add data source → Prometheus**.
    - Set the URL to [http://localhost:9090](http://localhost:9090) and save.

2. **Import RDC Dashboard:**
    - Click the **+** icon and select **Import**.
    - Upload `rdc_grafana_dashboard_example.json` from the `python_binding` folder.
    - Select the desired compute node for visualization.

### 🛡️ Reliability, Availability, and Serviceability (RAS) Plugin

The RAS plugin enables monitoring and counting of ECC (Error-Correcting Code) errors.

**Installation:**

1. **Ensure GPU Supports RAS:**
    - The GPU must support RAS features.

2. **RDC Installation Includes RAS Library:**
    - `librdc_ras.so` is located in `/opt/rocm-4.2.0/rdc/lib`.

**Usage:**

- **Monitor ECC Errors:**

    ```bash
    rdci dmon -i 0 -e 600,601
    ```

    **Sample Output:**

    ```
    GPU     ECC_CORRECT         ECC_UNCORRECT
    0       0                   0
    ```

---
> [!IMPORTANT]
>## 🐞 Troubleshooting
>
> ### Known Issues
>#### 🛑 dmon Fields Return N/A
>
>1. **Missing Libraries:**
>    - Verify `/opt/rocm/lib/rdc/librdc_*.so` exists.
>    - Ensure all related libraries (rocprofiler, rocruntime, etc.) are present.
>
>2. **Unsupported GPU:**
>    - Most metrics work on MI300 and newer.
>    - Limited metrics on MI200.
>    - Consumer GPUs (e.g., RX6800) have fewer supported metrics.
>
>#### 🐍 dmon RocProfiler Fields Return Zeros
>
>**Error Message:**
>
>```
>terminate called after throwing an instance of 'std::runtime_error'
>  what():  hsa error code: 4104 HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources. This error may also occur when the core runtime library needs to spawn threads or create internal OS-specific events.
>Aborted (core dumped)
>```
>
>**Solution:**
>
>1. **Missing Groups:**
>    - Ensure `video` and `render` groups exist.
>
>    ```bash
>    sudo usermod -aG video,render $USER
>    ```
>
>    - Log out and log back in to apply group changes.
>
>### 🐛 Troubleshooting RDCD
>
>- **View RDCD Logs:**
>
>    ```bash
>    sudo journalctl -u rdc
>    ```
>
>- **Run RDCD with Debug Logs:**
>
>    ```bash
>    RDC_LOG=DEBUG /opt/rocm/bin/rdcd
>    ```
>
>    - **Logging Levels Supported:** ERROR, INFO, DEBUG
>
>- **Enable Additional Logging Messages:**
>
>    ```bash
>    export RSMI_LOGGING=3
>    ```

---

## 📄 License

RDC is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

## 📧 Support

For support and further inquiries, please refer to the [**ROCm Documentation**](https://rocm.docs.amd.com/projects/rdc/en/latest/) or contact the maintainers through the repository's issue tracker.
