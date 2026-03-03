# AMDSMI Exporter

The `AMDSMI Exporter` is a tool that exports AMD SMI metrics and runs an HTTP server to serve these metrics. It provides options to list supported GPUs and metrics' information, as well as to export metrics with optional filtering using whitelist and blacklist. This enables fine-grained control over which metrics are included or excluded.

## Features

- List supported GPUs and detailed metrics information.
- Export AMD SMI metrics and serve them via an HTTP server.
- Filter metrics using whitelist and blacklist.
- Filter metrics by GPU BDF or ID.

## Usage

### List Supported GPUs and Metrics

To list the supported GPUs and metrics' information, use the `list` subcommand:

```sh
amdsmi_exporter list
```

### Export Metrics and Run HTTP Server

To export AMD SMI metrics and run the HTTP server, use the `metric` subcommand:

```sh
amdsmi_exporter metric [OPTIONS]
```

#### Options

- `-p, --port <PORT>`: Sets the port to listen on (default: 9898).
- `-g, --gpu <GPU>`: Sets the GPU BDF or the zero-based index of the socket ID and processor ID for export.
  Example:
  - `--gpu 0000:03:00.0`
  - `--gpu 0.0`
- `-w, --whitelist <METRIC>`: Comma-separated list of metric field names to include.
- `-b, --blacklist <METRIC>`: Comma-separated list of metric field names to exclude.

### Examples

#### List Supported GPUs and Metrics

```sh
amdsmi_exporter list
```

#### Export Metrics and Run HTTP Server on Port 8080 for a Specific GPU

```sh
amdsmi_exporter metric --port 8080 --gpu 0000:03:00.0
```

#### Export Metrics with Whitelist Filter

```sh
amdsmi_exporter metric --whitelist AMDSMI_FI_GFX_ACTIVITY,AMDSMI_FI_UMC_ACTIVITY
```

#### Export Metrics with Blacklist Filter

```sh
amdsmi_exporter metric --blacklist AMDSMI_FI_GFX_VOLTAGE,AMDSMI_FI_SOC_VOLTAGE
```

### Set Up Prometheus Docker Server to Connect to This Client Exporter

To set up a Prometheus Docker server to connect to this client exporter, follow these steps:

1. **Create a Prometheus Configuration File**

   Create a file named prometheus.yml with the following content:

   ```yaml
   global:
     scrape_interval: 15s

   scrape_configs:
     - job_name: 'amdsmi_exporter'
       static_configs:
         - targets: ['localhost:9898']
   ```

   Replace `localhost:9898` with the appropriate address and port if you are running the exporter on a different host or port.

2. **Run Prometheus Docker Container**

   Run the Prometheus Docker container with the configuration file:

   ```sh
   docker run -d --name=prometheus -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
   ```

3. **Access Prometheus Web Interface**

   Open your web browser and go to `http://localhost:9090` to access the Prometheus web interface. You should see the `amdsmi_exporter` job listed under the "Targets" section.
