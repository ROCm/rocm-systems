# Remote Access Tools for rocstorage

This directory contains tools for accessing rocstorage databases on remote machines,
particularly HPC clusters with restrictive security policies.

## Overview

| Tool | Description | Use Case | HPC-side Process |
|------|-------------|----------|------------------|
| `rocstorage-sshfs` | Mount remote filesystem via SSHFS | Ad-hoc analysis, development | None |
| `rocstorage-ssh-proxy` | Local HTTP server proxying queries over SSH | Interactive dashboards | None |
| `rocstorage-server` | HTTP server running on remote machine | Production, high-frequency queries | Required |

## Quick Start

### Option A: SSHFS (Simplest)

Mount the remote directory and access databases directly:

```bash
# Mount remote traces directory
./rocstorage-sshfs mount hpc-login /scratch/user/traces

# Query the database
./rocstorage-sshfs query ~/mnt/rocstorage/hpc-login/trace.db "SELECT COUNT(*) FROM rocpd_op"

# Or use sqlite3 directly (immutable mode recommended)
sqlite3 "file:~/mnt/rocstorage/hpc-login/trace.db?immutable=1" "SELECT * FROM rocpd_op LIMIT 10"

# Unmount when done
./rocstorage-sshfs unmount ~/mnt/rocstorage/hpc-login
```

**Pros:** Zero setup, works with existing tools
**Cons:** High latency for large queries

### Option B: SSH Query Proxy (Recommended for Dashboards)

Run a local HTTP server that proxies queries over SSH. This tool piggybacks on an
existing SSH ControlMaster connection, so you handle authentication separately in
your own terminal (supporting 2FA, security keys, etc.):

```bash
# Step 1: Create socket directory (one-time setup)
mkdir -p ~/.ssh/sockets

# Step 2: Establish SSH connection in your terminal
ssh -fNM -S ~/.ssh/sockets/hpc-login.sock hpc-login

# Step 3: Start the proxy (runs locally, uses existing connection)
./rocstorage-ssh-proxy --host hpc-login --db /scratch/user/trace.db --port 8080

# Query via HTTP from another terminal or browser
curl http://localhost:8080/tables
curl "http://localhost:8080/query?sql=SELECT%20COUNT(*)%20FROM%20rocpd_op"
curl -X POST http://localhost:8080/query -d '{"sql": "SELECT * FROM rocpd_op LIMIT 10"}'
```

**Pros:** User handles authentication, connection reuse, caching
**Cons:** Still has per-query SSH overhead (~50-200ms)

### Option C: Remote Server (Best Performance)

Run the server on the HPC machine, access via SSH tunnel:

```bash
# On HPC machine (or in a batch job)
./rocstorage-server --db /scratch/user/trace.db --port 8080

# On local machine - create SSH tunnel
ssh -L 8080:localhost:8080 hpc-login

# Query via HTTP
curl http://localhost:8080/tables
curl http://localhost:8080/query?sql=SELECT+COUNT(*)+FROM+rocpd_op
```

**Pros:** Best performance, supports concurrent users
**Cons:** Requires running process on HPC side

---

## Tool Reference

### rocstorage-sshfs

Mount remote filesystems via SSHFS for transparent database access.

```
Usage: rocstorage-sshfs <command> [options]

Commands:
  mount <ssh-host> <remote-path> [local-mount-point]
  unmount <local-mount-point>
  status
  query <local-db-path> <sql>

Options:
  --jump <host>       Use jump host (ProxyJump)
  --compress          Enable SSH compression
  --cache-timeout <s> Cache timeout in seconds (default: 60)
  --reconnect         Auto-reconnect on connection drop
```

**Examples:**

```bash
# Basic mount
rocstorage-sshfs mount hpc-login /scratch/user/traces

# Mount with jump host (for compute nodes)
rocstorage-sshfs mount compute-node /data/traces --jump bastion

# Mount with compression (good for slow networks)
rocstorage-sshfs mount hpc-login /scratch/user/traces --compress

# Check active mounts
rocstorage-sshfs status
```

**SQLite Tips for SSHFS:**

```bash
# Use immutable mode to avoid locking issues
sqlite3 "file:~/mnt/traces/db.sqlite?immutable=1" "SELECT * FROM rocpd_op"

# Enable read-ahead for better performance
sqlite3 -cmd "PRAGMA cache_size=-64000;" ~/mnt/traces/db.sqlite
```

---

### rocstorage-ssh-proxy

Local HTTP server that proxies SQL queries to remote SQLite via SSH. This tool
piggybacks on an existing SSH ControlMaster connection, allowing you to handle
authentication (2FA, security keys, etc.) in your own terminal.

**Setup:**
```bash
# Create socket directory (one-time)
mkdir -p ~/.ssh/sockets

# Establish master connection (do this first!)
ssh -fNM -S ~/.ssh/sockets/hpc-login.sock hpc-login
```

```
Usage: rocstorage-ssh-proxy [options]

Required:
  --host, -H <host>     SSH host to connect to
  --db, -d <path>       Path to database on remote host

Optional:
  --socket, -S <path>   Path to existing SSH ControlMaster socket
  --port, -p <port>     Local port (default: 8080)
  --bind, -b <addr>     Bind address (default: 127.0.0.1)
  --jump, -J <host>     Jump host for SSH
  --compress, -C        Enable SSH compression
  --config, -c <file>   JSON config file
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Server status and connection info |
| `/tables` | GET | List all tables |
| `/schema` | GET | Database schema |
| `/query` | GET/POST | Execute SQL query |

**Query Examples:**

```bash
# GET with query parameter
curl "http://localhost:8080/query?sql=SELECT%20*%20FROM%20rocpd_op%20LIMIT%2010"

# POST with JSON body
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT * FROM rocpd_op LIMIT 10", "cache": true}'

# POST with plain SQL
curl -X POST http://localhost:8080/query \
  -d "SELECT COUNT(*) FROM rocpd_op"
```

**Configuration File:**

```json
{
  "host": "hpc-login",
  "db": "/scratch/user/trace.db",
  "socket": "~/.ssh/sockets/hpc-login.sock",
  "port": 8080,
  "jump": "bastion",
  "compress": true
}
```

---

### rocstorage-server

Lightweight HTTP server for direct database access (runs on remote machine).

```
Usage: rocstorage-server [options]

Required:
  --db, -d <path>     Path to SQLite database

Optional:
  --port, -p <port>   Port to listen on (default: 8080)
  --bind, -b <addr>   Bind address (default: 127.0.0.1)
  --cache-ttl <sec>   Query cache TTL (default: 60)
  --read-write        Allow write operations (default: read-only)
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Server status |
| `/tables` | GET | List tables with row counts |
| `/schema` | GET | Database schema |
| `/metadata` | GET | rocstorage metadata |
| `/traces` | GET | Trace information (time range, tracks) |
| `/query` | GET/POST | Execute SQL query |
| `/cache/clear` | GET | Clear query cache |

**Query Parameters:**

- `sql` - SQL query string
- `limit` - Maximum rows to return
- `offset` - Skip first N rows
- `cache` - Use cache (default: true)

**Deployment Options:**

1. **SSH Tunnel (Recommended)**
   ```bash
   # On HPC
   rocstorage-server --db /scratch/trace.db

   # On local machine
   ssh -L 8080:localhost:8080 hpc-login
   ```

2. **Singularity/Apptainer Container**
   ```bash
   singularity exec rocstorage.sif rocstorage-server --db /data/trace.db
   ```

3. **Batch Job Script**
   ```bash
   #!/bin/bash
   #SBATCH --job-name=rocstorage-server
   #SBATCH --time=4:00:00

   echo "Server running on $(hostname):8080"
   rocstorage-server --db $SCRATCH/trace.db --port 8080
   ```

---

## SSH Configuration

For better performance with `rocstorage-ssh-proxy`, configure SSH ControlMaster:

```
# ~/.ssh/config

Host hpc-login
    HostName login.hpc.example.gov
    User myuser
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
    ServerAliveInterval 30
    ServerAliveCountMax 3
    Compression yes

Host compute-*
    ProxyJump hpc-login
    User myuser
```

Create the socket directory:
```bash
mkdir -p ~/.ssh/sockets
chmod 700 ~/.ssh/sockets
```

---

## Comparison

| Aspect | SSHFS | SSH Proxy | Remote Server |
|--------|-------|-----------|---------------|
| **Setup** | 1 command | 1 command | Requires access to remote |
| **HPC process** | None | None | Required |
| **Latency** | High (per page read) | Medium (per query) | Low |
| **Large queries** | Slow | Acceptable | Fast |
| **Caching** | OS-level | Built-in | Built-in |
| **Authentication** | Once at mount | Once at start | N/A (via tunnel) |
| **Best for** | Ad-hoc, debugging | Dashboards | Production |

---

## Troubleshooting

### SSHFS Issues

**"Transport endpoint is not connected"**
```bash
# Force unmount and remount
fusermount -uz ~/mnt/traces
rocstorage-sshfs mount hpc-login /scratch/traces
```

**Slow performance**
- Use `--compress` for slow networks
- Increase cache timeout: `--cache-timeout 300`
- For large databases, consider SSH proxy instead

### SSH Proxy Issues

**Connection drops**
- Check SSH ControlMaster configuration
- Increase `ServerAliveInterval` in SSH config

**Query timeouts**
- Increase timeout: `--timeout 120`
- Break large queries into smaller chunks

### Remote Server Issues

**"Address already in use"**
- Another process is using the port
- Use `--port` to specify different port
- Check with: `lsof -i :8080`

**Permission denied on database**
- Check file permissions: `ls -la /path/to/db`
- Ensure database isn't locked by another process

---

## Testing

### Local Tests

Run tests directly on your machine (requires sqlite3, curl, python3):

```bash
./tests/test_local.sh
```

### Docker Tests (Recommended)

Run tests in an isolated Docker container for security:

```bash
./tests/run_docker_tests.sh
```

This builds a minimal Docker image and runs the test suite with:
- Non-root user
- Read-only root filesystem
- Dropped capabilities
- No new privileges
- Isolated tmpfs for temporary files

The Docker tests verify:
- All dependencies are available
- `rocstorage-sshfs` query command works
- `rocstorage-server` HTTP endpoints function correctly
- `rocstorage-ssh-proxy` argument validation works
