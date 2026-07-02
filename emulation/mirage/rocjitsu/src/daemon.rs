//! In-process rocjitsu **daemon** — a Rust reimplementation of the
//! `rocjitsu --daemon` server, driven directly through the rocjitsu C
//! API ([`rocjitsu_sys`]) instead of the `rocjitsu` command-line tool.
//!
//! # What it does
//!
//! The mirage per-node host starts one of these on the node it serves.
//! It:
//!
//! 1. Loads `librocjitsu.so` (which exports the full `rj_vm_*` API)
//!    and creates a VM in [`RjVmMode::Daemon`] mode from the synthesised
//!    `SimulationConfig`. In daemon mode every GPU allocation is backed
//!    by a `memfd` so it can be shared with the workload process.
//! 2. Spawns the simulation engine on its own thread (`rj_vm_run`).
//! 3. Binds a Unix domain socket at `<runtime_dir>/daemon.sock` — the
//!    exact path the rocjitsu KMD interposer probes first when a
//!    workload opens `/dev/kfd` — and serves the daemon RPC protocol.
//!
//! Because the interposer connects to the daemon socket *before* falling
//! back to in-process (local) emulation, simply standing this server up
//! at the workload's `ROCJITSU_RUNTIME_DIR` switches the workload to
//! daemon-served emulation with no change to the injected environment.
//!
//! # Protocol
//!
//! The wire format mirrors `rocjitsu/lib/rocjitsu/src/rocjitsu/kmd/linux/rpc.h`:
//! a fixed 16-byte header followed by an opcode-specific payload, with
//! GPU `memfd`s passed as `SCM_RIGHTS` ancillary data. This server is
//! byte-compatible with the upstream C daemon and interposer.

use std::ffi::CString;
use std::os::raw::c_void;
use std::os::unix::io::RawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use mirage_core::emulator::EmulatorDaemon;
use rocjitsu_sys::{Lib, RjVm, RjVmCmd, RjVmGpuInfo, RjVmMap, RjVmMode, RjVmUnmap};

/// RPC opcodes (must match `enum RpcOpcode` in `rpc.h`).
const RPC_HANDSHAKE: u16 = 0;
#[allow(dead_code)]
const RPC_OPEN: u16 = 1;
const RPC_CLOSE: u16 = 2;
const RPC_MMAP: u16 = 3;
const RPC_MUNMAP: u16 = 4;
const RPC_IOCTL: u16 = 5;

/// RPC protocol version (`kRpcProtocolVersion` in `rpc.h`).
const RPC_PROTOCOL_VERSION: u32 = 3;

/// Size of the fixed RPC header in bytes.
const RPC_HEADER_LEN: usize = 16;

/// Size of the fixed `RpcHandshakeResponse` payload: four `u32` fields
/// (version, gpu_id, topology_path_len, drm_path_len) followed by the
/// 312-byte `RpcGpuInfo`. Must equal `sizeof(RpcHandshakeResponse)`
/// (== 328, asserted in `rpc.h`).
const RPC_HANDSHAKE_RESPONSE_LEN: usize = 16 + std::mem::size_of::<RjVmGpuInfo>();

/// Upper bound on an ioctl payload, mirroring the C daemon's guard
/// against a malicious or corrupt client.
const MAX_IOCTL_PAYLOAD: u32 = 16 * 1024 * 1024;

/// The VM pointer plus the loaded library, shared (read-only) across the
/// engine thread and every client thread.
///
/// rocjitsu serialises access to the VM internally (the daemon shares
/// one VM across all client connections via the `*_as` API), so sharing
/// the raw pointer across threads is sound.
struct Shared {
    lib: Lib,
    vm: *mut RjVm,
}

// Safety: see the type doc — the VM is internally synchronised and the
// resolved `Lib` entry points are plain function pointers.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// A running rocjitsu daemon. Dropping it (or calling
/// [`EmulatorDaemon::stop`]) tears the server down cleanly: it stops
/// accepting, unblocks and joins all client threads, stops the engine,
/// destroys the VM, and removes the socket.
pub struct Daemon {
    shared: Arc<Shared>,
    listen_fd: RawFd,
    socket_path: PathBuf,
    stop: Arc<AtomicBool>,
    /// fds of currently-connected clients, so shutdown can unblock them.
    clients: Arc<Mutex<Vec<RawFd>>>,
    accept_thread: Option<JoinHandle<()>>,
    engine_thread: Option<JoinHandle<()>>,
}

impl Daemon {
    /// Path of the Unix socket this daemon listens on.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Load `lib_path`, create a daemon-mode VM from `config_path`, and
    /// start serving on `<runtime_dir>/daemon.sock`.
    ///
    /// On success the engine is running and the socket is accepting
    /// connections. Returns a human-readable error otherwise.
    pub fn start(
        lib_path: &Path,
        config_path: &Path,
        runtime_dir: &Path,
    ) -> std::result::Result<Self, String> {
        // Load the rocjitsu library and create the VM in daemon mode.
        let lib = unsafe { Lib::open(lib_path) }
            .map_err(|e| format!("rocjitsu daemon: cannot load {}: {e}", lib_path.display()))?;
        let cfg = CString::new(config_path.as_os_str().as_encoded_bytes())
            .map_err(|e| format!("rocjitsu daemon: invalid config path: {e}"))?;
        let (status, vm) = unsafe { lib.vm_create(&cfg, RjVmMode::Daemon) };
        if status != rocjitsu_sys::ROCJITSU_STATUS_SUCCESS || vm.is_null() {
            return Err(format!(
                "rocjitsu daemon: rj_vm_create({}) failed with status {status}",
                config_path.display()
            ));
        }
        let shared = Arc::new(Shared { lib, vm });

        // Bind the listening socket *before* spawning the engine so a
        // bind failure leaves nothing to tear down but the VM.
        let socket_path = runtime_dir.join("daemon.sock");
        let listen_fd = match bind_listen(&socket_path) {
            Ok(fd) => fd,
            Err(e) => {
                // Roll back the VM we just created.
                unsafe {
                    let reason = CString::new("bind failed").unwrap();
                    shared.lib.vm_request_exit(shared.vm, &reason);
                    shared.lib.vm_destroy(shared.vm);
                }
                return Err(e);
            }
        };

        // Run the simulation engine. `rj_vm_run` blocks until
        // `rj_vm_request_exit` is called from `stop`/`drop`.
        let engine_shared = shared.clone();
        let engine_thread = std::thread::Builder::new()
            .name("rocjitsu-engine".to_string())
            .spawn(move || {
                unsafe { engine_shared.lib.vm_run(engine_shared.vm) };
            })
            .map_err(|e| format!("rocjitsu daemon: cannot spawn engine thread: {e}"))?;

        let stop = Arc::new(AtomicBool::new(false));
        let clients = Arc::new(Mutex::new(Vec::new()));
        let accept_shared = shared.clone();
        let accept_stop = stop.clone();
        let accept_clients = clients.clone();
        let accept_thread = std::thread::Builder::new()
            .name("rocjitsu-accept".to_string())
            .spawn(move || {
                accept_loop(listen_fd, accept_shared, accept_stop, accept_clients);
            })
            .map_err(|e| format!("rocjitsu daemon: cannot spawn accept thread: {e}"))?;

        tracing::info!(
            socket = %socket_path.display(),
            config = %config_path.display(),
            "rocjitsu daemon started"
        );

        Ok(Self {
            shared,
            listen_fd,
            socket_path,
            stop,
            clients,
            accept_thread: Some(accept_thread),
            engine_thread: Some(engine_thread),
        })
    }

    /// Tear the daemon down. Idempotent; called from both
    /// [`EmulatorDaemon::stop`] and [`Drop`].
    fn teardown(&mut self) {
        if self.stop.swap(true, Ordering::SeqCst) {
            // Already torn down.
            return;
        }
        // Stop accepting new connections and unblock the accept thread.
        unsafe { libc::shutdown(self.listen_fd, libc::SHUT_RDWR) };
        // Unblock any in-flight client recv()s so their threads exit.
        if let Ok(fds) = self.clients.lock() {
            for &fd in fds.iter() {
                unsafe { libc::shutdown(fd, libc::SHUT_RDWR) };
            }
        }
        // The accept thread joins all client threads before returning.
        if let Some(t) = self.accept_thread.take() {
            let _ = t.join();
        }
        unsafe { libc::close(self.listen_fd) };
        let _ = std::fs::remove_file(&self.socket_path);

        // Stop the engine and reclaim the VM.
        unsafe {
            let reason = CString::new("daemon shutdown").unwrap();
            self.shared.lib.vm_request_exit(self.shared.vm, &reason);
        }
        if let Some(t) = self.engine_thread.take() {
            let _ = t.join();
        }
        unsafe { self.shared.lib.vm_destroy(self.shared.vm) };
        tracing::info!(socket = %self.socket_path.display(), "rocjitsu daemon stopped");
    }
}

impl EmulatorDaemon for Daemon {
    fn stop(mut self: Box<Self>) {
        self.teardown();
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.teardown();
    }
}

/// Create, bind, and listen on the daemon Unix socket at `path`.
fn bind_listen(path: &Path) -> std::result::Result<RawFd, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("rocjitsu daemon: cannot create {}: {e}", parent.display()))?;
    }
    let path_bytes = path.as_os_str().as_encoded_bytes();
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    if path_bytes.len() >= std::mem::size_of_val(&addr.sun_path) {
        return Err(format!(
            "rocjitsu daemon: socket path too long ({} bytes): {}",
            path_bytes.len(),
            path.display()
        ));
    }
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (dst, &src) in addr.sun_path.iter_mut().zip(path_bytes.iter()) {
        *dst = src as libc::c_char;
    }

    // A stale socket from a previous run would make bind() fail with
    // EADDRINUSE, so remove it first.
    let _ = std::fs::remove_file(path);

    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(format!(
            "rocjitsu daemon: socket() failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let addr_len = std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
    let rc = unsafe {
        libc::bind(
            fd,
            &addr as *const libc::sockaddr_un as *const libc::sockaddr,
            addr_len,
        )
    };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(format!(
            "rocjitsu daemon: bind({}) failed: {err}",
            path.display()
        ));
    }
    if unsafe { libc::listen(fd, 16) } != 0 {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(format!("rocjitsu daemon: listen() failed: {err}"));
    }
    Ok(fd)
}

/// Accept connections until `stop` is set (signalled by shutting the
/// listening socket down), spawning one thread per client and joining
/// them all before returning.
fn accept_loop(
    listen_fd: RawFd,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    clients: Arc<Mutex<Vec<RawFd>>>,
) {
    let mut handles: Vec<JoinHandle<()>> = Vec::new();
    loop {
        let client = unsafe { libc::accept(listen_fd, std::ptr::null_mut(), std::ptr::null_mut()) };
        if client < 0 {
            break;
        }
        if stop.load(Ordering::SeqCst) {
            unsafe { libc::close(client) };
            break;
        }
        if let Ok(mut fds) = clients.lock() {
            fds.push(client);
        }
        let client_shared = shared.clone();
        let client_list = clients.clone();
        match std::thread::Builder::new()
            .name("rocjitsu-client".to_string())
            .spawn(move || {
                handle_client(client, &client_shared);
                // Drop our fd from the live set once we're done so
                // shutdown does not race a closed/reused descriptor.
                if let Ok(mut fds) = client_list.lock() {
                    fds.retain(|&f| f != client);
                }
                unsafe { libc::close(client) };
            }) {
            Ok(h) => handles.push(h),
            Err(_) => unsafe {
                libc::close(client);
            },
        }
    }
    for h in handles {
        let _ = h.join();
    }
}

/// Return the connecting client's OS PID via `SO_PEERCRED`, or 0 when it
/// cannot be determined. The rocjitsu daemon needs the real client PID to
/// access the workload's address space for cross-process GPU memory.
fn peer_pid(fd: RawFd) -> i32 {
    let mut cred = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    let rc = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut cred as *mut libc::ucred).cast(),
            &mut len,
        )
    };
    if rc == 0 && cred.pid > 0 {
        cred.pid
    } else {
        0
    }
}

/// Serve a single client connection until it closes or errors. Mirrors
/// the C daemon's `handle_client`.
fn handle_client(fd: RawFd, shared: &Shared) {
    let lib = &shared.lib;
    let vm = shared.vm;
    let mut process_id: u32 = 0;
    // The client's real OS PID, needed by the VM for daemon-mode
    // cross-process memory access (see `rj_vm_device_open`).
    let client_pid = peer_pid(fd);

    loop {
        let mut header = [0u8; RPC_HEADER_LEN];
        if !recv_exact(fd, &mut header) {
            break;
        }
        let (opcode, request_id, payload_bytes) = parse_header(&header);

        let keep_going = match opcode {
            RPC_HANDSHAKE => {
                let (status, pid) = unsafe { lib.vm_device_open(vm, client_pid) };
                if status != rocjitsu_sys::ROCJITSU_STATUS_SUCCESS {
                    let resp = build_header(0, request_id, 0, -1);
                    send_exact(fd, &resp);
                    false
                } else {
                    process_id = pid;
                    let (_s, gpu_id) = unsafe { lib.vm_gpu_id(vm) };
                    let topo = unsafe { lib.vm_topology_path(vm) }
                        .map(|c| c.to_bytes().to_vec())
                        .unwrap_or_default();
                    let drm = unsafe { lib.vm_drm_path(vm) }
                        .map(|c| c.to_bytes().to_vec())
                        .unwrap_or_default();
                    // Device metadata for the client's libdrm/DRM
                    // emulation. A zeroed payload (present == 0) is a
                    // valid fallback for libraries without the symbol.
                    let gpu_info = unsafe { lib.vm_gpu_info(vm) }.unwrap_or_default();
                    let payload = RPC_HANDSHAKE_RESPONSE_LEN + topo.len() + drm.len();
                    let mut msg = Vec::with_capacity(RPC_HEADER_LEN + payload);
                    msg.extend_from_slice(&build_header(0, request_id, payload as u32, 0));
                    // RpcHandshakeResponse: version, gpu_id, topo_len,
                    // drm_len, gpu_info, then the topo/drm path strings.
                    msg.extend_from_slice(&RPC_PROTOCOL_VERSION.to_ne_bytes());
                    msg.extend_from_slice(&gpu_id.to_ne_bytes());
                    msg.extend_from_slice(&(topo.len() as u32).to_ne_bytes());
                    msg.extend_from_slice(&(drm.len() as u32).to_ne_bytes());
                    msg.extend_from_slice(gpu_info.as_bytes());
                    msg.extend_from_slice(&topo);
                    msg.extend_from_slice(&drm);
                    send_exact(fd, &msg)
                }
            }

            RPC_CLOSE => {
                unsafe { lib.vm_device_close(vm, process_id) };
                process_id = 0;
                let resp = build_header(0, request_id, 0, 0);
                send_exact(fd, &resp);
                false
            }

            RPC_MMAP => {
                let mut req = [0u8; 32];
                if !recv_exact(fd, &mut req) {
                    false
                } else {
                    let mut map = RjVmMap {
                        addr: u64::from_ne_bytes(req[0..8].try_into().unwrap()),
                        length: u64::from_ne_bytes(req[8..16].try_into().unwrap()),
                        prot: i32::from_ne_bytes(req[16..20].try_into().unwrap()) as u32,
                        flags: i32::from_ne_bytes(req[20..24].try_into().unwrap()) as u32,
                        offset: i64::from_ne_bytes(req[24..32].try_into().unwrap()),
                        mapped_addr: 0,
                    };
                    let offset = map.offset;
                    unsafe { lib.vm_device_map_as(vm, process_id, &mut map) };
                    let result = if map.mapped_addr == u64::MAX {
                        -last_errno()
                    } else {
                        0
                    };
                    // Header + RpcMmapResponse{mapped_addr}.
                    let mut msg = Vec::with_capacity(RPC_HEADER_LEN + 8);
                    msg.extend_from_slice(&build_header(0, request_id, 8, result));
                    msg.extend_from_slice(&map.mapped_addr.to_ne_bytes());
                    match unsafe { lib.vm_get_shared_mem_as(vm, process_id, offset) } {
                        Some(memfd) => send_msg(fd, &msg, &[memfd]),
                        None => send_exact(fd, &msg),
                    }
                }
            }

            RPC_MUNMAP => {
                let mut req = [0u8; 16];
                if !recv_exact(fd, &mut req) {
                    false
                } else {
                    let mut unmap = RjVmUnmap {
                        addr: u64::from_ne_bytes(req[0..8].try_into().unwrap()),
                        length: u64::from_ne_bytes(req[8..16].try_into().unwrap()),
                    };
                    unsafe { lib.vm_device_unmap_as(vm, process_id, &mut unmap) };
                    let resp = build_header(0, request_id, 0, 0);
                    send_exact(fd, &resp)
                }
            }

            RPC_IOCTL => {
                if payload_bytes > MAX_IOCTL_PAYLOAD || (payload_bytes as usize) < 8 {
                    false
                } else {
                    let mut payload = vec![0u8; payload_bytes as usize];
                    if !recv_exact(fd, &mut payload) {
                        false
                    } else {
                        let ioctl_cmd = u32::from_ne_bytes(payload[0..4].try_into().unwrap());
                        let args_bytes = u32::from_ne_bytes(payload[4..8].try_into().unwrap());
                        let mut cmd = RjVmCmd {
                            cmd: ioctl_cmd,
                            buf: payload[8..].as_mut_ptr() as *mut c_void,
                            buf_size: args_bytes as usize,
                            result: 0,
                            shared_handle: -1,
                        };
                        unsafe { lib.vm_execute_as(vm, process_id, &mut cmd) };
                        // `buf_size` is updated in place; clamp the slice
                        // we read back to what the payload actually holds.
                        let out_len = cmd.buf_size.min(payload.len().saturating_sub(8));
                        let resp =
                            build_header(RPC_IOCTL, request_id, cmd.buf_size as u32, cmd.result);
                        if cmd.shared_handle >= 0 {
                            let mut msg = Vec::with_capacity(RPC_HEADER_LEN + out_len);
                            msg.extend_from_slice(&resp);
                            msg.extend_from_slice(&payload[8..8 + out_len]);
                            send_msg(fd, &msg, &[cmd.shared_handle])
                        } else if send_exact(fd, &resp) {
                            out_len == 0 || send_exact(fd, &payload[8..8 + out_len])
                        } else {
                            false
                        }
                    }
                }
            }

            _ => false,
        };

        if !keep_going {
            break;
        }
    }

    if process_id != 0 {
        unsafe { lib.vm_device_close(vm, process_id) };
    }
}

/// Build a 16-byte RPC header.
fn build_header(opcode: u16, request_id: u32, payload_bytes: u32, result: i32) -> [u8; 16] {
    let mut h = [0u8; 16];
    h[0..2].copy_from_slice(&opcode.to_ne_bytes());
    // bytes 2..4 reserved (zero)
    h[4..8].copy_from_slice(&request_id.to_ne_bytes());
    h[8..12].copy_from_slice(&payload_bytes.to_ne_bytes());
    h[12..16].copy_from_slice(&result.to_ne_bytes());
    h
}

/// Parse the `(opcode, request_id, payload_bytes)` fields from a header.
fn parse_header(h: &[u8; 16]) -> (u16, u32, u32) {
    let opcode = u16::from_ne_bytes(h[0..2].try_into().unwrap());
    let request_id = u32::from_ne_bytes(h[4..8].try_into().unwrap());
    let payload_bytes = u32::from_ne_bytes(h[8..12].try_into().unwrap());
    (opcode, request_id, payload_bytes)
}

/// `errno` from the most recent failing libc call.
fn last_errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// Read exactly `buf.len()` bytes, handling partial reads. Returns false
/// on EOF/error.
fn recv_exact(fd: RawFd, buf: &mut [u8]) -> bool {
    let mut read = 0;
    while read < buf.len() {
        let n = unsafe {
            libc::recv(
                fd,
                buf[read..].as_mut_ptr() as *mut c_void,
                buf.len() - read,
                0,
            )
        };
        if n < 0 && last_errno() == libc::EINTR {
            continue;
        }
        if n <= 0 {
            return false;
        }
        read += n as usize;
    }
    true
}

/// Write exactly `buf.len()` bytes, handling partial writes. Returns
/// false on error.
fn send_exact(fd: RawFd, buf: &[u8]) -> bool {
    let mut sent = 0;
    while sent < buf.len() {
        let n = unsafe {
            libc::send(
                fd,
                buf[sent..].as_ptr() as *const c_void,
                buf.len() - sent,
                libc::MSG_NOSIGNAL,
            )
        };
        if n < 0 && last_errno() == libc::EINTR {
            continue;
        }
        if n <= 0 {
            return false;
        }
        sent += n as usize;
    }
    true
}

/// Send a message together with `fds` passed as `SCM_RIGHTS` ancillary
/// data (a single `sendmsg`). Used to hand GPU `memfd`s to the workload.
fn send_msg(fd: RawFd, data: &[u8], fds: &[RawFd]) -> bool {
    if fds.is_empty() {
        return send_exact(fd, data);
    }
    let mut iov = libc::iovec {
        iov_base: data.as_ptr() as *mut c_void,
        iov_len: data.len(),
    };
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    let fd_bytes = std::mem::size_of_val(fds);
    let cmsg_space = unsafe { libc::CMSG_SPACE(fd_bytes as u32) } as usize;
    let mut cmsg_buf = vec![0u8; cmsg_space];

    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut c_void;
    msg.msg_controllen = cmsg_space as _;

    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        if cmsg.is_null() {
            return false;
        }
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(fd_bytes as u32) as _;
        std::ptr::copy_nonoverlapping(fds.as_ptr() as *const u8, libc::CMSG_DATA(cmsg), fd_bytes);
        let n = libc::sendmsg(fd, &msg, libc::MSG_NOSIGNAL);
        n >= 0
    }
}
