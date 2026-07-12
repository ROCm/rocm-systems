// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//! Safe, idiomatic Rust interface to the **gpumetrics** GPU-metrics collector.
//!
//! gpumetrics is a small ROCm tool that loads backend *plugins* (amdsmi,
//! rocprofiler, ...) and exposes a unified view of GPUs / partitions / sockets
//! and the metrics they publish. This crate wraps the flat C API
//! ([`gpumetrics-sys`]) in safe Rust: a [`Collector`] owns the underlying handle
//! and frees it on drop, topology and metric metadata come back as owned Rust
//! types, and reads return a [`Sample`] carrying a typed [`Value`] plus a
//! [`Status`].
//!
//! # Example
//!
//! ```no_run
//! use gpumetrics::Collector;
//!
//! // Load the default plugins discovered on the system.
//! let collector = Collector::new()?;
//! println!("{} GPU(s)", collector.gpu_count());
//!
//! for gpu in collector.gpus() {
//!     println!("gpu {}: {} [{}]", gpu.ordinal(), gpu.name(), gpu.bdf());
//! }
//!
//! // Read a single metric off GPU 0.
//! if let Some(entity) = collector.resolve("gpu:0") {
//!     let sample = collector.read(&entity, "temp.edge");
//!     if let Some(value) = sample.value() {
//!         println!("temp.edge = {value}");
//!     }
//! }
//! # Ok::<(), gpumetrics::Status>(())
//! ```

use std::ffi::CString;
use std::fmt;
use std::os::raw::c_char;
use std::ptr;

use gpumetrics_sys as sys;

/// Maximum length of a string value returned in a [`gpum_value`](sys::gpum_value).
const STRING_MAX: usize = sys::GPUM_STRING_MAX as usize;

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/// Result status returned by the collector, mirroring `gpum_status`.
///
/// [`Status::Ok`] means success. Everything else is an error; note that
/// `Unsupported` / `NotFound` are first-class statuses (a valid request the
/// backend simply cannot serve), not magic sentinel values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    /// The operation succeeded.
    Ok,
    /// An argument was invalid.
    InvalidArg,
    /// Unknown metric key or entity.
    NotFound,
    /// Valid request, but the backend/hardware cannot provide it.
    Unsupported,
    /// The collector or a backend was not initialized.
    NotInitialized,
    /// The resource already exists.
    AlreadyExists,
    /// Transiently unavailable; retry may succeed.
    NoData,
    /// The operation timed out.
    Timeout,
    /// A backend/driver call failed.
    Backend,
    /// Plugin ABI mismatch or otherwise bad plugin.
    Abi,
    /// Internal error.
    Internal,
    /// A status code not known to this binding.
    Unknown(i32),
}

impl Status {
    /// Returns `true` if this status is [`Status::Ok`].
    pub fn is_ok(self) -> bool {
        self == Status::Ok
    }

    /// Converts an OK status into `Ok(())` and any error into `Err(self)`.
    pub fn into_result(self) -> Result<(), Status> {
        if self.is_ok() {
            Ok(())
        } else {
            Err(self)
        }
    }

    fn from_raw(s: sys::gpum_status) -> Status {
        use sys::gpum_status::*;
        match s {
            GPUM_OK => Status::Ok,
            GPUM_ERR_INVALID_ARG => Status::InvalidArg,
            GPUM_ERR_NOT_FOUND => Status::NotFound,
            GPUM_ERR_UNSUPPORTED => Status::Unsupported,
            GPUM_ERR_NOT_INITIALIZED => Status::NotInitialized,
            GPUM_ERR_ALREADY_EXISTS => Status::AlreadyExists,
            GPUM_ERR_NO_DATA => Status::NoData,
            GPUM_ERR_TIMEOUT => Status::Timeout,
            GPUM_ERR_BACKEND => Status::Backend,
            GPUM_ERR_ABI => Status::Abi,
            GPUM_ERR_INTERNAL => Status::Internal,
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Status::Ok => "ok",
            Status::InvalidArg => "invalid argument",
            Status::NotFound => "not found",
            Status::Unsupported => "unsupported",
            Status::NotInitialized => "not initialized",
            Status::AlreadyExists => "already exists",
            Status::NoData => "no data",
            Status::Timeout => "timeout",
            Status::Backend => "backend error",
            Status::Abi => "plugin ABI error",
            Status::Internal => "internal error",
            Status::Unknown(code) => return write!(f, "unknown status ({code})"),
        };
        f.write_str(s)
    }
}

impl std::error::Error for Status {}

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------

/// A typed metric value, decoded from a `gpum_value` using its type tag.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 64-bit float.
    F64(f64),
    /// UTF-8 string (lossily decoded from a fixed C buffer).
    Str(String),
}

impl Value {
    /// Interprets a raw `gpum_value` according to its `type` tag.
    ///
    /// # Safety
    /// `raw` must be a fully-initialized value whose `type_` correctly describes
    /// which union arm / buffer is live (as guaranteed by the collector).
    unsafe fn from_raw(raw: &sys::gpum_value) -> Value {
        match raw.type_ {
            sys::gpum_value_type::GPUM_TYPE_U64 => Value::U64(raw.__bindgen_anon_1.u64_),
            sys::gpum_value_type::GPUM_TYPE_I64 => Value::I64(raw.__bindgen_anon_1.i64_),
            sys::gpum_value_type::GPUM_TYPE_F64 => Value::F64(raw.__bindgen_anon_1.f64_),
            sys::gpum_value_type::GPUM_TYPE_STRING => {
                Value::Str(cstr_to_string(raw.str_.as_ptr(), raw.str_.len()))
            }
        }
    }

    /// Returns the value as an `f64` if numeric (integers are widened).
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::U64(v) => Some(v as f64),
            Value::I64(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            Value::Str(_) => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Str(v) => f.write_str(v),
        }
    }
}

/// The kind of value a metric produces, mirroring `gpum_value_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// 64-bit float.
    F64,
    /// UTF-8 string.
    Str,
}

impl ValueType {
    fn from_raw(t: sys::gpum_value_type) -> ValueType {
        match t {
            sys::gpum_value_type::GPUM_TYPE_U64 => ValueType::U64,
            sys::gpum_value_type::GPUM_TYPE_I64 => ValueType::I64,
            sys::gpum_value_type::GPUM_TYPE_F64 => ValueType::F64,
            sys::gpum_value_type::GPUM_TYPE_STRING => ValueType::Str,
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ValueType::U64 => "u64",
            ValueType::I64 => "i64",
            ValueType::F64 => "f64",
            ValueType::Str => "str",
        })
    }
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------

/// The outcome of reading one metric on one entity.
///
/// A read that the backend cannot serve (e.g. [`Status::Unsupported`] or
/// [`Status::NotFound`]) is *not* an error: it comes back as a `Sample` whose
/// [`status`](Sample::status) is non-OK and whose [`value`](Sample::value) is
/// `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// Per-sample status. Only [`Status::Ok`] carries a valid value.
    pub status: Status,
    /// The decoded value, present only when `status` is OK.
    pub value: Option<Value>,
    /// Backend-provided timestamp in nanoseconds.
    pub timestamp_ns: u64,
}

impl Sample {
    /// Returns `true` if the sample completed successfully.
    pub fn is_ok(&self) -> bool {
        self.status.is_ok()
    }

    /// Returns the decoded value, if the read succeeded.
    pub fn value(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    fn err(status: Status) -> Sample {
        Sample {
            status,
            value: None,
            timestamp_ns: 0,
        }
    }

    /// # Safety
    /// `raw` must be a fully-initialized sample as written by the collector.
    unsafe fn from_raw(raw: &sys::gpum_sample) -> Sample {
        let status = Status::from_raw(raw.status);
        let value = if status.is_ok() {
            Some(Value::from_raw(&raw.value))
        } else {
            None
        };
        Sample {
            status,
            value,
            timestamp_ns: raw.timestamp_ns,
        }
    }
}

// ---------------------------------------------------------------------------
// Entity / topology
// ---------------------------------------------------------------------------

/// Which kind of addressable entity an id refers to, mirroring `gpum_entity_kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    /// A socket (group of GPUs).
    Socket,
    /// A whole physical GPU.
    Gpu,
    /// One partition instance of a GPU.
    Partition,
}

impl EntityKind {
    fn from_raw(k: sys::gpum_entity_kind) -> EntityKind {
        match k {
            sys::gpum_entity_kind::GPUM_ENTITY_SOCKET => EntityKind::Socket,
            sys::gpum_entity_kind::GPUM_ENTITY_GPU => EntityKind::Gpu,
            sys::gpum_entity_kind::GPUM_ENTITY_GPU_PARTITION => EntityKind::Partition,
        }
    }
}

/// A canonical, core-assigned identity of an addressable entity
/// (`gpum_entity_id`). Obtain one via [`Collector::resolve`] or
/// [`Collector::gpu_entity`], then pass it to [`Collector::read`].
#[derive(Debug, Clone, Copy)]
pub struct EntityId {
    raw: sys::gpum_entity_id,
}

impl PartialEq for EntityId {
    fn eq(&self, other: &Self) -> bool {
        self.raw.kind as u32 == other.raw.kind as u32
            && self.raw.socket == other.raw.socket
            && self.raw.gpu == other.raw.gpu
            && self.raw.partition == other.raw.partition
    }
}

impl Eq for EntityId {}

impl std::hash::Hash for EntityId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.raw.kind as u32).hash(state);
        self.raw.socket.hash(state);
        self.raw.gpu.hash(state);
        self.raw.partition.hash(state);
    }
}

impl EntityId {
    /// The kind of entity this id addresses.
    pub fn kind(&self) -> EntityKind {
        EntityKind::from_raw(self.raw.kind)
    }

    /// The socket ordinal.
    pub fn socket(&self) -> u32 {
        self.raw.socket
    }

    /// The canonical physical-GPU ordinal (valid for GPU / partition kinds).
    pub fn gpu(&self) -> u32 {
        self.raw.gpu
    }

    /// The partition index, or `None` for a whole GPU / socket.
    pub fn partition(&self) -> Option<u32> {
        if self.raw.partition < 0 {
            None
        } else {
            Some(self.raw.partition as u32)
        }
    }
}

/// The scope bitmask describing which entity kinds a metric is meaningful for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scope(pub u32);

impl Scope {
    /// Whether the metric applies at socket scope.
    pub fn socket(self) -> bool {
        self.0 & sys::gpum_scope_flags::GPUM_SCOPE_SOCKET as u32 != 0
    }
    /// Whether the metric applies at whole-GPU scope.
    pub fn gpu(self) -> bool {
        self.0 & sys::gpum_scope_flags::GPUM_SCOPE_GPU as u32 != 0
    }
    /// Whether the metric applies at partition scope.
    pub fn partition(self) -> bool {
        self.0 & sys::gpum_scope_flags::GPUM_SCOPE_PARTITION as u32 != 0
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.socket() {
            parts.push("socket");
        }
        if self.gpu() {
            parts.push("gpu");
        }
        if self.partition() {
            parts.push("partition");
        }
        f.write_str(&parts.join("|"))
    }
}

/// Description of a GPU discovered by the collector.
#[derive(Debug, Clone)]
pub struct Gpu {
    ordinal: u32,
    name: String,
    identity: sys::gpum_device_identity,
    partitions: Vec<i32>,
}

impl Gpu {
    /// The canonical GPU ordinal (0-based).
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// The human-readable device name (marketing name).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The packed PCIe BDF value.
    pub fn bdf_raw(&self) -> u64 {
        self.identity.bdf
    }

    /// The PCIe BDF formatted as `DDDD:BB:DD.F`, or an empty string if unknown.
    pub fn bdf(&self) -> String {
        format_bdf(self.identity.bdf)
    }

    /// The KFD topology node id (0 if the backend did not provide it).
    pub fn kfd_node_id(&self) -> u32 {
        self.identity.kfd_node_id
    }

    /// The socket id this GPU belongs to.
    pub fn socket_id(&self) -> u32 {
        self.identity.socket_id
    }

    /// The GPU UUID formatted as hex, or `None` if all-zero (unknown).
    pub fn uuid(&self) -> Option<String> {
        if self.identity.uuid.iter().all(|&b| b == 0) {
            return None;
        }
        let mut s = String::with_capacity(32);
        for b in self.identity.uuid {
            s.push_str(&format!("{b:02x}"));
        }
        Some(s)
    }

    /// The partition indices exposed by this GPU (empty for an unpartitioned GPU).
    pub fn partitions(&self) -> &[i32] {
        &self.partitions
    }
}

/// A socket and the GPUs it contains.
#[derive(Debug, Clone)]
pub struct Socket {
    /// The socket ordinal.
    pub index: u32,
    /// Ordinals of the GPUs grouped under this socket.
    pub gpus: Vec<u32>,
}

/// Description of a metric the loaded plugins can provide.
#[derive(Debug, Clone)]
pub struct MetricDesc {
    /// The metric key, e.g. `"temp.edge"`.
    pub key: String,
    /// The unit string, e.g. `"C"`, `"W"`, `"MHz"`, or empty.
    pub unit: String,
    /// The name of the plugin/provider that won this metric.
    pub provider: String,
    /// The value type produced by this metric.
    pub value_type: ValueType,
    /// The entity scopes this metric applies to.
    pub scope: Scope,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for a [`Collector`] with non-default options.
///
/// ```no_run
/// use gpumetrics::Collector;
/// let collector = Collector::builder()
///     .plugin_path("/opt/gpumetrics/plugins")
///     .plugin("amdsmi")
///     .build()?;
/// # Ok::<(), gpumetrics::Status>(())
/// ```
#[derive(Debug, Default, Clone)]
pub struct Builder {
    plugin_paths: Vec<String>,
    plugins: Vec<String>,
    provider_priority: Vec<String>,
    read_timeout_ms: u32,
}

impl Builder {
    /// Creates a builder with all-default options.
    pub fn new() -> Builder {
        Builder::default()
    }

    /// Adds an extra plugin search directory.
    pub fn plugin_path(mut self, path: impl Into<String>) -> Builder {
        self.plugin_paths.push(path.into());
        self
    }

    /// Adds several plugin search directories.
    pub fn plugin_paths<I, S>(mut self, paths: I) -> Builder
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.plugin_paths.extend(paths.into_iter().map(Into::into));
        self
    }

    /// Restricts loading to the named plugin (by name or filename). May be
    /// called repeatedly.
    pub fn plugin(mut self, name: impl Into<String>) -> Builder {
        self.plugins.push(name.into());
        self
    }

    /// Restricts loading to the given set of plugins.
    pub fn plugins<I, S>(mut self, names: I) -> Builder
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.plugins.extend(names.into_iter().map(Into::into));
        self
    }

    /// Sets the provider conflict-resolution priority order.
    pub fn provider_priority<I, S>(mut self, names: I) -> Builder
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.provider_priority
            .extend(names.into_iter().map(Into::into));
        self
    }

    /// Sets the per-read timeout in milliseconds (0 = backend default).
    pub fn read_timeout_ms(mut self, ms: u32) -> Builder {
        self.read_timeout_ms = ms;
        self
    }

    /// Creates the collector, consuming the builder.
    pub fn build(self) -> Result<Collector, Status> {
        // Own the CStrings for the duration of the create call; the C side only
        // borrows the arrays.
        let paths = to_cstrings(&self.plugin_paths);
        let plugins = to_cstrings(&self.plugins);
        let prio = to_cstrings(&self.provider_priority);

        let paths_ptrs = to_ptr_vec(&paths);
        let plugins_ptrs = to_ptr_vec(&plugins);
        let prio_ptrs = to_ptr_vec(&prio);

        let opts = sys::gpum_collector_options {
            plugin_paths: ptr_or_null(&paths_ptrs),
            plugin_paths_count: paths_ptrs.len() as u32,
            plugins: ptr_or_null(&plugins_ptrs),
            plugins_count: plugins_ptrs.len() as u32,
            provider_priority: ptr_or_null(&prio_ptrs),
            provider_priority_count: prio_ptrs.len() as u32,
            read_timeout_ms: self.read_timeout_ms,
        };

        let mut status = sys::gpum_status::GPUM_OK;
        // SAFETY: opts and all borrowed arrays live until this call returns.
        let handle = unsafe { sys::gpum_collector_create(&opts, &mut status) };
        if handle.is_null() {
            return Err(Status::from_raw(status));
        }
        Ok(Collector { handle })
    }
}

// ---------------------------------------------------------------------------
// Collector
// ---------------------------------------------------------------------------

/// A GPU-metrics collector: an owned handle over `gpum_collector`.
///
/// Created with [`Collector::new`] (default plugins) or [`Collector::builder`].
/// The underlying handle is destroyed automatically on drop.
pub struct Collector {
    handle: *mut sys::gpum_collector,
}

// The C collector is a self-contained handle with no thread-affinity, and this
// wrapper only ever hands out `&self`/`&mut self` access to it, so it is safe to
// move between threads.
unsafe impl Send for Collector {}

impl Collector {
    /// Creates a collector with default options (auto-discovered plugins).
    pub fn new() -> Result<Collector, Status> {
        Builder::new().build()
    }

    /// Returns a [`Builder`] for configuring a collector.
    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Number of physical GPUs discovered.
    pub fn gpu_count(&self) -> u32 {
        // SAFETY: handle is valid for the lifetime of `self`.
        unsafe { sys::gpum_collector_gpu_count(self.handle) }
    }

    /// Number of sockets discovered.
    pub fn socket_count(&self) -> u32 {
        // SAFETY: handle is valid for the lifetime of `self`.
        unsafe { sys::gpum_collector_socket_count(self.handle) }
    }

    /// Returns the canonical entity id for the GPU at ordinal `gpu`, or `None`
    /// if out of range.
    pub fn gpu_entity(&self, gpu: u32) -> Option<EntityId> {
        let mut raw = zeroed_entity();
        // SAFETY: out-param is valid; handle is valid.
        let st = unsafe { sys::gpum_collector_gpu_entity(self.handle, gpu, &mut raw) };
        match Status::from_raw(st) {
            Status::Ok => Some(EntityId { raw }),
            _ => None,
        }
    }

    /// Describes the GPU at ordinal `gpu`, or `None` if out of range.
    pub fn gpu(&self, gpu: u32) -> Option<Gpu> {
        let mut name = [0 as c_char; STRING_MAX];
        let mut identity = zeroed_identity();
        let mut partition_count: u32 = 0;
        // SAFETY: all out-params are valid and sized; handle is valid.
        let st = unsafe {
            sys::gpum_collector_gpu_info(
                self.handle,
                gpu,
                name.as_mut_ptr(),
                name.len(),
                &mut identity,
                &mut partition_count,
            )
        };
        if !Status::from_raw(st).is_ok() {
            return None;
        }
        let partitions = self.gpu_partitions(gpu, partition_count);
        Some(Gpu {
            ordinal: gpu,
            name: cstr_to_string(name.as_ptr(), name.len()),
            identity,
            partitions,
        })
    }

    /// All discovered GPUs, in ordinal order.
    pub fn gpus(&self) -> Vec<Gpu> {
        (0..self.gpu_count()).filter_map(|i| self.gpu(i)).collect()
    }

    fn gpu_partitions(&self, gpu: u32, hint: u32) -> Vec<i32> {
        if hint == 0 {
            return Vec::new();
        }
        let mut buf = vec![0i32; hint as usize];
        let mut count: u32 = 0;
        // SAFETY: buf has `hint` capacity; out_count is valid.
        let st = unsafe {
            sys::gpum_collector_gpu_partitions(
                self.handle,
                gpu,
                buf.as_mut_ptr(),
                buf.len() as u32,
                &mut count,
            )
        };
        if !Status::from_raw(st).is_ok() {
            return Vec::new();
        }
        buf.truncate((count as usize).min(buf.len()));
        buf
    }

    /// The sockets and the GPUs grouped under each.
    pub fn sockets(&self) -> Vec<Socket> {
        let gpus = self.gpus();
        let mut sockets: Vec<Socket> = Vec::new();
        for g in &gpus {
            let sid = g.socket_id();
            if let Some(s) = sockets.iter_mut().find(|s| s.index == sid) {
                s.gpus.push(g.ordinal());
            } else {
                sockets.push(Socket {
                    index: sid,
                    gpus: vec![g.ordinal()],
                });
            }
        }
        sockets.sort_by_key(|s| s.index);
        sockets
    }

    /// Number of metrics in the registry.
    pub fn metric_count(&self) -> u32 {
        // SAFETY: handle is valid.
        unsafe { sys::gpum_collector_metric_count(self.handle) }
    }

    /// All metric descriptors offered by the loaded plugins.
    pub fn metrics(&self) -> Vec<MetricDesc> {
        (0..self.metric_count())
            .filter_map(|i| self.metric_at(i))
            .collect()
    }

    /// Describes the metric at registry index `i`, or `None` if out of range.
    pub fn metric_at(&self, i: u32) -> Option<MetricDesc> {
        let mut key = [0 as c_char; STRING_MAX];
        let mut unit = [0 as c_char; 32];
        let mut provider = [0 as c_char; STRING_MAX];
        let mut value_type = sys::gpum_value_type::GPUM_TYPE_U64;
        let mut scope: u32 = 0;
        // SAFETY: all buffers are sized by their `.len()`; handle is valid.
        let st = unsafe {
            sys::gpum_collector_metric_at(
                self.handle,
                i,
                key.as_mut_ptr(),
                key.len(),
                unit.as_mut_ptr(),
                unit.len(),
                provider.as_mut_ptr(),
                provider.len(),
                &mut value_type,
                &mut scope,
            )
        };
        if !Status::from_raw(st).is_ok() {
            return None;
        }
        Some(MetricDesc {
            key: cstr_to_string(key.as_ptr(), key.len()),
            unit: cstr_to_string(unit.as_ptr(), unit.len()),
            provider: cstr_to_string(provider.as_ptr(), provider.len()),
            value_type: ValueType::from_raw(value_type),
            scope: Scope(scope),
        })
    }

    /// Resolves a selector string (`"gpu:0"`, `"g0.1"`, `"socket:1"`,
    /// `"bdf:..."`, `"uuid:..."`) into an [`EntityId`], or `None` if unresolved.
    pub fn resolve(&self, selector: &str) -> Option<EntityId> {
        let c = CString::new(selector).ok()?;
        let mut raw = zeroed_entity();
        // SAFETY: selector is a valid NUL-terminated string; out-param is valid.
        let st = unsafe { sys::gpum_collector_resolve(self.handle, c.as_ptr(), &mut raw) };
        match Status::from_raw(st) {
            Status::Ok => Some(EntityId { raw }),
            _ => None,
        }
    }

    /// Reads a single metric `key` on `entity`.
    ///
    /// A backend that cannot serve the read reports it via the returned
    /// [`Sample`]'s status; only a hard failure of the call itself (e.g. an
    /// invalid key string) yields an error-status `Sample`.
    pub fn read(&self, entity: &EntityId, key: &str) -> Sample {
        let c = match CString::new(key) {
            Ok(c) => c,
            Err(_) => return Sample::err(Status::InvalidArg),
        };
        let mut raw = zeroed_sample();
        // SAFETY: entity and key are valid; out_sample is valid.
        let st =
            unsafe { sys::gpum_collector_read(self.handle, &entity.raw, c.as_ptr(), &mut raw) };
        // If the call itself failed, surface that status; otherwise decode the
        // per-sample status the plugin wrote.
        match Status::from_raw(st) {
            Status::Ok => unsafe { Sample::from_raw(&raw) },
            other => Sample::err(other),
        }
    }

    /// Batch-reads several metric `keys` on a single `entity`. The returned
    /// vector matches `keys` positionally.
    pub fn read_batch(&self, entity: &EntityId, keys: &[&str]) -> Vec<Sample> {
        if keys.is_empty() {
            return Vec::new();
        }
        // Own CStrings, then build a borrowed pointer array for the C call.
        let cstrings: Vec<CString> = keys
            .iter()
            .map(|k| CString::new(*k).unwrap_or_default())
            .collect();
        let ptrs: Vec<*const c_char> = cstrings.iter().map(|c| c.as_ptr()).collect();
        let mut out = vec![zeroed_sample(); keys.len()];

        // SAFETY: ptrs/out both have `keys.len()` entries; entity is valid.
        let st = unsafe {
            sys::gpum_collector_read_batch(
                self.handle,
                &entity.raw,
                ptrs.as_ptr(),
                ptrs.len() as u32,
                out.as_mut_ptr(),
            )
        };
        match Status::from_raw(st) {
            Status::Ok => out.iter().map(|s| unsafe { Sample::from_raw(s) }).collect(),
            other => (0..keys.len()).map(|_| Sample::err(other)).collect(),
        }
    }
}

impl Drop for Collector {
    fn drop(&mut self) {
        // SAFETY: handle came from gpum_collector_create and is destroyed once.
        unsafe { sys::gpum_collector_destroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn zeroed_entity() -> sys::gpum_entity_id {
    sys::gpum_entity_id {
        kind: sys::gpum_entity_kind::GPUM_ENTITY_GPU,
        socket: 0,
        gpu: 0,
        partition: -1,
    }
}

fn zeroed_identity() -> sys::gpum_device_identity {
    // SAFETY: gpum_device_identity is a plain-old-data struct; all-zero is a
    // valid bit pattern ("field not provided").
    unsafe { std::mem::zeroed() }
}

fn zeroed_sample() -> sys::gpum_sample {
    // SAFETY: gpum_sample is POD; all-zero is a valid, if uninteresting, value
    // (status = GPUM_OK, type = U64, value = 0). The collector overwrites it.
    unsafe { std::mem::zeroed() }
}

fn to_cstrings(v: &[String]) -> Vec<CString> {
    v.iter()
        .map(|s| CString::new(s.as_str()).unwrap_or_default())
        .collect()
}

fn to_ptr_vec(v: &[CString]) -> Vec<*const c_char> {
    v.iter().map(|c| c.as_ptr()).collect()
}

fn ptr_or_null(v: &[*const c_char]) -> *const *const c_char {
    if v.is_empty() {
        ptr::null()
    } else {
        v.as_ptr()
    }
}

/// Decodes a NUL-terminated C string held in a fixed buffer of `cap` chars into
/// an owned `String` (lossily for non-UTF-8).
fn cstr_to_string(ptr: *const c_char, cap: usize) -> String {
    if ptr.is_null() || cap == 0 {
        return String::new();
    }
    // SAFETY: the buffer is `cap` chars and NUL-terminated by the C side; we
    // bound the scan by `cap` in case it is not.
    unsafe {
        let bytes = std::slice::from_raw_parts(ptr as *const u8, cap);
        let len = bytes.iter().position(|&b| b == 0).unwrap_or(cap);
        String::from_utf8_lossy(&bytes[..len]).into_owned()
    }
}

/// Formats a packed PCIe BDF as `DDDD:BB:DD.F`, matching the CLI. Returns an
/// empty string for a zero (unknown) BDF.
pub fn format_bdf(bdf: u64) -> String {
    if bdf == 0 {
        return String::new();
    }
    let domain = (bdf >> 16) as u32;
    let bus = ((bdf >> 8) & 0xff) as u32;
    let dev = ((bdf >> 3) & 0x1f) as u32;
    let func = (bdf & 0x7) as u32;
    format!("{domain:04x}:{bus:02x}:{dev:02x}.{func:x}")
}
